"""Utility functions for span labeling on the NLU+ dataset.
"""

import itertools

from collections import defaultdict
from sklearn import metrics


def tokenise_annotate_and_convert_slot_labels_to_bio_tags(data, nlp_analyser):
    """Tokenises and annotates NLU++ data, and annotates slots using BIO tags.

    - Tokenises and analyses using spacy, storing the result under a new
      'annotated_text' key as a spacy Doc object.
    - Converts span annotations on slots into BIO tags on tokens, storing tag
      under the attribute 'bio_slot_label' for each spacy Token object, which
      must default to 'O'.

    Args:
        data (dict): raw data in NLU++ JSON format
        nlp_analyser (spacy.lang.<Language>): spacy model
    """
    for sample in data:
        if 'text' in sample:
            sample['annotated_text'] = nlp_analyser(sample['text'])
        if 'slots' in sample:
            for slot in sample['slots']:
                if 'span' in sample['slots'][slot]:
                    (span_start, span_end) = sample['slots'][slot]['span']
                    for token in sample['annotated_text']:
                        if token.idx < span_start:
                            continue
                        if token.idx >= span_end:
                            break
                        if token.idx == span_start:
                            token._.bio_slot_label = f'B-{slot}'
                        else:
                            token._.bio_slot_label = f'I-{slot}'


def _extract_spans(tag_sequence):
    """Return a list of labeled spans from a tagged sequence.

    Args:
        tag_sequence: List(str): a list of tags in BIO format
    Returns:
        Bool: True if input BIO tag sequence is well-formed, False otherwise
        Dict(Tuple((int,int),str)): a dictionary that maps spans to their
          labels. If the first return value is False, this list is not
          guaranteed to be correct. The dictionary returns '<not_a_span>'
          if queried with a non-existent key.
    """

    # default return values
    wellformed_bio_tags = True
    spans = defaultdict(lambda: '<not_a_span>')

    # traverse the BIO tag sequence from left to right, extracting spans as we
    # go. in_label keeps track of whether we are currently inside a labeled
    # span, span_start keeps track of the beginning of that span, and label
    # keeps track of its identity. Whenever we are in a span, and we encounter
    # a B or an O, we record that as the end of the span. Whenever we
    # encounter a B, we record the beginning of a new span. Whenever we
    # encounter an I, we check that we have already in a label of that type;
    # otherwise, the BIO tags are not well-formed.
    label = ''
    in_label = False
    for idx, tag in enumerate(tag_sequence + ['O']):
        if tag.startswith('B'):
            if in_label:
                spans[(span_start, idx)] = label
            in_label = True
            label = tag[2:]
            span_start = idx
        elif tag.startswith('I'):
            if not (in_label and label == tag[2:]):
                wellformed_bio_tags = False
        elif tag == 'O':
            if in_label:
                spans[(span_start, idx)] = label
                in_label = False
        else:
            wellformed_bio_tags = False

    return wellformed_bio_tags, spans


def evaluate(predictions, validation_data):
    """Print classification report for predicted BIO tags and slot labels.

    The slot labeling report is printed  only if predicted BIO tags are
    well-formed. Otherwise, an error message is printed.

    Args:
        predictions (List(List(str))): predictions for each sample, each
          consisting of one tag per word
        validation_data: data in NLU++ format imported from JSON, with
          annotated spans and corresponding BIO tags
    """

    # To build a report, we need to know annotations and predictions.
    # These must match one-for-one: the i-th prediction must correspond
    # to the i-th annotation. For tags, this is straightforward: each
    # word has an annotation and a prediction, so we will record them
    # in order in these two variables.
    annotated_tags = []
    predicted_tags = []

    # For spans, there is no one-to-one correspondence. What we will do
    # is to first construct a list of all spans in a sentence that have
    # either an annotation or a prediction. Then we'll traverse that list of
    # and record both the annotation and prediction; if either is missing,
    # it will receive a label of '<not_a_span>' for that value. The final
    # report will compute precision, recall, and f-measure for all labels
    # except '<not_a_span>'
    annotated_spans = []
    predicted_spans = []

    # We will also collect the full set of possible labels in this variable
    span_labels = set()

    corpus_level_well_formed_bio_tags = True
    for sample, prediction in zip(validation_data, predictions):
        # Collect annotated tags and spans
        sample_tags = [token._.bio_slot_label for
                       token in sample['annotated_text']]
        annotated_tags.extend(sample_tags)
        _, sample_spans = _extract_spans(sample_tags)

        # Collect predicted tags and spans
        predicted_tags.extend(prediction)
        well_formed_bio_tags, prediction_spans = _extract_spans(prediction)
        corpus_level_well_formed_bio_tags &= well_formed_bio_tags

        # Construct the set of all spans
        labeled_spans = set(itertools.chain(sample_spans.keys(),
                                            prediction_spans.keys()))

        # Update the set of sampled spans and span labels
        for span in labeled_spans:
            annotated_spans.append(sample_spans[span])
            predicted_spans.append(prediction_spans[span])
            span_labels.add(sample_spans[span])
            span_labels.add(prediction_spans[span])

    span_labels -= set(['<not_a_span>'])
    span_labels = sorted(span_labels)
    tag_labels = sorted([f'B-{label}' for label in span_labels] +
                        [f'I-{label}' for label in span_labels])

    print()
    print('Classification report for BIO tags:')
    print()
    report = metrics.classification_report(annotated_tags,
                                           predicted_tags,
                                           labels=tag_labels,
                                           zero_division=0)
    print(report)

    if corpus_level_well_formed_bio_tags:
        print()
        print('Classification report for slot labels:')
        print()
        report = metrics.classification_report(annotated_spans,
                                               predicted_spans,
                                               labels=span_labels,
                                               zero_division=0)
        print(report)
        print()
    else:
        print('!!! Classification report for slots is unavailable because' + \
              'some predicted BIO tag sequences were malformed !!!')


def visualise_bio_tags(predictions, validation_data, attributes, col_widths):
    """Print token-level predictions and annotations.

    Args:
        predictions (List(List(str))): predictions for each sample, each
          consisting of one tag per word
        validation_data: data in NLU++ format imported from JSON, with
          annotated spans and corresponding BIO tags
        attributes: List[str]: list of spacy Token attributes to print for
          each word
        col_widths: List[int]: list containing column widths for each
          attribute to print. Must be same length as attributes
    """
    for sample, prediction, in zip(validation_data, predictions):
        print(f'Input: {sample["text"]}')
        print()

        print('Analysis:                                                     ')
        if attributes and len(attributes) > 0:
            print('Additional annotations:')

        header = 'idx word            BIO annotation  BIO prediction  correct?  '
        if attributes and len(attributes) > 0:
            for attribute, col_width in zip(attributes, col_widths):
                header += f'{attribute:<{col_width}}'
        print(header)

        hrule = '------------------------------------------------------------  '
        if attributes and len(attributes) > 0:
            hrule += '-'*sum(col_widths)
        print(hrule)

        for idx, (token, predicted_label) in \
            enumerate(zip(sample['annotated_text'], prediction)):

            match = '  ✔     ' if token._.bio_slot_label == predicted_label \
               else '     ✗  '

            token_line = f'{idx:<3} ' + \
                         f'{str(token):<15} ' + \
                         f'{token._.bio_slot_label:<15} ' + \
                         f'{predicted_label:<15} ' + \
                         f'{match:<9} '
            for attribute, col_width in zip(attributes, col_widths):
                token_line += f'{getattr(token, f"{attribute}"):<{col_width}}'
            print(token_line)

        print()
        _, annotated_spans = _extract_spans([token._.bio_slot_label for
                                             token in sample['annotated_text']])
        well_formed_bio_tags, predicted_spans = _extract_spans(prediction)

        if not well_formed_bio_tags:
            print('!!! Predicted BIO tag sequence is malformed. ' +
                  'Some predicted spans were not extracted correctly !!!')
            print()

        print('annotated spans:')
        for span, label in annotated_spans.items():
            match = ' '
            if span in predicted_spans and predicted_spans[span] == label:
                match = '✔'
            print(f'{match} {span}: {label:<15}')
        print()

        warning = '' if well_formed_bio_tags else \
                  ' (incorrectly extracted from BIO tags)'
        print(f'predicted spans{warning}:')
        for span, label in predicted_spans.items():
            match = ' '
            if span in annotated_spans and annotated_spans[span] == label:
                match = '✔'
            print(f'{match} {span}: {label}')
        print()

        hrule = '==============================================================' 
        if attributes and len(attributes) > 0:
            hrule += '='*sum(col_widths)
        print(hrule)
        print()
