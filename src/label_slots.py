"""ANLP coursework 2: Executable module for span labeling on NLU++"""

import argparse
import json
import math
import os
from collections import defaultdict
import numpy as np

import sklearn.linear_model
import sklearn.preprocessing
import spacy

import utils

# Default data files
DATA_DIR = "data"
TRAINING_FILE = "training_data.json"
VALIDATION_FILE = "validation_data.json"

spacy.tokens.token.Token.set_extension("bio_slot_label", default="O")



def _predict_tag_mle(token, model_parameters):
    """Predict most-frequent-tag baseline.

    Args:
        token (spacy.tokens.Token): token whose tag we want to predict
        model_parameters (dict): a mapping token from most frequent tag, and
          an unknown word tag

    Returns:
        (List(Tuple(str,float))): a list of (tag, logprob) pairs, sorted from
          most to least probable
    """
    if token.text in model_parameters['per_word_tag_mle']:
        return model_parameters['per_word_tag_mle'][token.text]
    return model_parameters['unknown_word_distribution']


def train_tag_mle(data):
    """Train a model p(<tag> | <word>).

    Args:
        data: data in NLU++ format imported from JSON. Each data sample
          should include spacy annotations in the variable 'annotated_text',
          and each token in the annotated text should include a 'bio_slot_label'.

    Returns:
        Callable: a function from a word to a sorted distribution over tags.
    """
    token_count = 0

    # Count the number of times each word/tag combination appears
    # and store them so that word_tag_count[word][tag] is the count
    word_tag_count = defaultdict(lambda: defaultdict(int))
    for sample in data:
        for token in sample['annotated_text']:
            word_tag_count[token.text][token._.bio_slot_label] += 1
            token_count += 1

    # For each word, normalise the associated tag counts to obtain
    # maximum likelihood estimate of p(<tag> | <word>), and store the
    # logarithm of the probability. For each word, sort its possible tags
    # by their logprobs
    per_word_tag_mle = {}
    for word, tag_count_pairs in word_tag_count.items():
        word_count = sum(tag_count_pairs.values())
        tag_logprob_pairs = [(tag, math.log(tag_word_count / word_count)) for
                             (tag, tag_word_count) in tag_count_pairs.items()]
        sorted_tag_logprob_pairs = \
            sorted(tag_logprob_pairs,
                   key=lambda tag_logprob_pair: -tag_logprob_pair[1])
        per_word_tag_mle[word] = sorted_tag_logprob_pairs

    # The model maps each word to its sorted table of logprobs. Return a
    # function that takes a word as input, and returns this sorted table.
    # If the word is unknown, return a distribution where p('O'|<word>) == 1
    model_parameters = {
        'per_word_tag_mle': per_word_tag_mle,
        'unknown_word_distribution': [('O', 1)]
    }

    print(f'> Trained most frequent tag model on {token_count} tokens')
    return lambda token: _predict_tag_mle(token, model_parameters)



def _predict_tag_logreg(token, model, tag_encoder):
    """Predict tag according to logistic regression on word embedding.

    Args:
        token: a spacy token to tag
        model: an sklearn model that predicts tags from token vectors
        tags: a mapping from tag ids to BIO tag strings

    Returns:
        (List(Tuple(str,float))): distribution over tags, sorted from most to
          least probable
    """
    log_probs = model.predict_log_proba([token.vector])[0]
    distribution = list(zip(tag_encoder.classes_, log_probs))
    sorted_distribution = \
        sorted(distribution, key=lambda tag_logprob_pair: -tag_logprob_pair[1])
    return sorted_distribution


def train_tag_logreg(data):
    """Train a logistic regression model for p(<tag> | <word embedding>).

    Args:
        data: data in NLU++ format imported from JSON. Each data sample
          should include spacy annotations in the variable 'annotated_text',
          and each token in the annotated text should include a 'bio_slot_label'.

    Returns:
        Callable: a function that returns a (sorted) distribution over tags,
          given a word.
    """

    train_X = [token.vector for
               sample in data for
               token in sample['annotated_text']]

    train_y = [token._.bio_slot_label for
               sample in data for
               token in sample['annotated_text']]

    # train_X is already numeric, but train_y is strings. We must encode train_y
    # numerically in order to train the sklearn logistic regression model.
    tag_encoder = sklearn.preprocessing.LabelEncoder()
    train_y_encoded = tag_encoder.fit_transform(train_y)

    model = sklearn.linear_model.LogisticRegression(multi_class='multinomial',
                                                    solver='newton-cg' \
                                                   ).fit(train_X,
                                                         train_y_encoded)

    print(f'> Finished training logistic regression on {len(train_X)} tokens')
    return lambda token: _predict_tag_logreg(token, model, tag_encoder)

def get_left_token(generator, current):
    tokens = list(generator)
    if tokens:
        return tokens[0].vector
    else:
        return np.zeros(len(current))
    
def get_right_token(generator, current):
    tokens = list(generator)
    if tokens:
        return tokens[0].vector
    else:
        return np.zeros(len(current))
    
def get_two_left_token(generator, current):
    tokens = list(generator)
    if len(tokens) > 1:
        return np.concatenate((tokens[0].vector,tokens[1].vector))
    elif len(tokens) == 1:
        return np.concatenate((tokens[0].vector, np.zeros(len(current))))
    else:
        return np.zeros(2*len(current))
    
def get_two_right_token(generator, current):
    tokens = list(generator)
    if len(tokens) > 1:
        return np.concatenate((tokens[0].vector,tokens[1].vector))
    elif len(tokens) == 1:
        return np.concatenate((tokens[0].vector, np.zeros(len(current))))
    else:
        return np.zeros(2*len(current))
    
def get_three_left_token(generator, current):
    tokens = list(generator)
    if len(tokens) > 2:
        return np.concatenate((tokens[0].vector, tokens[1].vector, tokens[2].vector))
    elif len(tokens) > 1:
        return np.concatenate((tokens[0].vector, tokens[1].vecgtor, np.zeros(len(current))))
    elif len(tokens):
        return np.concatenate((tokens[0].vector, np.zeros(2*len(current))))
    else:
        return np.zeros(3*len(current))
    
def get_three_left_token(generator, current):
    tokens = list(generator)
    if len(tokens) > 2:
        return np.concatenate((tokens[0].vector, tokens[1].vector, tokens[2].vector))
    elif len(tokens) > 1:
        return np.concatenate((tokens[0].vector, tokens[1].vecgtor, np.zeros(len(current))))
    elif len(tokens):
        return np.concatenate((tokens[0].vector, np.zeros(2*len(current))))
    else:
        return np.zeros(3*len(current))

def _predict_tag_mymodel(token, model, tag_encoder, encoder):
    """Predict tag according to logistic regression on word embedding.

    Args:
        token: a spacy token to tag
        model: an sklearn model that predicts tags from token vectors
        tags: a mapping from tag ids to BIO tag strings

    Returns:
        (List(Tuple(str,float))): distribution over tags, sorted from most to
          least probable
    """
    ent_type = np.array([token.ent_type_]).reshape(1, -1)
    log_probs = model.predict_log_proba([np.concatenate((token.vector, token.head.vector, encoder.transform(ent_type).flatten()))])[0]
    distribution = list(zip(tag_encoder.classes_, log_probs))
    sorted_distribution = \
        sorted(distribution, key=lambda tag_logprob_pair: -tag_logprob_pair[1])
    return sorted_distribution

def train_my_model(data):
    """Train a logistic regression model for p(<tag> | <word embedding>).

    Args:
        data: data in NLU++ format imported from JSON. Each data sample
          should include spacy annotations in the variable 'annotated_text',
          and each token in the annotated text should include a 'bio_slot_label'.

    Returns:
        Callable: a function that returns a (sorted) distribution over tags,
          given a word.
    """

    train_X = []
    train_y = []
    ent_type_list = \
    [["", "CARDINAL", 
        "DATE", 
        "EVENT", 
        "FAC", 
        "GPE", 
        "LANGUAGE", 
        "LAW", 
        "LOC", 
        "MONEY", 
        "NORP", 
        "ORDINAL", 
        "ORG", 
        "PERCENT", 
        "PERSON", 
        "PRODUCT", 
        "QUANTITY", 
        "TIME", 
        "WORK_OF_ART",
        "MISC", 
        "PER"]]
    encoder = sklearn.preprocessing.OneHotEncoder(categories=ent_type_list, sparse_output=False)
    types = np.array([token.ent_type_ for sample in data for token in sample['annotated_text']]).reshape(-1, 1)
    print(np.unique(types))
    encoder.fit(types)

    for sample in data:
        for token in sample['annotated_text']:
            # Base token vector
            token_vector = token.vector
            ent_type = np.array([token.ent_type_]).reshape(1, -1)
            # Concatenate token vector with dependency relation feature
            combined_vector = np.concatenate((token_vector, token.head.vector, encoder.transform(ent_type).flatten()))

            # Append the enriched embedding to train_X
            train_X.append(combined_vector)
            # Append the BIO slot label to train_y
            train_y.append(token._.bio_slot_label)

    # train_X is already numeric, but train_y is strings. We must encode train_y
    # numerically in order to train the sklearn logistic regression model.
    tag_encoder = sklearn.preprocessing.LabelEncoder()
    train_y_encoded = tag_encoder.fit_transform(train_y)

    model = sklearn.linear_model.LogisticRegression(multi_class='multinomial',
                                                    solver='newton-cg' \
                                                   ).fit(train_X,
                                                         train_y_encoded)

    print(f'> Finished training logistic regression on {len(train_X)} tokens')
    return lambda token: _predict_tag_mymodel(token, model, tag_encoder, encoder)
    ##########################################################################
    #
    # Your model code goes here.
    #
    # You may find it easiest to base your code on train_tag_logreg, and
    # _predict_tag_logreg, both above. You should make sure that you understand
    # every line in the code before attempting to extend it. If you aren't
    # sure what somethings does, refer to the scikit-learn documentation:
    # https://scikit-learn.org/
    #
    # In that code, notice that string values for the output tag are encoded
    # numerically. If you use string values as input features (e.g. linguistic
    # annotations from spacy), you must decide how best to encode them
    # numerically as well. There are multiple ways to do this, see:
    # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
    #
    # Be sure to describe and justify all decisions in your report.
    #
    ##########################################################################


def predict_independent_tags(tag_predictor, data):
    """Predict tags independently of each other.

    Args:
        tag_predictor: function that returns a sorted distribution over tags,
          given a spacy token
        data: data in NLU++ format imported from JSON. Each data sample
          should include spacy annotations in the variable 'annotated_text',
          and each token in the annotated text should include a 'bio_slot_label'.

    Returns:
        List[List[Str]]: a nested array in the same shape as data, where each
        element is the predicted tag
    """
    predictions = []
    for sample in data:
        predictions.append([tag_predictor(token)[0][0] for
                            token in sample['annotated_text']])
    return predictions


def predict_bio_tags(tag_predictor, data):
    """Predict tags respecting BIO tagging constraints.

    An I-<tag> can only follow I-<tag> or B-<tag>, for any <tag>.

    Args:
        tag_predictor: function that returns a sorted distribution over tags,
          given a spacy token
       data: data in NLU++ format imported from JSON. Each data sample
          should include spacy annotations in the variable 'annotated_text',
          and each token in the annotated text should include a 'bio_slot_label'.

    Returns:
        List[List[Str]]: a nested array in the same shape as data, where each
        element is the predicted tag
    """
    predictions = []
    for sample in data:
        prev_tag = 'O'
        predictions.append([])
        for token in sample['annotated_text']:
            label = 'O'
            for tag, _ in tag_predictor(token):
                if tag.startswith('I') and tag[1:] != prev_tag[1:]:
                    continue
                label = tag
                break
            predictions[-1].append(label)
            prev_tag = predictions[-1][-1]
    return predictions


def main():
    """Main program loop. Trains model and evaluates on a test set.
    """

    # Map command-line options to callable functions
    train = {
        'most_frequent_tag' : train_tag_mle,
        'logistic_regression' : train_tag_logreg,
        'my_model': train_my_model
    }
    default_train = 'most_frequent_tag'

    predict = {
        'independent_tags' : predict_independent_tags,
        'bio_tags' : predict_bio_tags,
    }
    default_predict = 'independent_tags'

    default_annotations = 'lemma_:15,tag_:8,dep_:8'

    # Read command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_path',
                        help='path to training dataset in JSON format ' + \
                             f'(default={DATA_DIR}/{TRAINING_FILE})',
                        default=os.path.join(DATA_DIR, TRAINING_FILE))
    parser.add_argument('-v', '--validation_path',
                        help='path to validation dataset in JSON format ' + \
                             f'(default={DATA_DIR}/{VALIDATION_FILE})',
                        default=os.path.join(DATA_DIR, VALIDATION_FILE))
    parser.add_argument('-m', '--model',
                        help='type of model to use ' + \
                             f'(default={default_train})',
                        choices=train.keys(),
                        default=default_train)
    parser.add_argument('-p', '--predictor',
                        help='type of prediction method to use ' + \
                             f'(default={default_predict})',
                        choices=predict.keys(),
                        default=default_predict)
    parser.add_argument('-f', '--full_report',
                        help='show tags and annotations for all test tokens,',
                        action='store_true')
    parser.add_argument('-a', '--annotations',
                        help='spacy annotations to show for each word in ' + \
                             'a full report, given as ' + \
                             'comma-separated "<annotation>:<width>" ' + \
                             'pairs without spaces ' + \
                             f'(default={default_annotations})',
                        default=default_annotations)
    args = parser.parse_args()
    
    attributes = []
    col_widths = []
    for attr_col_pair in args.annotations.split(','):
        (attr, col_width) = attr_col_pair.split(':')
        attributes.append(attr)
        col_widths.append(int(col_width))

    print(f'> Reading training data from {args.train_path}...')
    with open(args.train_path) as input_file:
        training_data = json.load(input_file)

    print(f'> Reading validation data from {args.validation_path}...')
    with open(args.validation_path) as input_file:
        validation_data = json.load(input_file)

    print('> Tokenising and annotating raw data with spacy...')
    nlp_analyser = spacy.load("en_core_web_sm")
    utils.tokenise_annotate_and_convert_slot_labels_to_bio_tags(training_data,
                                                                nlp_analyser)
    utils.tokenise_annotate_and_convert_slot_labels_to_bio_tags(validation_data,
                                                                nlp_analyser)

    print(f'> Training {args.model} model on training data...')
    model = train[args.model](training_data)

    print(f'> Predicting tags on validation data...')
    predictions = predict[args.predictor](model, validation_data)

    print(f'> Evaluating results on validation data...')
    if args.full_report:
        utils.visualise_bio_tags(predictions, validation_data,
                                 attributes, col_widths)
    utils.evaluate(predictions, validation_data)


if __name__ == "__main__":
    main()
