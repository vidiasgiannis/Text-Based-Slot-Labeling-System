# README: Slot Labeling 

## Overview
This project implements and improves a slot labeling model, a crucial component of task-oriented dialogue systems such as virtual assistants (e.g., Siri, Alexa). The primary focus is to enhance the accuracy of slot labeling by experimenting with different approaches, including logistic regression and BIO tagging.

## Project Structure
```
|-- data/
    |-- training_data.json   # Training data
    |-- validation_data.json # Validation data
    |-- test_data.json       # Test data
    |-- ontology.json        # Ontology describing intents and slots
|-- src/
    |-- label_slots.py       # Main executable script
    |-- utils.py             # Helper functions for preprocessing and evaluation
|-- README.md                # Documentation
```

## Requirements
- Python 3.x
- `spacy`
- `scikit-learn`

### Setup Environment
```bash
conda create -n anlp_hw2 python=3.x
conda activate anlp_hw2
conda install -c conda-forge spacy
conda install scikit-learn
python -m spacy download en_core_web_sm
```

## Running the Code

### Baseline Model (Most Frequent Tag)
Run the baseline model that uses the most frequent tag for each word:
```bash
python label_slots.py
```

### Logistic Regression Model with Word Embeddings
Run the logistic regression-based model:
```bash
python label_slots.py -m logistic_regression
```

### BIO Tagging with Constraints
To enforce correct tagging structure using BIO constraints:
```bash
python label_slots.py -m logistic_regression -p bio_tags
```

### Custom Model (with Feature Engineering)
To run your improved model after implementing additional features:
```bash
python label_slots.py -m my_model -p bio_tags
```

## Explanation of Components

### 1. Baseline Model
- Tags each word based on the most frequent label in the training data.
- Utilizes the BIO tagging scheme.

### 2. Logistic Regression Model
- Conditions the tag prediction on word embeddings instead of individual words.
- Addresses issues with unseen words by leveraging semantic similarities in embeddings.

### 3. BIO Tagging
- Enforces valid sequences (e.g., `I-<tag>` must follow `B-<tag>`).
- Improves consistency of labeled spans.

### 4. Error Analysis and Feature Engineering
- Identify common errors by examining mispredicted spans.
- Experiment with linguistic features such as:
  - Part-of-speech (POS) tags
  - Dependency relations
  - Neighboring word embeddings
- Implement improvements in `train_my_model` and rerun tests.

## Evaluation Metrics
- Precision, Recall, F1-score: Evaluated for each slot type and overall.
- Micro-averaged and macro-averaged F1-scores are reported for performance comparison.
