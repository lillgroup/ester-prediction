# -*- coding: utf-8 -*-
"""
    Trains random forest model using training datasets with hyper parameters and selected features.
"""

import os
import pickle
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier

def init_training():

    # 1. Import features and define target label
    feat = pd.read_csv('input/features.csv').feature
    y_label = 'y_true'

    # 2. Import hyper parameters
    params = import_yaml('input/parameters.yaml')

    # 3. Import data
    data = pd.read_csv('input/datasets.csv')

    # 4. Filter for training data set
    data = data[data['Set'] == 'TRAIN']

    # 5. Split data into predictors and labels
    X, y = data[feat], data[y_label]

    # 6. Train and save model as pickle file
    train(X, y, params)

    print("Training complete")

def train(X: pd.DataFrame, y: pd.Series, params: dict):
    """
    Trains data set and saves model as pickle file.

    :param X: predictors
    :param y: labels
    :param params: hyper parameters for random forest
    """

    # Init classification model with predefined parameters
    classifier = RandomForestClassifier(**params)

    # Train model
    classifier.fit(X, y)

    # Save the model for upcoming predictions
    pickle.dump(classifier, open('model/RF_classifier.pkl', 'wb'))

def import_yaml(yaml_path: os.path):
    """
    Opens yaml file containing hyper parameters.

    :param yaml_path: File path to yaml
    :return: dictionary with parameters
    """
    try:
        with open(yaml_path, 'r') as stream:
            return yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


if __name__ == '__main__':
    init_training()
