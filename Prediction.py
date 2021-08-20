# -*- coding: utf-8 -*-
"""
    Predicts preference of compound towards hCE1 or hCE2 using model
    created during training.
"""

import pickle
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef


def predict(data_set: str, set_name: str):

    # 1. Import features
    feat = pd.read_csv('input/features.csv').feature

    # 2. Import data set
    data = pd.read_csv('input/datasets.csv')
    data = data[data['Set'] == data_set]

    # 3. Split data into predictors and true labels
    X, y_true = data[feat], data['y_true']

    # 4. Classify and predict classification probabilities
    classifier = load_model()
    y_pred = classifier.predict(X)
    y_scores = classifier.predict_proba(X)

    # 5. Print simple classification metrics
    print_stats_report(y_true, y_pred, y_scores, set_name)

    # 6. Save predictions
    meta = ['id', 'cid', 'SMILES', 'cmpdname', 'y_true']
    output_variables = meta + feat.to_list()
    data = save_prediction(data[output_variables].copy(), y_pred, y_scores, y_true)

    # 7. Export classification results
    data.to_csv(f'prediction/{data_set}.csv', index=False)

def save_prediction(data: pd.DataFrame, y_pred: pd.Series, y_scores: pd.Series, y_true: pd.Series):
    """
    Merge predictor data with predictions.

    :param data: data (predictors)
    :param y_pred: predicted labels
    :param y_scores: classification probabilities
    :param y_true: true lables
    return: data set with predictions
    """

    data.insert(loc=5, column='misclassified', value=y_pred - y_true)

    # Label encoding: 0 -> CE1, 1 -> CE2
    le = preprocessing.LabelEncoder()
    le.fit(['CE1', 'CE2'])
    y_pred = le.inverse_transform(y_pred)
    data['y_true'] = le.inverse_transform(y_true)

    # Insert predictions and classification probabilities
    data.insert(loc=5, column='prediction', value=y_pred)
    data.insert(loc=6, column='prob_CE1', value=y_scores[:, 0].round(2))
    data.insert(loc=7, column='prob_CE2', value=y_scores[:, 1].round(3))

    return data

def print_stats_report(y_true: pd.Series, y_pred: pd.Series, y_scores: pd.Series, set_name: str):
    """
    Prints the number of misclassified compounds,
    the accuracy, area under the curve, matthews coefficient

    :param y_true: true labels
    :param y_pred: predicted labels
    :param y_scores: classification probabilities
    :param set_name: data set name: Training / Test
    """

    print(f'\033[1m {set_name} \033[0m')
    print(f'    Misclassified: {np.sum(np.abs(y_pred - y_true))}')
    print(f'    Accuracy:       {round(accuracy_score(y_true, y_pred), 2)}')
    print(f'    AUC:            {round(roc_auc_score(y_true, y_scores[:, 1]), 2)}')
    print(f'    MCC:            {round(matthews_corrcoef(y_true, y_pred), 2)}\n')

def load_model(model_path='model/RF_classifier.pkl'):
    """
    Load classifier model for prediction.

    :param model_path: path to pickle file with random forest
    :return: loaded classifier
    """
    try:
        return pickle.load(open(model_path, 'rb'))
    except FileNotFoundError:
        raise Exception('Model has not been trained yet.')


if __name__ == '__main__':

    # Predict training (MAIN) and test sets
    sets = {'TRAIN': 'Training set', 'EXT_A': 'Test set A', 'EXT_B': 'Test set B'}
    for data_set in sets:
        predict(data_set, sets[data_set])
