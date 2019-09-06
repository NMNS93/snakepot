"""evaluate.py - Evaluate model against test data"""

import sys
import os
from src.log import Logger
log = Logger('evaluate')

import argparse
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import CalibratedClassifierCV

def get_training_data(cli_args):
    """Load the training dataset.
    Args:
        cli_args: Argparse object with command line arguments
    """
    args = cli_args
    training = pd.read_csv(args.training, index_col=0, dtype=np.float64)
    X_train = training.drop(columns=[args.target]).to_numpy()
    y_train = training[args.target].to_numpy()
    return X_train, y_train

def model_with_proba(model, cli_args):
    """Return a model with the predict_proba method. A wrapper to catch models that do not implement
    this method by default.
    Args:
        model: An sklearn estimator object
    Returns:
        model: An sklearn estimator object with the predict_proba() method
    """
    # Models without probabilities to check
    known = ['LinearSVC']
    # Return models with predict_proba
    if hasattr(model, 'predict_proba'):
        return model
    # Wrap model with calibrator for probability prediction if it is in the known list
    # Do not refit model, simply calibrate internal data.
    elif model.__class__.__name__ in known:
        # Wrap input model with calibrator
        calib_model = CalibratedClassifierCV(base_estimator=model, cv="prefit")
        # Recalibrate on training data
        X_train, y_train = get_training_data(cli_args)
        calib_model.fit(X_train, y_train)
        return calib_model
    else:
        raise ValueError(f'Model is not in known list and does not have predict_proba() method')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test')
    parser.add_argument('--target')
    parser.add_argument('--training')
    args = parser.parse_args()

    log.info('BEGIN')
    # Read test data and split
    log.info('Reading test data')
    df_test = pd.read_csv(args.test, index_col=0, dtype=np.float64)
    X_test = df_test.drop(columns=[args.target]).to_numpy()
    y_test = df_test[args.target].to_numpy()

    # Load model
    log.info('Loading model')
    loaded_model = joblib.load('model.joblib')
    # Ensure predict_proba method is implemented
    model = model_with_proba(loaded_model, args)

    # Predict on unseen data
    log.info('Predicting')
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    target_1_prob = y_prob[:,1]

    # Calculate metrics
    log.info('Calculating metrics')
    scores = [
        ('acc', accuracy_score),
        ('roc', roc_auc_score),
        ('prec', precision_score),
        ('recall', recall_score),
        ('f1', f1_score),
        ('mcc', matthews_corrcoef)
    ]

    # Calculate metrics
    results = []
    for score in scores:
        results.append((score[0], score[1](y_test, y_pred)))
    df_results = pd.DataFrame(results, columns=[f'score', f'result'])

    # Get graph data for ROC and prec-rec
    log.info('Calculating ROC and prec-rec curve data')
    roc_data = roc_curve(y_test, target_1_prob)
    prec_data = precision_recall_curve(y_test, target_1_prob)
    df_roc = pd.DataFrame(roc_data, index=['fpr','tpr','thresholds']).T
    df_prec = pd.DataFrame(prec_data, index=['prec', 'rec', 'thresholds']).T

    # Write metrics to files
    df_results.to_csv(f'metrics.csv')
    df_roc.to_csv(f'roc_data.csv')
    df_prec.to_csv(f'precrec_data.csv')
    log.info('END')

if __name__ == '__main__':
    main()