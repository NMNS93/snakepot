"""predict.py - Predict scores for unlabelled data"""

import sys
import argparse
import pandas as pd
import numpy as np
import joblib
from evaluate import get_training_data, model_with_proba

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unlabelled')
    parser.add_argument('--training')
    parser.add_argument('--target') 
    args = parser.parse_args()    

    # Load data
    unlabelled = pd.read_csv(args.unlabelled, index_col=0, dtype=np.float64)

    # Load model
    loaded_model = joblib.load('model.joblib')
    # Ensure predict_proba method is implemented
    model = model_with_proba(loaded_model, args)

    # Predict
    y_pred = model.predict(unlabelled) 
    y_prob = model.predict_proba(unlabelled)[:,1] # Gets the probabilities for '1' class predictions

    # Bulid a dataframe of scores with indexes from 'unlabelled'
    pred_tuple = zip(y_pred, y_prob)
    pred_columns = ['prediction', 'probability']
    pred_index = unlabelled.index
    pred_dataframe = pd.DataFrame(pred_tuple, columns=pred_columns, index=pred_index)

    # Write scores out
    pred_dataframe.to_csv(f'unlabelled_predictions.csv')

if __name__ == '__main__':
    main()