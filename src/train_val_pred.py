"""train_val_pred.py - Split dataframe based on target variable"""

import sys
import os
from src.log import Logger
log = Logger('clean')

import argparse
import category_encoders as ce
import pandas as pd

class DataSplitter():
    def __init__(self, df, target, target_1, to_predict, perc_split):
        self.df = df
        self.target = target
        assert df[target].nunique() == 3
        self.target_1 = target_1
        self.to_predict = to_predict
        self.perc_split = perc_split
        self.encoder = ce.OneHotEncoder(cols=[target], use_cat_names=True, return_df=True, drop_invariant=True)
        self.encoded = self.encoder.fit_transform(self.df)

    def get_test_train(self):
        df = self._get_labelled()
        # Get dataframes for each of the binary outputs
        df_targ1 = df[df[self.target] == 1] 
        df_targ0 = df[df[self.target] == 0]
        # Split each dataframe by the input fraction. Test dataset contains the percentage split
        df_targ1_test, df_targ1_train = self._perc_splitter(df_targ1, self.perc_split)
        df_targ0_test, df_targ0_train = self._perc_splitter(df_targ0, self.perc_split)
        # Combine training and test datasets
        test = pd.concat([df_targ1_test, df_targ0_test])
        training = pd.concat([df_targ1_train, df_targ0_train])
        return (test, training)

    def _get_labelled(self):
        col_to_predict = f'{self.target}_{self.to_predict}'
        col_to_train = f'{self.target}_{self.target_1}'
        encoded_target_1 = self.encoded[col_to_train]
        to_train_bool = (self.encoded[col_to_predict] == 0)
        df_to_train = self.df[to_train_bool].drop(columns=[self.target])
        df_to_train[self.target] = encoded_target_1[to_train_bool]
        return df_to_train.sample(frac=1,random_state=42)

    def get_to_predict(self):
        col_to_predict = f'{self.target}_{self.to_predict}'
        to_predict_bool = (self.encoded[col_to_predict] == 1)
        df_to_predict = self.df[to_predict_bool].drop(columns=[self.target])
        return df_to_predict.sample(frac=1, random_state=42)

    def _perc_splitter(self, df, perc):
        """Splits a dataframe (df) into two by some fraction (perc).
        Returns:
            split_data(tuple): split_by_perc, data_remainder"""
        # Split the dataframe by percentage
        split_by_perc = df.sample(frac=perc)
        # Get the remainder dataframe using the split data index
        data_remainder = df.drop(index=split_by_perc.index)
        # Return result
        return split_by_perc, data_remainder


def main():
    # Read data
    parser = argparse.ArgumentParser()
    for argument in ['--cleaned', '--target_column', '--target_1', '--target_0', '--to_predict']:
        parser.add_argument(argument)
    parser.add_argument('--perc_split', type=float)
    args = parser.parse_args()

    log.info('BEGIN')

    # Read data
    cleaned = pd.read_csv(args.cleaned, index_col=0)

    # Encode target
    log.info('Encoding data and asserting 3 unique values in target column')
    ds = DataSplitter(cleaned, args.target_column, args.target_1, args.to_predict, args.perc_split)
    
    # Split and write validation data
    log.info('Getting unlabelled dataset')
    unlabelled = ds.get_to_predict()
    unlabelled.to_csv('unlabelled.csv')

    # Get test and train data. Write to output files.
    log.info('Getting training and test datasets')
    test, training = ds.get_test_train()
    training.to_csv('training.csv')
    test.to_csv('test.csv')

    log.info('END')

if __name__ == '__main__':
    main()