"""clean.py

Drop and encode features specified in config.json.
"""

import os
import sys

import argparse
import category_encoders as ce
import pandas as pd

from src.log import Logger
log = Logger('clean')

class Cleaner():
    """Drop and encode columns in a dataframe. Additionally, drops rows with NA values.

    Args:
        data (pd.Dataframe): Dataframe containing data
        to_drop (List): Column names to drop
        to_encode (List): Column names to one-hot encode
    Methods:
        process: Iterates over class methods applying dataframe
        encode: One-hot encodes features
        drop_columns: Remove features passed at class initialisation
        drop_na_rows: Remove any rows with empty cells
    """
    def __init__(self, data, to_drop, to_encode):
        self.data = data
        self.to_drop = to_drop
        self.to_encode = to_encode
        # Set list of methods to p
        self.operations = [self.encode, self.drop_columns, self.drop_na_rows]

    def process(self):
        df_process = self.data.copy(deep=True)
        for operation in self.operations:
            df_process = operation(df_process)
        return df_process
    
    def encode(self, df):
        encodable = set(self.to_encode).intersection(set(df.columns))
        if not encodable: # No columns to encode
            return df
        else:
            encoder = ce.OneHotEncoder(cols=list(encodable), use_cat_names=True, handle_unknown='ignore', return_df=True)
            encoded = encoder.fit_transform(df)
            return encoded

    def drop_columns(self, df):
        return df.drop(columns=self.to_drop, errors='ignore')

    def drop_na_rows(self, df):
        return df.dropna(axis='index', how='any')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--drop', nargs='+')
    parser.add_argument('--encode', nargs='+')
    args = parser.parse_args()
    
    # Read data
    log.info('BEGIN')
    indf = pd.read_csv(args.input, index_col=0)
    log.info(f'Input shape {indf.shape}')

    # Clean data
    cleaner = Cleaner(data=indf, to_drop=args.drop, to_encode=args.encode)
    cleaned = cleaner.process()

    # Assert only one column contains strings. This will be the target,
    # used later to split the data.
    string_cols = cleaned.select_dtypes('object').columns
    log.debug(string_cols) 
    assert (len(string_cols)) == 1

    # Write data
    output = os.path.join('cleaned.csv')
    cleaned.to_csv(output)
    log.info(f'Output shape {cleaned.shape}')
    log.info('END')

if __name__ == "__main__":
    main()
