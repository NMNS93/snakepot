#!/usr/bin/env python3
"""tpot.py
Run tpot on an input training dataset."""

import sys
import os
import importlib.util
import joblib
import tempfile
from src.log import Logger
log = Logger('tpot')

import argparse
import pandas as pd
import numpy as np
from tpot import TPOTClassifier


class TPOTCleaner():
    def __init__(self, tpot_file):
        with open(tpot_file, 'r') as f:
            self.lines = f.readlines()

    @property
    def import_lines(self):
        lines = self.lines
        import_break = lines.index('\n')
        import_lines = lines[:import_break]
        return import_lines

    @property
    def export_lines(self):
        lines = self.lines
        export_start_line = list(filter(lambda x: 'exported_pipeline = ' in x, lines))[0]
        export_list = lines[lines.index(export_start_line):]
        export_break = export_list.index('\n')
        export_lines = export_list[:export_break]
        return export_lines

    def write_out(self, outdir):
        with open(outdir, 'w') as f:
            f.write("".join(self.import_lines))
            f.write("".join(self.export_lines))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training')
    parser.add_argument('--target')
    parser.add_argument('--outdir')
    parser.add_argument('--max_time', type=int)
    args = parser.parse_args()
    
    log.info('BEGIN')
    log.info('Loading data')
    training = pd.read_csv(args.training, index_col=0, dtype=np.float64)
    X_train = training.drop(columns=[args.target]).to_numpy()
    y_train = training[args.target].to_numpy()
    
    # TPOT setup
    pipeline_optimizer = TPOTClassifier(max_time_mins=args.max_time, cv=10, n_jobs=-1, 
        random_state=42, verbosity=2, memory='auto')

    # TPOT run
    log.info('Running TPOT')
    pipeline_optimizer.fit(X_train, y_train)
    pipeline_optimizer.export(f'{args.outdir}/tpot_pipeline.py')

    # Create python file for refitting model
    log.info('Cleaning TPOT output file')
    # Read varialbe 'exported_pipeline' from TPOT output
    tc = TPOTCleaner(f'{args.outdir}/tpot_pipeline.py')
    tc.write_out(f'{args.outdir}/tpot_pipe.py')

    # Refit model on training data and save
    log.info('Refitting model')
    spec = importlib.util.spec_from_file_location("src", f"{args.outdir}/tpot_pipe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    model = mod.exported_pipeline
    model.fit(X_train, y_train)

    log.info('Saving model')
    joblib.dump(model, f'{args.outdir}/model.joblib')

    log.info('END')

if __name__=="__main__":
    main()