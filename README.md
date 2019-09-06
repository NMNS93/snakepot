# snakepot

![snakepot_logo](_assets/snakepot.png)

`snakepot` is a snakemake workflow designed to train and evaluate a binary classifier using the [TPOT auto-ML library](http://automl.info/tpot/).

## Quick Start

1. [Install snakepot](#Setup) (requires [conda](https://docs.conda.io/en/latest/miniconda.html))

1. Edit parameters in [config.json](#configjson)

1. Run `snakemake`

1. View outputs in new directory

## Features

I developed `snakepot` during my elective at the William Harvey Research Institute. We used `snakepot` to quickly train a baseline model on a variety of gene-phenotype datasets. The workflow takes the following steps:

1. Clean the data set (simple N/A drop by rows and explicit drop by columns)
2. Split data into train/test/validate sets
3. Call the TPOT automated machine learning algorithm to train a classifier
4. Save classifier and re-run it on the houldout/validation data
5. Evaluate the classifier on the holdout set
6. Call the classifier for predictions on the unlabelled data

An example dataset (/test/data.csv) and config file (config.json) are provided.

## Setup

```bash
# Build conda environment
conda env create -f environment.yaml
conda activate snakepot
# Install python helper scripts to path
pip install . 
# Run the workflow in Snakefile using config.json
snakemake 
```

## config.json

|Parameter|Description|
|-----|-----|
|directory|Output directory for new files|
|input|Input CSV file. All data must be encoded as binary or continuous variables|
|drop_columns|Features to drop from the data. Skipped if not found|
|encode_columns|Categorical features to encode. Skipped if not found|
|target_column|The name of the target variable|
|target_1|Target variable value to label as '1'|
|target_0|Target variable value to label as '0'|
|to_predict|Target variable value for final predictions|
|perc_split|Percentage of training data (target '1' or '0') to split for holdout set|
|TPOT_max_time|Maximum time to run TPOT in muntes|

## License

MIT License. Copyright (c) 2019 Nana Mensah
