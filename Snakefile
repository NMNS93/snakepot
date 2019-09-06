"""Snakefile

A snakemake workflow (snakepot) for calling the TPOT binary classifier:

    rule copy: Copy the input dataset to the working directory
    rule clean: Drop columns and rows with empty data
    rule split: Create train, test, and validation data subsets
    rule tpot: Run the TPOTClassifier with training and test data
    rule evaluate: Evaluate model on validation data
    rule predict: Predict unlabelled values in the input dataset 
    rule end: Copy config.json to working directory to complete snakepot
"""

def expand_config(inlist):
    """Quote argument lists to avoid errors in shell command.
    Args:
        inlist: A list of values to be converted to strings and quoted
    Returns:
        A list of quoted strings
    """
    return [ f"'{str(i)}'" for i in inlist ]

# Set config file
configfile: "config.json"
# Create and set working directory
workdir: config['directory']
# Set snakefile directory
snakedir: "../{}".format(config['directory'])

# Set build target
rule all:
    input: "config.json"

# STEP 0: Copy the input dataset into the working directory
rule copy:
    input: f"../{config['input']}"
    output: "data.csv"
    shell: "cp {input} data.csv"

# STEP 1: Clean the dataset.
rule clean:
    # Read the dataset
    input: "data.csv"
    output: "cleaned.csv"
    # Set parameters for columns to drop and one-hot encode
    params:
        drop=expand_config(config['drop_columns']),
        encode=expand_config(config['encode_columns']),
    # Call clean.py
    shell:
        "python ../src/clean.py --input {input} --drop {params.drop} --encode {params.encode}"

rule split:
    input: "cleaned.csv"
    output: "training.csv", "test.csv", "unlabelled.csv"
    params:
        target_column=config['target_column'],
        target_1=config['target_1'],
        target_0=config['target_0'],
        to_predict=config['to_predict'],
        perc_split=config['perc_split'],
    shell:
        "python ../src/train_val_pred.py --cleaned {input} --target_column {params.target_column}"
        " --target_1 {params.target_1} --target_0 {params.target_0} --to_predict {params.to_predict}"
        " --perc_split {params.perc_split}"

rule tpot:
    input: "training.csv"
    output: "tpot_pipe.py", "model.joblib"
    params:
        target=config['target_column'],
        max=config['TPOT_max_time'],
        outdir=config['directory'] # Note this argument is relative to /src
    shell:
        "python ../src/tpot_caller.py --training {input} --target {params.target} "
        "--max_time {params.max} --outdir ../{params.outdir}"


rule evaluate:
    input:
        test="test.csv",
        training="training.csv"
    output: "metrics.csv", "roc_data.csv", "precrec_data.csv"
    params:
        target=config['target_column'],
    shell:
        "python ../src/evaluate.py --test {input.test} --target {params.target} --training {input.training}"

rule predict:
    input: 
        unlabelled="unlabelled.csv",
        model="model.joblib",
        training="training.csv"
    output: "unlabelled_predictions.csv"
    params: target=config['target_column']
    shell:
        "python ../src/predict.py --unlabelled {input.unlabelled} --training {input.training} --target {params.target}"

rule end:
    input: "unlabelled_predictions.csv"
    output: "config.json"
    shell:
        "cp ../config.json ."