#!/bin/bash

# Define log file with timestamp
LOGFILE="pipeline_$(date +%Y%m%d_%H%M%S).log"

# Function to log and execute commands
log_and_run() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Running: $@" | tee -a "$LOGFILE"
    "$@" >> "$LOGFILE" 2>&1
    if [ $? -ne 0 ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] Error: Command failed: $@" | tee -a "$LOGFILE"
        exit 1
    fi
}

# Function to log messages
log_message() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOGFILE"
}

# Set environment variables
CONFIG=configs/config.yaml
LABEL=bootstrap
FOLDER=paper
cd "$PATH_REPO" || { echo "Failed to cd to $PATH_REPO"; exit 1; }

# Data Preparation
log_and_run python -m data.simulated-data -d config-05 -c "$CONFIG" -l "$LABEL"
log_and_run python -m data.build-prediction-master -c "$CONFIG" -l "$LABEL"
log_and_run python -m data.word2vec -c "$CONFIG" -l "$LABEL"
log_and_run python -m uplift.true -c "$CONFIG" -l "$LABEL"

# Model Baselines
log_and_run python -m baselines.frequency -c "$CONFIG" -l "$LABEL"
log_and_run python -m baselines.logit-cross-by-j -c "$CONFIG" -l "$LABEL"
log_and_run python -m baselines.lightgbm-cross-by-j -c "$CONFIG" -l "$LABEL"
log_and_run python -m baselines.lightgbm-cat-cross-by-j -c "$CONFIG" -l "$LABEL"
log_and_run python -m baselines.run-mxl -c "$CONFIG" -l "$LABEL"

# DNN
log_and_run python -m dnn.train -c "$CONFIG" -l "$LABEL" -p
log_and_run python -m dnn.predict -c "$CONFIG" -l "$LABEL" -t dnn/config_model.yaml -e 99

# Probability Simulations
log_and_run python -m uplift.logit-cross-by-j -c "$CONFIG" -l "$LABEL"
log_and_run python -m uplift.lightgbm-cross-by-j -c "$CONFIG" -l "$LABEL"
log_and_run python -m uplift.lightgbm-cat-cross-by-j -c "$CONFIG" -l "$LABEL"
log_and_run python -m uplift.dnn -c "$CONFIG" -l "$LABEL" -m model_010
log_and_run python -m uplift.combine-prob-estimates -c "$CONFIG" -l "$LABEL"

# Evaluations
log_and_run python -m results.run-benchmark -c "$CONFIG" -l "$LABEL"
log_and_run python -m uplift.elasticities -c "$CONFIG" -l "$LABEL"
log_and_run python -m uplift.coupon-optim -c "$CONFIG" -l "$LABEL"

# Figure 2
log_and_run python -m paper.figure-2-data -c "$CONFIG"
log_and_run python -m paper.figure-2-mxl -c "$CONFIG"

# Paper Results
log_and_run python -m paper.table_1 -c "$CONFIG" -f "$FOLDER" -m log-loss
log_and_run python -m paper.table_1 -c "$CONFIG" -f "$FOLDER" -m kl-divergence
log_and_run python -m paper.table_2 -c "$CONFIG" -f "$FOLDER"
log_and_run python -m paper.table_3 -c "$CONFIG" -f "$FOLDER"
log_and_run python -m paper.table_4 -c "$CONFIG" -f "$FOLDER"
log_and_run python -m paper.figure-2-plot -c "$CONFIG"
log_and_run python -m paper.figure_3 -c "$CONFIG"

# Log completion message
log_message "Pipeline completed successfully."
