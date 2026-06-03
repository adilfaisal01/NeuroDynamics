#!/bin/bash
set -e

if [ "$1" == "sweep" ]; then
    echo "Starting full hyperparameter sweep..."
    bash /deeplearningtransformers/runpod_sweep.sh
else
    echo "Starting single training run..."
    python3 /deeplearningtransformers/model_training.py "$@"
fi