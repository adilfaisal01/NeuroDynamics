#!/bin/bash
set -e

# Start SSH so we can connect and download results
service ssh start

if [ "$1" == "sweep" ]; then
    echo "Starting full hyperparameter sweep..."
    bash /deeplearningtransformers/runpod_sweep.sh
else
    echo "Starting single training run..."
    python3 /deeplearningtransformers/model_training.py "$@"
fi