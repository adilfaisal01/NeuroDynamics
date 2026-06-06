#!/bin/bash
set -e
service ssh start
case "$1" in
    sweep)
        echo "Starting full hyperparameter sweep..."
        exec bash /deeplearningtransformers/runpod_sweep.sh
        ;;
    finetune)
        echo "Starting Stage 2 fine-tuning..."
        exec bash /deeplearningtransformers/runpod_finetune.sh
        ;;
    *)
        echo "Starting single training run..."
        shift
        python3 /deeplearningtransformers/model_training.py "$@"
        ;;
esac