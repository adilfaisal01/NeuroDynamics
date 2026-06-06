#!/bin/bash
set -e

case "$1" in
    sweep)
        echo "Starting full hyperparameter sweep..."
        bash /deeplearningtransformers/runpod_sweep.sh
        ;;
    finetune)
        echo "Starting Stage 2 fine-tuning..."
        python3 /deeplearningtransformers/physics_tuning.py
        ;;
    *)
        echo "Starting single training run..."
        shift
        python3 /deeplearningtransformers/model_training.py "$@"
        ;;
esac