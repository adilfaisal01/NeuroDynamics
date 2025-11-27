#!/bin/bash
set -e
set -x

echo "starting transformers"

python3 /deeplearningtransformers/model_training.py "$@"
