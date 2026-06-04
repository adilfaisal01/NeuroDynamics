#!/bin/bash
# RunPod sweep: bigger model, 2 sequence lengths
# Usage: bash runpod_sweep.sh

set -e

for dlen in 2500 5000; do

    echo "=== Training with SEQUENCE LENGTH=$dlen ==="

    export LR=1e-3
    export HEAD=16
    export NAME="model_dlen${dlen}_big.pth"
    export TYPE="transformer"
    export NE=200
    export DS_LEN=$dlen
    export ED=256
    export HD=2048
    export OUTPUT_DIR="/workspace/outputs"

    mkdir -p /workspace/outputs

    python3 /deeplearningtransformers/model_training.py

    echo "=== Finished dlen=$dlen, saved as $NAME ==="

done
echo "Both runs completed!"
# Keep container alive so we can download
sleep infinity