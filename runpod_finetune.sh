#!/bin/bash
# RunPod sweep: bigger model, 2 sequence lengths
# Usage: bash runpod_sweep.sh

set -e
echo "=== Starting ==="

export LR=3e-5
export TYPE="transformer"
export BATCH=64
export LAMBDA=0.4
export NAME="model_dlenfinetune_${LR}_${LAMBDA}_big.pth"
export NE=200

export OUTPUT_DIR="/workspace/outputs"

mkdir -p /workspace/outputs

python3 -u /deeplearningtransformers/physics_tuning.py

echo "=== Finished saved as $NAME ==="
# Keep container alive so we can download
sleep infinity