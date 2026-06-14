#!/bin/bash
# RunPod sweep: bigger model, 2 sequence lengths
# Usage: bash runpod_sweep.sh

set -e
echo "=== Starting ==="

export LR=3e-5
export TYPE="transformer"
export BATCH=64
export NE=20
export OUTPUT_DIR="/workspace/outputs"

mkdir -p /workspace/outputs

LAMBDA_VALUES=(0.01 0.1 0.2)
for LAMBDA in "${LAMBDA_VALUES[@]}"; do
  export LAMBDA
  export NAME="model_dlenfinetune_${LR}_${LAMBDA}_big.pth"
  python3 -u /deeplearningtransformers/physics_tuning.py
  echo "=== Finished saved as $NAME ==="
done

# Keep container alive so we can download
sleep infinity