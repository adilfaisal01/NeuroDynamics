#!/bin/bash
# RunPod sweep: bigger model, 2 sequence lengths
# Usage: bash runpod_sweep.sh

set -e
echo "=== Starting ==="

export LR=1e-3
export TYPE="transformer"
export BATCH=16
export NE=200
export OUTPUT_DIR="/workspace/outputs/fixed_models/"

mkdir -p "$OUTPUT_DIR"

LAMBDA_VALUES=(1 2 6)
for LAMBDA in "${LAMBDA_VALUES[@]}"; do
  export LAMBDA
  export NAME="model_dlenfinetune_${LR}_${LAMBDA}_big.pth"
  python3 -u /deeplearningtransformers/physics_tuning.py
  echo "=== Finished saved as $NAME ==="
done

# Keep container alive so we can download
sleep infinity