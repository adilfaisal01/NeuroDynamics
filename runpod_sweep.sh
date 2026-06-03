#!/bin/bash
# RunPod-specific sweep: calls python3 directly (no docker compose needed)
# Usage: bash runpod_sweep.sh

set -e

for dlen in 1000 2500 5000; do
    for head in 4 8 16; do
        for ed in 64 128; do
            for hd in 512 1024; do

        echo "=== Training with HEAD=$head, SEQUENCE LENGTH=$dlen, ED=$ed, HD=$hd ==="

        export LR=1e-3
        export HEAD=$head
        export NAME="model_HEAD${head}_dlen${dlen}_ED${ed}_HD${hd}_transformer.pth"
        export TYPE="transformer"
        export NE=200
        export DS_LEN=$dlen
        export ED=$ed
        export HD=$hd

        mkdir -p /deeplearningtransformers/outputs

        python3 /deeplearningtransformers/model_training.py

        echo "=== Finished HEAD=$head, ED=$ed, HD=$hd, saved as $NAME ==="

            done
        done
    done
done
echo "All runs completed!"