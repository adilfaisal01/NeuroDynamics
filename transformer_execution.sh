#!/bin/bash

# Hyperparameter sweep: dlen × head × embed_dim × hidden_dim
for dlen in 1000 2500 5000; do
    for head in 4 8 16; do
        for ed in 64 128; do
            for hd in 512 1024; do

        echo "=== Training with HEAD=$head, SEQUENCE LENGTH=$dlen, ED=$ed, HD=$hd ==="

        # Set environment variables for this run
        export LR=1e-3
        export HEAD=$head
        export NAME="model_HEAD${head}_dlen${dlen}_ED${ed}_HD${hd}_transformer.pth"
        export TYPE="transformer"
        export NE=200
        export DS_LEN=$dlen
        export ED=$ed
        export HD=$hd

        # Create output folder if it doesn't exist
        OUTPUT_DIR="./outputs"
        mkdir -p "$OUTPUT_DIR"

        # Run the Docker container
        docker compose run --rm \
          -e HEAD=$HEAD \
          -e LR=$LR \
          -e NAME=$NAME \
          -e DS_LEN=$DS_LEN \
          -e NE=$NE \
          -e ED=$ED \
          -e HD=$HD \
          cs523docker

        echo "=== Finished HEAD=$head, ED=$ed, HD=$hd, saved as $NAME ==="

            done
        done
    done
done
echo "All runs completed!"