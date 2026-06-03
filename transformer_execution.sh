#!/bin/bash

# Loop over different attention heads
for dlen in 250 500 1000 2500; do
    for head in 4 8 16; do
        echo "=== Training with HEAD=$head, SEQUENCE LENGTH=$dlen==="

        # Set environment variables for this run
        export LR=1e-3
        export HEAD=$head
        export NAME="model_HEAD${head}_sequencelength${dlen}_transformer.pth"  # output filename
        export TYPE="transformer"
        export NE=200
        export DS_LEN=$dlen

        # Create output folder if it doesn't exist
        OUTPUT_DIR="./outputs"
        mkdir -p "$OUTPUT_DIR"

        # Run the Docker container
        docker compose run --rm -e HEAD=$HEAD -e LR=$LR -e NAME=$NAME -e DS_LEN=$DS_LEN cs523docker

        echo "=== Finished HEAD=$head, saved as $NAME ==="
    done
done
echo "All runs completed!"
