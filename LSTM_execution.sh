#!/bin/bash

# Loop over different attention heads

for batch_size in 8 16 32; do
    for hd in 32 64 128 256 512 1024; do
        echo "=== Training with hidden dimensions=$hd batchsize=$batch_size ==="

        # Set environment variables for this run
        export LR=1e-3
        export HD=$hd
        export NAME="model_hiddendim${hd}_batch${batch_size}.pth"  # output filename
        export TYPE="lstm"
        export NE=200
        export BATCH=$batch_size

        # Create output folder if it doesn't exist
        OUTPUT_DIR="./outputs"
        mkdir -p "$OUTPUT_DIR"

        # Run the Docker container
        docker compose run --rm -e HD=$HD -e LR=$LR -e NAME=$NAME -e BATCH=$BATCH cs523docker

        echo "=== Finished Hidden dim=$hd, batch=$batch_size saved as $NAME ==="
    done
done
echo "All runs completed!"
