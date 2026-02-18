#!/bin/bash

# This file runs benchmark for all the image res, on a specific model
IMAGE_SIZES=(112 224 336 448 560 672 784 896 1008 1120)

# in main folder change your desired model and run this file using 
# chmod +x run_all_res_benchmark.sh  - this gives permission
# ./run_all_res_benchmark.sh -runs this file

for size in "${IMAGE_SIZES[@]}"; do
    echo "=========================================="
    echo "Running with --image_size $size"
    echo "=========================================="
    python main.py --image_size "$size"
done

echo "=========================================="
echo "All runs complete"
echo "=========================================="