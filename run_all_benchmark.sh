#!/bin/bash
set -e

# This runs all the benchmark,
# chmod +x run_all_benchmark.sh  - this gives permission
# ./run_all_benchmark.sh -runs this file

MODELS=(
    "resnet18" 
    # "resnet34" 
    # "resnet50"
)

IMAGE_SIZES=(
    112
    224 
    336 
    448 
    560 
    672 
    784 
    896 
    1008 
    1120
)

DEVICES=(
  "Dragonwing IQ-9075 EVK"
  # "QCS8550 (Proxy)"
  "Google Pixel 10 Pro XL"
  "Samsung Galaxy S24 (Family)"
  "Google Pixel 3"
  "Google Pixel 6"
  # "Samsung Galaxy S24 Ultra"
)

for device in "${DEVICES[@]}"; do
  for model in "${MODELS[@]}"; do
    for size in "${IMAGE_SIZES[@]}"; do

      echo "=========================================="
      echo "Running: $model | $device | $size"
      echo "=========================================="

      python main.py \
        --model "$model" \
        --image_size "$size" \
        --ai_hub_device "$device"

    done
  done
done

echo "===================================================="
echo "All runs complete"
echo "===================================================="
