#!/bin/bash
set -e

MODELS=(
    "resnet18" 
    "resnet34" 
    "resnet50"
)

IMAGE_SIZES=(
    112
    # 224 
    # 336 
    # 448 
    # 560 
    # 672 
    # 784 
    # 896 
    # 1008 
    # 1120
)

DEVICES=(
  "Dragonwing IQ-9075 EVK"
  "QCS8550 (Proxy)"
  "Google Pixel 10 Pro XL"
  "Samsung Galaxy S24 (Family)"
#   "Samsung Galaxy S24 Ultra"
)

for device in "${DEVICES[@]}"; do
  for model in "${MODELS[@]}"; do
    for size in "${IMAGE_SIZES[@]}"; do

      echo "Running: $model | $device | $size"

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
