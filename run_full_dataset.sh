#!/bin/bash

################################################################################
# ViT Dataset Benchmark Script with Batch Processing
################################################################################
#
# Description:
#   Runs ViT benchmark with batch processing using ViTbatchProcessing
#   - Input: 5D tensor (total_samples, 1, bands, height, width)  
#   - Processes data in batches internally through the ViT model
#   - Example: 10259 samples with batch_size=32 → 321 batches
#
# How to Run:
#   1. Make the script executable:
#      chmod +x run_full_dataset.sh
#
#   2. Run the script:
#      ./run_full_dataset.sh
#
#   Or run directly without making executable:
#      bash run_full_dataset.sh
#
################################################################################

python main.py \
    --total_samples 10259 \
    --batch_size 32 \
    --band 30 \
    --patches 15 \
    --mode ViT \
    --num_classes 16 \
    --wandb_mode offline \
    --ai_hub_device "Snapdragon X2 Elite CRD"


