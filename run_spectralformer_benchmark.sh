#!/bin/bash

# This runs all the benchmark,
# chmod +x run_spectralformer_benchmark.sh  - this gives permission
# ./run_spectralformer_benchmark.sh -runs this file

# --------- CONFIG ---------
PATCHES=(7)
BAND_PATCHES=(
  3 
  # 5 
  7
)
MODES=(
  "ViT"
  "CAF"
)
BANDS=(100)
DEVICES=(
  # "Dragonwing IQ-9075 EVK"
  # "QCS8550 (Proxy)"
  # "Google Pixel 10 Pro XL"
  "Samsung Galaxy S24 (Family)"
  # "Google Pixel 3"
  # "Google Pixel 6"
  # "Samsung Galaxy S24 Ultra"
)

# --------- LOOP ---------
for device in "${DEVICES[@]}"; do
  for mode in "${MODES[@]}"; do
    for patch in "${PATCHES[@]}"; do
      for band_patch in "${BAND_PATCHES[@]}"; do
        for band in "${BANDS[@]}"; do

          echo "=========================================="
          echo "Running on device: $device"
          echo "Mode: $mode | Patch: $patch | BandPatch: $band_patch | Band: $band"
          echo "=========================================="

          python Spectralformer_main.py \
            --patches "$patch" \
            --band_patches "$band_patch" \
            --mode "$mode" \
            --band "$band" \
            --ai_hub_device "$device" \
            --wandb_mode online

        done
      done
    done
  done
done

echo "=========================================="
echo "All SpectralFormer runs complete"
echo "=========================================="
