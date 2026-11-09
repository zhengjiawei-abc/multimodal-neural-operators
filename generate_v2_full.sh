#!/bin/bash
# Full v2 dataset generation (5000 trajectories)
cd code
python expand_burgers_dataset_v2.py \
  --num-trajectories 5000 \
  --output-dir ../datasets/burgers_augmented_v2 \
  --include-v1 \
  --seed 42
