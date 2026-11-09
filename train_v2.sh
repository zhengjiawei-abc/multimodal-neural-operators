#!/bin/bash
# Train with v2 expanded dataset

cd code

python demo_burgers_dataset.py \
  --data-dir ../datasets/burgers_augmented_v2 \
  --train-file burgers_augmented_train.npz \
  --dev-file burgers_augmented_dev.npz \
  --test-file burgers_augmented_test.npz \
  --steps 36000 \
  --lr-decay-steps 18000 \
  --batch-size 32 \
  --d-model 128 \
  --n-layers 4 \
  --physics-weight 0.1 \
  --dropout 0.1 \
  --preset wide \
  --output v_aug3x_xl_v2 \
  --seed 42
