"""
Save splits directly by streaming.
"""

import numpy as np
import os

output_dir = "../datasets/burgers_augmented_v2"
v1_dir = "../datasets/burgers_augmented_v1"

v1_train = np.load(f"{v1_dir}/burgers_augmented_train.npz")['data']
v1_dev = np.load(f"{v1_dir}/burgers_augmented_dev.npz")['data']
v1_test = np.load(f"{v1_dir}/burgers_augmented_test.npz")['data']

combined_total = len(v1_train) + len(v1_dev) + len(v1_test) + 5000
test_n = int(combined_total * 0.15)
dev_n = int(combined_total * 0.10)

np.random.seed(42)
all_idx = np.random.permutation(combined_total)
test_set = set(all_idx[:test_n])
dev_set = set(all_idx[test_n:test_n+dev_n])
train_set = set(all_idx[test_n+dev_n:])

print("Saving splits...")

# Train
train_data = []
idx = 0
for u in v1_train:
    if idx in train_set:
        train_data.append(u)
    idx += 1
for u in v1_dev:
    if idx in train_set:
        train_data.append(u)
    idx += 1
for u in v1_test:
    if idx in train_set:
        train_data.append(u)
    idx += 1
for i in range(25):
    chunk = np.load(f"{output_dir}_chunk_{i:03d}.npz")['data']
    for u in chunk:
        if idx in train_set:
            train_data.append(u)
        idx += 1

np.savez_compressed(
    f"{output_dir}/burgers_augmented_train.npz",
    data=np.float32(train_data[0:1])
)
del train_data[0:1]  # Free memory
# Append rest
train_file = f"{output_dir}/burgers_augmented_train.npz"
npz = np.load(train_file)
existing = npz['data']
print(f"Train: {len(existing)} initial")

# Just use simple save without concatenation
print("Writing train data...")
with open(train_file, 'wb') as f:
    np.savez_compressed(f, data=np.float32(train_data))

print("✅ Done")
