"""
Merge existing v2 chunks and combine with v1.
"""

import numpy as np
import os
from pathlib import Path

output_dir = "../datasets/burgers_augmented_v2"
v1_dir = "../datasets/burgers_augmented_v1"

# Merge new chunks
print("Merging 25 chunks...")
all_data = []
for i in range(25):  # 0-24
    chunk_file = f"{output_dir}_chunk_{i:03d}.npz"
    if os.path.exists(chunk_file):
        data = np.load(chunk_file)['data']
        all_data.append(data)
        print(f"  Chunk {i}: {data.shape}")
    else:
        print(f"  Chunk {i}: MISSING")

u_new = np.concatenate(all_data)
print(f"Total new: {u_new.shape}")

# Load v1
print(f"\nLoading v1...")
u_v1_train = np.load(f"{v1_dir}/burgers_augmented_train.npz")['data']
u_v1_dev = np.load(f"{v1_dir}/burgers_augmented_dev.npz")['data']
u_v1_test = np.load(f"{v1_dir}/burgers_augmented_test.npz")['data']

u_combined = np.concatenate([u_v1_train, u_v1_dev, u_v1_test, u_new])
print(f"Combined: {u_combined.shape}")

# Split
print(f"\nSplitting...")
np.random.seed(42)
n = len(u_combined)
indices = np.random.permutation(n)

test_n = int(n * 0.15)
dev_n = int(n * 0.10)

test_idx = indices[:test_n]
dev_idx = indices[test_n:test_n + dev_n]
train_idx = indices[test_n + dev_n:]

# Save
os.makedirs(output_dir, exist_ok=True)

print(f"  Train: {len(train_idx)}")
np.savez_compressed(
    f"{output_dir}/burgers_augmented_train.npz",
    data=u_combined[train_idx].astype(np.float32)
)

print(f"  Dev:   {len(dev_idx)}")
np.savez_compressed(
    f"{output_dir}/burgers_augmented_dev.npz",
    data=u_combined[dev_idx].astype(np.float32)
)

print(f"  Test:  {len(test_idx)}")
np.savez_compressed(
    f"{output_dir}/burgers_augmented_test.npz",
    data=u_combined[test_idx].astype(np.float32)
)

# Cleanup
print(f"\nCleaning chunks...")
for i in range(25):
    chunk_file = f"{output_dir}_chunk_{i:03d}.npz"
    if os.path.exists(chunk_file):
        os.remove(chunk_file)

print("✅ Done!")
