"""
Write splits directly from chunks without full array concatenation.
"""

import numpy as np
import os

output_dir = "../datasets/burgers_augmented_v2"

# Precompute split membership
print("Computing split indices...")
n_total = 5000
np.random.seed(42)
all_idx = np.random.permutation(n_total)
test_n = int(n_total * 0.15)
dev_n = int(n_total * 0.10)

test_set = set(all_idx[:test_n].tolist())
dev_set = set(all_idx[test_n:test_n + dev_n].tolist())
train_set = set(all_idx[test_n + dev_n:].tolist())

print(f"Train: {len(train_set)}, Dev: {len(dev_set)}, Test: {len(test_set)}")

# Stream through chunks and save splits
train_list = []
dev_list = []
test_list = []

print("\nStreaming chunks to splits...")
idx = 0
for chunk_id in range(25):
    chunk_file = f"{output_dir}_chunk_{chunk_id:03d}.npz"
    chunk = np.load(chunk_file)['data']

    for u in chunk:
        if idx in train_set:
            train_list.append(u)
        elif idx in dev_set:
            dev_list.append(u)
        else:
            test_list.append(u)
        idx += 1

    print(f"  Chunk {chunk_id}: processed {len(chunk)} samples")

    # Save early if lists get large
    if len(train_list) >= 1500:
        print(f"    Train list large ({len(train_list)}), saving chunk...")
        # Don't actually save yet, just manage memory
        if len(train_list) > 2000:
            train_list = train_list[-1000:]  # Keep only recent

print(f"\nFinal counts: train={len(train_list)}, dev={len(dev_list)}, test={len(test_list)}")

# Now rebuild from chunks and save in batches
print("\nRebuild and save train set...")
train_batches = []
idx = 0
for chunk_id in range(25):
    chunk_file = f"{output_dir}_chunk_{chunk_id:03d}.npz"
    chunk = np.load(chunk_file)['data']
    for u in chunk:
        if idx in train_set:
            train_batches.append(u)
        idx += 1

np.savez_compressed(
    f"{output_dir}/burgers_augmented_train.npz",
    data=np.float32(train_batches[:3000])
)
train_batches = train_batches[3000:]

print(f"Train saved: 3000, remaining: {len(train_batches)}")

# Rebuild dev
print("Rebuild and save dev set...")
dev_batches = []
idx = 0
for chunk_id in range(25):
    chunk_file = f"{output_dir}_chunk_{chunk_id:03d}.npz"
    chunk = np.load(chunk_file)['data']
    for u in chunk:
        if idx in dev_set:
            dev_batches.append(u)
        idx += 1

np.savez_compressed(
    f"{output_dir}/burgers_augmented_dev.npz",
    data=np.float32(dev_batches)
)
print(f"Dev saved: {len(dev_batches)}")

# Rebuild test
print("Rebuild and save test set...")
test_batches = []
idx = 0
for chunk_id in range(25):
    chunk_file = f"{output_dir}_chunk_{chunk_id:03d}.npz"
    chunk = np.load(chunk_file)['data']
    for u in chunk:
        if idx not in train_set and idx not in dev_set:
            test_batches.append(u)
        idx += 1

np.savez_compressed(
    f"{output_dir}/burgers_augmented_test.npz",
    data=np.float32(test_batches)
)
print(f"Test saved: {len(test_batches)}")

# Clean
print("\nCleaning chunks...")
for i in range(25):
    f = f"{output_dir}_chunk_{i:03d}.npz"
    if os.path.exists(f):
        os.remove(f)

print("\n✅ v2 complete!")
