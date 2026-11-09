"""
Save v2 data without v1 merging (memory efficient).
Just use 5000 new diverse trajectories - still 1.7x data expansion.
"""

import numpy as np
import os

output_dir = "../datasets/burgers_augmented_v2"

print("Loading 5000 new trajectories from chunks...")
all_data = []
for i in range(25):
    chunk_file = f"{output_dir}_chunk_{i:03d}.npz"
    data = np.load(chunk_file)['data']
    all_data.append(data)
    print(f"  Chunk {i}: {data.shape}")

# Concatenate chunks
print("\nConcatenating chunks...")
u_all = np.concatenate(all_data, dtype=np.float32)
print(f"Total: {u_all.shape}")

# Split
print("Splitting into train/dev/test...")
np.random.seed(42)
n = len(u_all)
indices = np.random.permutation(n)

test_n = int(n * 0.15)
dev_n = int(n * 0.10)

test_idx = indices[:test_n]
dev_idx = indices[test_n:test_n + dev_n]
train_idx = indices[test_n + dev_n:]

# Save
os.makedirs(output_dir, exist_ok=True)

print(f"Train: {len(train_idx)}")
np.savez_compressed(
    f"{output_dir}/burgers_augmented_train.npz",
    data=u_all[train_idx].astype(np.float32)
)

print(f"Dev: {len(dev_idx)}")
np.savez_compressed(
    f"{output_dir}/burgers_augmented_dev.npz",
    data=u_all[dev_idx].astype(np.float32)
)

print(f"Test: {len(test_idx)}")
np.savez_compressed(
    f"{output_dir}/burgers_augmented_test.npz",
    data=u_all[test_idx].astype(np.float32)
)

# Clean chunks
print("Cleaning chunks...")
for i in range(25):
    f = f"{output_dir}_chunk_{i:03d}.npz"
    if os.path.exists(f):
        os.remove(f)

print("\n✅ v2 dataset complete!")
print(f"Location: {output_dir}/")
print(f"Size: {len(u_all)} trajectories (1.7x expansion)")
