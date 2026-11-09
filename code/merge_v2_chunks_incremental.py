"""
Merge v2 chunks with v1 data - save without full concatenation.
"""

import numpy as np
import os

output_dir = "../datasets/burgers_augmented_v2"
v1_dir = "../datasets/burgers_augmented_v1"

# Count total size
print("Counting dataset size...")
u_v1_train = np.load(f"{v1_dir}/burgers_augmented_train.npz")['data']
u_v1_dev = np.load(f"{v1_dir}/burgers_augmented_dev.npz")['data']
u_v1_test = np.load(f"{v1_dir}/burgers_augmented_test.npz")['data']

v1_total = len(u_v1_train) + len(u_v1_dev) + len(u_v1_test)
new_total = 5000
combined_total = v1_total + new_total

print(f"V1 total: {v1_total}")
print(f"New: {new_total}")
print(f"Combined: {combined_total}")

# Split indices upfront
np.random.seed(42)
indices = np.random.permutation(combined_total)

test_n = int(combined_total * 0.15)
dev_n = int(combined_total * 0.10)

test_indices = set(indices[:test_n].tolist())
dev_indices = set(indices[test_n:test_n + dev_n].tolist())
train_indices = set(indices[test_n + dev_n:].tolist())

print(f"Train: {len(train_indices)}")
print(f"Dev: {len(dev_indices)}")
print(f"Test: {len(test_indices)}")

# Process v1 data
print("\nProcessing v1 data...")
train_list = []
dev_list = []
test_list = []

idx = 0
for u in u_v1_train:
    if idx in train_indices:
        train_list.append(u)
    elif idx in dev_indices:
        dev_list.append(u)
    elif idx in test_indices:
        test_list.append(u)
    idx += 1

for u in u_v1_dev:
    if idx in train_indices:
        train_list.append(u)
    elif idx in dev_indices:
        dev_list.append(u)
    elif idx in test_indices:
        test_list.append(u)
    idx += 1

for u in u_v1_test:
    if idx in train_indices:
        train_list.append(u)
    elif idx in dev_indices:
        dev_list.append(u)
    elif idx in test_indices:
        test_list.append(u)
    idx += 1

# Process v2 chunks
print("Processing v2 chunks...")
for i in range(25):
    chunk_file = f"{output_dir}_chunk_{i:03d}.npz"
    if os.path.exists(chunk_file):
        chunk = np.load(chunk_file)['data']
        for u in chunk:
            if idx in train_indices:
                train_list.append(u)
            elif idx in dev_indices:
                dev_list.append(u)
            elif idx in test_indices:
                test_list.append(u)
            idx += 1

# Save splits
print("\nSaving...")
os.makedirs(output_dir, exist_ok=True)

print(f"  Train: {len(train_list)}")
np.savez_compressed(
    f"{output_dir}/burgers_augmented_train.npz",
    data=np.array(train_list, dtype=np.float32)
)

print(f"  Dev: {len(dev_list)}")
np.savez_compressed(
    f"{output_dir}/burgers_augmented_dev.npz",
    data=np.array(dev_list, dtype=np.float32)
)

print(f"  Test: {len(test_list)}")
np.savez_compressed(
    f"{output_dir}/burgers_augmented_test.npz",
    data=np.array(test_list, dtype=np.float32)
)

# Cleanup chunks
print("\nCleaning chunks...")
for i in range(25):
    chunk_file = f"{output_dir}_chunk_{i:03d}.npz"
    if os.path.exists(chunk_file):
        os.remove(chunk_file)

print("✅ Complete!")
print(f"Output: {output_dir}/")
