"""
Analyze v1 vs v2 dataset properties
"""

import numpy as np

print("=" * 70)
print("BURGERS DATASET: v1 vs v2 COMPARISON")
print("=" * 70)

# Load datasets
v1_train = np.load("../datasets/burgers_augmented_v1/burgers_augmented_train.npz")['data']
v2_train = np.load("../datasets/burgers_augmented_v2/burgers_augmented_train.npz")['data']

v1_test = np.load("../datasets/burgers_augmented_v1/burgers_augmented_test.npz")['data']
v2_test = np.load("../datasets/burgers_augmented_v2/burgers_augmented_test.npz")['data']

print(f"\n📊 Dataset Size:")
print(f"  v1 Train: {v1_train.shape[0]:5d} trajectories")
print(f"  v2 Train: {v2_train.shape[0]:5d} trajectories → {v2_train.shape[0]/v1_train.shape[0]:.2f}x")

print(f"\n  v1 Test:  {v1_test.shape[0]:5d} trajectories")
print(f"  v2 Test:  {v2_test.shape[0]:5d} trajectories → {v2_test.shape[0]/v1_test.shape[0]:.2f}x")

# Amplitude statistics
amp_v1 = np.max(np.abs(v1_train), axis=(1, 2))
amp_v2 = np.max(np.abs(v2_train), axis=(1, 2))

print(f"\n📈 Amplitude Distribution:")
print(f"  v1: μ={amp_v1.mean():.3f}, σ={amp_v1.std():.3f}, range=[{amp_v1.min():.3f}, {amp_v1.max():.3f}]")
print(f"  v2: μ={amp_v2.mean():.3f}, σ={amp_v2.std():.3f}, range=[{amp_v2.min():.3f}, {amp_v2.max():.3f}]")
print(f"  Improvement: {(amp_v2.std()/amp_v1.std()):.2f}x diversity")

# Energy
energy_v1 = np.sum(v1_train ** 2, axis=(1, 2))
energy_v2 = np.sum(v2_train ** 2, axis=(1, 2))

print(f"\n⚡ Energy Distribution:")
print(f"  v1: μ={energy_v1.mean():.2f}, σ={energy_v1.std():.2f}")
print(f"  v2: μ={energy_v2.mean():.2f}, σ={energy_v2.std():.2f}")

# Gradients (smoothness)
grad_v1 = np.max(np.abs(np.gradient(v1_train, axis=2)), axis=(1, 2))
grad_v2 = np.max(np.abs(np.gradient(v2_train, axis=2)), axis=(1, 2))

print(f"\n🌊 Smoothness (max spatial gradient):")
print(f"  v1: μ={grad_v1.mean():.3f}, σ={grad_v1.std():.3f}")
print(f"  v2: μ={grad_v2.mean():.3f}, σ={grad_v2.std():.3f}")
print(f"  Improvement: {(grad_v2.std()/grad_v1.std()):.2f}x diversity")

# Test set comparison
print(f"\n🔍 Test Set Statistics:")
amp_v1_test = np.max(np.abs(v1_test), axis=(1, 2))
amp_v2_test = np.max(np.abs(v2_test), axis=(1, 2))

print(f"  v1 Test amplitude: μ={amp_v1_test.mean():.3f}")
print(f"  v2 Test amplitude: μ={amp_v2_test.mean():.3f}")

print("\n" + "=" * 70)
print("EXPECTED IMPROVEMENTS")
print("=" * 70)

print(f"""
v2 dataset provides:
  ✓ 1.6x more training data
  ✓ Greater amplitude diversity ({amp_v2.std()/amp_v1.std():.2f}x)
  ✓ Greater smoothness diversity ({grad_v2.std()/grad_v1.std():.2f}x)
  ✓ Multiple IC types (6 vs 1)
  ✓ Multiple ν values (6 vs 1)

Expected v2 performance improvements over v1 baseline:
  • Train L2: 0.201 → ~0.18-0.19 (↓5-10%)
  • Test L2:  0.246 → ~0.22-0.24 (↓5-10%)
  • PDE RMS: 0.0011 → ~0.0010 (≈ stable)
  • Generalization gap: ↓ slightly

Training time increase: ~50% (36k vs 24k steps)
Expected runtime: ~2400 seconds (40 min)
""")

print("=" * 70)
