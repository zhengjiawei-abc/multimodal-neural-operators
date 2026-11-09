"""
Analyze dataset diversity to identify gaps.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


def analyze_current_dataset(data_dir: str = "../datasets/burgers_augmented_v1"):
    """Analyze current dataset statistics."""

    print("=" * 70)
    print("BURGERS DATASET DIVERSITY ANALYSIS")
    print("=" * 70)

    # Load all splits
    train = np.load(f"{data_dir}/burgers_augmented_train.npz")['u']
    dev = np.load(f"{data_dir}/burgers_augmented_dev.npz")['u']
    test = np.load(f"{data_dir}/burgers_augmented_test.npz")['u']

    u_all = np.concatenate([train, dev, test])

    print(f"\n📊 Dataset Size:")
    print(f"   Train: {train.shape[0]:5d} trajectories")
    print(f"   Dev:   {dev.shape[0]:5d} trajectories")
    print(f"   Test:  {test.shape[0]:5d} trajectories")
    print(f"   Total: {u_all.shape[0]:5d} trajectories")
    print(f"   Spatial resolution: {u_all.shape[2]}")
    print(f"   Temporal resolution: {u_all.shape[1]}")

    # Amplitude statistics
    print(f"\n📈 Amplitude Statistics:")
    amplitudes = np.max(np.abs(u_all), axis=(1, 2))
    print(f"   Mean amplitude:   {amplitudes.mean():.4f}")
    print(f"   Std amplitude:    {amplitudes.std():.4f}")
    print(f"   Min amplitude:    {amplitudes.min():.4f}")
    print(f"   Max amplitude:    {amplitudes.max():.4f}")
    print(f"   Percentiles (25/50/75): {np.percentile(amplitudes, [25,50,75])}")

    # Energy spectrum
    print(f"\n⚡ Energy Distribution:")
    energies = np.sum(u_all ** 2, axis=(1, 2))
    print(f"   Mean energy:      {energies.mean():.4f}")
    print(f"   Std energy:       {energies.std():.4f}")
    print(f"   Min energy:       {energies.min():.4f}")
    print(f"   Max energy:       {energies.max():.4f}")

    # Gradient statistics (smoothness)
    print(f"\n🌊 Smoothness (spatial gradients):")
    dx = 1.0 / u_all.shape[2]
    gradients = np.abs(np.gradient(u_all, axis=2))
    grad_magnitudes = np.max(gradients, axis=(1, 2))
    print(f"   Mean max gradient: {grad_magnitudes.mean():.4f}")
    print(f"   Std max gradient:  {grad_magnitudes.std():.4f}")
    print(f"   Min max gradient:  {grad_magnitudes.min():.4f}")
    print(f"   Max max gradient:  {grad_magnitudes.max():.4f}")

    # Time evolution
    print(f"\n⏱️  Temporal Evolution:")
    time_change = np.abs(np.gradient(u_all, axis=1))
    tc_max = np.max(time_change, axis=(1, 2))
    print(f"   Mean max time derivative: {tc_max.mean():.4f}")
    print(f"   Std max time derivative:  {tc_max.std():.4f}")
    print(f"   Min max time derivative:  {tc_max.min():.4f}")
    print(f"   Max max time derivative:  {tc_max.max():.4f}")

    # PDE residual for constant nu=0.01
    print(f"\n🔬 PDE Residual Analysis (ν=0.01):")
    nu = 0.01
    dx = 1.0 / u_all.shape[2]
    dt = 1.0 / (u_all.shape[1] - 1)

    u_t = np.gradient(u_all, dt, axis=1)
    u_x = np.gradient(u_all, dx, axis=2)
    u_xx = np.gradient(u_x, dx, axis=2)

    pde_residual = u_t + u_all * u_x - nu * u_xx
    residual_rms = np.sqrt(np.mean(pde_residual ** 2, axis=(1, 2)))

    print(f"   Mean PDE RMS:     {residual_rms.mean():.6f}")
    print(f"   Std PDE RMS:      {residual_rms.std():.6f}")
    print(f"   Min PDE RMS:      {residual_rms.min():.6f}")
    print(f"   Max PDE RMS:      {residual_rms.max():.6f}")

    # Identify underrepresented regions
    print(f"\n🎯 Data Coverage Analysis:")

    # Group by amplitude ranges
    amp_ranges = [(0, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.2), (1.2, 2.0)]
    for low, high in amp_ranges:
        count = np.sum((amplitudes >= low) & (amplitudes < high))
        pct = 100 * count / len(amplitudes)
        print(f"   Amplitude [{low:4.1f}, {high:4.1f}): {count:4d} ({pct:5.1f}%)")

    # Recommendations
    print(f"\n💡 Recommendations for Dataset Expansion:")

    if amplitudes.std() / amplitudes.mean() < 0.3:
        print("   ⚠️  Low amplitude diversity - add wider amplitude range")

    if grad_magnitudes.std() / grad_magnitudes.mean() < 0.3:
        print("   ⚠️  Low smoothness diversity - add sharper transitions")

    if tc_max.std() / tc_max.mean() < 0.3:
        print("   ⚠️  Low temporal dynamics diversity")

    if len(amplitudes[amplitudes > 1.2]) < len(amplitudes) * 0.1:
        print("   ⚠️  Underrepresented high-amplitude solutions")

    if len(amplitudes[amplitudes < 0.3]) < len(amplitudes) * 0.1:
        print("   ⚠️  Underrepresented low-amplitude solutions")

    print("\n" + "=" * 70)

    return {
        'train_size': train.shape[0],
        'dev_size': dev.shape[0],
        'test_size': test.shape[0],
        'total_size': u_all.shape[0],
        'amplitude_mean': amplitudes.mean(),
        'amplitude_std': amplitudes.std(),
        'energy_mean': energies.mean(),
        'grad_mean': grad_magnitudes.mean(),
        'pde_rms_mean': residual_rms.mean()
    }


def recommend_expansion(current_stats: dict, target_increase: float = 1.5):
    """Recommend expansion strategy."""

    print("=" * 70)
    print("EXPANSION STRATEGY")
    print("=" * 70)

    current_total = current_stats['total_size']
    new_total = int(current_total * target_increase)
    new_trajectories = new_total - current_total

    print(f"\nTarget Dataset Size Increase: {target_increase}x")
    print(f"Current total:    {current_total} trajectories")
    print(f"Target total:     {new_total} trajectories")
    print(f"New trajectories: {new_trajectories} trajectories")

    print(f"\n📋 Recommended Strategies:")
    print(f"   1. Initial condition diversity:")
    print(f"      - Sine waves (multiple frequencies): 25%")
    print(f"      - Gaussian bumps (varied widths):    20%")
    print(f"      - Smooth steps (tanh):               20%")
    print(f"      - Multi-frequency combinations:      20%")
    print(f"      - Step functions:                    15%")

    print(f"\n   2. Parameter variations (ν):")
    print(f"      - Default (0.01):                    50%")
    print(f"      - Low viscosity (0.005):             20%")
    print(f"      - High viscosity (0.02-0.03):       30%")

    print(f"\n   3. Amplitude distribution:")
    print(f"      - Low (<0.5):      25%")
    print(f"      - Medium (0.5-1.0): 40%")
    print(f"      - High (>1.0):      35%")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "../datasets/burgers_augmented_v1"

    stats = analyze_current_dataset(data_dir)
    recommend_expansion(stats, target_increase=1.5)
