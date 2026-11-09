"""
Expand Burgers dataset with new initial conditions and parameter variations.
Extends burgers_augmented_v1 to v2 with:
1. Different initial condition types (sine, gaussian, step)
2. Parameter variation (nu values)
3. Space-time resolution variations
"""

import numpy as np
import os
from pathlib import Path
from burgers_solver import BurgersSolver
import argparse


def generate_burgers_solutions(
    num_trajectories: int = 1000,
    nx: int = 256,
    nt: int = 101,
    nu_values: list = None,
    ic_types: list = None,
    seed: int = 42
) -> dict:
    """
    Generate Burgers equation solutions with varied parameters.

    Args:
        num_trajectories: Number of solution trajectories
        nx: Spatial grid points
        nt: Temporal grid points
        nu_values: List of viscosity coefficients
        ic_types: List of initial condition types
        seed: Random seed

    Returns:
        dict with keys: u, x, t, nu, ic_type
    """
    np.random.seed(seed)

    if nu_values is None:
        nu_values = [0.005, 0.01, 0.02, 0.03]  # Parameter variation
    if ic_types is None:
        ic_types = ['sine', 'gaussian', 'step', 'multi_sine', 'smooth_step']

    solver = BurgersSolver(nx=nx, nt=nt)

    solutions = {
        'u': [],
        'x': [],
        't': [],
        'nu': [],
        'ic_type': [],
        'ic_amplitude': []
    }

    x = np.linspace(0, 1, nx, endpoint=False)
    t = np.linspace(0, 1, nt)

    solutions['x'] = x
    solutions['t'] = t

    for i in range(num_trajectories):
        # Randomly select parameters
        nu = np.random.choice(nu_values)
        ic_type = np.random.choice(ic_types)

        # Generate diverse initial conditions
        if ic_type == 'sine':
            # Multiple frequency sine waves
            freq = np.random.randint(1, 5)
            amplitude = np.random.uniform(0.5, 1.5)
            u0 = amplitude * np.sin(2 * np.pi * freq * x)

        elif ic_type == 'gaussian':
            # Gaussian bump
            center = np.random.uniform(0.2, 0.8)
            width = np.random.uniform(0.05, 0.2)
            amplitude = np.random.uniform(0.5, 1.5)
            u0 = amplitude * np.exp(-((x - center) ** 2) / (2 * width ** 2))

        elif ic_type == 'step':
            # Step function
            threshold = np.random.uniform(0.3, 0.7)
            amplitude = np.random.uniform(0.5, 1.5)
            u0 = np.where(x < threshold, amplitude, -amplitude)

        elif ic_type == 'multi_sine':
            # Multiple sine components
            freq1 = np.random.randint(1, 3)
            freq2 = np.random.randint(3, 6)
            amp1 = np.random.uniform(0.3, 0.8)
            amp2 = np.random.uniform(0.2, 0.5)
            u0 = amp1 * np.sin(2 * np.pi * freq1 * x) + amp2 * np.sin(2 * np.pi * freq2 * x)

        elif ic_type == 'smooth_step':
            # Smooth step (tanh)
            center = np.random.uniform(0.3, 0.7)
            width = np.random.uniform(0.05, 0.15)
            amplitude = np.random.uniform(0.5, 1.5)
            u0 = amplitude * np.tanh((x - center) / width)

        # Solve PDE
        try:
            solver.nu = nu
            u = solver.solve(u0)

            solutions['u'].append(u)
            solutions['nu'].append(nu)
            solutions['ic_type'].append(ic_type)
            solutions['ic_amplitude'].append(np.max(np.abs(u0)))

            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_trajectories} trajectories")

        except Exception as e:
            print(f"Failed to solve for trajectory {i}: {e}")
            continue

    # Convert to numpy arrays
    solutions['u'] = np.array(solutions['u'])
    solutions['x'] = solutions['x']
    solutions['t'] = solutions['t']
    solutions['nu'] = np.array(solutions['nu'])
    solutions['ic_type'] = np.array(solutions['ic_type'])
    solutions['ic_amplitude'] = np.array(solutions['ic_amplitude'])

    return solutions


def load_augmented_v1(data_dir: str = "../datasets/burgers_augmented_v1") -> dict:
    """Load existing augmented_v1 dataset."""
    train = np.load(f"{data_dir}/burgers_augmented_train.npz")
    dev = np.load(f"{data_dir}/burgers_augmented_dev.npz")
    test = np.load(f"{data_dir}/burgers_augmented_test.npz")

    return {
        'train': {k: train[k] for k in train.files},
        'dev': {k: dev[k] for k in dev.files},
        'test': {k: test[k] for k in test.files}
    }


def split_dataset(u_all, x, t, test_ratio=0.15, dev_ratio=0.10, seed=42):
    """Split dataset into train/dev/test."""
    np.random.seed(seed)
    n = len(u_all)

    indices = np.random.permutation(n)
    test_idx = indices[:int(n * test_ratio)]
    dev_idx = indices[int(n * test_ratio):int(n * (test_ratio + dev_ratio))]
    train_idx = indices[int(n * (test_ratio + dev_ratio)):]

    return {
        'train': {'u': u_all[train_idx], 'x': x, 't': t},
        'dev': {'u': u_all[dev_idx], 'x': x, 't': t},
        'test': {'u': u_all[test_idx], 'x': x, 't': t}
    }


def main():
    parser = argparse.ArgumentParser(description="Expand Burgers dataset")
    parser.add_argument("--num-new-trajectories", type=int, default=2000,
                        help="Number of new trajectories to generate")
    parser.add_argument("--output-dir", type=str, default="../datasets/burgers_augmented_v2",
                        help="Output directory for v2 dataset")
    parser.add_argument("--include-v1", action="store_true", default=True,
                        help="Include v1 data in v2")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Burgers Dataset Expansion: v1 → v2")
    print("=" * 60)

    # Generate new data
    print(f"\n1. Generating {args.num_new_trajectories} new trajectories...")
    new_data = generate_burgers_solutions(
        num_trajectories=args.num_new_trajectories,
        seed=args.seed
    )

    # Load v1 data if requested
    if args.include_v1:
        print("\n2. Loading burgers_augmented_v1...")
        v1_data = load_augmented_v1()
        print(f"   Train: {v1_data['train']['u'].shape[0]} trajectories")
        print(f"   Dev:   {v1_data['dev']['u'].shape[0]} trajectories")
        print(f"   Test:  {v1_data['test']['u'].shape[0]} trajectories")

        # Combine v1 + new data
        print("\n3. Combining v1 + new data...")
        u_combined = np.concatenate([
            v1_data['train']['u'],
            v1_data['dev']['u'],
            v1_data['test']['u'],
            new_data['u']
        ])
        print(f"   Total trajectories: {u_combined.shape[0]}")
    else:
        u_combined = new_data['u']

    # Split into train/dev/test
    print("\n4. Splitting dataset...")
    splits = split_dataset(
        u_combined,
        new_data['x'],
        new_data['t'],
        test_ratio=0.15,
        dev_ratio=0.10,
        seed=args.seed
    )

    # Save dataset
    print("\n5. Saving v2 dataset...")
    for split_name, split_data in splits.items():
        filename = f"{args.output_dir}/burgers_augmented_{split_name}.npz"
        np.savez_compressed(
            filename,
            u=split_data['u'],
            x=split_data['x'],
            t=split_data['t']
        )
        print(f"   {split_name}: {split_data['u'].shape[0]} trajectories → {filename}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("Dataset Expansion Summary")
    print("=" * 60)
    print(f"Train set: {splits['train']['u'].shape[0]} trajectories")
    print(f"Dev set:   {splits['dev']['u'].shape[0]} trajectories")
    print(f"Test set:  {splits['test']['u'].shape[0]} trajectories")
    print(f"Total:     {u_combined.shape[0]} trajectories")
    print(f"Spatial resolution: {new_data['x'].shape[0]} points")
    print(f"Temporal resolution: {new_data['t'].shape[0]} points")

    if 'nu' in new_data and len(new_data['nu']) > 0:
        print(f"\nParameter variations (ν):")
        print(f"  Min: {new_data['nu'].min():.4f}")
        print(f"  Max: {new_data['nu'].max():.4f}")
        print(f"  Mean: {new_data['nu'].mean():.4f}")

    if 'ic_type' in new_data and len(new_data['ic_type']) > 0:
        print(f"\nInitial condition types:")
        unique_types, counts = np.unique(new_data['ic_type'], return_counts=True)
        for ic_type, count in zip(unique_types, counts):
            print(f"  {ic_type}: {count} trajectories")


if __name__ == "__main__":
    main()
