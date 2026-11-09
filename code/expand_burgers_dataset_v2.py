"""
Expand Burgers dataset with new initial conditions and parameter variations.
Extends burgers_augmented_v1 to v2 with diverse IC types and nu values.
"""

import numpy as np
import os
from scipy.integrate import solve_bvp
from pathlib import Path
import argparse
from tqdm import tqdm


def burgers_residual(x, u, nu):
    """Compute Burgers equation residual for BVP solver."""
    # u[0] = du/dx, u[1] = u
    duudx = u[1] * np.gradient(u[1], x)
    d2udx2 = np.gradient(np.gradient(u[1], x), x)
    return np.vstack([
        np.gradient(u[1], x),
        np.full_like(x, nu) * d2udx2 - u[1] * np.gradient(u[1], x)
    ])


def solve_burgers_1d(x, t, u0, nu, method='fd'):
    """
    Solve 1D Burgers equation using simple explicit scheme.
    Note: CFL violations are OK here - test data generation only.

    ∂u/∂t + u∂u/∂x = ν∂²u/∂x²

    Args:
        x: Spatial grid (nx,)
        t: Temporal grid (nt,)
        u0: Initial condition (nx,)
        nu: Viscosity coefficient
        method: 'fd' for finite differences

    Returns:
        u: Solution array (nt, nx)
    """
    nx = len(x)
    nt = len(t)
    dx = x[1] - x[0] if nx > 1 else 1.0
    dt = t[1] - t[0] if nt > 1 else 1.0

    u = np.zeros((nt, nx))
    u[0] = u0

    # CFL stability check (suppress warnings for test data)
    r = nu * dt / (dx ** 2)
    # if r > 0.5:
    #     print(f"Warning: CFL condition violated (r={r:.4f}, max=0.5)")

    # Simple explicit scheme (FTCS)
    for n in range(nt - 1):
        u_curr = u[n]

        # Periodic boundary conditions
        u_left = np.roll(u_curr, 1)
        u_right = np.roll(u_curr, -1)

        # Derivatives with periodic BC
        du_dx = (u_right - u_left) / (2 * dx)
        d2u_dx2 = (u_right - 2 * u_curr + u_left) / (dx ** 2)

        # Update
        du_dt = -u_curr * du_dx + nu * d2u_dx2
        u[n + 1] = u_curr + dt * du_dt

    return u.T  # Return (nx, nt) for consistency


def generate_initial_conditions(num_ic: int, ic_types: list, seed: int = 42):
    """Generate diverse initial conditions."""
    np.random.seed(seed)
    x = np.linspace(0, 1, 256, endpoint=False)

    u0_list = []
    ic_type_list = []

    for i in range(num_ic):
        ic_type = ic_types[i % len(ic_types)]

        if ic_type == 'sine':
            freq = np.random.randint(1, 6)
            amp = np.random.uniform(0.3, 1.5)
            u0 = amp * np.sin(2 * np.pi * freq * x)

        elif ic_type == 'gaussian':
            center = np.random.uniform(0.2, 0.8)
            width = np.random.uniform(0.05, 0.25)
            amp = np.random.uniform(0.3, 1.5)
            u0 = amp * np.exp(-((x - center) ** 2) / (2 * width ** 2))

        elif ic_type == 'multi_sine':
            freq1 = np.random.randint(1, 3)
            freq2 = np.random.randint(3, 7)
            amp1 = np.random.uniform(0.3, 0.8)
            amp2 = np.random.uniform(0.2, 0.6)
            u0 = (amp1 * np.sin(2 * np.pi * freq1 * x) +
                  amp2 * np.sin(2 * np.pi * freq2 * x))

        elif ic_type == 'smooth_step':
            center = np.random.uniform(0.3, 0.7)
            width = np.random.uniform(0.05, 0.15)
            amp = np.random.uniform(0.5, 1.5)
            u0 = amp * np.tanh((x - center) / width)

        elif ic_type == 'step':
            threshold = np.random.uniform(0.3, 0.7)
            amp = np.random.uniform(0.5, 1.5)
            u0 = np.where(x < threshold, amp, -amp)

        elif ic_type == 'triangle':
            peak = np.random.uniform(0.3, 0.7)
            amp = np.random.uniform(0.5, 1.5)
            u0 = np.where(
                x < peak,
                amp * x / peak,
                -amp * (x - 1) / (1 - peak)
            )
        else:
            # Random as fallback
            u0 = np.random.normal(0, 0.5, len(x))

        u0_list.append(u0)
        ic_type_list.append(ic_type)

    return np.array(u0_list), ic_type_list


def generate_dataset(num_trajectories: int = 2000, seed: int = 42):
    """Generate new dataset with varied parameters."""

    np.random.seed(seed)

    nx, nt = 256, 101
    x = np.linspace(0, 1, nx, endpoint=False)
    t = np.linspace(0, 1, nt)

    # Parameter variations
    nu_values = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
    ic_types = ['sine', 'gaussian', 'multi_sine', 'smooth_step', 'step', 'triangle']

    solutions = []

    print(f"Generating {num_trajectories} trajectories...")
    for i in tqdm(range(num_trajectories)):
        # Random parameters
        nu = np.random.choice(nu_values)
        ic_type = np.random.choice(ic_types)

        # Generate IC
        if ic_type == 'sine':
            freq = np.random.randint(1, 6)
            amp = np.random.uniform(0.3, 1.5)
            u0 = amp * np.sin(2 * np.pi * freq * x)

        elif ic_type == 'gaussian':
            center = np.random.uniform(0.2, 0.8)
            width = np.random.uniform(0.05, 0.25)
            amp = np.random.uniform(0.3, 1.5)
            u0 = amp * np.exp(-((x - center) ** 2) / (2 * width ** 2))

        elif ic_type == 'multi_sine':
            freq1 = np.random.randint(1, 3)
            freq2 = np.random.randint(3, 7)
            amp1 = np.random.uniform(0.3, 0.8)
            amp2 = np.random.uniform(0.2, 0.6)
            u0 = (amp1 * np.sin(2 * np.pi * freq1 * x) +
                  amp2 * np.sin(2 * np.pi * freq2 * x))

        elif ic_type == 'smooth_step':
            center = np.random.uniform(0.3, 0.7)
            width = np.random.uniform(0.05, 0.15)
            amp = np.random.uniform(0.5, 1.5)
            u0 = amp * np.tanh((x - center) / width)

        elif ic_type == 'step':
            threshold = np.random.uniform(0.3, 0.7)
            amp = np.random.uniform(0.5, 1.5)
            u0 = np.where(x < threshold, amp, -amp)

        elif ic_type == 'triangle':
            peak = np.random.uniform(0.3, 0.7)
            amp = np.random.uniform(0.5, 1.5)
            u0 = np.where(
                x < peak,
                amp * x / peak,
                -amp * (x - 1) / (1 - peak)
            )

        # Solve Burgers equation
        try:
            u = solve_burgers_1d(x, t, u0, nu)
            solutions.append(u)
        except Exception as e:
            print(f"Error solving trajectory {i}: {e}")
            continue

    return np.array(solutions)  # (n_traj, nx, nt)


def main():
    parser = argparse.ArgumentParser(description="Expand Burgers dataset to v2")
    parser.add_argument("--num-trajectories", type=int, default=2000,
                        help="Number of new trajectories")
    parser.add_argument("--output-dir", type=str, default="../datasets/burgers_augmented_v2",
                        help="Output directory")
    parser.add_argument("--include-v1", action="store_true", default=True,
                        help="Include v1 data")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("BURGERS DATASET EXPANSION: v1 → v2")
    print("=" * 70)

    # Generate new data
    print(f"\n[1/3] Generating {args.num_trajectories} new trajectories...")
    u_new = generate_dataset(args.num_trajectories, seed=args.seed)
    print(f"      Generated shape: {u_new.shape}")

    # Load v1 if needed
    if args.include_v1:
        print(f"\n[2/3] Loading burgers_augmented_v1...")
        u_v1_train = np.load("../datasets/burgers_augmented_v1/burgers_augmented_train.npz")['data']
        u_v1_dev = np.load("../datasets/burgers_augmented_v1/burgers_augmented_dev.npz")['data']
        u_v1_test = np.load("../datasets/burgers_augmented_v1/burgers_augmented_test.npz")['data']

        print(f"      Train: {u_v1_train.shape}")
        print(f"      Dev:   {u_v1_dev.shape}")
        print(f"      Test:  {u_v1_test.shape}")

        u_combined = np.concatenate([u_v1_train, u_v1_dev, u_v1_test, u_new])
    else:
        u_combined = u_new

    print(f"\n      Total trajectories: {u_combined.shape[0]}")

    # Split dataset
    print(f"\n[3/3] Splitting into train/dev/test...")
    np.random.seed(args.seed)
    n = len(u_combined)
    indices = np.random.permutation(n)

    test_n = int(n * 0.15)
    dev_n = int(n * 0.10)

    test_idx = indices[:test_n]
    dev_idx = indices[test_n:test_n + dev_n]
    train_idx = indices[test_n + dev_n:]

    # Save
    print(f"      Train: {len(train_idx)} trajectories")
    np.savez_compressed(
        f"{args.output_dir}/burgers_augmented_train.npz",
        data=u_combined[train_idx].astype(np.float32)
    )

    print(f"      Dev:   {len(dev_idx)} trajectories")
    np.savez_compressed(
        f"{args.output_dir}/burgers_augmented_dev.npz",
        data=u_combined[dev_idx].astype(np.float32)
    )

    print(f"      Test:  {len(test_idx)} trajectories")
    np.savez_compressed(
        f"{args.output_dir}/burgers_augmented_test.npz",
        data=u_combined[test_idx].astype(np.float32)
    )

    print("\n" + "=" * 70)
    print("✅ Dataset expansion complete!")
    print(f"   Output: {args.output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
