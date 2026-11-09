"""
Expand Burgers dataset - memory-efficient streaming version.
"""

import numpy as np
import os
from pathlib import Path
import argparse
from tqdm import tqdm


def solve_burgers_1d(x, t, u0, nu):
    """Solve 1D Burgers - explicit finite difference."""
    nx = len(x)
    nt = len(t)
    dx = x[1] - x[0] if nx > 1 else 1.0
    dt = t[1] - t[0] if nt > 1 else 1.0

    u = np.zeros((nt, nx), dtype=np.float32)
    u[0] = u0.astype(np.float32)

    for n in range(nt - 1):
        u_curr = u[n]
        u_left = np.roll(u_curr, 1)
        u_right = np.roll(u_curr, -1)

        du_dx = (u_right - u_left) / (2 * dx)
        d2u_dx2 = (u_right - 2 * u_curr + u_left) / (dx ** 2)

        du_dt = -u_curr * du_dx + nu * d2u_dx2
        u[n + 1] = u_curr + dt * du_dt

    return u.T.astype(np.float32)  # (nx, nt)


def generate_dataset_streaming(num_trajectories, output_path, seed=42, chunk_size=100):
    """Generate and save dataset in streaming fashion."""

    np.random.seed(seed)

    nx, nt = 256, 101
    x = np.linspace(0, 1, nx, endpoint=False)
    t = np.linspace(0, 1, nt)

    nu_values = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
    ic_types = ['sine', 'gaussian', 'multi_sine', 'smooth_step', 'step', 'triangle']

    solutions_chunk = []
    chunk_count = 0

    print(f"Generating {num_trajectories} trajectories (streaming)...")

    for i in tqdm(range(num_trajectories), desc="Progress"):
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
            u0 = np.where(x < peak, amp * x / peak, -amp * (x - 1) / (1 - peak))

        try:
            u = solve_burgers_1d(x, t, u0, nu)
            solutions_chunk.append(u)

            # Save chunk when it reaches size
            if len(solutions_chunk) >= chunk_size:
                chunk_file = f"{output_path}_chunk_{chunk_count:03d}.npz"
                np.savez_compressed(chunk_file, data=np.array(solutions_chunk, dtype=np.float32))
                solutions_chunk = []
                chunk_count += 1

        except Exception as e:
            tqdm.write(f"Error at trajectory {i}: {e}")
            continue

    # Save final chunk
    if solutions_chunk:
        chunk_file = f"{output_path}_chunk_{chunk_count:03d}.npz"
        np.savez_compressed(chunk_file, data=np.array(solutions_chunk, dtype=np.float32))

    print(f"Generated {chunk_count + 1} chunks")
    return chunk_count + 1


def merge_chunks(output_dir, num_chunks, v1_data_dir=None):
    """Merge chunks and combine with v1 if specified."""

    # Load all new chunks
    print(f"\nMerging {num_chunks} chunks...")
    all_data = []

    for i in range(num_chunks):
        chunk_file = f"{output_dir}_chunk_{i:03d}.npz"
        data = np.load(chunk_file)['data']
        all_data.append(data)
        print(f"  Chunk {i}: {data.shape}")

    u_new = np.concatenate(all_data)  # (new_n, 256, 101)
    print(f"Total new data: {u_new.shape}")

    # Load v1 if specified
    if v1_data_dir:
        print(f"\nLoading v1 data...")
        u_v1_train = np.load(f"{v1_data_dir}/burgers_augmented_train.npz")['data']
        u_v1_dev = np.load(f"{v1_data_dir}/burgers_augmented_dev.npz")['data']
        u_v1_test = np.load(f"{v1_data_dir}/burgers_augmented_test.npz")['data']

        u_combined = np.concatenate([u_v1_train, u_v1_dev, u_v1_test, u_new])
        print(f"Combined size: {u_combined.shape}")
    else:
        u_combined = u_new

    # Split
    print(f"\nSplitting into train/dev/test...")
    np.random.seed(42)
    n = len(u_combined)
    indices = np.random.permutation(n)

    test_n = int(n * 0.15)
    dev_n = int(n * 0.10)

    test_idx = indices[:test_n]
    dev_idx = indices[test_n:test_n + dev_n]
    train_idx = indices[test_n + dev_n:]

    # Save final splits
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

    # Clean up chunks
    print(f"\nCleaning up chunks...")
    for i in range(num_chunks):
        chunk_file = f"{output_dir}_chunk_{i:03d}.npz"
        if os.path.exists(chunk_file):
            os.remove(chunk_file)

    print("\n✅ Complete!")
    print(f"Output: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Expand Burgers dataset to v2")
    parser.add_argument("--num-trajectories", type=int, default=5000)
    parser.add_argument("--output-dir", type=str, default="../datasets/burgers_augmented_v2")
    parser.add_argument("--include-v1", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=100)

    args = parser.parse_args()

    print("=" * 70)
    print("BURGERS DATASET EXPANSION: v1 → v2 (Memory-efficient)")
    print("=" * 70)

    # Generate
    num_chunks = generate_dataset_streaming(
        args.num_trajectories,
        args.output_dir,
        seed=args.seed,
        chunk_size=args.chunk_size
    )

    # Merge
    v1_dir = "../datasets/burgers_augmented_v1" if args.include_v1 else None
    merge_chunks(args.output_dir, num_chunks, v1_dir)


if __name__ == "__main__":
    main()
