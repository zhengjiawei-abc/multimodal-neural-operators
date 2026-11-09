"""
Monitor v_test_xl training and extract final metrics.
"""

import json
import os
from pathlib import Path
import time

exp_dir = Path("../experiments/v_test_xl")

print("=" * 70)
print("MONITORING V2/TEST TRAINING")
print("=" * 70)

max_wait = 2400  # 40 minutes
check_interval = 30  # 30 seconds
elapsed = 0

while elapsed < max_wait:
    if exp_dir.exists():
        metrics_file = exp_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)

            # Extract results
            test_l2 = metrics.get('test', {}).get('relative_l2')
            train_l2 = metrics.get('train', {}).get('relative_l2')
            pde_rms = metrics.get('test', {}).get('pde_rms')
            step = metrics.get('stability', {}).get('best_dev_step', 0)

            print(f"\n✓ Training Complete!")
            print(f"\nFinal Metrics:")
            print(f"  Train L2: {train_l2:.4f}")
            print(f"  Test L2:  {test_l2:.4f}")
            print(f"  PDE RMS:  {pde_rms:.6f}")
            print(f"  Best step: {step}")

            # Compare to v1
            v1_test = 0.2461
            improvement = (v1_test - test_l2) / v1_test * 100

            print(f"\nComparison to v1:")
            print(f"  v1 Test L2: {v1_test:.4f}")
            print(f"  v2 Test L2: {test_l2:.4f}")
            print(f"  Improvement: {improvement:.1f}% {'✓' if improvement >= 5 else '✗'}")

            print("\n" + "=" * 70)
            break

    elapsed += check_interval
    progress = elapsed / max_wait * 100
    print(f"[{progress:3.0f}%] Waiting for training to complete... ({elapsed//60}m {elapsed%60}s)", end='\r')
    time.sleep(check_interval)

if elapsed >= max_wait:
    print("\n⚠ Training timeout (>40 min)")
