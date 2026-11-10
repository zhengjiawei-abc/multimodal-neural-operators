"""增强数据（修复版）+ 基线配置重跑"""
import subprocess, sys, json
from pathlib import Path

OUTPUT_DIR = Path("experiments/outputs_burgers_dataset/pb_ood_aug_baseline")
cmd = [
    sys.executable, "code/demo_burgers_dataset.py",
    "--preset", "wide",
    "--data-dir",   "datasets/burgers_pb_ood_augmented",
    "--train-file", "train_augmented.npz",
    "--dev-file",   "burgers_nu0.01_test.parquet",
    "--test-file",  "burgers_nu0.01_test.parquet",
    "--output-dir", str(OUTPUT_DIR),
    "--seed", "123", "--d-model", "64", "--n-layers", "3",
    "--dropout", "0.05", "--lr", "0.001", "--weight-decay", "0.0",
    "--batch-size", "28",
    "--n-heads", "4", "--cnn-channels", "32", "--token-count", "32",
    "--head-width", "32", "--latent-dim", "32", "--manifold-hidden", "256",
    "--highpass-scale", "0.5", "--temporal", "ode",
    "--steps", "24000", "--warmup-steps", "200",
    "--eval-every", "500", "--early-stop-patience", "1000",
    "--input-highpass", "--manifold-cnn", "--query-local-features",
    "--latent-ode", "--cross-attention",
]
print("增强数据（修复版）+ 基线配置")
r = subprocess.run(cmd, cwd=".")
if r.returncode == 0:
    m = json.load(open(OUTPUT_DIR / "metrics.json"))
    tl2 = m['test']['relative_l2']
    gap = (m['gap']['test_over_train_ratio'] - 1) * 100
    print(f"Test L2={tl2:.4f} ({'↓' if tl2<0.3363 else '↑'}{abs(tl2-0.3363)/0.3363*100:.1f}% vs基线), Gap={gap:+.1f}%")
else:
    sys.exit(r.returncode)
