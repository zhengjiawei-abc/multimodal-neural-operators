"""
PB_OOD 方向1: 物理约束增强
physics_weight: 0.05 -> 0.3，强制模型学习Burgers方程规律
"""
import subprocess, sys, json, time
from pathlib import Path

OUTPUT_DIR = Path("experiments/outputs_burgers_dataset/pb_ood_physics_v1")

cmd = [
    sys.executable, "code/demo_burgers_dataset.py",
    "--preset", "wide",
    "--data-dir", "datasets/burgers_custom_pb_ood",
    "--train-file", "burgers_nu0.01_train_000.parquet",
    "--dev-file", "burgers_nu0.01_test.parquet",
    "--test-file", "burgers_nu0.01_test.parquet",
    "--output-dir", str(OUTPUT_DIR),
    "--seed", "123",
    "--d-model", "64", "--n-layers", "3",
    "--dropout", "0.10", "--lr", "0.001", "--weight-decay", "1e-4",
    "--batch-size", "24",
    "--n-heads", "4", "--cnn-channels", "32", "--token-count", "32",
    "--head-width", "32", "--latent-dim", "32", "--manifold-hidden", "256",
    "--highpass-scale", "0.5", "--temporal", "ode",
    "--steps", "24000", "--warmup-steps", "200",
    "--eval-every", "500", "--early-stop-patience", "1000",
    "--physics-weight", "0.3",          # 核心：从0.05增加到0.3
    "--input-highpass", "--manifold-cnn", "--query-local-features",
    "--latent-ode", "--cross-attention",
]

print("方向1: 物理约束增强 (physics_weight=0.3)")
result = subprocess.run(cmd, cwd=".")
if result.returncode == 0:
    m = json.load(open(OUTPUT_DIR / "metrics.json"))
    print(f"Test L2={m['test']['relative_l2']:.4f}, Gap={m['gap']['test_minus_train_relative_l2']:+.4f}")
else:
    sys.exit(result.returncode)
