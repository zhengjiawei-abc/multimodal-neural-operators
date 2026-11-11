"""
PB_OOD 方向3: 多任务/元学习
小batch + 物理约束 + 慢学习率，让模型更关注跨参数的物理规律
"""
import subprocess, sys, json
from pathlib import Path

OUTPUT_DIR = Path("experiments/outputs_burgers_dataset/pb_ood_multitask_v1")

cmd = [
    sys.executable, "code/demo_burgers_dataset.py",
    "--preset", "wide",
    "--data-dir", "datasets/burgers_custom_pb_ood",
    "--train-file", "burgers_nu0.01_train_split.parquet",
    "--dev-file", "burgers_nu0.01_dev_split.parquet",
    "--test-file", "burgers_nu0.01_test.parquet",
    "--output-dir", str(OUTPUT_DIR),
    "--seed", "42",             # 不同seed增加多样性
    "--d-model", "64", "--n-layers", "3",
    "--dropout", "0.10", "--lr", "0.0008", "--weight-decay", "1e-4",
    "--batch-size", "16",       # 核心：小batch引入更多多样性
    "--n-heads", "4", "--cnn-channels", "32", "--token-count", "32",
    "--head-width", "32", "--latent-dim", "32", "--manifold-hidden", "256",
    "--highpass-scale", "0.5", "--temporal", "ode",
    "--steps", "24000", "--warmup-steps", "500",   # 更长warmup
    "--eval-every", "500", "--early-stop-patience", "1000",
    "--physics-weight", "0.15",  # 适度增加物理约束
    "--input-highpass", "--manifold-cnn", "--query-local-features",
    "--latent-ode", "--cross-attention",
]

print("方向3: 多任务/元学习 (小batch=16, physics=0.15, 慢lr=0.0008)")
result = subprocess.run(cmd, cwd=".")
if result.returncode == 0:
    m = json.load(open(OUTPUT_DIR / "metrics.json"))
    print(f"Test L2={m['test']['relative_l2']:.4f}, Gap={m['gap']['test_minus_train_relative_l2']:+.4f}")
else:
    sys.exit(result.returncode)
