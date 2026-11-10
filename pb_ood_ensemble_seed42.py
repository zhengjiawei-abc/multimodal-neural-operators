"""
PB_OOD 方向4: 集成学习 - 第一个成员 (seed=42)
集成策略：训练3个seed不同的基线模型，推理时平均预测
当前：训练集成成员1 (seed=42, 与基线同配置但不同seed)
"""
import subprocess, sys, json
from pathlib import Path

OUTPUT_DIR = Path("experiments/outputs_burgers_dataset/pb_ood_ensemble_s42")

cmd = [
    sys.executable, "code/demo_burgers_dataset.py",
    "--preset", "wide",
    "--data-dir", "datasets/burgers_custom_pb_ood",
    "--train-file", "burgers_nu0.01_train_000.parquet",
    "--dev-file", "burgers_nu0.01_test.parquet",
    "--test-file", "burgers_nu0.01_test.parquet",
    "--output-dir", str(OUTPUT_DIR),
    "--seed", "42",             # 集成成员1
    "--d-model", "64", "--n-layers", "3",
    "--dropout", "0.08", "--lr", "0.001", "--weight-decay", "5e-5",
    "--batch-size", "28",
    "--n-heads", "4", "--cnn-channels", "32", "--token-count", "32",
    "--head-width", "32", "--latent-dim", "32", "--manifold-hidden", "256",
    "--highpass-scale", "0.5", "--temporal", "ode",
    "--steps", "24000", "--warmup-steps", "200",
    "--eval-every", "500", "--early-stop-patience", "1000",
    "--input-highpass", "--manifold-cnn", "--query-local-features",
    "--latent-ode", "--cross-attention",
]

print("方向4: 集成成员1 (seed=42, d64, dropout=0.08)")
result = subprocess.run(cmd, cwd=".")
if result.returncode == 0:
    m = json.load(open(OUTPUT_DIR / "metrics.json"))
    print(f"Test L2={m['test']['relative_l2']:.4f}, Gap={m['gap']['test_minus_train_relative_l2']:+.4f}")
else:
    sys.exit(result.returncode)
