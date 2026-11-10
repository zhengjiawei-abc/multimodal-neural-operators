"""
PB_OOD d96变体 — 对应PB_ID自适应成功配置，测试同等容量在域转移场景的效果
目的：补全"容量×数据集"矩阵中 PB_OOD + d96 这个缺失的格子
"""
import subprocess, sys, json
from pathlib import Path

OUTPUT_DIR = Path("experiments/outputs_burgers_dataset/pb_ood_d96_variant")

cmd = [
    sys.executable, "code/demo_burgers_dataset.py",
    "--preset", "wide",
    "--data-dir", "datasets/burgers_custom_pb_ood",
    "--train-file", "burgers_nu0.01_train_000.parquet",
    "--dev-file",   "burgers_nu0.01_test.parquet",
    "--test-file",  "burgers_nu0.01_test.parquet",
    "--output-dir", str(OUTPUT_DIR),
    "--seed", "42",
    "--d-model", "96", "--n-layers", "4",
    "--dropout", "0.08", "--lr", "0.0009", "--weight-decay", "5e-5",
    "--batch-size", "12",           # 与PB_ID自适应相同（避免OOM）
    "--n-heads", "4", "--cnn-channels", "32", "--token-count", "32",
    "--head-width", "32", "--latent-dim", "32", "--manifold-hidden", "256",
    "--highpass-scale", "0.5", "--temporal", "ode",
    "--steps", "24000", "--warmup-steps", "200",
    "--eval-every", "500", "--early-stop-patience", "1000",
    "--input-highpass", "--manifold-cnn", "--query-local-features",
    "--latent-ode", "--cross-attention",
]

print("PB_OOD d96变体: 与PB_ID成功配置完全一致")
print("期望验证：大容量模型在域转移数据上是否复现 PB_ID 的24.4%提升")
result = subprocess.run(cmd, cwd=".")
if result.returncode == 0:
    m = json.load(open(OUTPUT_DIR / "metrics.json"))
    tl2 = m['test']['relative_l2']
    gap = (m['gap']['test_over_train_ratio'] - 1) * 100
    print(f"\nPB_OOD d96结果:")
    print(f"  Test L2: {tl2:.4f}  (基线 0.3363, {'↓' if tl2<0.3363 else '↑'}{abs(tl2-0.3363)/0.3363*100:.1f}%)")
    print(f"  Gap:     {gap:+.1f}%  (基线 +23.6%)")
else:
    sys.exit(result.returncode)
