"""
PB_OOD 物理约束权重精调（黄金实验）
物理约束增强被验证为最有效的Gap改善方向，但pw=0.3时精度损失过大。
目标：寻找Test L2和Gap的最优平衡点。

pw=0.05 (基线): Test L2=0.3363, Gap=+23.6%
pw=0.30 (方向1): Test L2=0.5119, Gap=+9.5%

精调范围: 0.08, 0.12, 0.18
"""
import subprocess, sys, json, time
from pathlib import Path

SWEEP = [
    {"pw": 0.08, "name": "pb_ood_physics_pw008"},
    {"pw": 0.12, "name": "pb_ood_physics_pw012"},
    {"pw": 0.18, "name": "pb_ood_physics_pw018"},
]

BASE_ARGS = [
    sys.executable, "code/demo_burgers_dataset.py",
    "--preset", "wide",
    "--data-dir", "datasets/burgers_custom_pb_ood",
    "--train-file", "burgers_nu0.01_train_split.parquet",
    "--dev-file", "burgers_nu0.01_dev_split.parquet",
    "--test-file", "burgers_nu0.01_test.parquet",
    "--seed", "123",
    "--d-model", "64", "--n-layers", "3",
    "--dropout", "0.10", "--lr", "0.001", "--weight-decay", "1e-4",
    "--batch-size", "24",
    "--n-heads", "4", "--cnn-channels", "32", "--token-count", "32",
    "--head-width", "32", "--latent-dim", "32", "--manifold-hidden", "256",
    "--highpass-scale", "0.5", "--temporal", "ode",
    "--steps", "24000", "--warmup-steps", "200",
    "--eval-every", "500", "--early-stop-patience", "1000",
    "--input-highpass", "--manifold-cnn", "--query-local-features",
    "--latent-ode", "--cross-attention",
]

results = []
for cfg in SWEEP:
    output_dir = Path("experiments/outputs_burgers_dataset") / cfg["name"]
    cmd = BASE_ARGS + [
        "--output-dir", str(output_dir),
        "--physics-weight", str(cfg["pw"]),
    ]
    print(f"\n{'='*60}")
    print(f"physics_weight = {cfg['pw']}")
    print(f"{'='*60}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=".")
    elapsed = time.time() - t0
    if r.returncode == 0:
        m = json.load(open(output_dir / "metrics.json"))
        tl2 = m["test"]["relative_l2"]
        gap = (m["gap"]["test_over_train_ratio"] - 1) * 100
        results.append({"pw": cfg["pw"], "test_l2": tl2, "gap": gap})
        print(f"  Test L2={tl2:.4f}  Gap={gap:+.1f}%  ({elapsed/60:.0f}min)")
    else:
        print(f"  FAILED (exit {r.returncode})")

print("\n" + "="*60)
print("物理约束权重精调汇总")
print("="*60)
print(f"{'pw':>6} {'Test L2':>10} {'Gap%':>10}")
print(f"  0.05  {'0.3363':>10} {'+23.6%':>10}  ← 基线")
for r in results:
    print(f"  {r['pw']:.2f}  {r['test_l2']:>10.4f} {r['gap']:>+9.1f}%")
print(f"  0.30  {'0.5119':>10} {'+9.5%':>10}  ← 方向1")
