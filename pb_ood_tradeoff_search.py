"""
PB_OOD Trade-off 折中实验
目标：在 Test L2 最优 (d96, pw=0.05) 和 PDE RMS 最优 (d64, pw=0.30) 之间寻找最佳均衡

已知结果：
  d96 pw=0.05 : Test L2=0.3026  PDE=9.67   Gap=+83%   <- L2最优但PDE差
  d64 pw=0.18 : Test L2=0.4722  PDE=2.36   Gap=+14%   <- PDE尚可、Gap好
  d64 pw=0.30 : Test L2=0.5119  PDE=1.30   Gap=+9.5%  <- PDE最优但L2差

折中策略：d96 容量优势 + pw=0.12/0.18 适度物理约束
  方案A: d96, l4, pw=0.12   <- L2和PDE双向折中
  方案B: d96, l4, pw=0.18   <- 偏向物理优化
"""
import subprocess, sys, json, time
from pathlib import Path

BASE = Path("experiments/outputs_burgers_dataset")
SHARED = [
    "--n-heads", "4", "--cnn-channels", "32", "--token-count", "32",
    "--head-width", "32", "--latent-dim", "32", "--manifold-hidden", "256",
    "--highpass-scale", "0.5", "--temporal", "ode",
    "--steps", "24000", "--warmup-steps", "200",
    "--eval-every", "500", "--early-stop-patience", "1000",
    "--input-highpass", "--manifold-cnn", "--query-local-features",
    "--latent-ode", "--cross-attention",
]

CONFIGS = [
    {
        "name":    "pb_ood_tradeoff_d96_pw012",
        "desc":    "折中A: d96/l4 + pw=0.12",
        "d_model": 96, "n_layers": 4,
        "dropout": 0.08, "lr": 0.0009, "wd": 5e-5, "bs": 12,
        "pw": 0.12, "seed": 42,
    },
    {
        "name":    "pb_ood_tradeoff_d96_pw018",
        "desc":    "折中B: d96/l4 + pw=0.18",
        "d_model": 96, "n_layers": 4,
        "dropout": 0.08, "lr": 0.0009, "wd": 5e-5, "bs": 12,
        "pw": 0.18, "seed": 42,
    },
]

BASELINE = {"test_l2": 0.3363, "pde_rms": 10.694, "gap": 40.3}

for cfg in CONFIGS:
    out = BASE / cfg["name"]
    cmd = [
        sys.executable, "code/demo_burgers_dataset.py",
        "--preset", "wide",
        "--data-dir",   "datasets/burgers_custom_pb_ood",
        "--train-file", "burgers_nu0.01_train_split.parquet",
        "--dev-file",   "burgers_nu0.01_dev_split.parquet",
        "--test-file",  "burgers_nu0.01_test.parquet",
        "--output-dir", str(out),
        "--d-model",    str(cfg["d_model"]),
        "--n-layers",   str(cfg["n_layers"]),
        "--dropout",    str(cfg["dropout"]),
        "--lr",         str(cfg["lr"]),
        "--weight-decay", str(cfg["wd"]),
        "--batch-size", str(cfg["bs"]),
        "--physics-weight", str(cfg["pw"]),
        "--seed", str(cfg["seed"]),
    ] + SHARED

    print(f"\n{'='*65}")
    print(f"{cfg['desc']}")
    print(f"{'='*65}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=".")
    elapsed = time.time() - t0

    if r.returncode == 0:
        m = json.load(open(out / "metrics.json"))
        tl2 = m["test"]["relative_l2"]
        pde = m["test"]["pde_rms"]
        gap = (m["gap"]["test_over_train_ratio"] - 1) * 100
        print(f"\n  Test L2: {tl2:.4f}  (基线 {BASELINE['test_l2']:.4f}, {(tl2-BASELINE['test_l2'])/BASELINE['test_l2']*100:+.1f}%)")
        print(f"  PDE RMS: {pde:.4f}  (基线 {BASELINE['pde_rms']:.4f}, {(pde-BASELINE['pde_rms'])/BASELINE['pde_rms']*100:+.1f}%)")
        print(f"  Gap:     {gap:+.1f}%  (基线 +{BASELINE['gap']:.1f}%)")
        print(f"  耗时:    {elapsed/60:.0f} 分钟")
    else:
        print(f"  失败 (exit {r.returncode})")

print("\n" + "="*65)
print("Trade-off 折中实验完成")
print("="*65)
