"""v4增强数据完成后立即启动三组训练"""
import subprocess, sys, json
from pathlib import Path

OUTPUT_BASE = Path("experiments/outputs_burgers_dataset")

configs = [
    {
        'name': 'pb_ood_aug_v4_baseline',
        'desc': '增强v4 + 基线',
        'seed': 123, 'pw': 0.05, 'dropout': 0.05, 'lr': 0.001, 'wd': 0.0, 'bs': 28,
    },
    {
        'name': 'pb_ood_aug_v4_physics',
        'desc': '增强v4 + 物理约束',
        'seed': 123, 'pw': 0.12, 'dropout': 0.10, 'lr': 0.001, 'wd': 1e-4, 'bs': 24,
    },
    {
        'name': 'pb_ood_aug_v4_seed42',
        'desc': '增强v4 + seed=42',
        'seed': 42, 'pw': 0.05, 'dropout': 0.08, 'lr': 0.001, 'wd': 5e-5, 'bs': 28,
    },
]

for cfg in configs:
    out_dir = OUTPUT_BASE / cfg['name']
    cmd = [
        sys.executable, "code/demo_burgers_dataset.py",
        "--preset", "wide",
        "--data-dir",   "datasets/burgers_pb_ood_augmented",
        "--train-file", "train_augmented.npz",
        "--dev-file",   "../burgers_custom_pb_ood/burgers_nu0.01_dev_split.parquet",
        "--test-file",  "burgers_nu0.01_test.parquet",
        "--output-dir", str(out_dir),
        "--seed", str(cfg['seed']), "--d-model", "64", "--n-layers", "3",
        "--dropout", str(cfg['dropout']), "--lr", str(cfg['lr']), "--weight-decay", str(cfg['wd']),
        "--batch-size", str(cfg['bs']),
        "--n-heads", "4", "--cnn-channels", "32", "--token-count", "32",
        "--head-width", "32", "--latent-dim", "32", "--manifold-hidden", "256",
        "--highpass-scale", "0.5", "--temporal", "ode",
        "--steps", "24000", "--warmup-steps", "200",
        "--eval-every", "500", "--early-stop-patience", "1000",
        "--input-highpass", "--manifold-cnn", "--query-local-features",
        "--latent-ode", "--cross-attention",
    ]

    print(f"\n{'='*60}")
    print(f"训练: {cfg['desc']}")
    print(f"{'='*60}")
    r = subprocess.run(cmd, cwd=".")
    if r.returncode == 0:
        m = json.load(open(out_dir / "metrics.json"))
        tl2 = m['test']['relative_l2']
        gap = (m['gap']['test_over_train_ratio'] - 1) * 100
        print(f"✓ Test L2={tl2:.4f}, Gap={gap:+.1f}%")
    else:
        print(f"✗ 训练失败")
        sys.exit(1)

print(f"\n{'='*60}")
print("所有v4训练已完成！")
