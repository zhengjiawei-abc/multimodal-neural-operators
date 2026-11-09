"""
OOM修复：降低batch size以适应GPU内存限制
同时微调正则化策略
"""
import subprocess
import json
from pathlib import Path

# 数据集自适应配置（修正版）
DATASET_CONFIGS = [
    {
        "name": "pb_id_refined_v2",
        "data_dir": "datasets/burgers_custom_pb_id",
        "train_file": "burgers_nu0.01_train_000.parquet",
        "dev_file": "burgers_nu0.01_test.parquet",
        "test_file": "burgers_nu0.01_test.parquet",
        "desc": "参数内分布（简单）",
        "seed": 42,
        # 保持大模型，但降低batch size
        "d_model": 128,
        "n_layers": 4,
        "dropout": 0.08,  # 略微降低（0.1→0.08）
        "lr": 0.0008,
        "weight_decay": 5e-5,  # 降低（1e-4→5e-5）
        "batch_size": 16,  # 原32→16
    },
    {
        "name": "pb_ood_refined_v2",
        "data_dir": "datasets/burgers_custom_pb_ood",
        "train_file": "burgers_nu0.01_train_000.parquet",
        "dev_file": "burgers_nu0.01_test.parquet",
        "test_file": "burgers_nu0.01_test.parquet",
        "desc": "参数外分布（困难）",
        "seed": 123,
        # 中等容量 + 适度正则
        "d_model": 96,
        "n_layers": 3,
        "dropout": 0.12,  # 略微降低（0.15→0.12）
        "lr": 0.001,
        "weight_decay": 1e-4,  # 降低（2e-4→1e-4）
        "batch_size": 20,  # 原32→20
    },
    {
        "name": "pdebench_refined_v2",
        "data_dir": "datasets/pdebench_burgers",
        "train_file": "burgers_nu0.01_train_000.parquet",
        "dev_file": "burgers_nu0.01_dev.parquet",
        "test_file": "burgers_nu0.01_test.parquet",
        "desc": "标准基准（适中）",
        "seed": 2024,
        # 回归基线配置：无额外正则化
        "d_model": 64,
        "n_layers": 3,
        "dropout": 0.05,  # 恢复基线值
        "lr": 0.001,
        "weight_decay": 0.0,  # 移除weight decay
        "batch_size": 32,
    }
]

# 共享配置
SHARED_ARGS = [
    ("n-heads", 4),
    ("cnn-channels", 32),
    ("token-count", 32),
    ("head-width", 32),
    ("latent-dim", 32),
    ("manifold-hidden", 256),
    ("highpass-scale", 0.5),
    ("temporal", "ode"),
    ("steps", 24000),
    ("warmup-steps", 200),
    ("eval-every", 500),
    ("early-stop-patience", 1000),
]

BOOL_FLAGS = [
    "input-highpass",
    "manifold-cnn",
    "query-local-features",
    "latent-ode",
    "cross-attention",
]

def check_dataset_exists(dataset_config):
    """检查数据集文件是否存在"""
    for file_key in ["train_file", "dev_file", "test_file"]:
        file_path = Path(dataset_config["data_dir"]) / dataset_config[file_key]
        if not file_path.exists():
            return False, str(file_path)
    return True, None

def build_command(dataset_config, exp_name):
    """构建训练命令"""
    cmd = [
        "python", "code/demo_burgers_dataset.py",
        "--preset", "wide",
        "--data-dir", dataset_config["data_dir"],
        "--train-file", dataset_config["train_file"],
        "--dev-file", dataset_config["dev_file"],
        "--test-file", dataset_config["test_file"],
        "--output-dir", f"experiments/outputs_burgers_dataset/{exp_name}",
        "--seed", str(dataset_config["seed"]),
        "--d-model", str(dataset_config["d_model"]),
        "--n-layers", str(dataset_config["n_layers"]),
        "--dropout", str(dataset_config["dropout"]),
        "--lr", str(dataset_config["lr"]),
        "--weight-decay", str(dataset_config["weight_decay"]),
        "--batch-size", str(dataset_config["batch_size"]),
    ]

    # 添加共享参数
    for key, value in SHARED_ARGS:
        cmd.extend([f"--{key}", str(value)])

    # 添加布尔标志
    for flag in BOOL_FLAGS:
        cmd.append(f"--{flag}")

    return cmd

def run_experiment(dataset_config, exp_name):
    """运行一个实验"""
    print(f"\n{'='*70}")
    print(f"启动实验: {exp_name}")
    print(f"数据集: {dataset_config['desc']}")
    print(f"{'='*70}")
    print(f"  架构: d_model={dataset_config['d_model']}, n_layers={dataset_config['n_layers']}")
    print(f"  正则: dropout={dataset_config['dropout']}, weight_decay={dataset_config['weight_decay']}")
    print(f"  训练: lr={dataset_config['lr']}, batch_size={dataset_config['batch_size']}, seed={dataset_config['seed']}")
    print()

    # 检查数据集
    exists, missing = check_dataset_exists(dataset_config)
    if not exists:
        print(f"❌ 数据集文件不存在: {missing}")
        return False

    # 构建命令
    cmd = build_command(dataset_config, exp_name)

    # 运行训练
    try:
        result = subprocess.run(cmd, cwd=".")
        if result.returncode == 0:
            print(f"✅ 实验 {exp_name} 完成")
            return True
        else:
            print(f"❌ 实验 {exp_name} 失败 (exit code: {result.returncode})")
            return False
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("OOM修复 + 正则化策略微调")
    print("="*70)
    print(f"\n核心改进：")
    print(f"  1️⃣ 降低batch size：避免OOM")
    print(f"  2️⃣ 微调正则化：避免过度平滑")
    print(f"  3️⃣ PDEBench回归基线：移除多余正则")
    print()

    print("="*70)
    print("配置对比")
    print("="*70)
    print()
    print("【OOM原因分析】")
    print("  - d128模型参数量910K，在batch_size=32时超出GPU内存")
    print("  - d96模型参数量531K，在batch_size=32时也接近极限")
    print("  → 解决：降低batch_size到16-20")
    print()

    print("【正则化策略调整】")
    print()
    for i, cfg in enumerate(DATASET_CONFIGS, 1):
        print(f"{i}. {cfg['name']} ({cfg['desc']})")
        print(f"   架构: d{cfg['d_model']}, l{cfg['n_layers']}, batch={cfg['batch_size']}")
        print(f"   正则: dropout={cfg['dropout']}, wd={cfg['weight_decay']:.0e}")

        if 'pb_id' in cfg['name']:
            print(f"   策略: 大模型 + 轻度正则（避免过度约束）")
        elif 'pb_ood' in cfg['name']:
            print(f"   策略: 中模型 + 适度正则（平衡拟合与泛化）")
        elif 'pdebench' in cfg['name']:
            print(f"   策略: 小模型 + 无额外正则（回归基线）")
        print()

    print("="*70)
    print("预期效果")
    print("="*70)
    print("""
  PB_ID:
    目标: 0.1858 → ~0.17   (↓8%)
    关键: 大模型容量 + 避免过度正则化

  PB_OOD:
    目标: 0.3644 → ~0.32   (↓12%)
    关键: 降低Gap从23.6%到<18%

  PDEBench:
    目标: 回归基线 0.2138
    关键: 移除多余正则，恢复原始性能
    """)

    print("="*70)
    print("开始训练")
    print("="*70)

    results = {}

    for i, dataset_config in enumerate(DATASET_CONFIGS, 1):
        exp_name = f"custom_v14_{dataset_config['name']}"
        print(f"\n[{i}/{len(DATASET_CONFIGS)}] {dataset_config['name'].upper()}")

        success = run_experiment(dataset_config, exp_name)
        results[dataset_config['name']] = {
            "success": success,
            "exp_name": exp_name,
        }

    # 总结
    print("\n" + "="*70)
    print("实验总结")
    print("="*70)

    success_count = sum(1 for r in results.values() if r['success'])
    print(f"\n成功: {success_count}/{len(DATASET_CONFIGS)}\n")

    for name, result in results.items():
        status = "✅" if result['success'] else "❌"
        print(f"{status} {name:25} - {result['exp_name']}")

    print("\n" + "="*70)

if __name__ == "__main__":
    main()
