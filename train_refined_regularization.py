"""
精细化改进：加强正则化 + 数据集自适应架构
策略：
1. 增强正则化：dropout 0.05→0.1, 添加权重衰减
2. 数据集自适应：根据数据复杂度调整模型容量
3. 学习率调优：大模型使用更保守的学习率
"""
import subprocess
import json
from pathlib import Path

# 数据集自适应配置
DATASET_CONFIGS = [
    {
        "name": "pb_id_refined",
        "data_dir": "datasets/burgers_custom_pb_id",
        "train_file": "burgers_nu0.01_train_split.parquet",
        "dev_file": "burgers_nu0.01_dev_split.parquet",
        "test_file": "burgers_nu0.01_test.parquet",
        "desc": "参数内分布（简单）",
        "seed": 42,
        # 简单数据 → 大模型有效
        "d_model": 128,
        "n_layers": 4,
        "dropout": 0.1,  # 增强正则化
        "lr": 0.0008,    # 略低学习率
        "weight_decay": 1e-4,  # 添加权重衰减
    },
    {
        "name": "pb_ood_refined",
        "data_dir": "datasets/burgers_custom_pb_ood",
        "train_file": "burgers_nu0.01_train_split.parquet",
        "dev_file": "burgers_nu0.01_dev_split.parquet",
        "test_file": "burgers_nu0.01_test.parquet",
        "desc": "参数外分布（困难）",
        "seed": 123,
        # OOD数据 → 中等容量 + 强正则化
        "d_model": 96,
        "n_layers": 3,
        "dropout": 0.15,  # 最强正则化
        "lr": 0.001,
        "weight_decay": 2e-4,  # 更强权重衰减
    },
    {
        "name": "pdebench_refined",
        "data_dir": "datasets/pdebench_burgers",
        "train_file": "burgers_nu0.01_train_000.parquet",
        "dev_file": "burgers_nu0.01_dev.parquet",
        "test_file": "burgers_nu0.01_test.parquet",
        "desc": "标准基准（适中）",
        "seed": 2024,
        # 标准数据 → 小模型足够 + 适度正则
        "d_model": 64,
        "n_layers": 3,
        "dropout": 0.08,  # 轻微增强
        "lr": 0.001,
        "weight_decay": 5e-5,
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
    ("early-stop-patience", 1000),  # 禁用early stop
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
    print(f"启动精细化实验: {exp_name}")
    print(f"数据集: {dataset_config['desc']}")
    print(f"{'='*70}")
    print(f"  架构: d_model={dataset_config['d_model']}, n_layers={dataset_config['n_layers']}")
    print(f"  正则: dropout={dataset_config['dropout']}, weight_decay={dataset_config['weight_decay']}")
    print(f"  学习: lr={dataset_config['lr']}, seed={dataset_config['seed']}")
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
    print("精细化改进：加强正则化 + 数据集自适应架构")
    print("="*70)
    print(f"\n核心策略：")
    print(f"  1️⃣ 正则化增强：dropout↑, weight_decay↑")
    print(f"  2️⃣ 数据集自适应：根据复杂度调整容量")
    print(f"  3️⃣ 学习率优化：大模型用更保守的lr")
    print()

    print("="*70)
    print("数据集 → 架构映射")
    print("="*70)
    for cfg in DATASET_CONFIGS:
        params_est = (cfg['d_model'] ** 2) * cfg['n_layers'] * 10  # 粗略估计
        print(f"\n{cfg['name']:20} ({cfg['desc']})")
        print(f"  d_model={cfg['d_model']:3}, n_layers={cfg['n_layers']}, params≈{params_est//1000}K")
        print(f"  dropout={cfg['dropout']:.2f}, weight_decay={cfg['weight_decay']:.0e}, lr={cfg['lr']}")

    print("\n" + "="*70)
    print("理论依据")
    print("="*70)
    print("""
  PB_ID (简单):
    - 基线Gap仅0.06%，泛化极好
    - 大模型(d128,l4)改善15.5% ✅
    → 策略：使用大容量 + 强正则防止过拟合

  PB_OOD (困难):
    - 基线Gap 40.3%，严重过拟合 🔴
    - 大模型反而恶化Gap到23.6%
    - PDE RMS从10.69降到2.83 (↓73.6%)
    → 策略：中等容量 + 最强正则 + 更强weight decay

  PDEBench (适中):
    - 基线Gap -5.56%，完美泛化
    - 大模型反而退化12.6% ❌
    → 策略：小模型(d64,l3) + 轻度正则即可
    """)

    print("\n" + "="*70)
    print("开始训练")
    print("="*70)

    results = {}

    for i, dataset_config in enumerate(DATASET_CONFIGS, 1):
        exp_name = f"custom_v14_refined_{dataset_config['name']}"
        print(f"\n[{i}/{len(DATASET_CONFIGS)}] {dataset_config['name'].upper()}")

        success = run_experiment(dataset_config, exp_name)
        results[dataset_config['name']] = {
            "success": success,
            "exp_name": exp_name,
            "desc": dataset_config['desc'],
            "d_model": dataset_config['d_model'],
            "n_layers": dataset_config['n_layers'],
            "dropout": dataset_config['dropout'],
            "weight_decay": dataset_config['weight_decay'],
        }

    # 总结结果
    print("\n" + "="*70)
    print("实验总结")
    print("="*70)

    success_count = sum(1 for r in results.values() if r['success'])
    print(f"\n成功: {success_count}/{len(DATASET_CONFIGS)}\n")

    for name, result in results.items():
        status = "✅" if result['success'] else "❌"
        print(f"{status} {name:20} - {result['exp_name']}")
        print(f"   (d{result['d_model']}, l{result['n_layers']}, drop={result['dropout']}, wd={result['weight_decay']:.0e})")

    print("\n" + "="*70)
    print("预期效果")
    print("="*70)
    print("""
  PB_ID:
    当前(d128,no reg):  0.1858
    预期(d128,+reg):    ~0.17   (↓8%, 目标：更强泛化)

  PB_OOD:
    当前(d128,no reg):  0.3644 (Gap 23.6%)
    预期(d96,++reg):    ~0.30  (↓18%, 目标：Gap<15%)

  PDEBench:
    当前(d128,no reg):  0.2408
    预期(d64,+reg):     ~0.20  (↓17%, 回归基线性能)

  关键指标：
    ✅ Gap全部控制在10%以内
    ✅ PB_OOD的严重过拟合得到缓解
    ✅ 每个数据集使用最适配的架构
    """)
    print("="*70)

if __name__ == "__main__":
    main()
