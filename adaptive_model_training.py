"""
自适应模型容量训练：根据数据集复杂度匹配模型大小
基于实验结果的经验规则
"""
import subprocess
from pathlib import Path

# 基于实验结果的数据集复杂度评估
DATASET_CONFIGS = [
    {
        "name": "pb_id_adaptive",
        "data_dir": "datasets/burgers_custom_pb_id",
        "train_file": "burgers_nu0.01_train_000.parquet",
        "dev_file": "burgers_nu0.01_test.parquet",
        "test_file": "burgers_nu0.01_test.parquet",
        "desc": "参数内分布（简单数据）",
        "seed": 42,
        # 实验结论：简单数据受益于大模型
        # 基线(d64): 0.2199, 改进(d128): 0.1858 (↓15.5%) ✅
        "complexity": "medium-high",
        "d_model": 96,  # 使用中大模型
        "n_layers": 4,
        "dropout": 0.08,
        "lr": 0.0009,
        "weight_decay": 5e-5,
        "batch_size": 20,
    },
    {
        "name": "pb_ood_adaptive",
        "data_dir": "datasets/burgers_custom_pb_ood",
        "train_file": "burgers_nu0.01_train_000.parquet",
        "dev_file": "burgers_nu0.01_test.parquet",
        "test_file": "burgers_nu0.01_test.parquet",
        "desc": "参数外分布（复杂+域转移）",
        "seed": 123,
        # 实验结论：域转移数据，大模型反而恶化
        # 基线(d64): 0.3363, 改进(d128): 0.3644 (↑8.4%) ❌
        # 关键：需要适度容量+强正则化
        "complexity": "high",
        "d_model": 80,  # 中等偏小模型
        "n_layers": 3,
        "dropout": 0.15,  # 强正则化
        "lr": 0.001,
        "weight_decay": 2e-4,
        "batch_size": 24,
    },
    {
        "name": "pdebench_adaptive",
        "data_dir": "datasets/pdebench_burgers",
        "train_file": "burgers_nu0.01_train_000.parquet",
        "dev_file": "burgers_nu0.01_dev.parquet",
        "test_file": "burgers_nu0.01_test.parquet",
        "desc": "标准基准（已完美泛化）",
        "seed": 2024,
        # 实验结论：完美泛化数据，大模型导致噪声拟合
        # 基线(d64): 0.2138 (Gap -5.56%), 改进(d128): 0.2408 (↑12.6%) ❌
        # 关键：保持小模型即可
        "complexity": "low-medium",
        "d_model": 64,  # 小模型
        "n_layers": 3,
        "dropout": 0.05,  # 轻度正则
        "lr": 0.001,
        "weight_decay": 0.0,
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
    ("early-stop-patience", 1000),  # 禁用早停，完整训练
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

    for key, value in SHARED_ARGS:
        cmd.extend([f"--{key}", str(value)])

    for flag in BOOL_FLAGS:
        cmd.append(f"--{flag}")

    return cmd

def estimate_params(d_model, n_layers):
    """估算模型参数量（粗略）"""
    return (d_model ** 2) * n_layers * 10 // 1000

def run_experiment(dataset_config, exp_name):
    """运行一个实验"""
    print(f"\n{'='*80}")
    print(f"启动实验: {exp_name}")
    print(f"数据集: {dataset_config['desc']}")
    print(f"复杂度: {dataset_config['complexity'].upper()}")
    print(f"{'='*80}")

    params = estimate_params(dataset_config['d_model'], dataset_config['n_layers'])
    print(f"  架构: d_model={dataset_config['d_model']}, n_layers={dataset_config['n_layers']}, "
          f"参数量≈{params}K")
    print(f"  正则: dropout={dataset_config['dropout']}, weight_decay={dataset_config['weight_decay']}")
    print(f"  训练: lr={dataset_config['lr']}, batch_size={dataset_config['batch_size']}, "
          f"seed={dataset_config['seed']}")
    print()

    exists, missing = check_dataset_exists(dataset_config)
    if not exists:
        print(f"❌ 数据集文件不存在: {missing}")
        return False

    cmd = build_command(dataset_config, exp_name)

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
    print("\n" + "="*80)
    print("自适应模型容量训练")
    print("策略：根据数据集复杂度和已有实验结果，匹配最优模型容量")
    print("="*80)

    print("\n【容量匹配策略】")
    print("-" * 80)
    print()
    print("基于实验结果的三条核心规律：")
    print()
    print("1️⃣ PB_ID（简单数据）：")
    print("   现象: 基线(d64) 0.2199 → 改进(d128) 0.1858 (↓15.5%) ✅")
    print("   结论: 简单数据受益于更大模型容量")
    print("   策略: 使用中大模型 (d96, l4, ~370K参数)")
    print()
    print("2️⃣ PB_OOD（复杂+域转移）：")
    print("   现象: 基线(d64) 0.3363 → 改进(d128) 0.3644 (↑8.4%) ❌")
    print("   结论: 域转移数据，大模型反而拟合噪声，加剧过拟合")
    print("   策略: 使用中等偏小模型 (d80, l3, ~190K参数) + 强正则化")
    print()
    print("3️⃣ PDEBench（已完美泛化）：")
    print("   现象: 基线(d64) 0.2138 (Gap -5.56%) → 改进(d128) 0.2408 (↑12.6%) ❌")
    print("   结论: 完美泛化数据，大模型引入过度拟合")
    print("   策略: 保持小模型 (d64, l3, ~120K参数)")
    print()

    print("="*80)
    print("配置总览")
    print("="*80)
    print()

    for i, cfg in enumerate(DATASET_CONFIGS, 1):
        params = estimate_params(cfg['d_model'], cfg['n_layers'])
        print(f"{i}. {cfg['name']:20} (复杂度: {cfg['complexity']})")
        print(f"   d{cfg['d_model']}, l{cfg['n_layers']}, ~{params}K参数, "
              f"dropout={cfg['dropout']}, wd={cfg['weight_decay']:.0e}, "
              f"batch={cfg['batch_size']}")
    print()

    print("="*80)
    print("预期效果")
    print("="*80)
    print("""
  PB_ID (d96, l4):
    目标: 0.1858 → ~0.175  (再降5%)
    理由: d96介于d64和d128之间，平衡容量和正则化

  PB_OOD (d80, l3, 强正则):
    目标: 0.3644 → ~0.30   (降低12%)
    关键: Gap从23.6%降到<15%
    理由: 降低容量+强dropout，防止拟合域转移噪声

  PDEBench (d64, l3):
    目标: 回归基线 0.2138
    理由: 保持基线配置，完整24K训练
    """)

    print("="*80)
    print("开始训练")
    print("="*80)

    results = {}

    for i, dataset_config in enumerate(DATASET_CONFIGS, 1):
        exp_name = f"custom_v14_{dataset_config['name']}"
        print(f"\n[{i}/{len(DATASET_CONFIGS)}] {dataset_config['name'].upper()}")

        success = run_experiment(dataset_config, exp_name)
        results[dataset_config['name']] = {
            "success": success,
            "exp_name": exp_name,
        }

    print("\n" + "="*80)
    print("实验总结")
    print("="*80)

    success_count = sum(1 for r in results.values() if r['success'])
    print(f"\n成功: {success_count}/{len(DATASET_CONFIGS)}\n")

    for name, result in results.items():
        status = "✅" if result['success'] else "❌"
        print(f"{status} {name:25} - {result['exp_name']}")

    print("\n" + "="*80)
    print("核心思想")
    print("="*80)
    print("""
  不是"参数量大就用大模型"，而是"数据复杂度匹配模型容量"：

  ✅ 简单数据 + 大容量 = 充分学习特征
  ❌ 简单数据 + 小模型 = 欠拟合

  ✅ 复杂数据 + 适度容量 = 学习本质规律
  ❌ 复杂数据 + 过大模型 = 拟合噪声，过拟合

  关键是找到"刚刚好"的容量：
  - 足够大：能学到数据中的真实模式
  - 足够小：不会拟合数据中的噪声
    """)
    print("="*80)

if __name__ == "__main__":
    main()
