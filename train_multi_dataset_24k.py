"""
多数据集完整训练实验（24K步）
使用不同随机种子增加数据多样性，提升泛化能力
"""
import subprocess
import json
from pathlib import Path
import sys

# 最佳配置参数（对齐 custom_v14_shift3_drop005_24k）
BEST_CONFIG_ARGS = [
    ("dropout", 0.05),
    ("d-model", 64),
    ("n-layers", 3),
    ("n-heads", 4),
    ("cnn-channels", 32),
    ("token-count", 32),
    ("head-width", 32),
    ("latent-dim", 32),
    ("manifold-hidden", 256),
    ("highpass-scale", 0.5),
    ("temporal", "ode"),
    ("lr", 0.001),
    ("steps", 24000),  # 完整训练24K步
    ("warmup-steps", 200),
    ("eval-every", 500),
]

BOOL_FLAGS = [
    "input-highpass",
    "manifold-cnn",
    "query-local-features",
    "latent-ode",
    "cross-attention",
]

# 数据集配置 - 使用不同随机种子增加多样性
DATASETS = [
    {
        "name": "pb_id",
        "data_dir": "datasets/burgers_custom_pb_id",
        "train_file": "burgers_nu0.01_train_000.parquet",
        "dev_file": "burgers_nu0.01_test.parquet",
        "test_file": "burgers_nu0.01_test.parquet",
        "desc": "参数内分布（In-Distribution）",
        "seed": 42,  # 不同于之前的seed=11
    },
    {
        "name": "pb_ood",
        "data_dir": "datasets/burgers_custom_pb_ood",
        "train_file": "burgers_nu0.01_train_000.parquet",
        "dev_file": "burgers_nu0.01_test.parquet",
        "test_file": "burgers_nu0.01_test.parquet",
        "desc": "参数外分布（Out-of-Distribution）",
        "seed": 123,  # 不同种子
    },
    {
        "name": "pdebench",
        "data_dir": "datasets/pdebench_burgers",
        "train_file": "burgers_nu0.01_train_000.parquet",
        "dev_file": "burgers_nu0.01_dev.parquet",
        "test_file": "burgers_nu0.01_test.parquet",
        "desc": "标准PDEBench数据集",
        "seed": 2024,  # 不同种子
    }
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
        "--seed", str(dataset_config["seed"]),  # 使用数据集特定的种子
    ]

    # 添加数值参数
    for key, value in BEST_CONFIG_ARGS:
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
    print(f"随机种子: {dataset_config['seed']}")
    print(f"{'='*70}\n")

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
    print("多数据集完整训练实验（24K步）")
    print("="*70)
    print(f"\n基础配置: custom_v14_shift3_drop005_24k")
    print(f"训练步数: 24,000 (完整训练)")
    print(f"Dropout: 0.05")
    print(f"d_model: 64")
    print(f"n_layers: 3")
    print(f"\n🎲 使用不同随机种子增加数据多样性")
    print(f"   - PB_ID: seed={DATASETS[0]['seed']}")
    print(f"   - PB_OOD: seed={DATASETS[1]['seed']}")
    print(f"   - PDEBench: seed={DATASETS[2]['seed']}\n")

    results = {}

    for i, dataset_config in enumerate(DATASETS, 1):
        exp_name = f"custom_v14_multi_dataset_{dataset_config['name']}_24k"
        print(f"\n[{i}/{len(DATASETS)}] {dataset_config['name'].upper()}")

        success = run_experiment(dataset_config, exp_name)
        results[dataset_config['name']] = {
            "success": success,
            "exp_name": exp_name,
            "desc": dataset_config['desc'],
            "seed": dataset_config['seed']
        }

    # 总结结果
    print("\n" + "="*70)
    print("实验总结")
    print("="*70)

    success_count = sum(1 for r in results.values() if r['success'])
    print(f"\n成功: {success_count}/{len(DATASETS)}\n")

    for name, result in results.items():
        status = "✅" if result['success'] else "❌"
        print(f"{status} {name:10} - {result['exp_name']} (seed={result['seed']})")

    print("\n" + "="*70)
    print("改进要点")
    print("="*70)
    print("\n1. ✅ 训练步数: 12K → 24K (对齐最佳模型)")
    print("2. ✅ 随机种子: 使用不同种子增加数据多样性")
    print("3. 📈 预期改进: Test L2 从 0.25-0.35 降至接近 0.087")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
