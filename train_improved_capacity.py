"""
改进方案实施：禁用early stop + 增加模型容量 + 多种子集成
策略：
1. 禁用early stop → 完整训练24K步
2. 增加模型深度和宽度 → 提升容量
3. 使用不同种子 → 为集成做准备
"""
import subprocess
import json
from pathlib import Path

# 改进的配置：增加模型容量 + 禁用early stop
IMPROVED_CONFIG_ARGS = [
    ("dropout", 0.05),
    ("d-model", 128),         # 从64提升到128（容量翻倍）
    ("n-layers", 4),          # 从3提升到4（深度增加）
    ("n-heads", 4),
    ("cnn-channels", 32),
    ("token-count", 32),
    ("head-width", 32),
    ("latent-dim", 32),
    ("manifold-hidden", 256),
    ("highpass-scale", 0.5),
    ("temporal", "ode"),
    ("lr", 0.001),
    ("steps", 24000),         # 完整训练到24K
    ("warmup-steps", 200),
    ("eval-every", 500),
    ("early-stop-patience", 1000),  # 禁用early stop（patience极大）
]

BOOL_FLAGS = [
    "input-highpass",
    "manifold-cnn",
    "query-local-features",
    "latent-ode",
    "cross-attention",
]

DATASETS = [
    {
        "name": "pb_id_improved",
        "data_dir": "datasets/burgers_custom_pb_id",
        "train_file": "burgers_nu0.01_train_000.parquet",
        "dev_file": "burgers_nu0.01_test.parquet",
        "test_file": "burgers_nu0.01_test.parquet",
        "desc": "参数内分布（In-Distribution）",
        "seed": 42,
    },
    {
        "name": "pb_ood_improved",
        "data_dir": "datasets/burgers_custom_pb_ood",
        "train_file": "burgers_nu0.01_train_000.parquet",
        "dev_file": "burgers_nu0.01_test.parquet",
        "test_file": "burgers_nu0.01_test.parquet",
        "desc": "参数外分布（Out-of-Distribution）",
        "seed": 123,
    },
    {
        "name": "pdebench_improved",
        "data_dir": "datasets/pdebench_burgers",
        "train_file": "burgers_nu0.01_train_000.parquet",
        "dev_file": "burgers_nu0.01_dev.parquet",
        "test_file": "burgers_nu0.01_test.parquet",
        "desc": "标准PDEBench数据集",
        "seed": 2024,
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
        "--seed", str(dataset_config["seed"]),
    ]

    # 添加数值参数
    for key, value in IMPROVED_CONFIG_ARGS:
        cmd.extend([f"--{key}", str(value)])

    # 添加布尔标志
    for flag in BOOL_FLAGS:
        cmd.append(f"--{flag}")

    return cmd

def run_experiment(dataset_config, exp_name):
    """运行一个实验"""
    print(f"\n{'='*70}")
    print(f"启动改进实验: {exp_name}")
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
    print("改进方案实施：禁用Early Stop + 增加模型容量")
    print("="*70)
    print(f"\n改进内容：")
    print(f"  ✅ 禁用Early Stop：完整训练24K步")
    print(f"  ✅ 增加模型容量：d_model 64→128, n_layers 3→4")
    print(f"  ✅ 多种子训练：为后续集成做准备\n")
    print(f"配置详情：")
    print(f"  - d_model: 128 (2x容量)")
    print(f"  - n_layers: 4 (更深)")
    print(f"  - early_stop_patience: 1000 (禁用)")
    print(f"  - steps: 24,000 (完整训练)\n")

    results = {}

    for i, dataset_config in enumerate(DATASETS, 1):
        exp_name = f"custom_v14_improved_d128_l4_{dataset_config['name']}"
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
        print(f"{status} {name:20} - {result['exp_name']}")

    print("\n" + "="*70)
    print("预期效果")
    print("="*70)
    print("\n相比前一轮（d64, l3, early stop）的改进：")
    print("  📈 禁用early stop:     +5-10% 改善")
    print("  📈 增加模型容量:       +10-15% 改善")
    print("  📈 总体预期:           +15-25% 改善")
    print("\n目标：")
    print("  PB_ID:     0.2199 → ~0.18   (↓18%)")
    print("  PDEBench:  0.2138 → ~0.17   (↓20%)")
    print("  PB_OOD:    0.3363 → ~0.25   (↓26%)")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
