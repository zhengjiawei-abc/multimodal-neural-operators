"""
PB_OOD优化实验：测试更小容量+更强正则化
基于分析报告的改进建议
"""
import subprocess
from pathlib import Path

# 基于分析报告的三种优化方案
OPTIMIZATION_CONFIGS = [
    {
        "name": "pb_ood_optimize_v1_smaller_stronger",
        "desc": "方案A: 更小容量(d56) + 更强正则(dropout 0.25)",
        "data_dir": "datasets/burgers_custom_pb_ood",
        "train_file": "burgers_nu0.01_train_000.parquet",
        "dev_file": "burgers_nu0.01_test.parquet",
        "test_file": "burgers_nu0.01_test.parquet",
        "seed": 123,
        "d_model": 56,  # 从d80降低到d56 (-30%)
        "n_layers": 3,
        "dropout": 0.25,  # 从0.15提升到0.25 (+67%)
        "lr": 0.001,
        "weight_decay": 5e-4,  # 从2e-4提升到5e-4 (+150%)
        "batch_size": 24,
    },
    {
        "name": "pb_ood_optimize_v2_baseline_extreme",
        "desc": "方案B: 保持d64 + 极强正则(dropout 0.30, l2)",
        "data_dir": "datasets/burgers_custom_pb_ood",
        "train_file": "burgers_nu0.01_train_000.parquet",
        "dev_file": "burgers_nu0.01_test.parquet",
        "test_file": "burgers_nu0.01_test.parquet",
        "seed": 123,
        "d_model": 64,  # 与基线相同
        "n_layers": 2,  # 从3减少到2
        "dropout": 0.30,  # 极强dropout
        "lr": 0.001,
        "weight_decay": 1e-3,  # 极强weight decay
        "batch_size": 24,
    },
    {
        "name": "pb_ood_optimize_v3_minimal",
        "desc": "方案C: 极小容量(d48, l2) + 强正则",
        "data_dir": "datasets/burgers_custom_pb_ood",
        "train_file": "burgers_nu0.01_train_000.parquet",
        "dev_file": "burgers_nu0.01_test.parquet",
        "test_file": "burgers_nu0.01_test.parquet",
        "seed": 123,
        "d_model": 48,  # 极小容量
        "n_layers": 2,
        "dropout": 0.20,
        "lr": 0.001,
        "weight_decay": 5e-4,
        "batch_size": 28,
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

def build_command(config, exp_name):
    """构建训练命令"""
    cmd = [
        "python", "code/demo_burgers_dataset.py",
        "--preset", "wide",
        "--data-dir", config["data_dir"],
        "--train-file", config["train_file"],
        "--dev-file", config["dev_file"],
        "--test-file", config["test_file"],
        "--output-dir", f"experiments/outputs_burgers_dataset/{exp_name}",
        "--seed", str(config["seed"]),
        "--d-model", str(config["d_model"]),
        "--n-layers", str(config["n_layers"]),
        "--dropout", str(config["dropout"]),
        "--lr", str(config["lr"]),
        "--weight-decay", str(config["weight_decay"]),
        "--batch-size", str(config["batch_size"]),
    ]

    for key, value in SHARED_ARGS:
        cmd.extend([f"--{key}", str(value)])

    for flag in BOOL_FLAGS:
        cmd.append(f"--{flag}")

    return cmd

def estimate_params(d_model, n_layers):
    """估算模型参数量"""
    return (d_model ** 2) * n_layers * 10 // 1000

def run_optimization(config, exp_name):
    """运行一个优化实验"""
    print(f"\n{'='*80}")
    print(f"启动优化: {exp_name}")
    print(f"方案: {config['desc']}")
    print(f"{'='*80}")

    params = estimate_params(config['d_model'], config['n_layers'])
    print(f"  架构: d_model={config['d_model']}, n_layers={config['n_layers']}, "
          f"参数量≈{params}K")
    print(f"  正则: dropout={config['dropout']}, weight_decay={config['weight_decay']}")
    print(f"  训练: lr={config['lr']}, batch_size={config['batch_size']}")
    print()

    cmd = build_command(config, exp_name)

    try:
        result = subprocess.run(cmd, cwd=".")
        if result.returncode == 0:
            print(f"✅ 优化 {exp_name} 完成")
            return True
        else:
            print(f"❌ 优化 {exp_name} 失败 (exit code: {result.returncode})")
            return False
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        return False

def main():
    print("\n" + "="*80)
    print("PB_OOD优化实验")
    print("目标：解决Gap从+23.6%恶化到+50.2%的过拟合问题")
    print("="*80)

    print("\n【当前问题】")
    print("-" * 80)
    print("""
基线 (d64, l3, dropout 0.05):
  Test L2 = 0.3363, Gap = +23.6% (已过拟合)

自适应 (d80, l3, dropout 0.15):
  Test L2 = 0.3166 (↓5.9% ✅)
  Gap = +50.2% (恶化26.6pp ❌)

问题诊断：
  虽然增加了正则化，但容量也增加了
  容量增加的负面影响 > 正则化的正面作用
    """)

    print("="*80)
    print("优化策略")
    print("="*80)
    print()

    for i, cfg in enumerate(OPTIMIZATION_CONFIGS, 1):
        params = estimate_params(cfg['d_model'], cfg['n_layers'])
        print(f"{i}. {cfg['name']}")
        print(f"   {cfg['desc']}")
        print(f"   d{cfg['d_model']}, l{cfg['n_layers']}, ~{params}K参数, "
              f"dropout={cfg['dropout']}, wd={cfg['weight_decay']:.0e}")
    print()

    print("="*80)
    print("预期效果")
    print("="*80)
    print("""
方案A (d56, dropout 0.25):
  目标: Test L2 ≈ 0.32, Gap < 30% (降低过拟合)

方案B (d64, l2, dropout 0.30):
  目标: Test L2≈0.33, Gap < 25% (最强正则)

方案C (d48, l2, dropout 0.20):
  目标: Test L2 ≈ 0.34, Gap < 20% (最小容量)

共同目标: Gap显著降低，Test L2可接受小幅下降
    """)

    print("="*80)
    print("开始训练")
    print("="*80)

    results = {}

    for i, config in enumerate(OPTIMIZATION_CONFIGS, 1):
        exp_name = config['name']
        print(f"\n[{i}/{len(OPTIMIZATION_CONFIGS)}] {exp_name.upper()}")

        success = run_optimization(config, exp_name)
        results[config['name']] = {
            "success": success,
            "desc": config['desc'],
        }

    print("\n" + "="*80)
    print("优化实验总结")
    print("="*80)

    success_count = sum(1 for r in results.values() if r['success'])
    print(f"\n成功: {success_count}/{len(OPTIMIZATION_CONFIGS)}\n")

    for name, result in results.items():
        status = "✅" if result['success'] else "❌"
        print(f"{status} {name:45} - {result['desc']}")

    print("\n" + "="*80)
    print("下一步")
    print("="*80)
    print("""
训练完成后需要：
1. 对比三种方案的Test L2和Gap
2. 选择最佳方案（Gap最低且Test L2可接受）
3. 如果仍不理想，考虑域适应技术
4. 更新ADAPTIVE_CAPACITY_ANALYSIS.md
    """)
    print("="*80)

if __name__ == "__main__":
    main()
