"""
PB_OOD 参数空间数据增强实验 (方向2)

核心思路:
  容量-正则调整已证明无法同时改善 Test L2 和 Gap，根本原因是域转移。
  本方案不改模型容量，而是通过在线数据增强扩大训练数据的有效覆盖，
  迫使模型学习平移不变的物理规律而非记忆特定空间模式。

增强策略:
  1. --online-shift-augment: 每个 batch 对初始场施加随机循环平移
     (random_shift_batch 在空间轴上 roll，利用周期边界条件的物理对称性)
  2. dropout=0.15: 适度正则化，防止模型记忆增强后的扰动
  3. lr=0.0008: 略低于基线 0.001，增强训练稳定性
  4. 保持 d_model=64, n_layers=3: 基线容量，不引入容量-正则矛盾

与之前实验的对比基线:
  基线 (d64, l3, dropout=0.05, 无增强):  Test L2=0.3363, Gap=+23.6%
  方案A (d56, l3, dropout=0.25):         Test L2=0.4211, Gap=+33.0%  (欠拟合)
  方案B (d64, l2, dropout=0.30):          Test L2=0.3289, Gap=+57.6%  (Gap恶化)
  方案C (d48, l2, dropout=0.20):          Test L2=0.3532, Gap=+38.9%  (Gap稍改善)

预期: 在保持基线容量的前提下，通过数据增强降低 Gap 同时维持 Test L2。
"""
import subprocess
import sys
import json
import time
from pathlib import Path

# ── 实验配置 ──────────────────────────────────────────────────────────────────

EXPERIMENT_NAME = "pb_ood_augment_v1"
OUTPUT_DIR = Path("experiments/outputs_burgers_dataset") / EXPERIMENT_NAME

CONFIG = {
    "name": EXPERIMENT_NAME,
    "desc": "方向2: 参数空间数据增强 (online shift + 适度正则)",
    "data_dir": "datasets/burgers_custom_pb_ood",
    "train_file": "burgers_nu0.01_train_split.parquet",
    "dev_file": "burgers_nu0.01_dev_split.parquet",
    "test_file": "burgers_nu0.01_test.parquet",
    "seed": 123,          # 与优化实验一致，便于对比
    # 基线容量 (不引入容量-正则矛盾)
    "d_model": 64,
    "n_layers": 3,
    # 数据增强核心参数
    "dropout": 0.15,       # 适度正则化，防止记忆增强扰动
    "lr": 0.0008,          # 略低于基线，增强稳定性
    "weight_decay": 1e-4,  # 基线水平
    "batch_size": 32,      # 基线水平
}

# 共享架构参数 (与基线命令一致)
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
    ("early-stop-patience", 1000),   # 不启用早停，跑满 24K 步
    ("lr-decay-steps", 12000),       # 与基线一致的衰减节奏
]

# 布尔标志 (与基线一致)
BOOL_FLAGS = [
    "input-highpass",
    "manifold-cnn",
    "query-local-features",
    "latent-ode",
    "cross-attention",
    # ── 数据增强核心标志 ──
    "online-shift-augment",   # 启用在线循环平移增强
]


# ── 命令构建 ──────────────────────────────────────────────────────────────────

def build_command(config: dict) -> list:
    """构建训练命令行参数列表"""
    cmd = [
        sys.executable, "code/demo_burgers_dataset.py",
        "--preset", "wide",
        "--data-dir", config["data_dir"],
        "--train-file", config["train_file"],
        "--dev-file", config["dev_file"],
        "--test-file", config["test_file"],
        "--output-dir", str(OUTPUT_DIR),
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


# ── 结果分析 ──────────────────────────────────────────────────────────────────

# 之前的实验结果 (供对比)
BASELINE_RESULTS = {
    "baseline":        {"test_l2": 0.3363, "gap": 23.6,  "desc": "d64, l3, dropout=0.05, 无增强"},
    "adaptive":        {"test_l2": 0.3166, "gap": 50.2,  "desc": "d80, l3, dropout=0.15"},
    "plan_a":          {"test_l2": 0.4211, "gap": 33.0,  "desc": "d56, l3, dropout=0.25"},
    "plan_b":          {"test_l2": 0.3289, "gap": 57.6,  "desc": "d64, l2, dropout=0.30"},
    "plan_c":          {"test_l2": 0.3532, "gap": 38.9,  "desc": "d48, l2, dropout=0.20"},
}


def load_results(output_dir: Path) -> dict:
    """加载训练结果的 metrics.json"""
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_and_compare(results: dict) -> None:
    """分析结果并与之前实验对比"""
    if results is None:
        print("无法加载结果文件")
        return

    test_l2 = results["test"]["relative_l2"]
    train_l2 = results["train"]["relative_l2"]
    gap = results["gap"]["test_minus_train_relative_l2"]
    gap_pct = results["gap"]["test_over_train_ratio"]
    gap_percent = (gap_pct - 1.0) * 100
    diagnosis = results["gap"]["diagnosis"]

    print("\n" + "=" * 80)
    print("方向2: 参数空间数据增强 - 结果分析")
    print("=" * 80)

    print(f"\n{'配置':<45} {'Test L2':>10} {'Gap%':>10} {'诊断':>10}")
    print("-" * 80)

    # 之前的实验
    for key, val in BASELINE_RESULTS.items():
        print(f"  {val['desc']:<43} {val['test_l2']:>10.4f} {val['gap']:>9.1f}% {'':>10}")

    # 当前实验
    print(f"  {'→ ' + CONFIG['desc']:<43} {test_l2:>10.4f} {gap_percent:>9.1f}% {diagnosis:>10}")
    print("-" * 80)

    # 改善分析
    baseline_test = BASELINE_RESULTS["baseline"]["test_l2"]
    baseline_gap = BASELINE_RESULTS["baseline"]["gap"]
    test_improvement = (baseline_test - test_l2) / baseline_test * 100
    gap_improvement = baseline_gap - gap_percent

    print(f"\n与基线对比:")
    print(f"  Test L2: {test_l2:.4f} vs 基线 {baseline_test:.4f}  "
          f"({'↓' if test_improvement > 0 else '↑'}{abs(test_improvement):.1f}%)")
    print(f"  Gap:     {gap_percent:.1f}% vs 基线 {baseline_gap:.1f}%  "
          f"({'↓' if gap_improvement > 0 else '↑'}{abs(gap_improvement):.1f}pp)")

    # 结论判断
    print(f"\n诊断: {diagnosis}")
    if gap_percent < baseline_gap and test_l2 <= baseline_test * 1.05:
        print("结论: 数据增强有效改善了 Gap 同时维持了 Test L2 ✅")
    elif gap_percent < baseline_gap:
        print("结论: Gap 有所改善，但 Test L2 有所上升 ⚠️")
    else:
        print("结论: 数据增强未能改善域转移问题，需要更强方法 ❌")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 80)
    print("PB_OOD 方向2: 参数空间数据增强实验")
    print("=" * 80)

    print(f"\n实验名称: {CONFIG['name']}")
    print(f"描述: {CONFIG['desc']}")
    print(f"输出目录: {OUTPUT_DIR}")

    print(f"\n核心策略:")
    print(f"  1. 在线循环平移增强 (--online-shift-augment)")
    print(f"     利用周期边界条件的空间对称性，每个 batch 随机平移初始场")
    print(f"  2. 适度正则化 (dropout={CONFIG['dropout']})")
    print(f"     防止模型记忆增强后的扰动模式")
    print(f"  3. 稳定学习率 (lr={CONFIG['lr']})")
    print(f"     略低于基线 0.001，增强训练稳定性")
    print(f"  4. 基线容量 (d_model={CONFIG['d_model']}, n_layers={CONFIG['n_layers']})")
    print(f"     不引入容量-正则矛盾")

    print(f"\n与之前实验的对比基线:")
    for key, val in BASELINE_RESULTS.items():
        print(f"  {val['desc']:<45} Test L2={val['test_l2']:.4f}, Gap={val['gap']:.1f}%")

    # 构建命令
    cmd = build_command(CONFIG)
    print(f"\n训练命令:")
    print("  " + " ".join(cmd[:2]))
    for arg in cmd[2:]:
        print(f"  {arg}")

    print(f"\n{'=' * 80}")
    print("开始训练")
    print(f"{'=' * 80}\n")

    start_time = time.time()

    # 运行训练
    result = subprocess.run(cmd, cwd=".")

    elapsed = time.time() - start_time
    print(f"\n训练耗时: {elapsed:.1f}s ({elapsed / 60:.1f}min)")

    if result.returncode == 0:
        print(f"训练完成，正在分析结果...")

        # 加载并分析结果
        results = load_results(OUTPUT_DIR)
        analyze_and_compare(results)

        print(f"\n输出文件:")
        print(f"  metrics.json:      {OUTPUT_DIR / 'metrics.json'}")
        print(f"  training_curve.png: {OUTPUT_DIR / 'training_curve.png'}")
        print(f"  test_prediction.png: {OUTPUT_DIR / 'test_prediction.png'}")
    else:
        print(f"训练失败 (exit code: {result.returncode})")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
