"""
过拟合诊断：分析train/test gap和学习曲线
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_metrics(exp_path):
    """加载实验指标"""
    metrics_file = Path(exp_path) / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            return json.load(f)
    return None

def diagnose_overfitting(metrics, exp_name):
    """诊断是否过拟合"""
    train_l2 = metrics["train"]["relative_l2"]
    test_l2 = metrics["test"]["relative_l2"]
    gap = metrics["gap"]["test_minus_train_relative_l2"]
    gap_pct = (gap / train_l2) * 100

    print(f"\n{'='*80}")
    print(f"{exp_name}")
    print(f"{'='*80}")
    print(f"Train L2:      {train_l2:.4f}")
    print(f"Test L2:       {test_l2:.4f}")
    print(f"Gap:           {gap:.4f} ({gap_pct:+.2f}%)")
    print(f"Test/Train:    {test_l2/train_l2:.3f}")
    print(f"诊断:          {metrics['gap']['diagnosis']}")

    # 过拟合判定
    if gap_pct > 10:
        print(f"🔴 严重过拟合 (gap > 10%)")
        severity = "severe"
    elif gap_pct > 5:
        print(f"🟡 轻微过拟合 (gap > 5%)")
        severity = "mild"
    elif gap_pct > 0:
        print(f"🟢 几乎完美 (gap < 5%)")
        severity = "good"
    else:
        print(f"✅ 欠拟合或完美泛化 (test ≤ train)")
        severity = "underfit"

    return {
        "name": exp_name,
        "train_l2": train_l2,
        "test_l2": test_l2,
        "gap_pct": gap_pct,
        "severity": severity
    }

# 基线模型 (d64, l3)
BASELINE_EXPS = [
    ("experiments/outputs_burgers_dataset/custom_v14_multi_dataset_pb_id_24k", "PB_ID (基线)"),
    ("experiments/outputs_burgers_dataset/custom_v14_multi_dataset_pb_ood_24k", "PB_OOD (基线)"),
    ("experiments/outputs_burgers_dataset/custom_v14_multi_dataset_pdebench_24k", "PDEBench (基线)"),
]

# 改进模型 (d128, l4)
IMPROVED_EXPS = [
    ("experiments/outputs_burgers_dataset/custom_v14_improved_d128_l4_pb_id_improved", "PB_ID (改进)"),
    ("experiments/outputs_burgers_dataset/custom_v14_improved_d128_l4_pb_ood_improved", "PB_OOD (改进)"),
    ("experiments/outputs_burgers_dataset/custom_v14_improved_d128_l4_pdebench_improved", "PDEBench (改进)"),
]

print("\n" + "="*80)
print("过拟合诊断报告")
print("="*80)

baseline_results = []
print("\n【基线模型：d_model=64, n_layers=3, 353K参数】")
for exp_path, exp_name in BASELINE_EXPS:
    metrics = load_metrics(exp_path)
    if metrics:
        baseline_results.append(diagnose_overfitting(metrics, exp_name))

improved_results = []
print("\n【改进模型：d_model=128, n_layers=4, 910K参数】")
for exp_path, exp_name in IMPROVED_EXPS:
    metrics = load_metrics(exp_path)
    if metrics:
        improved_results.append(diagnose_overfitting(metrics, exp_name))

# 对比分析
print("\n" + "="*80)
print("对比分析")
print("="*80)

for i in range(3):
    if i < len(baseline_results) and i < len(improved_results):
        base = baseline_results[i]
        impr = improved_results[i]
        dataset = base['name'].split()[0]

        print(f"\n{dataset}:")
        print(f"  基线Gap: {base['gap_pct']:+.2f}% ({base['severity']})")
        print(f"  改进Gap: {impr['gap_pct']:+.2f}% ({impr['severity']})")

        if impr['gap_pct'] > base['gap_pct'] + 5:
            print(f"  ⚠️ 增加容量后过拟合加剧")
        elif impr['gap_pct'] < base['gap_pct'] - 5:
            print(f"  ✅ 增加容量改善了泛化")
        else:
            print(f"  ➡️ 泛化能力基本不变")

# 结论
print("\n" + "="*80)
print("💡 结论")
print("="*80 + "\n")

severe_count = sum(1 for r in baseline_results + improved_results if r['severity'] == 'severe')
mild_count = sum(1 for r in baseline_results + improved_results if r['severity'] == 'mild')
good_count = sum(1 for r in baseline_results + improved_results if r['severity'] in ['good', 'underfit'])

print(f"过拟合统计 (6个模型):")
print(f"  🔴 严重过拟合: {severe_count}")
print(f"  🟡 轻微过拟合: {mild_count}")
print(f"  🟢 泛化良好:   {good_count}")
print()

if severe_count > 3:
    print("❌ 模型确实容易过拟合")
    print("   建议：增加dropout (0.05→0.1), 添加L2正则, 减少模型容量")
elif mild_count > 3:
    print("⚠️ 存在轻微过拟合倾向")
    print("   建议：微调dropout和early stop参数")
else:
    print("✅ 整体泛化能力良好，不是过拟合问题")
    print("   真正问题：数据集质量、训练策略、或超参数不匹配")

print()
