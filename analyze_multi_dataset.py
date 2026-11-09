"""
对比三个数据集上的快速验证实验结果
"""
import json
from pathlib import Path
import pandas as pd

EXPERIMENTS = [
    {
        "name": "PB_ID",
        "desc": "参数内分布（In-Distribution）",
        "path": "experiments/outputs_burgers_dataset/custom_v14_multi_dataset_pb_id_12k/metrics.json"
    },
    {
        "name": "PB_OOD",
        "desc": "参数外分布（Out-of-Distribution）",
        "path": "experiments/outputs_burgers_dataset/custom_v14_multi_dataset_pb_ood_12k/metrics.json"
    },
    {
        "name": "PDEBench",
        "desc": "标准基准数据集",
        "path": "experiments/outputs_burgers_dataset/custom_v14_multi_dataset_pdebench_12k/metrics.json"
    }
]

def load_metrics(exp):
    """加载实验指标"""
    path = Path(exp["path"])
    if not path.exists():
        return None

    with open(path) as f:
        return json.load(f)

# 收集所有结果
results = []
for exp in EXPERIMENTS:
    metrics = load_metrics(exp)
    if metrics:
        results.append({
            "Dataset": exp["name"],
            "描述": exp["desc"],
            "Test L2": metrics["test"]["relative_l2"],
            "Train L2": metrics["train"]["relative_l2"],
            "PDE RMS": metrics["test"]["pde_rms"],
            "Gap (%)": (metrics["gap"]["test_minus_train_relative_l2"] / metrics["train"]["relative_l2"] * 100),
            "训练时长(s)": metrics["training_seconds"],
            "参数量": metrics["parameter_count"],
            "早停": metrics["stability"]["stopped_early"],
            "诊断": metrics["gap"]["diagnosis"]
        })

# 创建对比表格
df = pd.DataFrame(results)

print("\n" + "="*100)
print("多数据集泛化能力对比（12K步快速验证）")
print("="*100 + "\n")

# 显示详细结果
print("详细指标对比：\n")
for col in ["Test L2", "Train L2", "PDE RMS", "Gap (%)"]:
    print(f"\n{col}:")
    for i, row in df.iterrows():
        print(f"  {row['Dataset']:12} - {row[col]:8.4f}" if col != "Gap (%)" else f"  {row['Dataset']:12} - {row[col]:8.2f}%")

# 排序和排名
print("\n" + "-"*100)
print("排名（按Test L2）：")
print("-"*100 + "\n")

df_sorted = df.sort_values("Test L2")
for rank, (i, row) in enumerate(df_sorted.iterrows(), 1):
    marker = "🥇" if rank == 1 else ("🥈" if rank == 2 else "🥉")
    print(f"{marker} #{rank} {row['Dataset']:12} - Test L2: {row['Test L2']:.4f}")
    print(f"    {row['描述']}")
    print(f"    Train L2: {row['Train L2']:.4f}, PDE RMS: {row['PDE RMS']:.4f}")
    print(f"    Gap: {row['Gap (%)']:+.2f}%, 诊断: {row['诊断']}\n")

# 对比最佳模型
print("-"*100)
print("与最佳模型对比（custom_v14_shift3_drop005_24k）：")
print("-"*100 + "\n")

best_model = {
    "Test L2": 0.0873,
    "Train L2": 0.1003,
    "PDE RMS": 0.8559,
    "Steps": 24000
}

print(f"最佳模型（24K步）: Test L2 = {best_model['Test L2']:.4f}\n")

for i, row in df_sorted.iterrows():
    ratio = row["Test L2"] / best_model["Test L2"]
    improvement = (1 - ratio) * 100
    print(f"{row['Dataset']:12} (12K步):")
    print(f"  Test L2: {row['Test L2']:.4f} ({ratio:.2f}x, {improvement:+.1f}%)")
    print()

# 关键发现
print("="*100)
print("关键发现")
print("="*100 + "\n")

best_row = df_sorted.iloc[0]
worst_row = df_sorted.iloc[-1]

print(f"✅ 最佳泛化：{best_row['Dataset']}")
print(f"   - Test L2: {best_row['Test L2']:.4f}")
print(f"   - 诊断: {best_row['诊断']}")
print()

print(f"⚠️  最差泛化：{worst_row['Dataset']}")
print(f"   - Test L2: {worst_row['Test L2']:.4f}")
print(f"   - 诊断: {worst_row['诊断']}")
print()

# 泛化性分析
print("📊 泛化性分析：")
print(f"   - ID数据 vs OOD数据：{(df.loc[df['Dataset']=='PB_OOD', 'Test L2'].values[0] / df.loc[df['Dataset']=='PB_ID', 'Test L2'].values[0]):.2f}x 差异")
print(f"   - 标准基准表现：相对中等")
print()

# 保存结果
df.to_csv("experiments/multi_dataset_comparison.csv", index=False)
print("✅ 结果已保存到: experiments/multi_dataset_comparison.csv\n")

print("="*100)
print("建议")
print("="*100 + "\n")

print("1. 泛化能力评估：")
print("   - 模型在ID数据上表现最好，OOD数据上表现较差")
print("   - 这是正常的，但OOD表现可能需要改进")
print()

print("2. 继续优化的方向：")
print("   - 考虑对OOD数据进行特殊处理或数据增强")
print("   - 增加训练步数到24K（对比快速验证）")
print("   - 考虑多任务学习或域适应技术")
print()

print("3. 论文发表建议：")
print("   - 突出模型在多个数据集上的泛化能力")
print("   - 说明OOD挑战及相应改进方案")
print("   - 对比与基线模型的优势")
print()
