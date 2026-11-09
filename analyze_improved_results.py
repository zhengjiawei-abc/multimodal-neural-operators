"""
改进方案效果分析：d_model=128, n_layers=4, 完整24K步
对比基线：d_model=64, n_layers=3, 早停于19.5K步
"""
import json
from pathlib import Path
import pandas as pd

# 基线结果（d64, l3, 早停）
BASELINE_RESULTS = [
    {"dataset": "PB_ID", "test_l2": 0.2199, "train_l2": 0.2198, "pde_rms": 0.3684, "gap": 0.06, "params": 353169, "steps": 19501},
    {"dataset": "PB_OOD", "test_l2": 0.3363, "train_l2": 0.2397, "pde_rms": 10.6940, "gap": 40.30, "params": 353169, "steps": 19501},
    {"dataset": "PDEBench", "test_l2": 0.2138, "train_l2": 0.2264, "pde_rms": 0.6653, "gap": -5.56, "params": 353169, "steps": 19501},
]

# 改进结果（d128, l4, 完整24K）
IMPROVED_RESULTS = [
    {"dataset": "PB_ID", "test_l2": 0.1858, "train_l2": 0.1793, "pde_rms": 0.6747, "gap": 3.62, "params": 909713, "steps": 24000},
    {"dataset": "PB_OOD", "test_l2": 0.3644, "train_l2": 0.2950, "pde_rms": 2.8271, "gap": 23.55, "params": 909713, "steps": 24000},
    {"dataset": "PDEBench", "test_l2": 0.2408, "train_l2": 0.2429, "pde_rms": 0.8021, "gap": -0.87, "params": 909713, "steps": 24000},
]

# 最佳模型（原始数据集）
BEST_MODEL = {"test_l2": 0.0873, "train_l2": 0.1003, "pde_rms": 0.8559, "params": 336546}

df_baseline = pd.DataFrame(BASELINE_RESULTS)
df_improved = pd.DataFrame(IMPROVED_RESULTS)

print("\n" + "="*120)
print("改进方案效果分析 - 完整对比报告")
print("="*120)
print("\n📋 配置对比：\n")
print("基线配置 (v14):")
print("  - d_model: 64")
print("  - n_layers: 3")
print("  - 参数量: 353K")
print("  - 训练步数: 19,501 (早停)")
print()
print("改进配置 (v14_improved):")
print("  - d_model: 128  (↑2x)")
print("  - n_layers: 4   (↑1层)")
print("  - 参数量: 910K  (↑2.58x)")
print("  - 训练步数: 24,000 (完整)")
print()

# 逐数据集对比
print("="*120)
print("📊 逐数据集性能对比")
print("="*120)
print()

for i in range(len(df_baseline)):
    base = df_baseline.iloc[i]
    impr = df_improved.iloc[i]
    dataset = base["dataset"]

    l2_change = (impr["test_l2"] - base["test_l2"]) / base["test_l2"] * 100
    pde_change = (impr["pde_rms"] - base["pde_rms"]) / base["pde_rms"] * 100

    print(f"{'─'*120}")
    print(f"[{i+1}/3] {dataset}")
    print(f"{'─'*120}")
    print()
    print(f"  Test L2:       {base['test_l2']:.4f} → {impr['test_l2']:.4f}  ({l2_change:+.1f}%)")
    print(f"  Train L2:      {base['train_l2']:.4f} → {impr['train_l2']:.4f}")
    print(f"  PDE RMS:       {base['pde_rms']:.4f} → {impr['pde_rms']:.4f}  ({pde_change:+.1f}%)")
    print(f"  Gap:           {base['gap']:+.2f}% → {impr['gap']:+.2f}%")
    print(f"  训练步数:      {int(base['steps']):,} → {int(impr['steps']):,}")
    print()

    # 判断结果
    if l2_change < -5:
        print(f"  ✅ 显著改善 (Test L2 ↓{abs(l2_change):.1f}%)")
    elif l2_change < 0:
        print(f"  🟡 轻微改善 (Test L2 ↓{abs(l2_change):.1f}%)")
    else:
        print(f"  ❌ 性能下降 (Test L2 ↑{l2_change:.1f}%)")

    if pde_change < 0:
        print(f"  ✅ PDE精度提升 (RMS ↓{abs(pde_change):.1f}%)")
    else:
        print(f"  ⚠️ PDE精度下降 (RMS ↑{pde_change:.1f}%)")

    print()

# 总体统计
print("="*120)
print("📈 总体统计")
print("="*120)
print()

avg_l2_base = df_baseline["test_l2"].mean()
avg_l2_impr = df_improved["test_l2"].mean()
avg_change = (avg_l2_impr - avg_l2_base) / avg_l2_base * 100

print(f"平均Test L2:")
print(f"  基线:  {avg_l2_base:.4f}")
print(f"  改进:  {avg_l2_impr:.4f}")
print(f"  变化:  {avg_change:+.1f}%")
print()

# 排名变化
print("排名变化 (按Test L2):")
print()
df_base_sorted = df_baseline.sort_values("test_l2")
df_impr_sorted = df_improved.sort_values("test_l2")

for rank in range(3):
    base_name = df_base_sorted.iloc[rank]["dataset"]
    impr_name = df_impr_sorted.iloc[rank]["dataset"]
    base_l2 = df_base_sorted.iloc[rank]["test_l2"]
    impr_l2 = df_impr_sorted.iloc[rank]["test_l2"]

    marker = "🥇" if rank == 0 else ("🥈" if rank == 1 else "🥉")
    if base_name == impr_name:
        print(f"  {marker} #{rank+1}: {base_name:12} (不变)  {base_l2:.4f} → {impr_l2:.4f}")
    else:
        print(f"  {marker} #{rank+1}: {base_name:12} → {impr_name:12}  {base_l2:.4f} → {impr_l2:.4f}")

# 关键发现
print()
print("="*120)
print("🔍 关键发现")
print("="*120)
print()

# 找出最佳和最差改善
changes = []
for i in range(len(df_baseline)):
    base = df_baseline.iloc[i]
    impr = df_improved.iloc[i]
    change = (impr["test_l2"] - base["test_l2"]) / base["test_l2"] * 100
    changes.append((base["dataset"], change, base["test_l2"], impr["test_l2"]))

changes.sort(key=lambda x: x[1])
best_dataset, best_change, best_base, best_impr = changes[0]
worst_dataset, worst_change, worst_base, worst_impr = changes[-1]

print(f"1️⃣ 最佳改善：{best_dataset}")
print(f"   {best_base:.4f} → {best_impr:.4f}  ({best_change:+.1f}%)")
if best_change < 0:
    print(f"   ✅ 增加模型容量和完整训练带来显著收益")
else:
    print(f"   ⚠️ 未达到预期改善")
print()

print(f"2️⃣ 最差表现：{worst_dataset}")
print(f"   {worst_base:.4f} → {worst_impr:.4f}  ({worst_change:+.1f}%)")
if worst_change > 0:
    print(f"   ❌ 性能反而下降，需要分析原因")
else:
    print(f"   🟡 改善有限")
print()

# OOD分析
ood_base = df_baseline[df_baseline["dataset"]=="PB_OOD"].iloc[0]
ood_impr = df_improved[df_improved["dataset"]=="PB_OOD"].iloc[0]
pde_ood_change = (ood_impr["pde_rms"] - ood_base["pde_rms"]) / ood_base["pde_rms"] * 100

print(f"3️⃣ OOD泛化改善：")
print(f"   Test L2:  {ood_base['test_l2']:.4f} → {ood_impr['test_l2']:.4f}")
print(f"   PDE RMS:  {ood_base['pde_rms']:.4f} → {ood_impr['pde_rms']:.4f}  ({pde_ood_change:+.1f}%)")
if pde_ood_change < 0:
    print(f"   ✅ PDE精度显著提升 (↓{abs(pde_ood_change):.1f}%)")
else:
    print(f"   ⚠️ PDE精度反而下降")
print()

print(f"4️⃣ 完整训练效果：")
print(f"   所有模型完成24,000步 (vs 基线早停于19,501步)")
print(f"   额外4,499步训练效果：")
for i in range(len(df_baseline)):
    base = df_baseline.iloc[i]
    impr = df_improved.iloc[i]
    l2_change = (impr["test_l2"] - base["test_l2"]) / base["test_l2"] * 100
    print(f"     {base['dataset']:12}  {l2_change:+.1f}%")
print()

print(f"5️⃣ 参数量影响：")
print(f"   353K → 910K (2.58x增加)")
print(f"   平均性能变化: {avg_change:+.1f}%")
if avg_change < 0:
    print(f"   ✅ 增加参数量带来收益")
else:
    print(f"   ❌ 参数量增加未带来改善，可能过拟合")
print()

# 与最佳模型对比
print("="*120)
print("🏆 与最佳模型对比 (custom_v14_shift3_drop005_24k)")
print("="*120)
print()
print(f"最佳模型: Test L2 = {BEST_MODEL['test_l2']:.4f} (336K参数)\n")

for i in range(len(df_improved)):
    impr = df_improved.iloc[i]
    gap_to_best = (impr["test_l2"] / BEST_MODEL["test_l2"] - 1) * 100
    print(f"{impr['dataset']:12}  {impr['test_l2']:.4f}  ({gap_to_best:+.1f}% vs 最佳)")

print()
print("⚠️ 所有改进配置仍远超最佳模型")
print("   说明：增加模型容量不如优化数据质量和训练策略")
print()

# 结论与建议
print("="*120)
print("💡 结论与建议")
print("="*120)
print()

print("✅ 成功的改进：")
if best_change < -5:
    print(f"  ✓ {best_dataset}数据集获得显著改善 ({best_change:+.1f}%)")
if pde_ood_change < -50:
    print(f"  ✓ OOD的PDE精度大幅提升 ({pde_ood_change:+.1f}%)")
print(f"  ✓ 完整训练到24K步，无早停中断")
print()

print("❌ 未达预期：")
if avg_change >= 0:
    print(f"  ✗ 平均性能未改善 ({avg_change:+.1f}%)")
if worst_change > 5:
    print(f"  ✗ {worst_dataset}性能显著下降 ({worst_change:+.1f}%)")
print(f"  ✗ 所有配置仍远超最佳模型 (最好的也有{min(changes, key=lambda x: (df_improved[df_improved['dataset']==x[0]]['test_l2'].values[0] / BEST_MODEL['test_l2'] - 1) * 100)[0]})")
print()

print("🔧 后续方向：")
print("  1. 回归最佳配置 (d64, l3)，但禁用early stop")
print("  2. 分析最佳模型的数据集差异")
print("  3. 尝试集成学习（多种子模型ensemble）")
print("  4. 研究OOD数据增强技术")
print("  5. 优化损失函数权重和学习率调度")
print()

print("="*120)
