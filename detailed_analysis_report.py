"""
详细的实验分析报告：诊断过拟合/欠拟合 + 改进建议
"""
import json
from pathlib import Path
import pandas as pd

EXPERIMENTS_ANALYSIS = {
    "baseline": {
        "pb_id": "experiments/outputs_burgers_dataset/custom_v14_multi_dataset_pb_id_24k/metrics.json",
        "pb_ood": "experiments/outputs_burgers_dataset/custom_v14_multi_dataset_pb_ood_24k/metrics.json",
        "pdebench": "experiments/outputs_burgers_dataset/custom_v14_multi_dataset_pdebench_24k/metrics.json",
        "config": "d_model=64, n_layers=3, dropout=0.05, 参数量=353K, 早停于19.5K步"
    },
    "improved": {
        "pb_id": "experiments/outputs_burgers_dataset/custom_v14_improved_d128_l4_pb_id_improved/metrics.json",
        "pb_ood": "experiments/outputs_burgers_dataset/custom_v14_improved_d128_l4_pb_ood_improved/metrics.json",
        "pdebench": "experiments/outputs_burgers_dataset/custom_v14_improved_d128_l4_pdebench_improved/metrics.json",
        "config": "d_model=128, n_layers=4, dropout=0.05, 参数量=910K, 完整24K步"
    },
    "refined": {
        "pdebench": "experiments/outputs_burgers_dataset/custom_v14_refined_pdebench_refined/metrics.json",
        "config": "d_model=64, n_layers=3, dropout=0.08, weight_decay=5e-5, batch_size=32"
    }
}

def load_metrics(filepath):
    """加载metrics.json"""
    path = Path(filepath)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

def diagnose_model(train_l2, test_l2):
    """诊断过拟合/欠拟合"""
    gap = (test_l2 - train_l2) / train_l2 * 100

    if gap > 20:
        return "🔴 严重过拟合", gap
    elif gap > 10:
        return "🟠 中度过拟合", gap
    elif gap > 5:
        return "🟡 轻微过拟合", gap
    elif gap > 0:
        return "🟢 泛化良好", gap
    else:
        return "✅ 欠拟合/完美泛化", gap

def main():
    print("\n" + "="*100)
    print("深层训练分析报告")
    print("="*100 + "\n")

    # 1. 基线分析
    print("【1】基线模型分析 (d64, l3, 早停)")
    print("-" * 100)
    print()

    baseline_data = []
    for dataset in ['pb_id', 'pb_ood', 'pdebench']:
        metrics = load_metrics(EXPERIMENTS_ANALYSIS["baseline"][dataset])
        if metrics:
            diagnosis, gap = diagnose_model(metrics['train']['relative_l2'],
                                           metrics['test']['relative_l2'])
            baseline_data.append({
                'dataset': dataset.upper(),
                'train_l2': metrics['train']['relative_l2'],
                'test_l2': metrics['test']['relative_l2'],
                'gap': gap,
                'diagnosis': diagnosis,
                'pde_rms': metrics['test']['pde_rms'],
                'params': metrics['parameter_count'],
                'steps': metrics['stability']['completed_steps'],
                'stopped_early': metrics['stability']['stopped_early'],
                'best_step': metrics['stability']['best_dev_step'],
            })

            print(f"{dataset.upper()}:")
            print(f"  Train L2:         {metrics['train']['relative_l2']:.4f}")
            print(f"  Test L2:          {metrics['test']['relative_l2']:.4f}")
            print(f"  Gap:              {gap:+.2f}% {diagnosis}")
            print(f"  PDE RMS:          {metrics['test']['pde_rms']:.4f}")
            print(f"  完成步数:         {int(metrics['stability']['completed_steps']):,}/{24000:,}")
            print(f"  最佳验证步:       {int(metrics['stability']['best_dev_step']):,}")
            print(f"  早停:             {'是' if metrics['stability']['stopped_early'] else '否'}")
            print()

    # 2. 改进分析
    print("\n【2】改进模型分析 (d128, l4, 完整24K)")
    print("-" * 100)
    print()

    improved_data = []
    for dataset in ['pb_id', 'pb_ood', 'pdebench']:
        metrics = load_metrics(EXPERIMENTS_ANALYSIS["improved"][dataset])
        if metrics:
            diagnosis, gap = diagnose_model(metrics['train']['relative_l2'],
                                           metrics['test']['relative_l2'])
            improved_data.append({
                'dataset': dataset.upper(),
                'train_l2': metrics['train']['relative_l2'],
                'test_l2': metrics['test']['relative_l2'],
                'gap': gap,
                'diagnosis': diagnosis,
                'pde_rms': metrics['test']['pde_rms'],
                'params': metrics['parameter_count'],
            })

            print(f"{dataset.upper()}:")
            print(f"  Train L2:         {metrics['train']['relative_l2']:.4f}")
            print(f"  Test L2:          {metrics['test']['relative_l2']:.4f}")
            print(f"  Gap:              {gap:+.2f}% {diagnosis}")
            print(f"  PDE RMS:          {metrics['test']['pde_rms']:.4f}")
            print(f"  参数量:           {metrics['parameter_count']:,}")
            print()

    # 3. 精细化分析
    if Path(EXPERIMENTS_ANALYSIS["refined"]["pdebench"]).exists():
        print("\n【3】精细化模型分析 (PDEBench - 自适应正则)")
        print("-" * 100)
        print()

        metrics = load_metrics(EXPERIMENTS_ANALYSIS["refined"]["pdebench"])
        if metrics:
            diagnosis, gap = diagnose_model(metrics['train']['relative_l2'],
                                           metrics['test']['relative_l2'])

            print(f"PDEBench (精细化):")
            print(f"  Train L2:         {metrics['train']['relative_l2']:.4f}")
            print(f"  Test L2:          {metrics['test']['relative_l2']:.4f}")
            print(f"  Gap:              {gap:+.2f}% {diagnosis}")
            print(f"  PDE RMS:          {metrics['test']['pde_rms']:.4f}")
            print()

    # 4. 对比分析
    print("\n【4】改进效果对比分析")
    print("-" * 100)
    print()

    for dataset in ['pb_id', 'pb_ood', 'pdebench']:
        base = next((d for d in baseline_data if d['dataset'] == dataset.upper()), None)
        impr = next((d for d in improved_data if d['dataset'] == dataset.upper()), None)

        if base and impr:
            l2_change = (impr['test_l2'] - base['test_l2']) / base['test_l2'] * 100
            gap_change = impr['gap'] - base['gap']

            print(f"{dataset.upper()}:")
            print(f"  Test L2变化:      {base['test_l2']:.4f} → {impr['test_l2']:.4f} ({l2_change:+.1f}%)")
            print(f"  Gap变化:          {base['gap']:+.2f}% → {impr['gap']:+.2f}% ({gap_change:+.2f}pp)")

            if l2_change < 0:
                print(f"  ✅ 性能改善")
            else:
                print(f"  ❌ 性能下降")

            if gap_change < 0:
                print(f"  ✅ 过拟合缓解")
            else:
                print(f"  ⚠️ 过拟合加重")
            print()

    # 5. 关键发现
    print("\n【5】关键发现 & 深度分析")
    print("-" * 100)
    print()

    print("🔴 问题1：PB_OOD数据集严重过拟合")
    print("   基线Gap:   40.30% (严重过拟合)")
    print("   改进Gap:   23.56% (仍然严重)")
    print("   根本原因:  train/test数据分布严重不匹配（domain shift）")
    print("   解决方案:  需要域适应或数据增强，不是简单的模型改进")
    print()

    print("🟡 问题2：增加容量（353K→910K）的负面效应")
    print("   PDEBench性能下降12.6% (0.2138→0.2408)")
    print("   原因分析:")
    print("     - PDEBench基线Gap=-5.56%，已是完美泛化")
    print("     - 增加容量后模型容易拟合数据中的噪声")
    print("     - 9倍参数增加但性能下降 → 过度参数化的弊端")
    print()

    print("🟢 成功点：PB_ID数据集显著改善")
    print("   基线Gap:   0.06% (完美泛化)")
    print("   改进Gap:   3.62% (仍然良好)")
    print("   性能改善:  Test L2 ↓15.5% (0.2199→0.1858)")
    print("   说明:      简单数据集确实受益于大模型容量")
    print()

    # 6. 改进建议
    print("\n【6】改进建议（按优先级）")
    print("-" * 100)
    print()

    print("🔴 高优先级改进（立即行动）：")
    print()
    print("1️⃣ 对PB_OOD应用域适应技术")
    print("   当前: Gap 23.56% (严重过拟合)")
    print("   建议:")
    print("     - 数据增强: 随机平移、缩放、旋转")
    print("     - 逐步训练: 先在ID数据训练，再用少量OOD微调")
    print("     - 对抗训练: 增加domain adversarial loss")
    print("   预期效果: Gap降到 <15%")
    print()

    print("2️⃣ PDEBench保持小模型(d64,l3)")
    print("   当前: d128导致性能下降12.6%")
    print("   建议: 用基线配置(d64,l3)，但禁用early stop")
    print("   理由: 基线Gap=-5.56%，已经完美泛化，不需要更大模型")
    print("   预期效果: 恢复到0.2138或更好")
    print()

    print("3️⃣ PB_ID继续用大模型(d128,l4)")
    print("   当前: 已获得15.5%改善")
    print("   建议: 保持配置，微调学习率和dropout")
    print("   下一步: 尝试dropout=0.08或0.1微调")
    print()

    print("🟡 中优先级改进（后续优化）：")
    print()
    print("4️⃣ 集成学习(Ensemble)")
    print("   使用不同种子(42, 123, 2024)训练多个模型")
    print("   集成方式: 平均预测或加权投票")
    print("   预期效果: 在当前基础上再降5-10%")
    print()

    print("5️⃣ 学习率调度优化")
    print("   当前: 固定学习率0.001")
    print("   建议: 余弦退火 + warmup")
    print("   预期效果: 收敛更稳定，精度提升3-5%")
    print()

    print("🟢 低优先级改进（长期方向）：")
    print()
    print("6️⃣ 混合精度训练")
    print("   使用FP16+FP32混合，减少显存占用")
    print("   允许更大的batch size或模型")
    print()

    print("7️⃣ 知识蒸馏")
    print("   用大模型(改进)指导小模型(基线)")
    print("   预期: 小模型性能接近大模型但参数少2.5倍")
    print()

    # 7. 总结
    print("\n【7】最终总结")
    print("-" * 100)
    print()

    print("✅ 模型整体泛化能力评估:")
    print("   - 4/6配置过拟合程度可控 (Gap < 5%)")
    print("   - 2/6配置存在严重过拟合 (仅PB_OOD)")
    print("   - 结论: 不是模型容易过拟合，而是特定数据集(OOD)的域转移问题")
    print()

    print("📊 最优配置推荐:")
    print("   PB_ID:      d128, l4, dropout=0.05  → Test L2=0.1858")
    print("   PDEBench:   d64,  l3, dropout=0.05  → Test L2=0.2138 (基线)")
    print("   PB_OOD:     需要域适应，不只是调超参  → 目标 Test L2<0.25")
    print()

    print("🎯 下一步:")
    print("   1. 对PB_OOD应用数据增强和域适应")
    print("   2. 验证PDEBench用基线配置+禁用early stop的效果")
    print("   3. 对三个模型做ensemble")
    print("   4. 生成最终论文结果")
    print()
    print("="*100)

if __name__ == "__main__":
    main()
