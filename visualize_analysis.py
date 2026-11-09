"""
全面可视化分析：训练曲线、过拟合/欠拟合诊断、性能对比
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (16, 12)

# 实验列表
EXPERIMENTS = {
    "基线 (d64,l3,早停)": {
        "pb_id": "experiments/outputs_burgers_dataset/custom_v14_multi_dataset_pb_id_24k/metrics.json",
        "pb_ood": "experiments/outputs_burgers_dataset/custom_v14_multi_dataset_pb_ood_24k/metrics.json",
        "pdebench": "experiments/outputs_burgers_dataset/custom_v14_multi_dataset_pdebench_24k/metrics.json",
        "color": "blue",
        "marker": "o"
    },
    "改进 (d128,l4,完整)": {
        "pb_id": "experiments/outputs_burgers_dataset/custom_v14_improved_d128_l4_pb_id_improved/metrics.json",
        "pb_ood": "experiments/outputs_burgers_dataset/custom_v14_improved_d128_l4_pb_ood_improved/metrics.json",
        "pdebench": "experiments/outputs_burgers_dataset/custom_v14_improved_d128_l4_pdebench_improved/metrics.json",
        "color": "red",
        "marker": "s"
    },
    "精细化 (自适应+微正则)": {
        "pdebench": "experiments/outputs_burgers_dataset/custom_v14_refined_pdebench_refined/metrics.json",
        "color": "green",
        "marker": "^"
    }
}

def load_metrics(filepath):
    """加载metrics.json"""
    path = Path(filepath)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

def create_overview_chart():
    """创建总体性能对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('模型性能对比总览', fontsize=16, fontweight='bold')

    datasets = ['pb_id', 'pb_ood', 'pdebench']
    metrics_names = ['Test L2', 'Train L2', 'PDE RMS', 'Gap (%)', 'Parameters', 'Steps']

    results = {}
    for exp_name, exp_paths in EXPERIMENTS.items():
        results[exp_name] = {}
        for dataset in datasets:
            if dataset in exp_paths:
                metrics = load_metrics(exp_paths[dataset])
                if metrics:
                    results[exp_name][dataset] = {
                        'test_l2': metrics['test']['relative_l2'],
                        'train_l2': metrics['train']['relative_l2'],
                        'pde_rms': metrics['test']['pde_rms'],
                        'gap': (metrics['gap']['test_minus_train_relative_l2'] /
                               metrics['train']['relative_l2'] * 100),
                        'params': metrics['parameter_count'],
                        'steps': metrics['stability']['completed_steps'],
                    }

    # 1. Test L2 对比
    ax = axes[0, 0]
    for dataset in datasets:
        values = []
        labels = []
        for exp_name in EXPERIMENTS.keys():
            if dataset in results.get(exp_name, {}):
                values.append(results[exp_name][dataset]['test_l2'])
                labels.append(exp_name)
        x = np.arange(len(labels))
        ax.bar(x, values, alpha=0.7)
    ax.set_title('Test L2 Error', fontweight='bold')
    ax.set_ylabel('Error')
    ax.legend(labels[:len(labels)//3] if labels else [])

    # 2. Train L2 vs Test L2（Gap展示）
    ax = axes[0, 1]
    for dataset in datasets:
        train_vals = []
        test_vals = []
        labels = []
        for exp_name in EXPERIMENTS.keys():
            if dataset in results.get(exp_name, {}):
                train_vals.append(results[exp_name][dataset]['train_l2'])
                test_vals.append(results[exp_name][dataset]['test_l2'])
                labels.append(exp_name)
        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width/2, train_vals, width, label='Train', alpha=0.7)
        ax.bar(x + width/2, test_vals, width, label='Test', alpha=0.7)
    ax.set_title('Train vs Test L2', fontweight='bold')
    ax.set_ylabel('L2 Error')
    ax.legend()

    # 3. Gap (%) 过拟合指标
    ax = axes[0, 2]
    for dataset in datasets:
        gaps = []
        labels = []
        for exp_name in EXPERIMENTS.keys():
            if dataset in results.get(exp_name, {}):
                gaps.append(results[exp_name][dataset]['gap'])
                labels.append(exp_name)
        x = np.arange(len(labels))
        colors = ['red' if g > 10 else 'orange' if g > 5 else 'green' for g in gaps]
        ax.bar(x, gaps, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.axhline(y=10, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_title('Gap % (过拟合指标)', fontweight='bold')
    ax.set_ylabel('Gap (%)')
    ax.set_ylim(-20, 50)

    # 4. PDE RMS
    ax = axes[1, 0]
    for dataset in datasets:
        pde_vals = []
        labels = []
        for exp_name in EXPERIMENTS.keys():
            if dataset in results.get(exp_name, {}):
                pde_vals.append(results[exp_name][dataset]['pde_rms'])
                labels.append(exp_name)
        x = np.arange(len(labels))
        ax.bar(x, pde_vals, alpha=0.7)
    ax.set_title('PDE RMS Error', fontweight='bold')
    ax.set_ylabel('PDE RMS')

    # 5. 参数量对比
    ax = axes[1, 1]
    params_data = {}
    for exp_name in EXPERIMENTS.keys():
        params_data[exp_name] = []
        for dataset in datasets:
            if dataset in results.get(exp_name, {}):
                params_data[exp_name].append(results[exp_name][dataset]['params'] / 1000)
    x = np.arange(len(datasets))
    width = 0.25
    for i, (exp_name, params) in enumerate(params_data.items()):
        if params:
            ax.bar(x + i*width, params, width, label=exp_name, alpha=0.7)
    ax.set_title('Model Parameters', fontweight='bold')
    ax.set_ylabel('Parameters (K)')
    ax.set_xticks(x + width)
    ax.set_xticklabels(datasets)
    ax.legend()

    # 6. 训练步数
    ax = axes[1, 2]
    steps_data = {}
    for exp_name in EXPERIMENTS.keys():
        steps_data[exp_name] = []
        for dataset in datasets:
            if dataset in results.get(exp_name, {}):
                steps_data[exp_name].append(results[exp_name][dataset]['steps'] / 1000)
    x = np.arange(len(datasets))
    width = 0.25
    for i, (exp_name, steps) in enumerate(steps_data.items()):
        if steps:
            ax.bar(x + i*width, steps, width, label=exp_name, alpha=0.7)
    ax.set_title('Training Steps', fontweight='bold')
    ax.set_ylabel('Steps (K)')
    ax.set_xticks(x + width)
    ax.set_xticklabels(datasets)
    ax.legend()

    plt.tight_layout()
    plt.savefig('analysis_overview.png', dpi=150, bbox_inches='tight')
    print("✅ 已保存: analysis_overview.png")

def create_dataset_analysis():
    """为每个数据集创建详细分析"""
    datasets = ['pb_id', 'pb_ood', 'pdebench']

    for dataset in datasets:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'{dataset.upper()} 数据集 - 详细分析', fontsize=14, fontweight='bold')

        # 收集数据
        test_l2_vals = []
        train_l2_vals = []
        gap_vals = []
        exp_names = []

        for exp_name, exp_paths in EXPERIMENTS.items():
            if dataset in exp_paths:
                metrics = load_metrics(exp_paths[dataset])
                if metrics:
                    test_l2_vals.append(metrics['test']['relative_l2'])
                    train_l2_vals.append(metrics['train']['relative_l2'])
                    gap = (metrics['gap']['test_minus_train_relative_l2'] /
                           metrics['train']['relative_l2'] * 100)
                    gap_vals.append(gap)
                    exp_names.append(exp_name)

        # 左图：Train vs Test
        ax = axes[0]
        x = np.arange(len(exp_names))
        width = 0.35
        ax.bar(x - width/2, train_l2_vals, width, label='Train L2', alpha=0.8, color='skyblue')
        ax.bar(x + width/2, test_l2_vals, width, label='Test L2', alpha=0.8, color='coral')

        # 添加连线显示Gap
        for i in range(len(exp_names)):
            ax.plot([i-width/2, i+width/2], [train_l2_vals[i], test_l2_vals[i]],
                   'r--', alpha=0.5, linewidth=2)

        ax.set_xlabel('实验配置')
        ax.set_ylabel('L2 Error')
        ax.set_title('Train vs Test L2（Gap连线）')
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # 右图：Gap诊断
        ax = axes[1]
        colors = ['red' if g > 10 else 'orange' if g > 5 else 'green' for g in gap_vals]
        bars = ax.bar(x, gap_vals, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.axhline(y=10, color='red', linestyle=':', linewidth=2, label='过拟合阈值')
        ax.axhline(y=-5, color='blue', linestyle=':', linewidth=2, label='欠拟合阈值')

        # 添加数值标签
        for i, (bar, gap) in enumerate(zip(bars, gap_vals)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{gap:.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                   fontweight='bold')

        ax.set_xlabel('实验配置')
        ax.set_ylabel('Gap (%)')
        ax.set_title('过拟合/欠拟合诊断')
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'analysis_{dataset}.png', dpi=150, bbox_inches='tight')
        print(f"✅ 已保存: analysis_{dataset}.png")
        plt.close()

def create_summary_table():
    """创建汇总表格"""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')

    # 收集所有数据
    data = []
    for exp_name in EXPERIMENTS.keys():
        for dataset in ['pb_id', 'pb_ood', 'pdebench']:
            if dataset in EXPERIMENTS[exp_name]:
                metrics = load_metrics(EXPERIMENTS[exp_name][dataset])
                if metrics:
                    gap = (metrics['gap']['test_minus_train_relative_l2'] /
                           metrics['train']['relative_l2'] * 100)

                    # 诊断
                    if gap > 10:
                        diagnosis = '🔴 严重过拟合'
                    elif gap > 5:
                        diagnosis = '🟡 轻微过拟合'
                    elif gap > 0:
                        diagnosis = '🟢 泛化良好'
                    else:
                        diagnosis = '✅ 欠拟合/完美'

                    data.append([
                        exp_name,
                        dataset.upper(),
                        f"{metrics['test']['relative_l2']:.4f}",
                        f"{metrics['train']['relative_l2']:.4f}",
                        f"{gap:.2f}%",
                        f"{metrics['test']['pde_rms']:.4f}",
                        diagnosis,
                        f"{metrics['parameter_count']:,}",
                        f"{int(metrics['stability']['completed_steps']):,}"
                    ])

    columns = ['实验配置', '数据集', 'Test L2', 'Train L2', 'Gap', 'PDE RMS', '诊断', '参数量', '步数']

    table = ax.table(cellText=data, colLabels=columns, cellLoc='center', loc='center',
                     colWidths=[0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.1, 0.1])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)

    # 着色header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 交替着色rows
    for i in range(1, len(data) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('white')

    plt.title('完整实验结果汇总表', fontsize=14, fontweight='bold', pad=20)
    plt.savefig('analysis_summary_table.png', dpi=150, bbox_inches='tight')
    print("✅ 已保存: analysis_summary_table.png")
    plt.close()

def main():
    print("\n" + "="*80)
    print("开始生成可视化分析")
    print("="*80 + "\n")

    print("📊 生成总体性能对比图...")
    create_overview_chart()

    print("📊 生成数据集详细分析...")
    create_dataset_analysis()

    print("📊 生成汇总表格...")
    create_summary_table()

    print("\n" + "="*80)
    print("✅ 所有可视化分析已完成！")
    print("="*80)
    print("\n生成的图表：")
    print("  1. analysis_overview.png      - 总体性能对比（6个子图）")
    print("  2. analysis_pb_id.png         - PB_ID数据集详细分析")
    print("  3. analysis_pb_ood.png        - PB_OOD数据集详细分析")
    print("  4. analysis_pdebench.png      - PDEBench数据集详细分析")
    print("  5. analysis_summary_table.png - 完整实验结果汇总表")
    print()

if __name__ == "__main__":
    main()
