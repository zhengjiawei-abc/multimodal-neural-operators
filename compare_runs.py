"""
V1 vs V2 Clean Dataset Training Comparison
"""
import json
from pathlib import Path
import pandas as pd

def load_metrics(exp_name):
    """加载实验的metrics.json"""
    metrics_file = Path(f"experiments/{exp_name}/metrics.json")
    if metrics_file.exists():
        with open(metrics_file) as f:
            return json.load(f)
    return None

def compare_runs():
    """对比多个训练运行"""

    experiments = {
        'v1_baseline_12k': 'V1基线 (12K步)',
        'v2_clean_fixed': 'V2清理版 (18K步)',
    }

    results = {}
    print("="*70)
    print("V1 vs V2 CLEAN DATASET COMPARISON")
    print("="*70)

    for exp_name, label in experiments.items():
        metrics = load_metrics(exp_name)
        if metrics:
            results[exp_name] = {
                'label': label,
                'train_l2': metrics.get('train', {}).get('relative_l2'),
                'test_l2': metrics.get('test', {}).get('relative_l2'),
                'dev_l2': metrics.get('dev', {}).get('relative_l2'),
                'pde_rms': metrics.get('test', {}).get('pde_rms'),
                'train_time': metrics.get('training_seconds', 0),
            }
            print(f"\n{label}")
            print(f"  Train L2: {results[exp_name]['train_l2']:.4f}")
            print(f"  Dev L2:   {results[exp_name]['dev_l2']:.4f}")
            print(f"  Test L2:  {results[exp_name]['test_l2']:.4f}")
            print(f"  PDE RMS:  {results[exp_name]['pde_rms']:.6f}")
            print(f"  Time:     {results[exp_name]['train_time']:.0f}s")
        else:
            print(f"\n{label}: 未找到metrics文件")

    # 对比分析
    if len(results) >= 2:
        exp_list = list(results.keys())
        v1 = results[exp_list[0]]
        v2 = results[exp_list[1]]

        print("\n" + "="*70)
        print("IMPROVEMENT ANALYSIS")
        print("="*70)

        if v1['test_l2'] and v2['test_l2']:
            improvement = (v1['test_l2'] - v2['test_l2']) / v1['test_l2'] * 100
            print(f"\nTest L2 改善: {improvement:+.1f}%")
            print(f"  {v1['label']}: {v1['test_l2']:.4f}")
            print(f"  {v2['label']}: {v2['test_l2']:.4f}")
            if improvement > 5:
                print("  ✓ 显著改善 (>5%)")
            elif improvement > 0:
                print("  ✓ 小幅改善 (>0%)")
            else:
                print("  ✗ 性能下降")

    print("\n" + "="*70)

if __name__ == '__main__':
    compare_runs()
