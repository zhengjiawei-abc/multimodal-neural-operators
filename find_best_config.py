"""
查找并对比所有实验的最佳结果
"""
import json
from pathlib import Path
import pandas as pd

experiments_dir = Path("experiments")
results = []

# 遍历所有实验目录
for metrics_file in experiments_dir.rglob("metrics.json"):
    try:
        with open(metrics_file) as f:
            metrics = json.load(f)

        exp_name = metrics_file.parent.name
        test_l2 = metrics.get('test', {}).get('relative_l2')
        train_l2 = metrics.get('train', {}).get('relative_l2')
        pde_rms = metrics.get('test', {}).get('pde_rms')

        if test_l2 and test_l2 < 0.3:  # 只看合理范围内的结果
            results.append({
                'experiment': exp_name,
                'test_l2': test_l2,
                'train_l2': train_l2,
                'pde_rms': pde_rms,
                'path': str(metrics_file.parent)
            })
    except:
        pass

# 按test_l2排序
results_sorted = sorted(results, key=lambda x: x['test_l2'])

print("="*80)
print("TOP 10 BEST EXPERIMENTS (by Test L2)")
print("="*80)
for i, r in enumerate(results_sorted[:10], 1):
    print(f"\n{i}. {r['experiment']}")
    print(f"   Test L2:  {r['test_l2']:.4f}")
    print(f"   Train L2: {r['train_l2']:.4f}")
    print(f"   PDE RMS:  {r['pde_rms']:.6f}")
    print(f"   Path: {r['path']}")

# 保存到CSV
df = pd.DataFrame(results_sorted)
df.to_csv('experiments/best_results_summary.csv', index=False)
print(f"\n保存结果到: experiments/best_results_summary.csv")
print(f"总计找到 {len(results)} 个有效实验")
