"""
分析数据集复杂度，决定最优模型容量
策略：参数量小用中小模型，参数量大用大模型
"""
import numpy as np
import pandas as pd
from pathlib import Path

def analyze_dataset(data_dir, train_file):
    """分析数据集复杂度"""
    file_path = Path(data_dir) / train_file

    if not file_path.exists():
        return None

    # 读取数据
    df = pd.read_parquet(file_path)

    # 提取u字段
    u_data = np.stack([np.load(np.frombuffer(x, dtype=np.uint8))
                       for x in df['u'].values])

    # 计算复杂度指标
    n_samples = len(u_data)
    spatial_res = u_data.shape[1] if len(u_data.shape) > 1 else 256
    temporal_res = u_data.shape[2] if len(u_data.shape) > 2 else 101

    # 数据统计
    data_mean = u_data.mean()
    data_std = u_data.std()
    data_range = u_data.max() - u_data.min()

    # 空间变化率（梯度）
    spatial_grad = np.abs(np.diff(u_data, axis=1)).mean()

    # 时间变化率
    temporal_grad = np.abs(np.diff(u_data, axis=2)).mean() if len(u_data.shape) > 2 else 0

    # 数据熵（信息量）
    hist, _ = np.histogram(u_data.flatten(), bins=50)
    prob = hist / hist.sum()
    entropy = -np.sum(prob * np.log(prob + 1e-10))

    # 复杂度评分（0-1，越高越复杂）
    complexity_score = (
        0.3 * min(spatial_grad / 0.1, 1.0) +  # 空间复杂度
        0.2 * min(temporal_grad / 0.05, 1.0) +  # 时间复杂度
        0.3 * min(entropy / 4.0, 1.0) +  # 信息熵
        0.2 * min(data_std / 1.0, 1.0)  # 标准差
    )

    return {
        'n_samples': n_samples,
        'spatial_res': spatial_res,
        'temporal_res': temporal_res,
        'data_mean': data_mean,
        'data_std': data_std,
        'data_range': data_range,
        'spatial_grad': spatial_grad,
        'temporal_grad': temporal_grad,
        'entropy': entropy,
        'complexity_score': complexity_score,
    }

def recommend_model_size(complexity_score, n_samples):
    """根据复杂度推荐模型容量"""
    # 考虑数据集大小和复杂度
    data_factor = min(n_samples / 512, 1.0)  # 标准化到512样本

    # 综合评分
    total_score = 0.7 * complexity_score + 0.3 * data_factor

    if total_score < 0.3:
        return "small", 64, 3, "数据简单，小模型足够"
    elif total_score < 0.6:
        return "medium", 96, 3, "中等复杂度，中等模型"
    else:
        return "large", 128, 4, "高复杂度，需要大模型"

def main():
    print("\n" + "="*90)
    print("数据集复杂度分析 & 自适应模型容量推荐")
    print("="*90 + "\n")

    datasets = [
        {
            "name": "PB_ID",
            "data_dir": "datasets/burgers_custom_pb_id",
            "train_file": "burgers_nu0.01_train_000.parquet",
            "desc": "参数内分布"
        },
        {
            "name": "PB_OOD",
            "data_dir": "datasets/burgers_custom_pb_ood",
            "train_file": "burgers_nu0.01_train_000.parquet",
            "desc": "参数外分布"
        },
        {
            "name": "PDEBench",
            "data_dir": "datasets/pdebench_burgers",
            "train_file": "burgers_nu0.01_train_000.parquet",
            "desc": "标准基准"
        }
    ]

    recommendations = []

    for dataset in datasets:
        print(f"【{dataset['name']}】 - {dataset['desc']}")
        print("-" * 90)

        stats = analyze_dataset(dataset['data_dir'], dataset['train_file'])

        if stats is None:
            print(f"  ❌ 数据集不存在: {dataset['data_dir']}/{dataset['train_file']}\n")
            continue

        # 打印统计信息
        print(f"  样本数量:         {stats['n_samples']}")
        print(f"  空间分辨率:       {stats['spatial_res']}")
        print(f"  时间分辨率:       {stats['temporal_res']}")
        print(f"  数据均值:         {stats['data_mean']:.4f}")
        print(f"  数据标准差:       {stats['data_std']:.4f}")
        print(f"  数据范围:         {stats['data_range']:.4f}")
        print(f"  空间梯度:         {stats['spatial_grad']:.6f}")
        print(f"  时间梯度:         {stats['temporal_grad']:.6f}")
        print(f"  信息熵:           {stats['entropy']:.4f}")
        print(f"  复杂度评分:       {stats['complexity_score']:.3f} / 1.0")
        print()

        # 推荐模型
        size, d_model, n_layers, reason = recommend_model_size(
            stats['complexity_score'],
            stats['n_samples']
        )

        params_est = (d_model ** 2) * n_layers * 10 // 1000

        print(f"  ✅ 推荐模型: {size.upper()}")
        print(f"     d_model={d_model}, n_layers={n_layers}, 参数量≈{params_est}K")
        print(f"     理由: {reason}")
        print()

        recommendations.append({
            'dataset': dataset['name'],
            'complexity': stats['complexity_score'],
            'size': size,
            'd_model': d_model,
            'n_layers': n_layers,
            'params': params_est,
            'reason': reason
        })

    # 总结推荐
    print("\n" + "="*90)
    print("最终配置推荐")
    print("="*90 + "\n")

    for rec in recommendations:
        print(f"{rec['dataset']:12} (复杂度={rec['complexity']:.3f})  →  "
              f"d{rec['d_model']}, l{rec['n_layers']}, {rec['params']}K参数  |  {rec['reason']}")

    print("\n" + "="*90)
    print("策略说明")
    print("="*90)
    print("""
  📊 复杂度评分计算：
      - 30% 空间梯度：反映数据空间变化剧烈程度
      - 20% 时间梯度：反映数据时间演化速度
      - 30% 信息熵：反映数据信息量
      - 20% 标准差：反映数据波动范围

  🎯 模型容量匹配：
      - 复杂度 < 0.3  →  小模型 (d64, l3, ~120K参数)
      - 0.3 ≤ 复杂度 < 0.6  →  中模型 (d96, l3, ~280K参数)
      - 复杂度 ≥ 0.6  →  大模型 (d128, l4, ~910K参数)

  ✅ 优势：
      - 避免小数据用大模型（过拟合）
      - 避免复杂数据用小模型（欠拟合）
      - 自动平衡性能和计算成本
    """)
    print("="*90)

if __name__ == "__main__":
    main()
