"""
基于现有稳定数据进行扩展增强
策略：不重新求解PDE，而是对已验证数据进行物理合理的变换
"""
import numpy as np
import os

def apply_advanced_augmentations(data, num_augmentations=2):
    """
    对现有数据应用更多样化的增强

    Args:
        data: 原始数据 [N, nx, nt]
        num_augmentations: 每个样本生成多少个增强版本

    Returns:
        augmented_data: 增强后的数据
    """
    N, nx, nt = data.shape
    augmented_samples = []

    print(f"对{N}个样本进行{num_augmentations}倍增强...")

    for i in range(N):
        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{N}")

        sample = data[i]

        # 原样本
        augmented_samples.append(sample)

        for aug_idx in range(num_augmentations):
            augmented = sample.copy()

            # 策略1: 空间翻转 (70%概率)
            if np.random.rand() < 0.7:
                augmented = np.flip(augmented, axis=0)

            # 策略2: 循环平移 (70%概率)
            if np.random.rand() < 0.7:
                shift = np.random.randint(-nx//4, nx//4)
                augmented = np.roll(augmented, shift, axis=0)

            # 策略3: 时间子序列 (30%概率)
            if np.random.rand() < 0.3:
                # 随机选择时间窗口
                window_size = int(nt * np.random.uniform(0.6, 1.0))
                start_t = np.random.randint(0, nt - window_size + 1)
                end_t = start_t + window_size

                # 插值回到原始时间长度
                time_slice = augmented[:, start_t:end_t]
                indices = np.linspace(0, window_size-1, nt)
                augmented_time = np.zeros((nx, nt))
                for x_idx in range(nx):
                    augmented_time[x_idx] = np.interp(
                        indices,
                        np.arange(window_size),
                        time_slice[x_idx]
                    )
                augmented = augmented_time

            # 策略4: 幅值缩放 (50%概率)
            if np.random.rand() < 0.5:
                scale = np.random.uniform(0.85, 1.15)
                augmented = augmented * scale

            # 策略5: 小噪声注入 (80%概率)
            if np.random.rand() < 0.8:
                noise_level = np.random.uniform(0.001, 0.005)
                noise = np.random.normal(0, noise_level * np.abs(augmented).mean(), augmented.shape)
                augmented = augmented + noise

            # 策略6: 时间部分反转 (20%概率)
            if np.random.rand() < 0.2:
                # 反转后半段时间
                mid_t = nt // 2
                augmented[:, mid_t:] = np.flip(augmented[:, mid_t:], axis=1)

            augmented_samples.append(augmented)

    return np.array(augmented_samples, dtype=np.float32)

def combine_existing_datasets():
    """
    合并现有的多个数据集以扩大覆盖范围
    """
    datasets_to_combine = [
        'burgers_augmented_v1',
        'burgers_custom_v1'
    ]

    all_train = []
    all_dev = []
    all_test = []

    base_path = 'C:/Users/郑炜锋/Desktop/pinn_hybrid_demo_20261001/datasets'

    print("=== 合并现有数据集 ===")

    for ds_name in datasets_to_combine:
        ds_path = os.path.join(base_path, ds_name)

        # 尝试不同的文件名模式
        train_patterns = [
            'burgers_augmented_train.npz',
            'burgers_train.npz',
            'train.npz'
        ]

        train_file = None
        for pattern in train_patterns:
            candidate = os.path.join(ds_path, pattern)
            if os.path.exists(candidate):
                train_file = candidate
                break

        if train_file is None:
            print(f"  跳过 {ds_name} (找不到训练文件)")
            continue

        try:
            train_data = np.load(train_file)

            # 检查数据格式
            if 'data' in train_data:
                train_samples = train_data['data']
            elif 'u' in train_data:
                train_samples = train_data['u']
            else:
                print(f"  跳过 {ds_name} (未知格式)")
                continue

            print(f"  {ds_name}: {train_samples.shape[0]} 训练样本")
            all_train.append(train_samples)

            # 同样处理验证和测试集
            # ...这里简化处理

        except Exception as e:
            print(f"  跳过 {ds_name}: {e}")
            continue

    if len(all_train) > 0:
        combined_train = np.concatenate(all_train, axis=0)
        print(f"\n合并后训练样本总数: {combined_train.shape[0]}")
        return combined_train
    else:
        return None

def create_super_augmented_dataset():
    """
    创建超级增强数据集
    策略：基于最稳定的burgers_augmented_v1，应用5倍增强
    """
    base_path = 'C:/Users/郑炜锋/Desktop/pinn_hybrid_demo_20261001/datasets'
    source_dir = os.path.join(base_path, 'burgers_augmented_v1')
    output_dir = os.path.join(base_path, 'burgers_super_augmented')

    os.makedirs(output_dir, exist_ok=True)

    print("=== 创建超级增强数据集 ===")
    print(f"源: {source_dir}")
    print(f"目标: {output_dir}")
    print()

    # 加载原始数据
    train_orig = np.load(os.path.join(source_dir, 'burgers_augmented_train.npz'))['data']
    dev_orig = np.load(os.path.join(source_dir, 'burgers_augmented_dev.npz'))['data']
    test_orig = np.load(os.path.join(source_dir, 'burgers_augmented_test.npz'))['data']

    print(f"原始训练集: {train_orig.shape}")
    print(f"原始验证集: {dev_orig.shape}")
    print(f"原始测试集: {test_orig.shape}")
    print()

    # 对训练集进行5倍增强
    print("增强训练集...")
    train_augmented = apply_advanced_augmentations(train_orig, num_augmentations=5)

    # 对验证集进行2倍增强
    print("增强验证集...")
    dev_augmented = apply_advanced_augmentations(dev_orig, num_augmentations=2)

    # 测试集保持原样(不增强)
    test_augmented = test_orig

    print()
    print("=== 增强后数据集统计 ===")
    print(f"训练集: {train_orig.shape[0]} → {train_augmented.shape[0]} (x{train_augmented.shape[0]/train_orig.shape[0]:.1f})")
    print(f"验证集: {dev_orig.shape[0]} → {dev_augmented.shape[0]} (x{dev_augmented.shape[0]/dev_orig.shape[0]:.1f})")
    print(f"测试集: {test_augmented.shape[0]} (未增强)")
    print()

    # 数值统计
    print("=== 数值范围检查 ===")
    print(f"训练集 - 范围: [{train_augmented.min():.4f}, {train_augmented.max():.4f}]")
    print(f"         均值: {train_augmented.mean():.4f}, std: {train_augmented.std():.4f}")
    print(f"验证集 - 范围: [{dev_augmented.min():.4f}, {dev_augmented.max():.4f}]")
    print(f"测试集 - 范围: [{test_augmented.min():.4f}, {test_augmented.max():.4f}]")
    print()

    # 保存
    print("保存增强数据集...")
    np.savez_compressed(
        os.path.join(output_dir, 'burgers_super_train.npz'),
        data=train_augmented
    )
    np.savez_compressed(
        os.path.join(output_dir, 'burgers_super_dev.npz'),
        data=dev_augmented
    )
    np.savez_compressed(
        os.path.join(output_dir, 'burgers_super_test.npz'),
        data=test_augmented
    )

    print(f"✓ 完成！数据集已保存到: {output_dir}")

    return output_dir

if __name__ == '__main__':
    create_super_augmented_dataset()
