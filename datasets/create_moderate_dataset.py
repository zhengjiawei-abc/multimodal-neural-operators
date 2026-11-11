"""
创建适度扩大的数据集（控制大小）
"""
import numpy as np
import os

def apply_moderate_augmentations(data, num_augmentations=2):
    """适度增强，控制数据量"""
    N, nx, nt = data.shape
    augmented_samples = []

    print(f"对{N}个样本进行{num_augmentations}倍增强...")

    for i in range(N):
        if (i + 1) % 200 == 0:
            print(f"  进度: {i+1}/{N}")

        sample = data[i]
        augmented_samples.append(sample)

        for aug_idx in range(num_augmentations):
            augmented = sample.copy()

            # 空间翻转
            if np.random.rand() < 0.7:
                augmented = np.flip(augmented, axis=0)

            # 循环平移
            if np.random.rand() < 0.7:
                shift = np.random.randint(-nx//4, nx//4)
                augmented = np.roll(augmented, shift, axis=0)

            # 小噪声
            if np.random.rand() < 0.8:
                noise_level = np.random.uniform(0.002, 0.008)
                noise = np.random.normal(0, noise_level * np.abs(augmented).mean(), augmented.shape)
                augmented = augmented + noise

            augmented_samples.append(augmented)

    return np.array(augmented_samples, dtype=np.float32)

# 加载现有最优数据集（但只用一部分来避免磁盘满）
base_path = 'C:/Users/郑炜锋/Desktop/pinn_hybrid_demo_20261001/datasets'
source_dir = os.path.join(base_path, 'burgers_augmented_v1')
output_dir = os.path.join(base_path, 'burgers_moderate_aug')

os.makedirs(output_dir, exist_ok=True)

print("=== 创建适度扩大数据集 ===")

# 只用一半数据进行增强（避免磁盘满）
train_orig = np.load(os.path.join(source_dir, 'burgers_augmented_train.npz'))['data']
dev_orig = np.load(os.path.join(source_dir, 'burgers_augmented_dev.npz'))['data']
test_orig = np.load(os.path.join(source_dir, 'burgers_augmented_test.npz'))['data']

# 采样减少规模
sample_rate = 0.6
train_sampled = train_orig[::int(1/sample_rate)]
dev_sampled = dev_orig

print(f"采样后训练集: {train_sampled.shape[0]} 样本")

# 3倍增强
train_augmented = apply_moderate_augmentations(train_sampled, num_augmentations=2)
dev_augmented = dev_sampled  # 验证集不增强

print()
print(f"最终训练集: {train_augmented.shape[0]} 样本")
print(f"最终验证集: {dev_augmented.shape[0]} 样本")
print(f"最终测试集: {test_orig.shape[0]} 样本")

# 保存
np.savez_compressed(
    os.path.join(output_dir, 'burgers_moderate_train.npz'),
    data=train_augmented
)
np.savez_compressed(
    os.path.join(output_dir, 'burgers_moderate_dev.npz'),
    data=dev_augmented
)
np.savez_compressed(
    os.path.join(output_dir, 'burgers_moderate_test.npz'),
    data=test_orig
)

print(f"\n✓ 数据集已保存到: {output_dir}")
