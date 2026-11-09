"""
清理V2数据集：移除包含NaN/Inf的样本
"""
import numpy as np
from pathlib import Path
import gc

def clean_dataset(input_file, output_file, verbose=True):
    """移除NaN/Inf样本，保留有效数据"""

    # 加载原始数据
    data = np.load(input_file)['data']
    original_count = data.shape[0]

    if verbose:
        print(f'Loading: {input_file.name}')
        print(f'  Original shape: {data.shape}')
        print(f'  Original NaN: {np.isnan(data).sum()}')
        print(f'  Original Inf: {np.isinf(data).sum()}')

    # 识别有效样本（无NaN也无Inf）
    valid_mask = ~(np.isnan(data).any(axis=(1,2)) | np.isinf(data).any(axis=(1,2)))
    valid_data = data[valid_mask]
    valid_count = valid_data.shape[0]
    removed_count = original_count - valid_count

    if verbose:
        print(f'\n  Valid samples: {valid_count} / {original_count}')
        print(f'  Removed samples: {removed_count} ({100*removed_count/original_count:.1f}%)')
        print(f'  Cleaned shape: {valid_data.shape}')
        print(f'  Cleaned NaN: {np.isnan(valid_data).sum()}')
        print(f'  Cleaned Inf: {np.isinf(valid_data).sum()}')

    # 保存清理后的数据
    np.savez_compressed(output_file, data=valid_data.astype(np.float32))

    if verbose:
        size_mb = output_file.stat().st_size / 1e6
        print(f'  Saved to: {output_file.name} ({size_mb:.1f} MB)')

    return valid_count, removed_count

# 清理三个数据集
base_dir = Path('datasets/burgers_augmented_test')
splits = ['train', 'dev', 'test']

print('='*60)
print('CLEANING V2 DATASET')
print('='*60)

for split in splits:
    input_file = base_dir / f'burgers_augmented_{split}.npz'
    output_file = base_dir / f'burgers_augmented_{split}_clean.npz'

    valid_count, removed_count = clean_dataset(input_file, output_file)
    print()

print('='*60)
print('CLEANUP COMPLETE')
print('='*60)
