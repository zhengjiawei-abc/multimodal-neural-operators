import numpy as np
from pathlib import Path

# 检查v2数据集结构
train_file = Path('datasets/burgers_augmented_test/burgers_augmented_train.npz')
print('Train file exists:', train_file.exists())
print('Train file size:', train_file.stat().st_size / 1e6, 'MB')

# 加载并检查keys
data = np.load(train_file)
print('\nKeys in NPZ:', list(data.keys()))
for key in data.keys():
    arr = data[key]
    print(f'  {key}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.6f}, max={arr.max():.6f}')
    print(f'       has_nan={np.isnan(arr).any()}, has_inf={np.isinf(arr).any()}')
