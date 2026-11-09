import numpy as np
from pathlib import Path

# 检查v1数据集结构
train_file = Path('datasets/burgers_augmented_v1/burgers_augmented_train.npz')
data = np.load(train_file)
print('Keys in V1 train NPZ:', list(data.keys()))
for key in data.keys():
    arr = data[key]
    print(f'  {key}: shape={arr.shape}, dtype={arr.dtype}')
    print(f'       has_nan={np.isnan(arr).any()}, has_inf={np.isinf(arr).any()}')
    print(f'       min={np.nanmin(arr):.6f}, max={np.nanmax(arr):.6f}')
