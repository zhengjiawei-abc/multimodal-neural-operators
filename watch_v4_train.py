"""监控v4数据生成，完成后立即启动三组训练"""
import time, subprocess, sys
from pathlib import Path

v4_data = Path('datasets/burgers_pb_ood_augmented/train_augmented.npz')

print('等待 v4 增强数据生成...')
while not v4_data.exists():
    time.sleep(5)

# 检查v4数据是否已写完
for i in range(3):
    size_before = v4_data.stat().st_size
    time.sleep(3)
    size_after = v4_data.stat().st_size
    if size_before == size_after:
        break

print(f'✓ v4数据生成完成: {v4_data.stat().st_size / 1024 / 1024:.1f}MB')
print('\n启动三组v4训练...')
r = subprocess.run([sys.executable, 'pb_ood_aug_v4_train_all.py'], cwd='.')
sys.exit(r.returncode)
