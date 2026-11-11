"""
PB_OOD 参数空间数据增强
分析：
  PB_OOD train: mean_grad=0.0070 → nu ≈ 0.010（原始训练集）
  PB_OOD test:  mean_grad=0.0085 → nu 更小（更低粘度，更强激波，域外分布）

策略：生成3组不同nu范围的增强数据，拼接到训练集，扩大参数空间覆盖。
  组1 (aug_a): nu ~ [0.005, 0.009]  —— 向测试集方向靠近
  组2 (aug_b): nu ~ [0.003, 0.007]  —— 更远的域外区域
  组3 (aug_c): nu ~ [0.001, 0.015]  —— 宽范围覆盖（包含域内和域外）

每组 300 样本，shape=(300, nx=256, nt=101)
合并后训练集：原始 512 + 900 增强 = 1412 样本
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from scipy.integrate import solve_ivp
from scipy.fft import fft, ifft
import pyarrow.parquet as pq
from pathlib import Path

# ── 谱方法求解器（复用 generate_wider_burgers.py 的实现）────────────────────

def burgers_spectral_rhs(t, u_hat, k, nu):
    u = np.real(ifft(u_hat))
    u_ux = u * np.real(ifft(1j * k * u_hat))
    return -fft(u_ux) - nu * (k ** 2) * u_hat


def gen_ic(x):
    """随机初始条件（高斯波包 + 正弦叠加）"""
    ic_type = np.random.choice(['gaussian', 'sine', 'shock'])
    if ic_type == 'gaussian':
        u0 = np.zeros_like(x)
        for _ in range(np.random.randint(2, 5)):
            amp = np.random.uniform(0.5, 2.0)
            c   = np.random.uniform(x[0], x[-1])
            w   = np.random.uniform(0.15, 0.6)
            u0 += amp * np.exp(-((x - c) / w) ** 2)
        u0 += np.random.normal(0, 0.05, x.shape)
    elif ic_type == 'sine':
        u0 = np.zeros_like(x)
        for _ in range(np.random.randint(2, 6)):
            u0 += np.random.uniform(0.3, 1.5) * np.sin(
                np.random.uniform(1, 8) * x + np.random.uniform(0, 2*np.pi))
    else:  # shock
        c = np.random.uniform(x[0]+0.5, x[-1]-0.5)
        w = np.random.uniform(0.05, 0.25)
        a = np.random.uniform(1.0, 2.5)
        u0 = a * (1 - np.tanh((x - c) / w)) / 2 - a * (1 + np.tanh((x - c) / w)) / 2
    return u0


def solve_one(nu, nx=256, nt=101, t_end=1.0):
    L = 2 * np.pi
    x = np.linspace(0, L, nx, endpoint=False)
    k = 2 * np.pi * np.fft.fftfreq(nx, d=x[1]-x[0])
    t_eval = np.linspace(0, t_end, nt)

    u0 = gen_ic(x)
    u0_hat = fft(u0)

    sol = solve_ivp(
        fun=lambda t, y: burgers_spectral_rhs(t, y, k, nu),
        t_span=(0, t_end), y0=u0_hat, t_eval=t_eval,
        method='RK45', rtol=1e-6, atol=1e-8
    )
    u = np.array([np.real(ifft(sol.y[:, i])) for i in range(nt)])  # (nt, nx)
    return u.T  # (nx, nt)


def generate_group(nu_range, n_samples, label):
    print(f"\n生成增强组 {label}: nu ∈ [{nu_range[0]}, {nu_range[1]}], n={n_samples}")
    samples = []
    failed = 0
    for i in range(n_samples):
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{n_samples} ...")
        nu = np.random.uniform(*nu_range)
        try:
            u = solve_one(nu)
            if np.isfinite(u).all() and np.abs(u).max() < 50:
                samples.append(u[:, :, np.newaxis])  # (256, 101, 1)
            else:
                failed += 1
        except Exception:
            failed += 1
    print(f"  成功: {len(samples)}/{n_samples}, 失败: {failed}")
    return np.array(samples, dtype=np.float32)  # (N, 256, 101, 1)


def load_original_parquet(path, max_samples=512):
    """加载原始 parquet 训练集 → numpy (N, 256, 101, 1)"""
    pf = pq.ParquetFile(path)
    samples = []
    for batch in pf.iter_batches(batch_size=64):
        col = batch.column('tensor')
        for i in range(batch.num_rows):
            if len(samples) >= max_samples:
                break
            arr = np.array(col[i].as_py(), dtype=np.float32)
            samples.append(arr)
        if len(samples) >= max_samples:
            break
    data = np.stack(samples)             # (N, 256, 101, 1) or (N, 256, 101)
    if data.ndim == 3:
        data = data[:, :, :, np.newaxis]
    return data


# ── 主流程 ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    np.random.seed(2024)
    out_dir = Path('datasets/burgers_pb_ood_augmented')
    out_dir.mkdir(parents=True, exist_ok=True)

    # 生成三组增强数据
    aug_a = generate_group((0.005, 0.009), 300, 'A (nu 0.005~0.009)')
    aug_b = generate_group((0.003, 0.007), 300, 'B (nu 0.003~0.007)')
    aug_c = generate_group((0.001, 0.015), 300, 'C (nu 0.001~0.015)')

    # 加载原始训练集
    print('\n加载原始 PB_OOD 训练集...')
    orig = load_original_parquet(
        'datasets/burgers_custom_pb_ood/burgers_nu0.01_train_split.parquet')
    print(f'  原始: {orig.shape}')

    # 拼接
    combined = np.concatenate([orig, aug_a, aug_b, aug_c], axis=0)
    print(f'\n合并后训练集: {combined.shape}  '
          f'(原始 {len(orig)} + 增强 {len(aug_a)+len(aug_b)+len(aug_c)})')

    # 打乱顺序
    idx = np.random.permutation(len(combined))
    combined = combined[idx]

    # 保存训练集（NPZ，训练脚本可直接读取）
    train_path = out_dir / 'train_augmented.npz'
    np.savez_compressed(train_path, data=combined[:, :, :, 0])  # (N, 256, 101)
    print(f'  → 训练集已保存: {train_path}')

    # 统计
    print(f'\n数据统计:')
    s = combined[:, :, :, 0]
    print(f'  shape={s.shape}, min={s.min():.3f}, max={s.max():.3f}')
    print(f'  mean_grad(首帧)={np.abs(np.diff(s[:, :, 0], axis=1)).mean():.4f}')

    print('\n✅ 参数空间增强数据集生成完成！')
