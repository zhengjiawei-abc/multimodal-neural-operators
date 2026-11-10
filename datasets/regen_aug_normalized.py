"""
重新生成增强数据 —— 归一化到原始数据的统计特性
原始PB_OOD数据范围: [-1.05, 0.82]，均值≈0, std≈0.25
原因：原始数据使用的IC幅值较小，本脚本匹配这一特性
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.fft import fft, ifft
import pyarrow.parquet as pq
from pathlib import Path

def gen_ic_small(x):
    """生成与原始数据幅值相近的初始条件（幅值0.2~0.8）"""
    ic_type = np.random.choice(['gaussian', 'sine', 'shock'])
    if ic_type == 'gaussian':
        u0 = np.zeros_like(x)
        for _ in range(np.random.randint(2, 4)):
            amp = np.random.uniform(0.1, 0.5)   # 小幅值
            c   = np.random.uniform(x[0], x[-1])
            w   = np.random.uniform(0.3, 1.0)   # 较宽
            u0 += amp * np.exp(-((x - c) / w) ** 2)
        u0 += np.random.normal(0, 0.02, x.shape)
    elif ic_type == 'sine':
        u0 = np.zeros_like(x)
        for _ in range(np.random.randint(2, 5)):
            u0 += np.random.uniform(0.1, 0.4) * np.sin(
                np.random.uniform(1, 6) * x + np.random.uniform(0, 2*np.pi))
    else:  # shock（小幅）
        c = np.random.uniform(x[0]+0.5, x[-1]-0.5)
        w = np.random.uniform(0.1, 0.4)
        a = np.random.uniform(0.3, 0.8)
        u0 = a * (1 - np.tanh((x - c) / w)) / 2 - a * (1 + np.tanh((x - c) / w)) / 2
    return u0

def burgers_rhs(t, u_hat, k, nu):
    u = np.real(ifft(u_hat))
    return -fft(u * np.real(ifft(1j * k * u_hat))) - nu * (k**2) * u_hat

def solve_one(nu, nx=256, nt=101):
    L = 2 * np.pi
    x = np.linspace(0, L, nx, endpoint=False)
    k = 2 * np.pi * np.fft.fftfreq(nx, d=x[1]-x[0])
    t_eval = np.linspace(0, 1.0, nt)
    u0_hat = fft(gen_ic_small(x))
    sol = solve_ivp(lambda t, y: burgers_rhs(t, y, k, nu),
                    t_span=(0, 1.0), y0=u0_hat, t_eval=t_eval,
                    method='RK45', rtol=1e-6, atol=1e-8)
    u = np.array([np.real(ifft(sol.y[:, i])) for i in range(nt)]).T  # (nx, nt)
    return u

def gen_group(nu_range, n, label):
    print(f'  {label}: nu∈{nu_range}, n={n}')
    out, fail = [], 0
    for i in range(n):
        if (i+1) % 100 == 0: print(f'    {i+1}/{n}')
        try:
            u = solve_one(np.random.uniform(*nu_range))
            if np.isfinite(u).all() and np.abs(u).max() < 10:
                out.append(u[:, :, np.newaxis])  # (256, 101, 1)
            else: fail += 1
        except: fail += 1
    print(f'    成功{len(out)}, 失败{fail}')
    return np.array(out, dtype=np.float32) if out else np.zeros((0,256,101,1))

def load_orig(path, n=512):
    pf = pq.ParquetFile(path)
    s = []
    for batch in pf.iter_batches(batch_size=64):
        col = batch.column('tensor')
        for i in range(batch.num_rows):
            if len(s) >= n: break
            s.append(np.array(col[i].as_py(), dtype='f4'))
        if len(s) >= n: break
    d = np.stack(s)
    if d.ndim == 3: d = d[:, :, :, np.newaxis]
    return d

if __name__ == '__main__':
    np.random.seed(42)
    out_dir = Path('datasets/burgers_pb_ood_augmented')
    out_dir.mkdir(exist_ok=True)

    print('分析原始数据统计...')
    orig = load_orig('datasets/burgers_custom_pb_ood/burgers_nu0.01_train_000.parquet')
    print(f'  原始: shape={orig.shape}, range=[{orig.min():.3f}, {orig.max():.3f}], std={orig.std():.3f}')

    print('生成三组增强数据（小幅值IC）...')
    aug_a = gen_group((0.005, 0.009), 300, 'A nu∈[0.005,0.009]')
    aug_b = gen_group((0.003, 0.007), 300, 'B nu∈[0.003,0.007]')
    aug_c = gen_group((0.001, 0.015), 300, 'C nu∈[0.001,0.015]')

    all_aug = [x for x in [aug_a, aug_b, aug_c] if len(x) > 0]
    aug = np.concatenate(all_aug, axis=0)
    print(f'\n增强数据: shape={aug.shape}, range=[{aug.min():.3f}, {aug.max():.3f}], std={aug.std():.3f}')

    combined = np.concatenate([orig, aug], axis=0)
    idx = np.random.permutation(len(combined))
    combined = combined[idx]
    print(f'合并后: shape={combined.shape}, range=[{combined.min():.3f}, {combined.max():.3f}]')

    np.savez_compressed(out_dir / 'train_augmented.npz', data=combined[:, :, :, 0])
    print(f'✓ 已保存: {out_dir}/train_augmented.npz')
