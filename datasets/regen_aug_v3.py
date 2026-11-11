"""
增强数据 v3：匹配OOD测试集分布
目标：
  OOD测试集统计: t=0 mean|u|≈1.31, t=-1 mean|u|≈0.47, std≈0.76
  OOD训练集统计: t=0 mean|u|≈0.62, t=-1 mean|u|≈0.08, std≈0.33

  增强数据需覆盖测试集分布 → 大IC幅值 + 小nu（少粘性耗散）
  IC幅值: 0.5~2.0（比训练集大，接近测试集的1.31）
  nu范围: 0.001~0.008（比训练集的0.01更小，能量耗散少，t末解场更大）
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.fft import fft, ifft
import pyarrow.parquet as pq
from pathlib import Path

def gen_ic(x, amp_range=(0.5, 2.0)):
    """生成与测试集幅值相近的IC"""
    ic_type = np.random.choice(['gaussian', 'sine', 'shock', 'mixed'])
    if ic_type == 'gaussian':
        u0 = np.zeros_like(x)
        for _ in range(np.random.randint(2, 5)):
            amp = np.random.uniform(*amp_range)
            c   = np.random.uniform(x[0], x[-1])
            w   = np.random.uniform(0.2, 0.8)
            u0 += amp * np.exp(-((x - c) / w) ** 2) * np.random.choice([-1, 1])
        u0 += np.random.normal(0, 0.05, x.shape)
    elif ic_type == 'sine':
        u0 = np.zeros_like(x)
        for _ in range(np.random.randint(2, 6)):
            amp = np.random.uniform(amp_range[0]*0.5, amp_range[1]*0.5)
            u0 += amp * np.sin(np.random.uniform(1, 6) * x + np.random.uniform(0, 2*np.pi))
    elif ic_type == 'shock':
        c = np.random.uniform(x[0]+0.5, x[-1]-0.5)
        w = np.random.uniform(0.1, 0.4)
        a = np.random.uniform(*amp_range)
        u0 = a * (1 - np.tanh((x - c) / w)) / 2 - a * (1 + np.tanh((x - c) / w)) / 2
    else:  # mixed
        u0 = np.zeros_like(x)
        for _ in range(np.random.randint(1, 3)):
            a = np.random.uniform(*amp_range)
            c = np.random.uniform(x[0], x[-1])
            w = np.random.uniform(0.15, 0.6)
            u0 += a * np.exp(-((x - c) / w) ** 2) * np.random.choice([-1, 1])
        for _ in range(np.random.randint(1, 3)):
            a = np.random.uniform(amp_range[0]*0.3, amp_range[1]*0.3)
            u0 += a * np.sin(np.random.uniform(2, 8) * x + np.random.uniform(0, 2*np.pi))
    return u0

def burgers_rhs(t, u_hat, k, nu):
    u = np.real(ifft(u_hat))
    return -fft(u * np.real(ifft(1j * k * u_hat))) - nu * (k**2) * u_hat

def solve_one(nu, nx=256, nt=101, t_end=1.0):
    L = 2 * np.pi
    x = np.linspace(0, L, nx, endpoint=False)
    k = 2 * np.pi * np.fft.fftfreq(nx, d=x[1]-x[0])
    t_eval = np.linspace(0, t_end, nt)
    u0_hat = fft(gen_ic(x))
    sol = solve_ivp(lambda t, y: burgers_rhs(t, y, k, nu),
                    t_span=(0, t_end), y0=u0_hat, t_eval=t_eval,
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
            if np.isfinite(u).all() and np.abs(u).max() < 20:
                out.append(u[:, :, np.newaxis])
            else:
                fail += 1
        except:
            fail += 1
    print(f'    成功{len(out)}/{n}, 失败{fail}')
    return np.array(out, dtype=np.float32)

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
    np.random.seed(2026)
    out_dir = Path('datasets/burgers_pb_ood_augmented')
    out_dir.mkdir(exist_ok=True)

    print('分析原始数据...')
    orig = load_orig('datasets/burgers_custom_pb_ood/burgers_nu0.01_train_000.parquet')
    o = orig[:, :, :, 0]
    print(f'  原始train: std={o.std():.3f}, t0={abs(o[:,:,0]).mean():.3f}, t-1={abs(o[:,:,-1]).mean():.3f}')

    print('\n生成增强数据 v3（大幅值IC + 小nu）...')
    # 组A: 小nu（接近测试集），中幅值
    aug_a = gen_group((0.002, 0.006), 300, 'A nu∈[0.002,0.006] 小nu')
    # 组B: 更小nu，大幅值
    aug_b = gen_group((0.001, 0.004), 300, 'B nu∈[0.001,0.004] 极小nu')
    # 组C: 宽范围
    aug_c = gen_group((0.001, 0.015), 300, 'C nu∈[0.001,0.015] 宽范围')

    all_aug = [x for x in [aug_a, aug_b, aug_c] if len(x) > 0]
    aug = np.concatenate(all_aug, axis=0)
    a = aug[:, :, :, 0]
    print(f'\n增强数据: n={len(aug)}, std={a.std():.3f}, t0={abs(a[:,:,0]).mean():.3f}, t-1={abs(a[:,:,-1]).mean():.3f}')

    combined = np.concatenate([orig, aug], axis=0)
    idx = np.random.permutation(len(combined))
    combined = combined[idx]
    c = combined[:, :, :, 0]
    print(f'合并: n={len(combined)}, std={c.std():.3f}, range=[{c.min():.3f},{c.max():.3f}]')

    np.savez_compressed(out_dir / 'train_augmented.npz', data=c)
    print(f'✓ 已保存: {out_dir}/train_augmented.npz')
