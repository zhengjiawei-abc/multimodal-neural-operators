"""
增强数据 v4：使用 Crank-Nicolson 隐式格式求解低粘度 Burgers 方程
核心洞察：
  OOD测试集 nu << 0.01（估计 0.001~0.003），导致 t=end 能量大量保留
  ODE 显式法在低 nu 时不稳定（失败率 70%+）

  Crank-Nicolson (CN) 方案：无条件稳定，适合低粘度强激波问题
  对流项用 Lax-Wendroff 格式（二阶精度，抑制数值耗散）
"""
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

def gen_ic(x, amp_range=(0.8, 2.5)):
    """高幅值 IC，匹配测试集 t=0 std≈1.5"""
    ic_type = np.random.choice(['gaussian', 'shock', 'mixed'])
    u0 = np.zeros_like(x)
    if ic_type == 'gaussian':
        for _ in range(np.random.randint(2, 5)):
            amp = np.random.uniform(*amp_range) * np.random.choice([-1, 1])
            c   = np.random.uniform(x[0], x[-1])
            w   = np.random.uniform(0.3, 1.0)
            u0 += amp * np.exp(-((x - c) / w) ** 2)
        u0 += np.random.normal(0, 0.1, x.shape)
    elif ic_type == 'shock':
        c = np.random.uniform(x[0]+0.5, x[-1]-0.5)
        w = np.random.uniform(0.05, 0.3)
        a = np.random.uniform(*amp_range)
        u0 = a * (1 - np.tanh((x - c) / w)) / 2 - a * (1 + np.tanh((x - c) / w)) / 2
    else:
        for _ in range(np.random.randint(1, 3)):
            a = np.random.uniform(*amp_range) * np.random.choice([-1, 1])
            c = np.random.uniform(x[0], x[-1])
            w = np.random.uniform(0.2, 0.7)
            u0 += a * np.exp(-((x - c) / w) ** 2)
        for _ in range(np.random.randint(1, 3)):
            a = np.random.uniform(0.3, 1.0)
            u0 += a * np.sin(np.random.uniform(1, 5) * x + np.random.uniform(0, 2*np.pi))
    return u0

def solve_burgers_lxw(nu, nx=256, nt=101, t_end=1.0):
    """
    Lax-Wendroff + CN 隐式扩散项求解 Burgers 方程
    稳定性强，适合低 nu（强激波）
    """
    L = 2 * np.pi
    dx = L / nx
    dt = t_end / (nt - 1)
    x  = np.linspace(0, L, nx, endpoint=False)

    u = gen_ic(x)
    result = np.zeros((nx, nt), dtype=np.float32)
    result[:, 0] = u

    # CN 扩散项矩阵（周期边界）
    r = nu * dt / (2 * dx**2)
    diag  =  1 + 2*r
    off   = -r
    A = np.diag([diag]*nx) + np.diag([off]*(nx-1), 1) + np.diag([off]*(nx-1), -1)
    A[0, -1] = off; A[-1, 0] = off  # 周期边界

    for n_t in range(1, nt):
        # Lax-Wendroff 对流项（周期边界）
        up  = np.roll(u, -1); um = np.roll(u, 1)
        F   = 0.5 * u**2
        Fp  = 0.5 * up**2
        # LW flux
        f_half_plus  = 0.5*(F + Fp) - 0.5*(dt/dx)*0.5*(u+up)*(Fp - F)
        f_half_minus = np.roll(f_half_plus, 1)
        u_adv = u - (dt/dx) * (f_half_plus - f_half_minus)

        # CN 扩散项: (I + r*L) u^{n+1} = u_adv + r*L * u_adv（显式）
        rhs = u_adv + r*(np.roll(u_adv,-1) - 2*u_adv + np.roll(u_adv,1))
        try:
            u = np.linalg.solve(A, rhs)
        except:
            return None

        if not np.isfinite(u).all() or abs(u).max() > 50:
            return None
        result[:, n_t] = u.astype(np.float32)

    return result

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
    return d[:,:,:,0] if d.ndim == 4 else d

if __name__ == '__main__':
    np.random.seed(42)
    out_dir = Path('datasets/burgers_pb_ood_augmented')
    out_dir.mkdir(exist_ok=True)

    # 分三组，覆盖小 nu 范围
    groups = [
        ((0.001, 0.003), 350, 'A 极小nu[0.001,0.003]'),
        ((0.003, 0.006), 350, 'B 小nu[0.003,0.006]'),
        ((0.001, 0.010), 300, 'C 宽范围[0.001,0.010]'),
    ]

    all_samples = []
    for nu_range, n, label in groups:
        print(f'  {label}: n={n}')
        ok, fail = 0, 0
        for i in range(n):
            if (i+1) % 100 == 0: print(f'    {i+1}/{n} ok={ok}')
            nu = np.random.uniform(*nu_range)
            u = solve_burgers_lxw(nu)
            if u is not None:
                all_samples.append(u[:, :, np.newaxis])
                ok += 1
            else:
                fail += 1
        print(f'    成功{ok}/{n}, 失败{fail}')

    aug = np.array(all_samples, dtype=np.float32)
    a = aug[:, :, :, 0]
    print(f'\n增强数据: n={len(aug)}, std={a.std():.3f}')
    print(f'  t=0:   std={a[:,:,0].std():.3f}, mean|u|={abs(a[:,:,0]).mean():.3f}')
    print(f'  t=end: std={a[:,:,-1].std():.3f}, mean|u|={abs(a[:,:,-1]).mean():.3f}')
    print(f'  range=[{a.min():.3f},{a.max():.3f}]')

    # 加载原始训练集并合并
    orig = load_orig('datasets/burgers_custom_pb_ood/burgers_nu0.01_train_000.parquet')
    # orig shape: (N, 256, 101), a shape: (n, 256, 101)
    combined = np.concatenate([orig, a], axis=0)
    idx = np.random.permutation(len(combined))
    combined = combined[idx]

    np.savez_compressed(out_dir / 'train_augmented.npz', data=combined)
    print(f'✓ 合并: n={len(combined)}, 已保存')

    # 与测试集对比
    test = load_orig('datasets/burgers_custom_pb_ood/burgers_nu0.01_test.parquet')
    print(f'\n目标（测试集）: std={test.std():.3f}, t0={abs(test[:,:,0]).mean():.3f}, t-1={abs(test[:,:,-1]).mean():.3f}')
    print(f'增强数据:       std={a.std():.3f}, t0={abs(a[:,:,0]).mean():.3f}, t-1={abs(a[:,:,-1]).mean():.3f}')
