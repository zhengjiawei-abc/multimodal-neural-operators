"""
生成更广泛参数范围的Burgers方程数据集
目标: 扩大物理参数覆盖范围，提升模型泛化能力
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.fft import fft, ifft
import os

def burgers_pde_spectral(t, u_hat, k, nu):
    """
    Burgers方程的谱方法求解器
    ∂u/∂t + u∂u/∂x = ν∂²u/∂x²

    Args:
        u_hat: 频域中的u
        k: 波数
        nu: 粘度系数
    """
    # 转到物理空间
    u = np.real(ifft(u_hat))

    # 计算非线性项 u*∂u/∂x
    u_ux = u * np.real(ifft(1j * k * u_hat))
    u_ux_hat = fft(u_ux)

    # 组合: -u*∂u/∂x - ν*k²*u
    dudt_hat = -u_ux_hat - nu * (k**2) * u_hat

    return dudt_hat

def generate_initial_condition(x, ic_type='random_gaussian', params=None):
    """
    生成多样化的初始条件

    Args:
        x: 空间网格
        ic_type: 初始条件类型
        params: 参数字典
    """
    if params is None:
        params = {}

    if ic_type == 'random_gaussian':
        # 随机高斯波包组合
        num_modes = np.random.randint(2, 6)  # 2-5个模式
        u0 = np.zeros_like(x)
        for _ in range(num_modes):
            amp = np.random.uniform(0.3, 2.0)
            center = np.random.uniform(x.min(), x.max())
            width = np.random.uniform(0.1, 0.5)
            u0 += amp * np.exp(-((x - center) / width) ** 2)

        # 添加小扰动
        u0 += np.random.normal(0, 0.05, size=x.shape)

    elif ic_type == 'shock':
        # 激波结构
        center = np.random.uniform(x.min() + 0.3, x.max() - 0.3)
        width = np.random.uniform(0.05, 0.2)
        amp_left = np.random.uniform(1.0, 2.5)
        amp_right = np.random.uniform(-2.5, -1.0)
        u0 = amp_left * (1 - np.tanh((x - center) / width)) / 2 + \
             amp_right * (1 + np.tanh((x - center) / width)) / 2

    elif ic_type == 'multi_sine':
        # 多频率正弦叠加
        num_freqs = np.random.randint(3, 8)
        u0 = np.zeros_like(x)
        for _ in range(num_freqs):
            freq = np.random.uniform(1, 10)
            amp = np.random.uniform(0.2, 1.5)
            phase = np.random.uniform(0, 2*np.pi)
            u0 += amp * np.sin(freq * x + phase)

    elif ic_type == 'rarefaction':
        # 稀疏波
        center = np.random.uniform(x.min() + 0.3, x.max() - 0.3)
        width = np.random.uniform(0.2, 0.5)
        amp = np.random.uniform(1.5, 3.0)
        u0 = amp * np.exp(-((x - center) / width) ** 2)
        u0 -= amp * 0.5  # 负偏移

    else:
        raise ValueError(f"Unknown IC type: {ic_type}")

    return u0

def solve_burgers_wider_param(nx=256, nt=101,
                               nu_range=(0.001, 0.1),
                               ic_types=None):
    """
    求解更广参数范围的Burgers方程

    Args:
        nx: 空间点数
        nt: 时间步数
        nu_range: 粘度系数范围
        ic_types: 初始条件类型列表
    """
    # 空间网格
    L = 2 * np.pi
    x = np.linspace(0, L, nx, endpoint=False)
    dx = x[1] - x[0]

    # 波数
    k = 2 * np.pi * np.fft.fftfreq(nx, d=dx)

    # 时间网格 (更长的演化时间以观察更多物理现象)
    t_span = (0, 2.0)  # 从1.0扩展到2.0
    t_eval = np.linspace(t_span[0], t_span[1], nt)

    # 随机选择粘度系数
    nu = np.random.uniform(nu_range[0], nu_range[1])

    # 随机选择初始条件类型
    if ic_types is None:
        ic_types = ['random_gaussian', 'shock', 'multi_sine', 'rarefaction']
    ic_type = np.random.choice(ic_types)

    # 生成初始条件
    u0 = generate_initial_condition(x, ic_type)
    u0_hat = fft(u0)

    # 求解ODE
    sol = solve_ivp(
        fun=lambda t, y: burgers_pde_spectral(t, y, k, nu),
        t_span=t_span,
        y0=u0_hat,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-6,
        atol=1e-8
    )

    # 转回物理空间
    u_solution = np.zeros((nt, nx))
    for i in range(nt):
        u_solution[i, :] = np.real(ifft(sol.y[:, i]))

    metadata = {
        'nu': nu,
        'ic_type': ic_type,
        'x': x,
        't': t_eval
    }

    return u_solution, metadata

def generate_wider_dataset(num_samples=2000,
                           output_dir='burgers_wider_params',
                           nu_range=(0.001, 0.1)):
    """
    生成更广参数范围的数据集

    核心改进:
    1. 粘度范围扩大: (0.001, 0.1) vs 原来可能的(0.01, 0.1)
    2. 4种不同初始条件类型
    3. 更长演化时间(t=2.0)
    4. 更多样本(2000个基础样本)
    """
    os.makedirs(output_dir, exist_ok=True)

    samples = []
    metadata_list = []

    print(f"生成{num_samples}个样本...")
    print(f"粘度范围: {nu_range}")
    print(f"初始条件: random_gaussian, shock, multi_sine, rarefaction")

    for i in range(num_samples):
        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{num_samples}")

        u_sol, meta = solve_burgers_wider_param(nu_range=nu_range)
        samples.append(u_sol)
        metadata_list.append(meta)

    # 转换为数组
    data = np.array(samples, dtype=np.float32)

    # 统计信息
    print(f"\n=== 数据集统计 ===")
    print(f"形状: {data.shape}")
    print(f"u值范围: [{data.min():.4f}, {data.max():.4f}]")
    print(f"u均值: {data.mean():.4f}, std: {data.std():.4f}")

    # 粘度分布
    nu_values = [m['nu'] for m in metadata_list]
    print(f"\n粘度统计:")
    print(f"  范围: [{min(nu_values):.5f}, {max(nu_values):.5f}]")
    print(f"  均值: {np.mean(nu_values):.5f}")

    # 初始条件分布
    ic_counts = {}
    for m in metadata_list:
        ic_type = m['ic_type']
        ic_counts[ic_type] = ic_counts.get(ic_type, 0) + 1
    print(f"\n初始条件分布:")
    for ic_type, count in sorted(ic_counts.items()):
        print(f"  {ic_type}: {count} ({count/num_samples*100:.1f}%)")

    # 分割数据集
    train_ratio = 0.70
    dev_ratio = 0.15
    # test_ratio = 0.15

    n_train = int(num_samples * train_ratio)
    n_dev = int(num_samples * dev_ratio)

    indices = np.random.permutation(num_samples)
    train_idx = indices[:n_train]
    dev_idx = indices[n_train:n_train+n_dev]
    test_idx = indices[n_train+n_dev:]

    # 保存
    np.savez_compressed(
        os.path.join(output_dir, 'burgers_wider_train.npz'),
        data=data[train_idx]
    )
    np.savez_compressed(
        os.path.join(output_dir, 'burgers_wider_dev.npz'),
        data=data[dev_idx]
    )
    np.savez_compressed(
        os.path.join(output_dir, 'burgers_wider_test.npz'),
        data=data[test_idx]
    )

    # 保存元数据
    np.save(
        os.path.join(output_dir, 'metadata_train.npy'),
        [metadata_list[i] for i in train_idx],
        allow_pickle=True
    )

    print(f"\n✓ 数据集已保存到: {output_dir}")
    print(f"  训练集: {len(train_idx)} 样本")
    print(f"  验证集: {len(dev_idx)} 样本")
    print(f"  测试集: {len(test_idx)} 样本")

    return output_dir

if __name__ == '__main__':
    # 生成更广参数范围的数据集
    generate_wider_dataset(
        num_samples=2000,
        output_dir='burgers_wider_params',
        nu_range=(0.001, 0.1)  # 粘度范围扩大10倍
    )
