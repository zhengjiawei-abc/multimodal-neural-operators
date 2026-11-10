"""
重新生成三张论文图（使用最新实验数据）
图1_架构地图.png   — 模型架构流程图
图2_性能对比.png   — 完整实验结果对比
图3_消融实验.png   — 自适应容量消融分析
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.gridspec as gridspec
import numpy as np

plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# ═══════════════════════════════════════════════════════════════════
# 图1：架构地图
# ═══════════════════════════════════════════════════════════════════
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16); ax.set_ylim(0, 10); ax.axis('off')
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('#F8F9FA')

    def box(ax, x, y, w, h, label, fc, ec, fs=10, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.15", fc=fc, ec=ec, lw=2, zorder=3))
        ax.text(x+w/2, y+h/2, label, ha='center', va='center',
                fontsize=fs, fontweight='bold' if bold else 'normal', zorder=4)

    def arrow(ax, x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.8),
                    zorder=5)

    # 标题
    ax.text(8, 9.6, '多模态混合PINN神经算子 — 架构地图',
            ha='center', fontsize=16, fontweight='bold', color='#1a1a2e')

    # ── 输入层 ──
    inputs = [('初始场\nu₀(x)', 0.5), ('PDE约束\n∂u/∂t + u∂u/∂x = ν∂²u/∂x²', 4.0), ('查询点\n(x*, t*)', 11.5)]
    colors_in = ['#AED6F1', '#A9DFBF', '#F9E79F']
    for (lbl, x), c in zip(inputs, colors_in):
        box(ax, x, 8.3, 3.2, 0.9, lbl, c, '#2874A6', fs=9)

    # ── CNN分词器 ──
    box(ax, 0.3, 6.6, 3.6, 0.9, 'CNN 分词器\n(token_count=32)', '#D2B4DE', '#8E44AD', fs=9, bold=True)
    arrow(ax, 2.1, 8.3, 2.1, 7.5)

    # ── Manifold编码器 ──
    box(ax, 4.2, 6.6, 7.6, 0.9, 'Manifold编码器  (MLP + CNN降维)  → latent z ∈ ℝ³²',
        '#A9DFBF', '#1E8449', fs=10, bold=True)
    arrow(ax, 5.6, 8.3, 7.0, 7.5)
    arrow(ax, 8.0, 7.5, 8.0, 7.5)

    # ── Latent ODE ──
    box(ax, 4.2, 5.2, 3.6, 0.9, 'Latent ODE\n(时序演化)', '#FDEBD0', '#E67E22', fs=9, bold=True)
    arrow(ax, 8.0, 6.6, 7.0, 6.1)

    # ── Transformer编码器 ──
    box(ax, 0.3, 5.2, 3.6, 0.9,
        f'Transformer编码器\n(d=64/96/128, L=3/4)', '#AED6F1', '#1A5276', fs=9, bold=True)
    arrow(ax, 2.1, 6.6, 2.1, 6.1)

    # ── 交叉注意力 ──
    box(ax, 0.3, 3.7, 7.5, 0.9, '交叉注意力融合  (context tokens ← query local features)',
        '#F9E79F', '#B7950B', fs=10, bold=True)
    arrow(ax, 2.1, 5.2, 3.0, 4.6)
    arrow(ax, 6.0, 5.2, 5.0, 4.6)

    # ── High-pass滤波 ──
    box(ax, 11.5, 6.6, 3.5, 0.9, 'High-Pass滤波\n(scale=0.5)', '#FADBD8', '#922B21', fs=9)
    arrow(ax, 13.1, 8.3, 13.1, 7.5)

    # ── 查询解码器 ──
    box(ax, 8.5, 3.7, 6.5, 0.9, '查询解码器  (local feature + cross-attention)',
        '#D5DBDB', '#566573', fs=10, bold=True)
    arrow(ax, 13.1, 6.6, 13.1, 4.6)
    arrow(ax, 7.8, 4.15, 8.5, 4.15)

    # ── 损失函数 ──
    box(ax, 1.5, 2.1, 5.5, 0.9,
        '多任务损失  L = L_data + λ_phy·L_pde + L_ic + L_bc + L_recon',
        '#FDFEFE', '#1C2833', fs=9, bold=True)
    arrow(ax, 4.0, 3.7, 4.0, 3.0)

    box(ax, 8.5, 2.1, 6.5, 0.9, '预测输出  û(x*, t*)', '#A9DFBF', '#1E8449', fs=10, bold=True)
    arrow(ax, 11.75, 3.7, 11.75, 3.0)

    # ── 物理权重标注 ──
    ax.text(1.5, 1.5, '★ 关键超参数: physics_weight (λ_phy = 0.05~0.30)\n'
            '  较大λ_phy → Gap↓ (泛化↑)，但精度损失；最优: 0.12~0.18',
            fontsize=9, color='#C0392B', style='italic',
            bbox=dict(boxstyle='round', fc='#FDEDEC', ec='#C0392B', alpha=0.8))

    ax.text(8.5, 1.5, '★ 自适应容量结论:\n'
            '  简单数据 d96/l4 → Test L2 ↓24.4% (PB_ID)\n'
            '  复杂数据 d64/l3 → 保持基线泛化性',
            fontsize=9, color='#1A5276', style='italic',
            bbox=dict(boxstyle='round', fc='#EBF5FB', ec='#1A5276', alpha=0.8))

    plt.tight_layout()
    plt.savefig('图1_架构地图.png', dpi=200, bbox_inches='tight', facecolor='#F8F9FA')
    print('✓ 图1_架构地图.png')
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# 图2：性能对比（使用真实实验数据）
# ═══════════════════════════════════════════════════════════════════
def fig2_performance():
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor('#F8F9FA')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── 实验数据（真实结果）──
    # PB_ID
    pb_id_configs  = ['基线\nd64,l3', '自适应\nd96,l4']
    pb_id_test_l2  = [0.2199, 0.1663]
    pb_id_gaps     = [4.8, -3.9]

    # PB_OOD 全量对比（包含d96变体占位，训练完后更新）
    pb_ood_configs = ['基线\nd64,l3', 'd80,l3\ndo=0.15', 'pw=0.12\nd64', 'pw=0.18\nd64', 'seed42\nd64']
    pb_ood_test_l2 = [0.3363,  0.3166,   0.4579,   0.4722,   0.3288]
    pb_ood_gaps    = [23.6,    50.2,     17.2,     14.2,     35.1]

    # 物理权重扫描
    pw_vals   = [0.05, 0.08, 0.12, 0.18, 0.30]
    pw_test   = [0.3363, 0.3865, 0.4579, 0.4722, 0.5119]
    pw_gaps   = [23.6,   27.8,   17.2,   14.2,    9.5]

    # 自适应容量 — 完整演进
    cap_names  = ['基线\nd64,l3', '改进\nd128,l4', '自适应\nd96,l4']
    cap_test   = [0.2199, 0.1858, 0.1663]
    cap_colors = ['#E74C3C', '#F39C12', '#27AE60']

    C = {'base': '#E74C3C', 'good': '#27AE60', 'mid': '#F39C12', 'blue': '#2980B9', 'purple': '#8E44AD'}

    # ── 子图1: PB_ID 自适应容量 ──
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(pb_id_configs, pb_id_test_l2,
                   color=[C['base'], C['good']], alpha=0.85, edgecolor='k', lw=1.5)
    for b, v in zip(bars, pb_id_test_l2):
        ax1.text(b.get_x()+b.get_width()/2, v+0.003, f'{v:.4f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_title('PB_ID: 自适应容量效果', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test L2 (↓越好)', fontsize=10)
    ax1.set_ylim(0, 0.28)
    ax1.text(0.95, 0.95, '↓24.4%', transform=ax1.transAxes,
             ha='right', va='top', fontsize=13, color='#27AE60', fontweight='bold',
             bbox=dict(boxstyle='round', fc='#EAFAF1', ec='#27AE60'))
    ax1.grid(axis='y', alpha=0.3)

    # ── 子图2: PB_ID Gap对比 ──
    ax2 = fig.add_subplot(gs[0, 1])
    colors_gap = [C['mid'] if g > 0 else C['good'] for g in pb_id_gaps]
    bars = ax2.bar(pb_id_configs, pb_id_gaps, color=colors_gap, alpha=0.85, edgecolor='k', lw=1.5)
    ax2.axhline(0, color='k', lw=1, ls='--', alpha=0.5)
    for b, v in zip(bars, pb_id_gaps):
        ax2.text(b.get_x()+b.get_width()/2, v + (0.3 if v >= 0 else -0.8),
                 f'{v:+.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_title('PB_ID: 泛化Gap变化', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Gap = (Test-Train)/Train (%)', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    # ── 子图3: 自适应容量演进 ──
    ax3 = fig.add_subplot(gs[0, 2])
    bars = ax3.bar(cap_names, cap_test, color=cap_colors, alpha=0.85, edgecolor='k', lw=1.5)
    for b, v in zip(bars, cap_test):
        ax3.text(b.get_x()+b.get_width()/2, v+0.002, f'{v:.4f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax3.set_title('PB_ID: 容量演进 (d64→d128→d96)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Test L2 (↓越好)', fontsize=10)
    ax3.set_ylim(0, 0.27)
    ax3.grid(axis='y', alpha=0.3)

    # ── 子图4: PB_OOD 多策略对比 ──
    ax4 = fig.add_subplot(gs[1, 0])
    bar_colors = [C['base'], '#E67E22', C['blue'], C['purple'], C['good']]
    bars = ax4.bar(pb_ood_configs, pb_ood_test_l2, color=bar_colors, alpha=0.85, edgecolor='k', lw=1.5)
    ax4.axhline(0.3363, color=C['base'], lw=1.5, ls='--', alpha=0.7, label='基线')
    for b, v in zip(bars, pb_ood_test_l2):
        ax4.text(b.get_x()+b.get_width()/2, v+0.003, f'{v:.4f}',
                 ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax4.set_title('PB_OOD: 多策略 Test L2 对比', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Test L2 (↓越好)', fontsize=10)
    ax4.set_ylim(0, 0.6)
    ax4.grid(axis='y', alpha=0.3)

    # ── 子图5: PB_OOD Gap对比 ──
    ax5 = fig.add_subplot(gs[1, 1])
    bars = ax5.bar(pb_ood_configs, pb_ood_gaps, color=bar_colors, alpha=0.85, edgecolor='k', lw=1.5)
    ax5.axhline(23.6, color=C['base'], lw=1.5, ls='--', alpha=0.7, label='基线')
    for b, v in zip(bars, pb_ood_gaps):
        ax5.text(b.get_x()+b.get_width()/2, v+0.5, f'{v:.1f}%',
                 ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax5.set_title('PB_OOD: 多策略 Gap 对比', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Gap% (↓越好)', fontsize=10)
    ax5.grid(axis='y', alpha=0.3)
    ax5.text(0.98, 0.95, 'pw=0.18\n最优Gap', transform=ax5.transAxes,
             ha='right', va='top', fontsize=9, color=C['purple'],
             bbox=dict(boxstyle='round', fc='#F5EEF8', ec=C['purple']))

    # ── 子图6: 物理权重扫描曲线 ──
    ax6 = fig.add_subplot(gs[1, 2])
    ax6_r = ax6.twinx()
    l1, = ax6.plot(pw_vals, pw_test, 'o-', color=C['base'], lw=2, ms=8, label='Test L2 (左轴)')
    l2, = ax6_r.plot(pw_vals, pw_gaps, 's--', color=C['blue'], lw=2, ms=8, label='Gap% (右轴)')
    ax6.axvline(0.12, color='#27AE60', lw=1.5, ls=':', alpha=0.8)
    ax6.axvline(0.18, color='#F39C12', lw=1.5, ls=':', alpha=0.8)
    ax6.text(0.12, 0.52, 'pw=0.12\n折中', fontsize=8, color='#27AE60', ha='center')
    ax6.text(0.18, 0.49, 'pw=0.18\n最优', fontsize=8, color='#F39C12', ha='center')
    ax6.set_xlabel('physics_weight', fontsize=10)
    ax6.set_ylabel('Test L2', fontsize=10, color=C['base'])
    ax6_r.set_ylabel('Gap%', fontsize=10, color=C['blue'])
    ax6.set_title('物理约束权重扫描', fontsize=12, fontweight='bold')
    ax6.legend(handles=[l1, l2], loc='upper left', fontsize=8)
    ax6.grid(alpha=0.3)

    fig.suptitle('多模态混合PINN — 完整实验性能对比', fontsize=15, fontweight='bold', y=1.01)
    plt.savefig('图2_性能对比.png', dpi=200, bbox_inches='tight', facecolor='#F8F9FA')
    print('✓ 图2_性能对比.png')
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# 图3：消融实验 — 自适应容量策略分析
# ═══════════════════════════════════════════════════════════════════
def fig3_ablation():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#F8F9FA')

    # ── 子图A: 容量 × 数据集 交叉矩阵 ──
    ax = axes[0]
    datasets  = ['PB_ID', 'PB_OOD', 'PDEBench']
    capacities = ['d64,l3\n(基线)', 'd96,l4\n(自适应)', 'd128,l4\n(改进)']
    # Test L2 值（NaN = 未测试）
    data = np.array([
        [0.2199, 0.1663, 0.1858],   # PB_ID
        [0.3363, np.nan, 0.3644],   # PB_OOD
        [0.2138, 0.2491, 0.2408],   # PDEBench
    ])
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, cmap='RdYlGn_r', aspect='auto', vmin=0.15, vmax=0.45)
    ax.set_xticks(range(3)); ax.set_xticklabels(capacities, fontsize=9)
    ax.set_yticks(range(3)); ax.set_yticklabels(datasets, fontsize=11)
    ax.set_title('Test L2 热力图\n(绿=好, 红=差)', fontsize=11, fontweight='bold')
    for i in range(3):
        for j in range(3):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.4f}', ha='center', va='center',
                        fontsize=10, fontweight='bold',
                        color='white' if v > 0.32 else 'black')
            else:
                ax.text(j, i, 'N/A', ha='center', va='center', fontsize=10, color='gray')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # ── 子图B: 四方向优化对比（PB_OOD） ──
    ax = axes[1]
    strategies = ['基线', '1.物理\n约束', '2.数据\n增强', '3.元\n学习', '4.集成\n成员']
    test_l2    = [0.3363, 0.5119, 0.3629, 0.4279, 0.3288]
    gaps       = [23.6,    9.5,   40.2,   19.1,   35.1]

    x = np.arange(len(strategies))
    w = 0.35
    b1 = ax.bar(x - w/2, test_l2, w, label='Test L2 (左轴)',
                color='#E74C3C', alpha=0.8, edgecolor='k', lw=1)
    ax2b = ax.twinx()
    b2 = ax2b.bar(x + w/2, gaps, w, label='Gap% (右轴)',
                  color='#3498DB', alpha=0.6, edgecolor='k', lw=1)
    ax.set_xticks(x); ax.set_xticklabels(strategies, fontsize=9)
    ax.set_ylabel('Test L2', fontsize=10, color='#E74C3C')
    ax2b.set_ylabel('Gap%', fontsize=10, color='#3498DB')
    ax.set_title('四方向优化: PB_OOD Trade-off\n(方向1 Gap最优, 方向4 L2最优)',
                 fontsize=10, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax2b.legend(loc='upper right', fontsize=8)
    ax.axhline(0.3363, color='#E74C3C', ls='--', lw=1, alpha=0.5)
    ax2b.axhline(23.6, color='#3498DB', ls='--', lw=1, alpha=0.5)
    ax.grid(axis='y', alpha=0.2)

    # ── 子图C: 核心发现总结 ──
    ax = axes[2]
    ax.axis('off')
    summary = [
        ('PB_ID 自适应容量', '[成功]', 'd96 → Test L2 ↓24.4%\nGap -3.9%（完美泛化）', '#EAFAF1', '#27AE60'),
        ('PB_OOD 物理约束', '[权衡]', 'pw=0.18 → Gap ↓40%\n但 Test L2 ↑40%', '#FEF9E7', '#F39C12'),
        ('PB_OOD 集成学习', '[小幅]', 'seed=42 → Test L2 ↓2.2%\nGap +35%（泛化差）', '#FEF9E7', '#F39C12'),
        ('数据/正则调整', '[无效]', '容量/dropout/augment\n均无法同时改善两者', '#FDEDEC', '#E74C3C'),
        ('参数空间增强', '[进行中]', '生成900个不同nu样本\n(0.001~0.015)', '#EBF5FB', '#2980B9'),
    ]
    y = 0.95
    ax.text(0.5, 1.02, '核心发现汇总', ha='center', va='top',
            transform=ax.transAxes, fontsize=13, fontweight='bold')
    for title, status, detail, fc, ec in summary:
        ax.add_patch(FancyBboxPatch((0.02, y-0.17), 0.96, 0.17,
            boxstyle="round,pad=0.01", fc=fc, ec=ec, lw=1.5,
            transform=ax.transAxes, zorder=2))
        ax.text(0.05, y-0.04, f'{status}  {title}',
                transform=ax.transAxes, fontsize=9, fontweight='bold', color=ec, zorder=3)
        ax.text(0.05, y-0.12, detail,
                transform=ax.transAxes, fontsize=8, color='#2C3E50', zorder=3)
        y -= 0.19

    fig.suptitle('自适应模型容量策略 — 消融实验分析', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('图3_消融实验.png', dpi=200, bbox_inches='tight', facecolor='#F8F9FA')
    print('✓ 图3_消融实验.png')
    plt.close()


if __name__ == '__main__':
    print('生成论文图表...')
    fig1_architecture()
    fig2_performance()
    fig3_ablation()
    print('\n三张图全部生成完成！')
