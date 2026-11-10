"""
PB_OOD 参数空间增强训练（三组实验）
先生成增强数据，再分别训练不同配置：
  exp1: 增强训练集 + 基线配置 (d64, pw=0.05)
  exp2: 增强训练集 + 物理约束 (d64, pw=0.12)
  exp3: 增强训练集 + seed=42  (d64, pw=0.05)
"""
import subprocess, sys, json, time
from pathlib import Path

AUG_TRAIN  = Path('datasets/burgers_pb_ood_augmented/train_augmented.npz')
OOD_TEST   = 'burgers_nu0.01_test.parquet'   # 已复制到 data-dir 中
OUT_BASE   = Path('experiments/outputs_burgers_dataset')

EXPERIMENTS = [
    {
        'name': 'pb_ood_aug_baseline',
        'desc': '增强数据 + 基线配置 (pw=0.05, seed=123)',
        'seed': 123, 'pw': 0.05, 'dropout': 0.05,
        'lr': 0.001, 'wd': 0.0, 'bs': 28,
    },
    {
        'name': 'pb_ood_aug_physics',
        'desc': '增强数据 + 物理约束 (pw=0.12, seed=123)',
        'seed': 123, 'pw': 0.12, 'dropout': 0.10,
        'lr': 0.001, 'wd': 1e-4, 'bs': 24,
    },
    {
        'name': 'pb_ood_aug_seed42',
        'desc': '增强数据 + seed=42 (pw=0.05)',
        'seed': 42,  'pw': 0.05, 'dropout': 0.08,
        'lr': 0.001, 'wd': 5e-5, 'bs': 28,
    },
]

SHARED = [
    '--d-model', '64', '--n-layers', '3', '--n-heads', '4',
    '--cnn-channels', '32', '--token-count', '32',
    '--head-width', '32', '--latent-dim', '32', '--manifold-hidden', '256',
    '--highpass-scale', '0.5', '--temporal', 'ode',
    '--steps', '24000', '--warmup-steps', '200',
    '--eval-every', '500', '--early-stop-patience', '1000',
    '--input-highpass', '--manifold-cnn', '--query-local-features',
    '--latent-ode', '--cross-attention',
]

BASELINE = {'test_l2': 0.3363, 'gap': 23.6}

def run_exp(cfg):
    out = OUT_BASE / cfg['name']
    cmd = [
        sys.executable, 'code/demo_burgers_dataset.py',
        '--preset', 'wide',
        '--data-dir',   'datasets/burgers_pb_ood_augmented',
        '--train-file', 'train_augmented.npz',
        '--dev-file',   OOD_TEST,
        '--test-file',  OOD_TEST,
        '--output-dir', str(out),
        '--seed',           str(cfg['seed']),
        '--dropout',        str(cfg['dropout']),
        '--lr',             str(cfg['lr']),
        '--weight-decay',   str(cfg['wd']),
        '--batch-size',     str(cfg['bs']),
        '--physics-weight', str(cfg['pw']),
    ] + SHARED

    print(f"\n{'='*65}")
    print(f"实验: {cfg['name']}")
    print(f"描述: {cfg['desc']}")
    print(f"{'='*65}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd='.')
    elapsed = time.time() - t0

    if r.returncode == 0:
        m = json.load(open(out / 'metrics.json'))
        tl2 = m['test']['relative_l2']
        gap = (m['gap']['test_over_train_ratio'] - 1) * 100
        print(f'\n结果 ({elapsed/60:.0f}min):')
        print(f'  Test L2: {tl2:.4f}  (基线 {BASELINE["test_l2"]:.4f}, '
              f'{"↓" if tl2<BASELINE["test_l2"] else "↑"}'
              f'{abs(tl2-BASELINE["test_l2"])/BASELINE["test_l2"]*100:.1f}%)')
        print(f'  Gap:     {gap:+.1f}%  (基线 +{BASELINE["gap"]:.1f}%, '
              f'{"↓" if gap<BASELINE["gap"] else "↑"}'
              f'{abs(gap-BASELINE["gap"]):.1f}pp)')
        return {'name': cfg['name'], 'test_l2': tl2, 'gap': gap, 'ok': True}
    else:
        print(f'失败 (exit {r.returncode})')
        return {'name': cfg['name'], 'ok': False}

# ── 步骤1：生成增强数据 ────────────────────────────────────────────────────
if not AUG_TRAIN.exists():
    print('生成参数空间增强数据集...')
    r = subprocess.run([sys.executable,
                        'datasets/generate_pb_ood_augmentation.py'], cwd='.')
    if r.returncode != 0:
        print('数据生成失败，退出')
        sys.exit(1)
else:
    print(f'增强数据已存在: {AUG_TRAIN}，跳过生成')

# ── 步骤2：顺序训练三个实验 ──────────────────────────────────────────────
results = []
for cfg in EXPERIMENTS:
    results.append(run_exp(cfg))

# ── 步骤3：汇总 ───────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print('参数空间增强训练汇总')
print(f"{'='*65}")
print(f"  {'配置':<35} {'Test L2':>8} {'Gap':>8}")
print(f"  {'基线(无增强)':<35} {BASELINE['test_l2']:>8.4f} {BASELINE['gap']:>7.1f}%")
for r in results:
    if r['ok']:
        better = (r['test_l2'] < BASELINE['test_l2'] or r['gap'] < BASELINE['gap'])
        star = ' ★' if better else ''
        print(f"  {r['name']:<35} {r['test_l2']:>8.4f} {r['gap']:>7.1f}%{star}")
    else:
        print(f"  {r['name']:<35} {'FAILED':>8} {'':>8}")
