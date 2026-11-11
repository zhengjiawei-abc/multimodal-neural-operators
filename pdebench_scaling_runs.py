"""PDEBench 上 FNO 与混合 PINN 的样本量扩展实验（512 / 1024 / 2375）。

- FNO：code/baselines_1d_burgers.py，24k 步，3 个 seed，评估全部 250 个 test 样本。
- 混合 PINN：复现 lowdata/opt5_n512_drop10_steps24k 的配置（code/demo_burgers_resnet.py），
  只改 --train-samples 和 --seed；512 档已有 8 个 seed，这里只跑 1024 和 2375。
已有输出的运行会跳过，可重复执行续跑。
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path("experiments/outputs_burgers_dataset/pdebench_scaling")
N_TEST = 250

FNO_SEEDS = (23, 7, 11)
HYBRID_SEEDS = (7, 23, 11, 101)

# 与 opt5_n512_drop10_steps24k 的 config 逐项对应（其余字段来自 --preset wide）
HYBRID_ARGS = [
    # demo_burgers_resnet.py 没有数据路径参数，默认就是 datasets/pdebench_burgers 的 train/dev/test
    "--preset", "wide",
    "--manifold-arch", "conv", "--dropout", "0.1",
    "--manifold-cnn", "--query-local-features", "--latent-ode",
    "--manifold-steps", "1500", "--steps", "24000", "--lr-decay-steps", "24000",
    "--loss-balance-warmup", "500", "--physics-weight", "0.0",
    "--eval-every", "500", "--early-stop-patience", "100",
    "--save-model",
]


def run(name, cmd, done_file):
    out = ROOT / name
    if (out / done_file).exists():
        print(f"skip {name}", flush=True)
        return True
    print(f"\n===== {name} =====", flush=True)
    return subprocess.run([*cmd, "--output-dir", str(out)]).returncode == 0


if __name__ == "__main__":
    failed = []
    for n in (512, 1024, 2375):
        for seed in FNO_SEEDS:
            cmd = [sys.executable, "code/baselines_1d_burgers.py", "--model", "fno",
                   "--train-samples", str(n), "--test-samples", str(N_TEST),
                   "--steps", "24000", "--seed", str(seed)]
            if not run(f"fno_n{n}_s{seed}", cmd, "metrics.json"):
                failed.append(f"fno_n{n}_s{seed}")
    for n in (1024, 2375):
        for seed in HYBRID_SEEDS:
            cmd = [sys.executable, "code/demo_burgers_resnet.py", *HYBRID_ARGS,
                   "--train-samples", str(n), "--seed", str(seed)]
            if not run(f"hybrid_n{n}_s{seed}", cmd, "model.pt"):
                failed.append(f"hybrid_n{n}_s{seed}")
    print("\nfailed:", failed or "none")
    sys.exit(1 if failed else 0)
