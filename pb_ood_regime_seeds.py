"""基线 / 流态加权 d96/l4 各补跑 seed 7、42（seed 123 已有），成对交替运行。"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".")
from pb_ood_clean_split_runs import COMMON

WEIGHTS = "experiments/lda_kmeans_pb_ood/regime_weights.npy"
RUNS = []
for seed in (7, 42):
    RUNS.append((f"pb_ood_clean_d96_l4_saved_s{seed}", seed, []))
    RUNS.append((f"pb_ood_clean_d96_l4_regime_weighted_s{seed}", seed, ["--sample-weights", WEIGHTS]))

failed = []
for name, seed, extra in RUNS:
    out = Path("experiments/outputs_burgers_dataset") / name
    if (out / "model.pt").exists():
        print(f"skip {name} (already done)", flush=True)
        continue
    print(f"\n===== {name} =====", flush=True)
    # COMMON 里已有 --seed 123；argparse 取最后一次出现的值
    cmd = [sys.executable, "code/demo_burgers_dataset.py", *COMMON,
           "--d-model", "96", "--n-layers", "4", "--save-model", "--seed", str(seed),
           *extra, "--output-dir", str(out)]
    if subprocess.run(cmd).returncode != 0:
        failed.append(name)
print("\nfailed:", failed or "none")
sys.exit(1 if failed else 0)
