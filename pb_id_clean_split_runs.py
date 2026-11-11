"""PB_ID 在独立 dev 集上的干净对照（d64/l3 与 d96/l4），配置与 PB_OOD 的干净对照相同。

先等 pb_ood_regime_seeds.py 的 4 个模型都生成，避免两个训练同时占 GPU。
"""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
from pb_ood_clean_split_runs import COMMON

ROOT = Path("experiments/outputs_burgers_dataset")
WAIT_FOR = [ROOT / f"pb_ood_clean_d96_l4_{n}" / "model.pt"
            for n in ("saved_s7", "regime_weighted_s7", "saved_s42", "regime_weighted_s42")]

args = list(COMMON)
args[args.index("--data-dir") + 1] = "datasets/burgers_custom_pb_id"
RUNS = [
    ("pb_id_clean_d64_l3", ["--d-model", "64", "--n-layers", "3"]),
    ("pb_id_clean_d96_l4", ["--d-model", "96", "--n-layers", "4"]),
]

if __name__ == "__main__":
    while not all(p.exists() for p in WAIT_FOR):
        time.sleep(60)
    failed = []
    for name, extra in RUNS:
        out = ROOT / name
        if (out / "model.pt").exists():
            print(f"skip {name}", flush=True)
            continue
        print(f"\n===== {name} =====", flush=True)
        cmd = [sys.executable, "code/demo_burgers_dataset.py", *args, *extra,
               "--save-model", "--output-dir", str(out)]
        if subprocess.run(cmd).returncode != 0:
            failed.append(name)
    print("\nfailed:", failed or "none")
    sys.exit(1 if failed else 0)
