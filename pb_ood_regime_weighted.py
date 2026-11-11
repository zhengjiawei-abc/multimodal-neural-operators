"""方案 1：d96/l4 + 按流态加权采样，其余设置与 pb_ood_clean_d96_l4 完全相同。"""
import subprocess
import sys

sys.path.insert(0, ".")
from pb_ood_clean_split_runs import COMMON

name = "pb_ood_clean_d96_l4_regime_weighted"
cmd = [sys.executable, "code/demo_burgers_dataset.py", *COMMON,
       "--d-model", "96", "--n-layers", "4", "--save-model",
       "--sample-weights", "experiments/lda_kmeans_pb_ood/regime_weights.npy",
       "--output-dir", f"experiments/outputs_burgers_dataset/{name}"]
sys.exit(subprocess.run(cmd).returncode)
