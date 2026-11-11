"""重训干净划分下的 d96/l4（配置与 pb_ood_clean_d96_l4 完全相同），并保存模型供随机森林使用。"""
import subprocess
import sys

sys.path.insert(0, ".")
from pb_ood_clean_split_runs import COMMON

name = "pb_ood_clean_d96_l4_saved"
cmd = [sys.executable, "code/demo_burgers_dataset.py", *COMMON,
       "--d-model", "96", "--n-layers", "4", "--save-model",
       "--output-dir", f"experiments/outputs_burgers_dataset/{name}"]
sys.exit(subprocess.run(cmd).returncode)
