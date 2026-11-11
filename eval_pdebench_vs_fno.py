"""混合 PINN（opt5_n512_drop10_steps24k_seed7）在 PDEBench 前 128 个 test 样本上的 L2，
与 FNO 基线（baselines_1d_burgers.py 同样取前 128 个）同口径比较。

该模型由 code/demo_burgers_resnet.py 训练，所以用那个模块自带的 evaluate_split。
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "code")
import demo_burgers_resnet as nb  # noqa: E402
from rf_common import load_fields  # noqa: E402
import rf_common  # noqa: E402

rf_common.DATA_DIR = Path("datasets/pdebench_burgers")
MODEL = Path("experiments/outputs_burgers_dataset/lowdata/opt5_n512_drop10_steps24k_seed7/model.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test = torch.from_numpy(load_fields("burgers_nu0.01_test.parquet", slice(0, 128))[..., None]).to(device)
ckpt = torch.load(MODEL, map_location=device, weights_only=False)
cfg = nb.DatasetConfig(**ckpt["config"])
model = nb.BurgersHybridPINN(cfg).to(device)
model.load_state_dict(ckpt["state_dict"])
for n in (32, 128):
    r = nb.evaluate_split(model, test, n, cfg, device)
    print(f"params {ckpt['parameter_count']}, test first{n}: L2 {r['relative_l2']:.4f}")
