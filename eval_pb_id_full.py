"""PB_ID 干净划分模型在全部 dev/test 样本上的 L2（训练脚本只用前 32 个）。"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "code")
import demo_burgers_dataset as nb  # noqa: E402
import rf_common  # noqa: E402

rf_common.DATA_DIR = Path("datasets/burgers_custom_pb_id")
from nn_predict_cache import predict_fields  # noqa: E402

ROOT = Path("experiments/outputs_burgers_dataset")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = {"dev": rf_common.load_fields("burgers_nu0.01_dev_split.parquet"),
        "test": rf_common.load_fields("burgers_nu0.01_test.parquet")}
out = {}
for name in ("pb_id_clean_d64_l3", "pb_id_clean_d96_l4"):
    ckpt = torch.load(ROOT / name / "model.pt", map_location=device, weights_only=False)
    cfg = nb.DatasetConfig(**ckpt["config"])
    model = nb.BurgersHybridPINN(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    out[name] = {"params": ckpt["parameter_count"]}
    for k, y in data.items():
        p = predict_fields(model, cfg, y, device)
        out[name][f"{k}_l2_all{len(y)}"] = rf_common.relative_l2(p, y)
        out[name][f"{k}_l2_first32"] = rf_common.relative_l2(p[:32], y[:32])
    torch.cuda.empty_cache()
    print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in out[name].items()})
Path("experiments/pb_id_clean_full_eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
