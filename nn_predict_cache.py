"""用保存的 d96/l4 模型对 train/dev/test 做整场推理，结果缓存到 npz 供分析脚本使用。"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "code")
import demo_burgers_dataset as nb  # noqa: E402
from rf_common import load_fields, pde_rms_fd, relative_l2  # noqa: E402

MODEL = Path(sys.argv[1] if len(sys.argv) > 1 else
             "experiments/outputs_burgers_dataset/pb_ood_clean_d96_l4_saved/model.pt")
CACHE = Path(sys.argv[2] if len(sys.argv) > 2 else "experiments/rf_pb_ood/nn_predictions.npz")


@torch.no_grad()
def predict_fields(model, cfg, fields: np.ndarray, device) -> np.ndarray:
    """复刻 evaluate_split 的整场推理循环。"""
    model.eval()
    x_axis = torch.linspace(0.0, 1.0, cfg.nx, device=device)
    t_axis = torch.linspace(0.0, 1.0, cfg.nt, device=device)
    x_flat = x_axis.repeat_interleave(cfg.nt).unsqueeze(1)
    t_flat = t_axis.repeat(cfg.nx).unsqueeze(1)
    tensor = torch.from_numpy(fields[..., None]).to(device)
    out = np.empty_like(fields)
    for i in range(len(fields)):
        u0 = tensor[i, :, 0, 0].unsqueeze(0)
        sensors = nb.build_sensor_tokens(tensor[i:i + 1], torch.zeros(1, dtype=torch.long, device=device), cfg, device)
        latent, context = model.encode(u0, sensors)
        latent = model.prepare_eval_latent(latent)
        chunks = []
        for s in range(0, x_flat.size(0), 4096):
            e = min(s + 4096, x_flat.size(0))
            q = nb.query_local_features(model, u0.expand(e - s, -1), x_flat[s:e], cfg.initial_patch_radius)
            chunks.append(model.decode_queries(context, nb.expand_latent(latent, e - s), x_flat[s:e], t_flat[s:e], q))
        out[i] = torch.cat(chunks).reshape(cfg.nx, cfg.nt).cpu().numpy()
    return out



if __name__ == "__main__":
    splits = {
        "train": load_fields("burgers_nu0.01_train_split.parquet"),
        "dev": load_fields("burgers_nu0.01_dev_split.parquet"),
        "test": load_fields("burgers_nu0.01_test.parquet"),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MODEL, map_location=device, weights_only=False)
    cfg = nb.DatasetConfig(**ckpt["config"])
    model = nb.BurgersHybridPINN(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    pred = {k: predict_fields(model, cfg, v, device) for k, v in splits.items()}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **pred)
    for k in splits:
        print(f"{k:5} L2 {relative_l2(pred[k], splits[k]):.4f}")
    print("saved", CACHE)
