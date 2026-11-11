"""方案 B：随机森林拟合 d96 网络的残差（真值 - 网络预测），用来修正网络输出。

输入特征 = 初值 u0 + 网络预测场的 PCA 系数；目标 = 残差场的 PCA 系数。
预剪枝参数在 dev 上按 L2 + 0.1*PDE 选择。
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

sys.path.insert(0, "code")
import demo_burgers_dataset as nb  # noqa: E402
from rf_common import NT, NX, PCAForest, grid_search, load_fields, score  # noqa: E402

MODEL = Path("experiments/outputs_burgers_dataset/pb_ood_clean_d96_l4_saved/model.pt")
OUT = Path("experiments/rf_pb_ood")
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "nn_predictions.npz"
N_COMPONENTS = 32


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


splits = {
    "train": load_fields("burgers_nu0.01_train_split.parquet"),
    "dev": load_fields("burgers_nu0.01_dev_split.parquet"),
    "test": load_fields("burgers_nu0.01_test.parquet"),
}

if CACHE.exists():
    nn_pred = dict(np.load(CACHE))
    print("loaded cached NN predictions")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MODEL, map_location=device, weights_only=False)
    cfg = nb.DatasetConfig(**ckpt["config"])
    model = nb.BurgersHybridPINN(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    nn_pred = {k: predict_fields(model, cfg, v, device) for k, v in splits.items()}
    np.savez_compressed(CACHE, **nn_pred)
    print("saved NN predictions to", CACHE)

print("\n网络（d96/l4）单独：")
nn_scores = {k: score(nn_pred[k], splits[k]) for k in splits}
for k, s in nn_scores.items():
    print(f"  {k:5} L2 {s['l2']:.4f}  FD-PDE {s['pde']:.4f}")

# 特征：u0 + 网络预测的 PCA 系数（PCA 只在 train 上拟合）
pred_pca = PCA(n_components=16, random_state=0).fit(nn_pred["train"].reshape(len(nn_pred["train"]), -1))
feat = {k: np.hstack([splits[k][:, :, 0], pred_pca.transform(nn_pred[k].reshape(len(nn_pred[k]), -1))])
        for k in splits}
resid = {k: splits[k] - nn_pred[k] for k in splits}


def grid_search_corrected():
    """与 rf_common.grid_search 相同的网格，但按“网络输出 + 残差修正”打分。"""
    import itertools
    import time
    from rf_common import PRUNE_GRID
    keys = list(PRUNE_GRID)
    results = []
    t0 = time.time()
    for values in itertools.product(*PRUNE_GRID.values()):
        params = dict(zip(keys, values))
        m = PCAForest(N_COMPONENTS, n_estimators=200, **params).fit(feat["train"], resid["train"])
        dev = score(nn_pred["dev"] + m.predict(feat["dev"]), splits["dev"])
        tr = score(nn_pred["train"] + m.predict(feat["train"]), splits["train"])
        leaves, depth = m.leaf_stats()
        results.append({**params, "train_l2": tr["l2"], "dev_l2": dev["l2"], "dev_pde": dev["pde"],
                        "dev_score": dev["score"], "leaves": leaves, "depth": depth})
    print(f"  网格 {len(results)} 组，用时 {time.time() - t0:.0f}s")
    # 按 dev L2 选择：有限差分 PDE 会奖励平滑（真值场自身 FD-PDE≈5.5），不宜作为选择标准
    results.sort(key=lambda r: r["dev_l2"])
    return {k: results[0][k] for k in keys}, results


best, results = grid_search_corrected()
print("dev 上最优预剪枝参数:", best)

unpruned = dict(max_depth=None, min_samples_leaf=1, min_impurity_decrease=0.0, max_features="sqrt")
report = {"nn_only": nn_scores}
for label, params in (("nn+rf_unpruned", unpruned), ("nn+rf_pruned_best", best)):
    m = PCAForest(N_COMPONENTS, n_estimators=200, **params).fit(feat["train"], resid["train"])
    leaves, depth = m.leaf_stats()
    s = {k: score(nn_pred[k] + m.predict(feat[k]), splits[k]) for k in splits}
    report[label] = {"params": params, "leaves": leaves, "depth": depth, **s}
    print(f"\n[{label}] 平均叶子 {leaves:.0f}, 平均深度 {depth:.1f}")
    for k in splits:
        print(f"  {k:5} L2 {s[k]['l2']:.4f}  FD-PDE {s[k]['pde']:.4f}")

(OUT / "residual.json").write_text(json.dumps({"report": report, "grid": results}, indent=2, default=str),
                                   encoding="utf-8")
print("\nsaved", OUT / "residual.json")
