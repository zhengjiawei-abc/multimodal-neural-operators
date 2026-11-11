"""基线 d96 与流态加权 d96 的对比：全部 test 样本的 L2、各流态 L2、固定采样点的 PDE RMS。

PDE RMS 用模型自身的自动微分残差（与训练脚本的 pde_residual 相同），
但用固定随机种子在每个 test 样本上取 32 个点（共 4096 点），两个模型用同一批点，
比训练脚本默认的 128 个随机点稳定得多。
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sys.path.insert(0, "code")
import demo_burgers_dataset as nb  # noqa: E402
from nn_predict_cache import predict_fields  # noqa: E402
from rf_common import load_fields, relative_l2  # noqa: E402

RUNS = {
    "baseline": "experiments/outputs_burgers_dataset/pb_ood_clean_d96_l4_saved/model.pt",
    "regime_weighted": "experiments/outputs_burgers_dataset/pb_ood_clean_d96_l4_regime_weighted/model.pt",
}
POINTS_PER_SAMPLE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train = load_fields("burgers_nu0.01_train_split.parquet")
test = load_fields("burgers_nu0.01_test.parquet")
# 流态标签：与 lda_kmeans.py 相同（train 输出场 PCA32 + K-means K=8），test 用真值场分配
pca = PCA(n_components=32, random_state=0).fit(train.reshape(len(train), -1))
km = KMeans(n_clusters=8, n_init=20, random_state=0).fit(pca.transform(train.reshape(len(train), -1)))
test_regime = km.predict(pca.transform(test.reshape(len(test), -1)))


def pde_rms(model, cfg, fields):
    u0 = torch.from_numpy(fields[:, :, 0]).to(device)
    g = torch.Generator(device="cpu").manual_seed(0)
    sq = []
    for _ in range(POINTS_PER_SAMPLE):
        x = torch.rand(len(u0), 1, generator=g).to(device)
        t = torch.rand(len(u0), 1, generator=g).to(device)
        latent, context = model.encode(u0)
        r = nb.pde_residual(model, context, latent, x, t, u0, cfg.initial_patch_radius, cfg.nu, create_graph=False)
        sq.append((r.detach() ** 2).flatten().cpu())
    sq = torch.cat(sq)
    return float(sq.mean().sqrt()), float(sq.median().sqrt())


def evaluate_model(path):
    """返回一个模型在全部 test 上的 L2、固定点 PDE 统计和各流态 L2。"""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = nb.DatasetConfig(**ckpt["config"])
    model = nb.BurgersHybridPINN(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    pred = predict_fields(model, cfg, test, device)
    rms, med = pde_rms(model, cfg, test)
    torch.cuda.empty_cache()
    return {
        "test_l2_all128": relative_l2(pred, test),
        "test_l2_first32": relative_l2(pred[:32], test[:32]),
        "pde_rms_4096pts": rms,
        "pde_median_abs_4096pts": med,
        "per_regime_l2": {int(c): relative_l2(pred[test_regime == c], test[test_regime == c])
                          for c in np.unique(test_regime)},
    }


if __name__ == "__main__":
    report = {}
    for name, path in RUNS.items():
        report[name] = evaluate_model(path)

    counts = {int(c): int((test_regime == c).sum()) for c in np.unique(test_regime)}
    print(f"{'':20} {'L2(128)':>9} {'L2(前32)':>9} {'PDE RMS':>9} {'PDE 中位':>9}")
    for name, r in report.items():
        print(f"{name:20} {r['test_l2_all128']:>9.4f} {r['test_l2_first32']:>9.4f} "
              f"{r['pde_rms_4096pts']:>9.3f} {r['pde_median_abs_4096pts']:>9.3f}")
    print("\n各流态 L2（样本数）:")
    for c, n in counts.items():
        b, w = report["baseline"]["per_regime_l2"][c], report["regime_weighted"]["per_regime_l2"][c]
        print(f"  流态 {c} ({n:2}): {b:.4f} -> {w:.4f}  ({(w - b) / b * 100:+.1f}%)")
    Path("experiments/lda_kmeans_pb_ood/regime_weighted_comparison.json").write_text(
        json.dumps({"counts": counts, "report": report}, indent=2), encoding="utf-8")
