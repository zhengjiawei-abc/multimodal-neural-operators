"""LDA（有监督）+ K-means：能否从初值 u0 识别出输出场的流态，并与 PCA（无监督）对照。

1. 标签：对训练集输出场 u(x,t)（先压到 32 维 PCA）做 K-means，k 按轮廓系数选。
2. LDA：在 u0 上用这些标签拟合，得到 k-1 个判别方向（类间/类内方差比最大）。
3. K-means：在 LDA 投影后的 u0 上聚类；对照组在 u0 的 PCA 投影（同维度）上聚类。
4. 用 ARI 衡量聚类与流态标签的一致性，并看分布外 test 落在哪些簇。
所有变换只在 train 上拟合。
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import adjusted_rand_score, silhouette_score

from rf_common import load_fields, relative_l2

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path("experiments/lda_kmeans_pb_ood")
OUT.mkdir(parents=True, exist_ok=True)
NN_CACHE = Path("experiments/rf_pb_ood/nn_predictions.npz")

fields = {
    "train": load_fields("burgers_nu0.01_train_split.parquet"),
    "dev": load_fields("burgers_nu0.01_dev_split.parquet"),
    "test": load_fields("burgers_nu0.01_test.parquet"),
}
u0 = {k: v[:, :, 0] for k, v in fields.items()}
flat = {k: v.reshape(len(v), -1) for k, v in fields.items()}

# 1. 输出场流态标签
out_pca = PCA(n_components=32, random_state=0).fit(flat["train"])
z = {k: out_pca.transform(v) for k, v in flat.items()}
sil = {}
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(z["train"])
    sil[k] = float(silhouette_score(z["train"], km.labels_))
K = max(sil, key=sil.get)
regime_km = KMeans(n_clusters=K, n_init=20, random_state=0).fit(z["train"])
labels = {k: regime_km.predict(v) for k, v in z.items()}
print("输出场 K-means 轮廓系数:", {k: round(v, 3) for k, v in sil.items()}, "-> K =", K)
print("各流态样本数:", {s: np.bincount(labels[s], minlength=K).tolist() for s in labels})

# 2. LDA：256 维输入、851 样本，用收缩估计避免类内协方差奇异
lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto", n_components=K - 1).fit(u0["train"], labels["train"])
lda_acc = {s: float((lda.predict(u0[s]) == labels[s]).mean()) for s in u0}
print("LDA 从 u0 预测流态的准确率:", {k: round(v, 3) for k, v in lda_acc.items()})

# 3. 两种投影上的 K-means
proj = {
    "LDA": {s: lda.transform(u0[s]) for s in u0},
    "PCA": None,
}
in_pca = PCA(n_components=K - 1, random_state=0).fit(u0["train"])
proj["PCA"] = {s: in_pca.transform(u0[s]) for s in u0}

ari = {}
for name, p in proj.items():
    km = KMeans(n_clusters=K, n_init=20, random_state=0).fit(p["train"])
    ari[name] = {s: float(adjusted_rand_score(labels[s], km.predict(p[s]))) for s in p}
    print(f"{name}+K-means 与流态标签的 ARI:", {k: round(v, 3) for k, v in ari[name].items()})

# 4. 分布外 test 离训练流态多远：到最近簇中心的距离（LDA 空间，按 train 距离归一化）
km_lda = KMeans(n_clusters=K, n_init=20, random_state=0).fit(proj["LDA"]["train"])
dist = {s: km_lda.transform(proj["LDA"][s]).min(axis=1) for s in u0}
ref = float(np.median(dist["train"]))
shift = {s: float(np.median(dist[s]) / ref) for s in dist}
print("到最近簇中心的中位距离（相对 train）:", {k: round(v, 2) for k, v in shift.items()})

# 5. 若有 d96 网络预测，按流态统计误差
per_regime = None
if NN_CACHE.exists():
    nn = dict(np.load(NN_CACHE))
    per_regime = {}
    for s in ("train", "dev", "test"):
        per_regime[s] = {int(c): (relative_l2(nn[s][labels[s] == c], fields[s][labels[s] == c]),
                                  int((labels[s] == c).sum()))
                         for c in range(K) if (labels[s] == c).any()}
    print("d96 网络各流态 L2 (L2, 样本数):", per_regime)
else:
    print("未找到 d96 预测缓存，跳过分流态误差统计")

# 图：LDA 与 PCA 前两维，颜色 = 流态，形状 = 划分
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, name in zip(axes, ("LDA", "PCA")):
    for s, marker, alpha in (("train", "o", 0.35), ("test", "^", 0.9)):
        p = proj[name][s]
        y = p[:, 1] if p.shape[1] > 1 else np.zeros(len(p))
        ax.scatter(p[:, 0], y, c=labels[s], cmap="tab10", vmin=0, vmax=9,
                   marker=marker, s=18 if s == "train" else 40, alpha=alpha,
                   edgecolors="black" if s == "test" else "none", linewidths=0.5, label=s)
    ax.set_title(f"u0 的 {name} 投影（ARI test = {ari[name]['test']:.2f}）")
    ax.set_xlabel(f"{name} 1")
    ax.set_ylabel(f"{name} 2")
    ax.legend()
    ax.grid(alpha=0.3)
fig.suptitle(f"PB_OOD：颜色 = 输出场 K-means 流态（K={K}），三角 = 分布外 test")
fig.tight_layout()
fig.savefig(OUT / "lda_vs_pca.png", dpi=150)

(OUT / "results.json").write_text(json.dumps({
    "silhouette": sil, "K": K, "lda_accuracy": lda_acc, "ari": ari,
    "nearest_center_shift": shift, "nn_per_regime_l2": per_regime,
    "regime_counts": {s: np.bincount(labels[s], minlength=K).tolist() for s in labels},
}, indent=2), encoding="utf-8")
print("saved", OUT)
