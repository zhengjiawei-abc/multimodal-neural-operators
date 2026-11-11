"""方案 A：随机森林单独做算子学习 u0 -> u(x, t)，带预剪枝网格搜索。"""
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from rf_common import PCAForest, grid_search, load_fields, pde_rms_fd, relative_l2, score

OUT = Path("experiments/rf_pb_ood")
OUT.mkdir(parents=True, exist_ok=True)

train = load_fields("burgers_nu0.01_train_split.parquet")
dev = load_fields("burgers_nu0.01_dev_split.parquet")
test = load_fields("burgers_nu0.01_test.parquet")
print("shapes:", train.shape, dev.shape, test.shape)
x_tr, x_dev, x_te = train[:, :, 0], dev[:, :, 0], test[:, :, 0]

# PCA 维度：看重建误差下限，选能把 test 重建误差压到远低于 NN 误差的最小维度
for k in (16, 32, 64):
    pca = PCA(n_components=k, random_state=0).fit(train.reshape(len(train), -1))
    rec = pca.inverse_transform(pca.transform(test.reshape(len(test), -1))).reshape(test.shape)
    print(f"PCA k={k:3}: 解释方差 {pca.explained_variance_ratio_.sum():.4f}, test 重建 L2 下限 {relative_l2(rec, test):.4f}")
N_COMPONENTS = 32

print(f"真值场的有限差分 PDE RMS（离散化噪声下限）: test {pde_rms_fd(test):.4f}")

best, results = grid_search(x_tr, train, x_dev, dev, N_COMPONENTS)
print("dev 上最优预剪枝参数:", best)

unpruned = dict(max_depth=None, min_samples_leaf=1, min_impurity_decrease=0.0, max_features="sqrt")
report = {}
for label, params in (("unpruned", unpruned), ("pruned_best", best)):
    m = PCAForest(N_COMPONENTS, n_estimators=200, **params).fit(x_tr, train)
    leaves, depth = m.leaf_stats()
    s = {split: score(m.predict(x), y) for split, x, y in
         (("train", x_tr, train), ("dev", x_dev, dev), ("test", x_te, test))}
    gap = (s["test"]["l2"] - s["train"]["l2"]) / s["train"]["l2"] * 100
    report[label] = {"params": params, "leaves": leaves, "depth": depth, "gap_pct": gap, **s}
    print(f"\n[{label}] 平均叶子 {leaves:.0f}, 平均深度 {depth:.1f}")
    for split in ("train", "dev", "test"):
        print(f"  {split:5} L2 {s[split]['l2']:.4f}  FD-PDE {s[split]['pde']:.4f}")
    print(f"  gap {gap:+.1f}%")

(OUT / "standalone.json").write_text(
    json.dumps({"n_components": N_COMPONENTS, "report": report, "grid": results}, indent=2,
               default=str), encoding="utf-8")
print("\nsaved", OUT / "standalone.json")
