"""按流态做重要性加权：w(train 样本) = p_test(流态) / p_train(流态)。

流态标签：train 输出场的 K-means（与 lda_kmeans.py 相同设置，K=8）。
test 的流态分布只用 test 的初值 u0 经 LDA 预测得到，不使用 test 输出场。
"""
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from rf_common import load_fields

K = 8
CLIP = 5.0  # 限制最大/最小权重比例，避免少数样本被反复抽到
OUT = Path("experiments/lda_kmeans_pb_ood")

train = load_fields("burgers_nu0.01_train_split.parquet")
test_u0 = load_fields("burgers_nu0.01_test.parquet")[:, :, 0]

z = PCA(n_components=32, random_state=0).fit_transform(train.reshape(len(train), -1))
labels = KMeans(n_clusters=K, n_init=20, random_state=0).fit_predict(z)
lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(train[:, :, 0], labels)

p_train = np.bincount(labels, minlength=K) / len(labels)
p_test = np.bincount(lda.predict(test_u0), minlength=K) / len(test_u0)
ratio = p_test / p_train
ratio = np.clip(ratio, ratio[ratio > 0].min() if (ratio > 0).any() else 1.0, None)
ratio = np.clip(ratio, ratio.max() / CLIP, None)
weights = ratio[labels].astype(np.float32)

OUT.mkdir(parents=True, exist_ok=True)
np.save(OUT / "regime_weights.npy", weights)
summary = {"K": K, "clip": CLIP,
           "p_train": p_train.round(3).tolist(),
           "p_test_pred_from_u0": p_test.round(3).tolist(),
           "regime_weight": ratio.round(3).tolist(),
           "effective_share_first512": (np.bincount(labels[:512], weights=weights[:512], minlength=K)
                                        / weights[:512].sum()).round(3).tolist()}
(OUT / "regime_weights.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
