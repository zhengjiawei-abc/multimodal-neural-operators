"""随机森林实验的公共工具：数据加载、PCA 压缩、L2 / 有限差分 PDE 残差、预剪枝网格搜索。"""
from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

DATA_DIR = Path("datasets/burgers_custom_pb_ood")
NU = 0.01
NX, NT = 256, 101
# 与 demo_burgers_dataset.py 一致：x, t 都是 [0, 1] 上的归一化坐标
DX, DT = 1.0 / (NX - 1), 1.0 / (NT - 1)
# 与训练脚本的 validation_physics_weight 一致
PHYSICS_WEIGHT = 0.1


def load_fields(name: str, rows: slice | None = None) -> np.ndarray:
    """返回 (N, 256, 101) float32。"""
    col = pq.read_table(DATA_DIR / name).column("tensor")
    idx = range(len(col))[rows] if rows is not None else range(len(col))
    return np.stack([np.asarray(col[i].as_py(), dtype=np.float32)[..., 0] for i in idx])


def relative_l2(pred: np.ndarray, true: np.ndarray) -> float:
    """逐样本相对 L2 的平均，与训练脚本的定义一致。"""
    num = np.linalg.norm((pred - true).reshape(len(true), -1), axis=1)
    den = np.maximum(np.linalg.norm(true.reshape(len(true), -1), axis=1), 1e-8)
    return float(np.mean(num / den))


def pde_rms_fd(u: np.ndarray) -> float:
    """有限差分 Burgers 残差 RMS（只取内部点，避开边界单边差分）。"""
    u_t = np.gradient(u, DT, axis=2)
    u_x = np.gradient(u, DX, axis=1)
    u_xx = np.gradient(u_x, DX, axis=1)
    r = (u_t + u * u_x - NU * u_xx)[:, 2:-2, 2:-2]
    return float(np.sqrt(np.mean(r**2)))


def score(pred: np.ndarray, true: np.ndarray) -> dict:
    l2 = relative_l2(pred, true)
    pde = pde_rms_fd(pred)
    return {"l2": l2, "pde": pde, "score": l2 + PHYSICS_WEIGHT * pde}


class PCAForest:
    """u0 -> PCA 系数 的多输出随机森林，预测后再投影回整场。"""

    def __init__(self, n_components: int, **rf_kwargs):
        self.pca = PCA(n_components=n_components, random_state=0)
        self.rf = RandomForestRegressor(random_state=0, n_jobs=8, **rf_kwargs)

    def fit(self, x: np.ndarray, y_fields: np.ndarray):
        coefs = self.pca.fit_transform(y_fields.reshape(len(y_fields), -1))
        # 系数按总方差归一化，使 min_impurity_decrease 成为“相对总方差”的阈值
        self.scale = float(np.sqrt(coefs.var(axis=0).sum()))
        self.rf.fit(x, coefs / self.scale)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        coefs = self.rf.predict(x) * self.scale
        return self.pca.inverse_transform(coefs).reshape(len(x), NX, NT).astype(np.float32)

    def leaf_stats(self) -> tuple[float, float]:
        trees = self.rf.estimators_
        return (float(np.mean([t.get_n_leaves() for t in trees])),
                float(np.mean([t.get_depth() for t in trees])))


# 预剪枝网格：None / 1 / 0.0 是“不剪枝”
PRUNE_GRID = {
    "max_depth": [None, 16, 10, 6],
    "min_samples_leaf": [1, 3, 8, 16],
    "min_impurity_decrease": [0.0, 1e-4, 1e-3, 5e-3],
    "max_features": ["sqrt", 0.3],
}


def grid_search(x_tr, y_tr, x_dev, y_dev, n_components, n_estimators=200, log=print):
    """遍历预剪枝网格，按 dev 上的 L2 + 0.1*PDE 选最优。返回 (最优参数, 全部结果)。"""
    keys = list(PRUNE_GRID)
    results = []
    t0 = time.time()
    for values in itertools.product(*PRUNE_GRID.values()):
        params = dict(zip(keys, values))
        model = PCAForest(n_components, n_estimators=n_estimators, **params).fit(x_tr, y_tr)
        dev = score(model.predict(x_dev), y_dev)
        tr = score(model.predict(x_tr), y_tr)
        leaves, depth = model.leaf_stats()
        results.append({**params, "train_l2": tr["l2"], "dev_l2": dev["l2"],
                        "dev_pde": dev["pde"], "dev_score": dev["score"],
                        "leaves": leaves, "depth": depth})
    log(f"  网格 {len(results)} 组，用时 {time.time() - t0:.0f}s")
    # 按 dev L2 选择：有限差分 PDE 会奖励平滑（真值场自身 FD-PDE≈5.5），不宜作为选择标准
    results.sort(key=lambda r: r["dev_l2"])
    best = {k: results[0][k] for k in keys}
    return best, results
