"""基线 vs 流态加权 d96/l4，三个 seed（7, 42, 123）的配对比较。"""
import json
from pathlib import Path

import numpy as np

from compare_regime_weighted import evaluate_model

ROOT = Path("experiments/outputs_burgers_dataset")
SEEDS = {123: "", 7: "_s7", 42: "_s42"}
METRICS = ("test_l2_all128", "pde_rms_4096pts", "pde_median_abs_4096pts")

rows = {}
for seed, suffix in SEEDS.items():
    paths = {"baseline": ROOT / f"pb_ood_clean_d96_l4_saved{suffix}" / "model.pt",
             "weighted": ROOT / f"pb_ood_clean_d96_l4_regime_weighted{suffix}" / "model.pt"}
    if not all(p.exists() for p in paths.values()):
        print(f"seed {seed}: 缺少模型，跳过")
        continue
    rows[seed] = {k: evaluate_model(p) for k, p in paths.items()}

print(f"\n{'seed':>5} | {'L2 基线':>8} {'L2 加权':>8} {'差':>8} | {'PDE中位 基线':>12} {'加权':>8} {'差':>8} | {'PDE RMS 基线':>12} {'加权':>8}")
for seed, r in rows.items():
    b, w = r["baseline"], r["weighted"]
    print(f"{seed:>5} | {b['test_l2_all128']:>8.4f} {w['test_l2_all128']:>8.4f} "
          f"{w['test_l2_all128'] - b['test_l2_all128']:>+8.4f} | "
          f"{b['pde_median_abs_4096pts']:>12.3f} {w['pde_median_abs_4096pts']:>8.3f} "
          f"{w['pde_median_abs_4096pts'] - b['pde_median_abs_4096pts']:>+8.3f} | "
          f"{b['pde_rms_4096pts']:>12.2f} {w['pde_rms_4096pts']:>8.2f}")

summary = {}
if len(rows) >= 2:
    print("\n均值 ± 标准差（跨 seed）；配对差 = 加权 - 基线")
    for m in METRICS:
        b = np.array([r["baseline"][m] for r in rows.values()])
        w = np.array([r["weighted"][m] for r in rows.values()])
        d = w - b
        summary[m] = {"baseline_mean": b.mean(), "baseline_std": b.std(ddof=1),
                      "weighted_mean": w.mean(), "weighted_std": w.std(ddof=1),
                      "diff_mean": d.mean(), "diff_std": d.std(ddof=1),
                      "weighted_better_in": int((d < 0).sum()), "n": len(d)}
        s = summary[m]
        print(f"  {m:24} 基线 {s['baseline_mean']:.4f}±{s['baseline_std']:.4f}  "
              f"加权 {s['weighted_mean']:.4f}±{s['weighted_std']:.4f}  "
              f"差 {s['diff_mean']:+.4f}±{s['diff_std']:.4f}  加权更好 {s['weighted_better_in']}/{s['n']}")

Path("experiments/lda_kmeans_pb_ood/regime_seeds_comparison.json").write_text(
    json.dumps({"per_seed": {str(k): v for k, v in rows.items()}, "summary": summary},
               indent=2, default=float), encoding="utf-8")
