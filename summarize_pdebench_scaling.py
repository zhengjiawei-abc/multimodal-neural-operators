"""汇总 PDEBench 样本量扩展实验：FNO 与混合 PINN 都在全部 250 个 test 样本上比较。

- FNO：直接读 metrics.json 的 test_relative_l2（训练时已用 --test-samples 250）。
- 混合 PINN：加载 model.pt，用 demo_burgers_resnet.evaluate_split 在全部 250 个样本上评估。
  512 档用 lowdata/opt5_n512_drop10_steps24k_seed*，1024 / 2375 档用 pdebench_scaling/hybrid_*。
"""
import json
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, "code")
import demo_burgers_resnet as nb  # noqa: E402
import rf_common  # noqa: E402

rf_common.DATA_DIR = Path("datasets/pdebench_burgers")
ROOT = Path("experiments/outputs_burgers_dataset")
SCALING = ROOT / "pdebench_scaling"
N_TEST = 250

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test = torch.from_numpy(rf_common.load_fields("burgers_nu0.01_test.parquet")[..., None]).to(device)
assert test.size(0) == N_TEST, test.shape


def hybrid_l2(model_path: Path) -> float:
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = nb.DatasetConfig(**ckpt["config"])
    model = nb.BurgersHybridPINN(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    l2 = nb.evaluate_split(model, test, N_TEST, cfg, device)["relative_l2"]
    del model
    torch.cuda.empty_cache()
    return float(l2)


results = {"fno": {}, "hybrid": {}}
for n in (512, 1024, 2375):
    results["fno"][n] = {int(p.parent.name.split("_s")[-1]):
                         json.loads(p.read_text(encoding="utf-8"))["test_relative_l2"]
                         for p in sorted(SCALING.glob(f"fno_n{n}_s*/metrics.json"))}
    paths = (sorted(ROOT.glob("lowdata/opt5_n512_drop10_steps24k_seed*/model.pt")) if n == 512
             else sorted(SCALING.glob(f"hybrid_n{n}_s*/model.pt")))
    results["hybrid"][n] = {}
    for p in paths:
        seed = int(p.parent.name.split("seed")[-1] if "seed" in p.parent.name else p.parent.name.split("_s")[-1])
        results["hybrid"][n][seed] = hybrid_l2(p)
        print(f"hybrid n={n} seed={seed}: {results['hybrid'][n][seed]:.4f}", flush=True)


def fmt(vals):
    v = list(vals)
    if not v:
        return "—"
    return f"{st.mean(v):.4f} ± {st.stdev(v):.4f} (n={len(v)})" if len(v) > 1 else f"{v[0]:.4f} (n=1)"


from scipy.stats import ttest_ind  # noqa: E402

print(f"\n全部 {N_TEST} 个 test 样本的相对 L2；p 值为 Welch t 检验（双侧）")
print(f"{'训练样本':>8} | {'FNO':>24} | {'混合 PINN':>24} | {'混合低多少':>8} | p")
tests = {}
for n in (512, 1024, 2375):
    f, h = list(results["fno"][n].values()), list(results["hybrid"][n].values())
    gap = f"{(1 - st.mean(h) / st.mean(f)) * 100:+.1f}%" if f and h else "—"
    p = float(ttest_ind(h, f, equal_var=False).pvalue) if len(f) > 1 and len(h) > 1 else float("nan")
    tests[str(n)] = {"p_welch": p, "hybrid_minus_fno": st.mean(h) - st.mean(f) if f and h else None}
    print(f"{n:>8} | {fmt(f):>24} | {fmt(h):>24} | {gap:>8} | {p:.3f}")
results["welch"] = tests

out = Path("experiments/pdebench_scaling_summary.json")
out.write_text(json.dumps({k: {str(n): v for n, v in d.items()} for k, d in results.items()}, indent=2),
               encoding="utf-8")
print("saved", out)
