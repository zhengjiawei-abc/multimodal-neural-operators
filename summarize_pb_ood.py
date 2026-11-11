"""汇总所有 PB_OOD 相关实验的 metrics.json。"""
import json
from pathlib import Path

root = Path("experiments/outputs_burgers_dataset")
rows = []
for d in sorted(root.glob("*pb_ood*")):
    f = d / "metrics.json"
    if not f.exists():
        rows.append((d.name, None))
        continue
    m = json.loads(f.read_text(encoding="utf-8"))
    c = m.get("config", {})
    rows.append((d.name, {
        "d": c.get("d_model"), "l": c.get("n_layers"),
        "pw": c.get("physics_weight"), "drop": c.get("dropout"),
        "wd": c.get("weight_decay"), "seed": c.get("seed"),
        "params": m.get("parameter_count"),
        "train": m["train"]["relative_l2"], "test": m["test"]["relative_l2"],
        "pde": m["test"]["pde_rms"],
        "steps": m.get("stability", {}).get("completed_steps"),
        "best": m.get("stability", {}).get("best_dev_step"),
    }))

hdr = f"{'experiment':48} {'d':>4} {'l':>2} {'pw':>5} {'drop':>5} {'seed':>5} {'train':>7} {'test':>7} {'gap%':>7} {'PDE':>8} {'steps':>6} {'best':>6}"
print(hdr)
print("-" * len(hdr))
for name, r in rows:
    if r is None:
        print(f"{name:48} (no metrics.json)")
        continue
    gap = (r["test"] - r["train"]) / r["train"] * 100
    fmt = lambda v, s: s.format(v) if v is not None else "-"
    print(f"{name:48} {fmt(r['d'],'{:>4}')} {fmt(r['l'],'{:>2}')} {fmt(r['pw'],'{:>5.2f}')} "
          f"{fmt(r['drop'],'{:>5.2f}')} {fmt(r['seed'],'{:>5}')} {r['train']:>7.4f} {r['test']:>7.4f} "
          f"{gap:>+7.1f} {r['pde']:>8.3f} {fmt(r['steps'],'{:>6.0f}')} {fmt(r['best'],'{:>6.0f}')}")
