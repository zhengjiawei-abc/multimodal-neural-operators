"""PB_OOD 实验：test L2 vs PDE RMS 权衡图（排除 v4 失败的运行）。"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

root = Path("experiments/outputs_burgers_dataset")
pts = []
for d in sorted(root.glob("*pb_ood*")):
    f = d / "metrics.json"
    if not f.exists() or "aug_v4" in d.name:
        continue
    m = json.loads(f.read_text(encoding="utf-8"))
    c = m["config"]
    pts.append((d.name, c["d_model"], c["physics_weight"], m["test"]["relative_l2"], m["test"]["pde_rms"]))

colors = {48: "#9aa5b1", 56: "#7b8794", 64: "#2f6fbd", 80: "#2a9d8f", 96: "#e07a1f", 128: "#b23a48"}
fig, ax = plt.subplots(figsize=(9, 6))
for name, d, pw, l2, pde in pts:
    hi = "tradeoff" in name
    ax.scatter(l2, pde, s=140 if hi else 60, c=colors.get(d, "gray"),
               edgecolors="black" if hi else "none", linewidths=1.5, zorder=3)
    if hi or l2 < 0.33 or pde < 1.5:
        ax.annotate(f"{name.replace('pb_ood_', '').replace('custom_v14_', '')}\n(d{d}, pw={pw})",
                    (l2, pde), fontsize=7, xytext=(5, 4), textcoords="offset points")
for d, col in colors.items():
    ax.scatter([], [], c=col, label=f"d_model={d}")
ax.set_yscale("log")
ax.set_xlabel("Test relative L2 (越小越好)")
ax.set_ylabel("Test PDE RMS (log，越小越好)")
ax.set_title("PB_OOD：预测精度与物理一致性的权衡（黑框 = 折中实验）")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
out = Path("experiments/pb_ood_tradeoff.png")
fig.savefig(out, dpi=150)
print("saved", out)
