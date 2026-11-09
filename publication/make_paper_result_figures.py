"""Generate evidence-consistent result figures for the bilingual paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"

SEEDS = ["11", "7", "23"]
BASE_L2 = np.array([0.097000325, 0.102856729, 0.110596523])
SHIFT_L2 = np.array([0.090097056, 0.095845718, 0.105621935])
BASE_PDE = np.array([0.838240, 1.354726, 1.524887])
SHIFT_PDE = np.array([0.855308, 1.738279, 1.471190])

SENSOR_L2 = [0.097000325, 0.115318924]
SENSOR_PDE = [0.838240, 1.038709]

INK = "#202124"
MUTED = "#676d73"
GRID = "#d9dde1"
BASE = "#8b4a4a"
SHIFT = "#365f7d"
ACCENT = "#b36b39"


def setup_fonts(language: str) -> None:
    candidates = [
        Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ]
    family = "DejaVu Sans"
    for path in candidates:
        if not path.exists():
            continue
        try:
            font_manager.fontManager.addfont(str(path))
            family = font_manager.FontProperties(fname=str(path)).get_name()
            break
        except Exception:
            continue
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "axes.edgecolor": "#858b90",
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.unicode_minus": False,
        }
    )


def clean_axes(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_axisbelow(True)


def labels(language: str) -> dict[str, str]:
    if language == "zh":
        return {
            "baseline": "原始训练集",
            "shifted": "周期平移增强",
            "test_l2": "测试相对 L2",
            "pde": "PDE RMS",
            "seed": "随机种子",
            "mean": "平均",
            "field": "纯场模型",
            "sensor": "八传感器",
            "sensor_note": "单 seed 初步结果",
            "paired_title": "三个配对种子的周期平移增强效果",
            "tradeoff_title": "预测精度改善，但 PDE 残差没有一致改善",
            "sensor_title": "直接拼接传感器 token 的初步负结果",
            "relative_improvement": "相对改善",
            "parameter_count": "参数量",
        }
    return {
        "baseline": "Original training set",
        "shifted": "Periodic-shift augmented",
        "test_l2": "Test relative L2",
        "pde": "PDE RMS",
        "seed": "Seed",
        "mean": "Mean",
        "field": "Field only",
        "sensor": "Eight sensors",
        "sensor_note": "Single-seed preliminary result",
        "paired_title": "Paired periodic-shift effect across three seeds",
        "tradeoff_title": "Accuracy improves while PDE residual does not",
        "sensor_title": "Direct sensor-token fusion is preliminary negative",
        "relative_improvement": "Relative change",
        "parameter_count": "Parameters",
    }


def paired_l2(language: str) -> None:
    setup_fonts(language)
    text = labels(language)
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(SEEDS))
    width = 0.34
    axis.bar(
        x - width / 2,
        BASE_L2,
        width,
        label=f"{text['baseline']} ({BASE_L2.mean():.4f})",
        color=BASE,
        edgecolor="white",
        linewidth=1.0,
    )
    axis.bar(
        x + width / 2,
        SHIFT_L2,
        width,
        label=f"{text['shifted']} ({SHIFT_L2.mean():.4f})",
        color=SHIFT,
        edgecolor="white",
        linewidth=1.0,
    )
    for index, (before, after) in enumerate(zip(BASE_L2, SHIFT_L2)):
        change = (after - before) / before * 100
        axis.text(
            index,
            max(before, after) + 0.0025,
            f"{change:.2f}%",
            ha="center",
            va="bottom",
            color=SHIFT,
            fontsize=9.5,
            fontweight="bold",
        )
    axis.axhline(
        BASE_L2.mean(),
        color=BASE,
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
    )
    axis.axhline(
        SHIFT_L2.mean(),
        color=SHIFT,
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
    )
    axis.set_xticks(x, SEEDS)
    axis.set_xlabel(text["seed"])
    axis.set_ylabel(text["test_l2"])
    axis.set_title(text["paired_title"], loc="left", fontweight="bold")
    axis.set_ylim(0, 0.122)
    axis.legend(frameon=False, loc="upper left", ncol=2)
    clean_axes(axis)
    figure.tight_layout()
    figure.savefig(
        ASSETS / f"figure_paired_l2_{language}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def tradeoff(language: str) -> None:
    setup_fonts(language)
    text = labels(language)
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    for axis, base, shift, ylabel in [
        (axes[0], BASE_L2, SHIFT_L2, text["test_l2"]),
        (axes[1], BASE_PDE, SHIFT_PDE, text["pde"]),
    ]:
        means = [base.mean(), shift.mean()]
        bars = axis.bar(
            [0, 1],
            means,
            width=0.55,
            color=[BASE, SHIFT],
            edgecolor="white",
            linewidth=1.0,
        )
        jitter = np.linspace(-0.11, 0.11, len(base))
        axis.scatter(
            jitter,
            base,
            color=BASE,
            edgecolor="white",
            zorder=4,
            s=32,
        )
        axis.scatter(
            1 + jitter,
            shift,
            color=SHIFT,
            edgecolor="white",
            zorder=4,
            s=32,
        )
        for bar, value in zip(bars, means):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )
        axis.set_xticks(
            [0, 1],
            [text["baseline"], text["shifted"]],
        )
        axis.set_ylabel(ylabel)
        clean_axes(axis)
    change = (SHIFT_PDE.mean() - BASE_PDE.mean()) / BASE_PDE.mean() * 100
    axes[1].text(
        0.5,
        max(SHIFT_PDE.mean(), BASE_PDE.mean()) * 1.08,
        f"+{change:.1f}%",
        ha="center",
        va="bottom",
        color=ACCENT,
        fontsize=11,
        fontweight="bold",
    )
    figure.suptitle(text["tradeoff_title"], x=0.02, ha="left", fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(
        ASSETS / f"figure_l2_pde_tradeoff_{language}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def sensor_negative(language: str) -> None:
    setup_fonts(language)
    text = labels(language)
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.3))
    for axis, values, ylabel in [
        (axes[0], SENSOR_L2, text["test_l2"]),
        (axes[1], SENSOR_PDE, text["pde"]),
    ]:
        bars = axis.bar(
            [0, 1],
            values,
            width=0.55,
            color=[SHIFT, ACCENT],
            edgecolor="white",
            linewidth=1.0,
        )
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )
        change = (values[1] - values[0]) / values[0] * 100
        axis.text(
            0.5,
            max(values) * 1.08,
            f"+{change:.1f}%",
            ha="center",
            va="bottom",
            color=ACCENT,
            fontsize=11,
            fontweight="bold",
        )
        axis.set_xticks([0, 1], [text["field"], text["sensor"]])
        axis.set_ylabel(ylabel)
        clean_axes(axis)
    figure.suptitle(text["sensor_title"], x=0.02, ha="left", fontweight="bold")
    figure.text(
        0.98,
        0.02,
        text["sensor_note"],
        ha="right",
        va="bottom",
        color=MUTED,
        fontsize=8.8,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    figure.savefig(
        ASSETS / f"figure_sensor_negative_{language}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for language in ("zh", "en"):
        paired_l2(language)
        tradeoff(language)
        sensor_negative(language)


if __name__ == "__main__":
    main()

