"""Generate the architecture diagram for the current v_aug3x_xl_d128_l4 run."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "assets" / "architecture_clean_zh.png"

INK = "#17202a"
MUTED = "#5f6b76"
ARROW = "#43505c"
BLUE = "#2f6384"
BLUE_FILL = "#edf5f9"
GREEN = "#2f7662"
GREEN_FILL = "#edf7f3"
ORANGE = "#a4672e"
ORANGE_FILL = "#fbf3e8"
RED = "#a34f45"
RED_FILL = "#fbefed"


def setup_font() -> str:
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
    plt.rcParams.update({"font.family": family, "font.size": 9})
    return family


def box(
    axis,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    color: str,
    fill: str,
    family: str,
    fontsize: float = 9.0,
    bold: bool = False,
) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.07",
            facecolor=fill,
            edgecolor=color,
            linewidth=1.25,
        )
    )
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        color=INK,
        family=family,
        fontsize=fontsize * 1.22,
        fontweight="bold" if bold else "normal",
        linespacing=1.24,
    )


def arrow(
    axis,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = ARROW,
    linestyle: str = "-",
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.25,
            linestyle=linestyle,
            color=color,
            connectionstyle="arc3",
            shrinkA=2,
            shrinkB=2,
        )
    )


def lane_label(axis, x: float, y: float, text: str, color: str, family: str) -> None:
    axis.text(
        x,
        y,
        text,
        ha="left",
        va="center",
        color=color,
        family=family,
        fontsize=10.0,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.20,rounding_size=0.10",
            "facecolor": "white",
            "edgecolor": color,
            "linewidth": 0.8,
        },
    )


def main() -> None:
    family = setup_font()
    figure, axis = plt.subplots(figsize=(13.0, 9.0))
    axis.set_xlim(0, 16.0)
    axis.set_ylim(0, 8.8)
    axis.axis("off")
    figure.patch.set_facecolor("white")

    axis.text(
        0.25,
        8.52,
        "当前实验架构：Q-MMNO（v_aug3x_xl_d128_l4）",
        ha="left",
        va="top",
        family=family,
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    axis.text(
        15.72,
        8.52,
        "874,209 参数 · d=128 · L=4 · H=4",
        ha="right",
        va="top",
        family=family,
        fontsize=10.0,
        color=MUTED,
    )

    lane_label(axis, 0.30, 7.65, "场编码", BLUE, family)
    box(axis, 0.30, 6.48, 2.05, 0.92, "密集初始场\n$u_0(x)$（256 点）", BLUE, BLUE_FILL, family, bold=True)
    box(axis, 2.75, 6.48, 3.05, 0.92, "CNN tokenizer\n2×Conv1d + Pool + Linear", BLUE, BLUE_FILL, family, 8.6)
    box(
        axis,
        6.20,
        6.48,
        2.05,
        0.92,
        "32 个场 token\n宽度 128\n位置嵌入 + $z_0$ 投影",
        BLUE,
        BLUE_FILL,
        family,
        7.9,
    )
    box(axis, 11.10, 6.30, 4.35, 1.22, "Transformer 上下文编码器\n4 层 · 4 个注意力头 · 宽度 128", BLUE, BLUE_FILL, family, 9.6, bold=True)
    arrow(axis, (2.35, 6.94), (2.75, 6.94))
    arrow(axis, (5.80, 6.94), (6.20, 6.94))
    arrow(axis, (8.25, 6.94), (11.10, 6.94))

    lane_label(axis, 6.30, 6.22, "传感器模态", ORANGE, family)
    box(axis, 7.55, 5.05, 2.10, 0.92, "8 个传感器\n$(x_i,0.5,u_i)$", ORANGE, ORANGE_FILL, family, 8.7)
    box(axis, 9.95, 5.05, 1.95, 0.92, "两层 MLP\n+ 类型嵌入", ORANGE, ORANGE_FILL, family, 8.7)
    box(axis, 12.20, 5.05, 1.80, 0.92, "8 个\n传感器 token", ORANGE, ORANGE_FILL, family, 8.7)
    arrow(axis, (9.65, 5.51), (9.95, 5.51), color=ORANGE)
    arrow(axis, (11.90, 5.51), (12.20, 5.51), color=ORANGE)
    arrow(axis, (13.10, 5.97), (13.10, 6.30), color=ORANGE)

    lane_label(axis, 0.30, 4.72, "流形表示", GREEN, family)
    arrow(axis, (1.32, 6.48), (1.32, 4.88), color=GREEN)
    box(axis, 0.30, 3.55, 2.55, 1.02, "MLP 流形编码器\n256 → 256 → 32", GREEN, GREEN_FILL, family, 8.7)
    box(axis, 3.15, 3.55, 1.75, 1.02, "潜状态 $z_0$\n32 维", GREEN, GREEN_FILL, family, 8.8)
    arrow(axis, (2.85, 4.06), (3.15, 4.06), color=GREEN)
    arrow(axis, (4.05, 3.55), (4.05, 2.50), color=GREEN)
    arrow(axis, (4.05, 2.50), (5.80, 2.50), color=GREEN)

    lane_label(axis, 5.55, 3.18, "查询解码", RED, family)
    box(
        axis,
        5.80,
        1.92,
        2.95,
        1.06,
        "查询特征\n$[\\gamma(x,t), z_0, u_0^{local}(x)]$",
        RED,
        RED_FILL,
        family,
        8.1,
    )
    box(axis, 9.15, 1.92, 1.45, 1.06, "查询 MLP\n81 → 128 → 128", RED, RED_FILL, family, 8.2)
    box(axis, 10.95, 1.92, 1.95, 1.06, "4 头交叉注意力\n读取 40 个上下文 token", RED, RED_FILL, family, 8.4, bold=True)
    box(axis, 13.25, 1.92, 1.30, 1.06, "前馈网络\n+ 输出头", RED, RED_FILL, family, 8.2)
    box(axis, 14.90, 1.92, 0.80, 1.06, "预测\n$\\hat{u}(x,t)$", RED, RED_FILL, family, 7.8, bold=True)
    arrow(axis, (8.75, 2.45), (9.15, 2.45), color=RED)
    arrow(axis, (10.60, 2.45), (10.95, 2.45), color=RED)
    arrow(axis, (12.90, 2.45), (13.25, 2.45), color=RED)
    arrow(axis, (14.55, 2.45), (14.90, 2.45), color=RED)

    box(
        axis,
        5.80,
        0.60,
        2.95,
        0.82,
        "查询坐标 $(x,t)$\n初值 1 维 + 局部 CNN 特征 32 维",
        RED,
        RED_FILL,
        family,
        8.0,
    )
    arrow(axis, (7.27, 1.42), (7.27, 1.92), color=RED)
    arrow(axis, (11.92, 6.30), (11.92, 2.98), color=BLUE, linestyle="--")
    axis.text(
        12.05,
        4.60,
        "40 个上下文 token",
        ha="left",
        va="center",
        family=family,
        fontsize=9.8,
        color=BLUE,
    )

    axis.plot([0.30, 15.70], [0.36, 0.36], color="#c8d0d8", linewidth=0.9)
    axis.text(
        0.30,
        0.10,
        "训练损失：数据拟合 + 初始条件 + 周期边界 + $\\lambda_{\\mathrm{PDE}}=0.1$ PDE 残差 + 流形重构",
        ha="left",
        va="bottom",
        family=family,
        fontsize=9.8,
        color=MUTED,
    )
    axis.text(
        15.70,
        0.10,
        "当前配置：无输入高通支路 · 无潜空间 ODE · 无流形 CNN 支路",
        ha="right",
        va="bottom",
        family=family,
        fontsize=9.4,
        color=MUTED,
    )

    figure.tight_layout(pad=0.35)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()

