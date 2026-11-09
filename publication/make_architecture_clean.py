"""Generate a clean architecture diagram for the Q-MMNO paper."""

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
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 9,
            "axes.linewidth": 0,
        }
    )
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
    fontsize: float = 9.2,
    bold: bool = False,
) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.08",
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
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        linespacing=1.25,
    )


def arrow(
    axis,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    connection: str = "arc3",
    color: str = ARROW,
    linewidth: float = 1.25,
    linestyle: str = "-",
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            connectionstyle=connection,
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
        fontsize=8.2,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.22,rounding_size=0.12",
            "facecolor": "white",
            "edgecolor": color,
            "linewidth": 0.8,
        },
    )


def main() -> None:
    family = setup_font()
    figure, axis = plt.subplots(figsize=(15.5, 8.8))
    axis.set_xlim(0, 15.8)
    axis.set_ylim(0, 9.0)
    axis.axis("off")
    figure.patch.set_facecolor("white")

    axis.text(
        0.25,
        8.72,
        "查询式多模态神经算子（Q-MMNO）",
        ha="left",
        va="top",
        family=family,
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    axis.text(
        15.55,
        8.72,
        "四条信息路径 · 无交叉连接",
        ha="right",
        va="top",
        family=family,
        fontsize=8.8,
        color=MUTED,
    )

    lane_label(axis, 0.30, 7.78, "场编码", BLUE, family)
    box(
        axis,
        0.30,
        6.55,
        2.05,
        0.92,
        "密集初始场\n$u_0(x)$",
        BLUE,
        BLUE_FILL,
        family,
        bold=True,
    )
    box(
        axis,
        2.75,
        6.55,
        3.25,
        0.92,
        "CNN tokenizer\n（高通滤波可选）",
        BLUE,
        BLUE_FILL,
        family,
    )
    box(
        axis,
        6.40,
        6.55,
        2.55,
        0.92,
        "32 个场 token\n宽度 $d$",
        BLUE,
        BLUE_FILL,
        family,
    )
    box(
        axis,
        11.05,
        6.40,
        4.25,
        1.22,
        "Transformer 上下文编码器\n$L$ 层 · $H$ 个注意力头",
        BLUE,
        BLUE_FILL,
        family,
        10.2,
        bold=True,
    )
    arrow(axis, (2.35, 7.01), (2.75, 7.01))
    arrow(axis, (6.00, 7.01), (6.40, 7.01))
    arrow(axis, (8.95, 7.01), (11.05, 7.01))

    lane_label(axis, 6.55, 6.22, "传感器模态", ORANGE, family)
    box(
        axis,
        7.65,
        5.10,
        2.25,
        0.92,
        "传感器三元组\n$(x_i,t_s,u_i)$",
        ORANGE,
        ORANGE_FILL,
        family,
    )
    box(
        axis,
        10.25,
        5.10,
        2.15,
        0.92,
        "MLP + 类型嵌入",
        ORANGE,
        ORANGE_FILL,
        family,
    )
    box(
        axis,
        12.75,
        5.10,
        2.15,
        0.92,
        "传感器 token",
        ORANGE,
        ORANGE_FILL,
        family,
    )
    arrow(axis, (9.90, 5.56), (10.25, 5.56), color=ORANGE)
    arrow(axis, (12.40, 5.56), (12.75, 5.56), color=ORANGE)
    arrow(
        axis,
        (13.82, 6.02),
        (13.82, 6.40),
        connection="arc3",
        color=ORANGE,
    )

    lane_label(axis, 0.30, 4.92, "潜变量路径", GREEN, family)
    arrow(axis, (1.32, 6.55), (1.32, 4.92), color=GREEN)
    box(
        axis,
        0.30,
        3.65,
        2.45,
        1.02,
        "流形编码器\n（MLP / Conv）",
        GREEN,
        GREEN_FILL,
        family,
    )
    box(
        axis,
        3.10,
        3.65,
        1.65,
        1.02,
        "潜状态\n$z_0$（32 维）",
        GREEN,
        GREEN_FILL,
        family,
    )
    box(
        axis,
        5.10,
        3.65,
        2.05,
        1.02,
        "潜空间 ODE\n4 步 Euler（可选）",
        GREEN,
        GREEN_FILL,
        family,
        8.8,
    )
    box(
        axis,
        7.50,
        3.65,
        1.60,
        1.02,
        "演化状态\n$z(t)$",
        GREEN,
        GREEN_FILL,
        family,
    )
    arrow(axis, (2.75, 4.16), (3.10, 4.16), color=GREEN)
    arrow(axis, (4.75, 4.16), (5.10, 4.16), color=GREEN)
    arrow(axis, (7.15, 4.16), (7.50, 4.16), color=GREEN)

    lane_label(axis, 7.90, 3.47, "查询解码", RED, family)
    box(
        axis,
        8.10,
        2.20,
        2.85,
        0.98,
        "查询特征\n$[\\gamma(x,t), z(t), u_0^{local}(x)]$",
        RED,
        RED_FILL,
        family,
        8.3,
    )
    box(
        axis,
        11.25,
        2.20,
        1.55,
        0.98,
        "查询 MLP",
        RED,
        RED_FILL,
        family,
    )
    box(
        axis,
        11.00,
        0.62,
        2.20,
        1.05,
        "交叉注意力\n读取上下文",
        RED,
        RED_FILL,
        family,
        9.3,
        bold=True,
    )
    box(
        axis,
        13.35,
        0.62,
        1.30,
        1.05,
        "前馈网络\n+ 输出头",
        RED,
        RED_FILL,
        family,
        8.5,
    )
    arrow(axis, (9.10, 4.67), (9.10, 3.18), color=GREEN)
    arrow(axis, (10.95, 2.69), (11.25, 2.69), color=RED)
    arrow(axis, (11.92, 2.20), (11.92, 1.67), color=RED)
    arrow(axis, (13.20, 1.15), (13.35, 1.15), color=RED)
    arrow(axis, (13.08, 6.40), (13.08, 1.76), color=BLUE, linestyle="--")
    arrow(axis, (13.08, 1.76), (12.58, 1.76), color=BLUE, linestyle="--")
    axis.text(
        13.20,
        4.75,
        "上下文",
        ha="left",
        va="center",
        family=family,
        fontsize=8.2,
        color=BLUE,
    )

    box(
        axis,
        14.72,
        0.62,
        0.88,
        1.05,
        "预测\n$\\hat{u}(x,t)$",
        RED,
        RED_FILL,
        family,
        8.0,
        bold=True,
    )
    arrow(axis, (14.65, 1.15), (14.72, 1.15), color=RED)

    axis.plot([0.30, 15.50], [0.25, 0.25], color="#c8d0d8", linewidth=0.9)
    axis.text(
        0.30,
        0.08,
        "损失：数据拟合 + 初始条件 + 周期边界 + 可选 PDE 残差与流形重构",
        ha="left",
        va="bottom",
        family=family,
        fontsize=8.7,
        color=MUTED,
    )
    axis.text(
        15.48,
        0.08,
        "Wide：d=64，L=3，高通/ODE 开，Conv 流形\nXL：d=128，L=4，高通/ODE 关，MLP 流形",
        ha="right",
        va="bottom",
        family=family,
        fontsize=8.3,
        color=MUTED,
    )

    figure.tight_layout(pad=0.35)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()

