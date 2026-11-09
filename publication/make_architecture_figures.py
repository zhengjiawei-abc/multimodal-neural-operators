"""Create English and Chinese architecture diagrams for the publication."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"


def font_family(language: str) -> str:
    if language == "zh":
        candidates = [
            "Microsoft YaHei",
            "Noto Sans SC",
            "Source Han Sans CN",
            "SimHei",
        ]
        available = {font.name for font in font_manager.fontManager.ttflist}
        for candidate in candidates:
            if candidate in available:
                return candidate
    return "DejaVu Sans"


def draw_box(
    axis,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    edge: str,
    face: str,
    fontsize: float,
    family: str,
    bold: bool = False,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.5,
        edgecolor=edge,
        facecolor=face,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        family=family,
        color="#172333",
        fontweight="bold" if bold else "normal",
        linespacing=1.25,
    )


def draw_arrow(
    axis,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#5f6f7d",
    connection: str = "arc3",
    rad: float = 0.0,
    linewidth: float = 1.5,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=linewidth,
        color=color,
        connectionstyle=f"{connection},rad={rad}",
        shrinkA=2,
        shrinkB=2,
    )
    axis.add_patch(arrow)


def make_figure(language: str) -> Path:
    labels = {
        "en": {
            "dense": "Dense initial field\n$u_0(x)$",
            "tokenizer": "High-pass filter +\nCNN tokenizer",
            "tokens": "32 field tokens\nwidth 64",
            "context": "Transformer context encoder\n3 layers - 4 heads",
            "manifold": "Convolutional\nmanifold encoder",
            "latent": "Latent state\n$z_0$ (32-d)",
            "ode": "Latent ODE\n$v_\\theta$ - 4 Euler steps",
            "evolved": "Evolved state\n$z(t)$",
            "sensor": "Sensor triples\n$(x_i,t_s,u_i)$",
            "sensor_mlp": "MLP + type\nembedding",
            "sensor_token": "Sensor tokens",
            "concat": "concatenate",
            "query": "Query features\n$[\\gamma(x,t),z(t),u_0^{local}]$",
            "query_mlp": "Query MLP",
            "attention": "Cross-attention\nread context",
            "head": "Feed-forward\n+ output head",
            "output": "Prediction\n$\\hat{u}(x,t)$",
            "loss": "Loss: data + initial condition + periodic boundary + optional PDE residual",
            "main": "Main runs: $\\lambda_{\\mathrm{PDE}}=0$",
            "lane_a": "FIELD CONTEXT",
            "lane_b": "TEMPORAL LATENT",
            "lane_c": "SENSOR MODALITY",
            "lane_d": "QUERY DECODER",
        },
        "zh": {
            "dense": "密集初始场\n$u_0(x)$",
            "tokenizer": "高通滤波 +\nCNN tokenizer",
            "tokens": "32 个场 token\n宽度 64",
            "context": "Transformer 上下文编码器\n3 层 · 4 个注意力头",
            "manifold": "卷积流形\n编码器",
            "latent": "潜状态\n$z_0$（32 维）",
            "ode": "潜空间 ODE\n$v_\\theta$ · 4 步 Euler",
            "evolved": "演化状态\n$z(t)$",
            "sensor": "传感器三元组\n$(x_i,t_s,u_i)$",
            "sensor_mlp": "MLP + 类型\n嵌入",
            "sensor_token": "传感器 token",
            "concat": "拼接",
            "query": "查询特征\n$[\\gamma(x,t),z(t),u_0^{local}]$",
            "query_mlp": "查询 MLP",
            "attention": "交叉注意力\n读取上下文",
            "head": "前馈网络 +\n输出头",
            "output": "预测\n$\\hat{u}(x,t)$",
            "loss": "损失：数据 + 初值 + 周期边界 + 可选 PDE 残差",
            "main": "主实验：$\\lambda_{\\mathrm{PDE}}=0$",
            "lane_a": "场上下文",
            "lane_b": "时间潜变量",
            "lane_c": "传感器模态",
            "lane_d": "查询解码",
        },
    }[language]

    family = font_family(language)
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    figure, axis = plt.subplots(figsize=(14.2, 7.2))
    axis.set_xlim(0, 14.2)
    axis.set_ylim(0, 7.2)
    axis.axis("off")
    figure.patch.set_facecolor("white")

    field_color = "#2f5f87"
    latent_color = "#2d7a6d"
    sensor_color = "#a96b18"
    query_color = "#b65b35"
    ink = "#263241"

    axis.text(
        0.35,
        6.78,
        labels["lane_a"],
        fontsize=10.5,
        family=family,
        color=field_color,
        fontweight="bold",
        va="top",
    )
    draw_box(
        axis,
        0.35,
        5.45,
        1.8,
        0.88,
        labels["dense"],
        edge=field_color,
        face="#eef4f8",
        fontsize=12,
        family=family,
    )
    draw_box(
        axis,
        2.55,
        5.45,
        2.05,
        0.88,
        labels["tokenizer"],
        edge=field_color,
        face="#eef4f8",
        fontsize=12,
        family=family,
    )
    draw_box(
        axis,
        5.0,
        5.45,
        1.9,
        0.88,
        labels["tokens"],
        edge=field_color,
        face="#eef4f8",
        fontsize=12,
        family=family,
    )
    draw_box(
        axis,
        9.25,
        5.3,
        3.95,
        1.12,
        labels["context"],
        edge=field_color,
        face="#e7f0f6",
        fontsize=13,
        family=family,
        bold=True,
    )
    draw_arrow(axis, (2.15, 5.89), (2.55, 5.89), color=field_color)
    draw_arrow(axis, (4.60, 5.89), (5.0, 5.89), color=field_color)
    draw_arrow(axis, (6.90, 5.89), (9.25, 5.86), color=field_color)

    axis.text(
        0.35,
        4.93,
        labels["lane_b"],
        fontsize=10.5,
        family=family,
        color=latent_color,
        fontweight="bold",
        va="top",
    )
    draw_box(
        axis,
        0.35,
        3.55,
        1.8,
        0.88,
        labels["manifold"],
        edge=latent_color,
        face="#edf6f3",
        fontsize=12,
        family=family,
    )
    draw_box(
        axis,
        2.55,
        3.55,
        1.55,
        0.88,
        labels["latent"],
        edge=latent_color,
        face="#edf6f3",
        fontsize=12,
        family=family,
    )
    draw_box(
        axis,
        4.5,
        3.55,
        2.3,
        0.88,
        labels["ode"],
        edge=latent_color,
        face="#edf6f3",
        fontsize=12,
        family=family,
    )
    draw_box(
        axis,
        7.2,
        3.55,
        1.35,
        0.88,
        labels["evolved"],
        edge=latent_color,
        face="#edf6f3",
        fontsize=12,
        family=family,
    )
    draw_arrow(axis, (2.15, 3.99), (2.55, 3.99), color=latent_color)
    draw_arrow(axis, (4.10, 3.99), (4.50, 3.99), color=latent_color)
    draw_arrow(axis, (6.80, 3.99), (7.20, 3.99), color=latent_color)
    draw_arrow(axis, (1.25, 5.45), (1.25, 4.43), color=latent_color, rad=0.0)
    draw_arrow(axis, (7.88, 4.43), (10.1, 5.30), color=latent_color, rad=-0.08)

    axis.text(
        9.25,
        4.87,
        labels["lane_c"],
        fontsize=10.5,
        family=family,
        color=sensor_color,
        fontweight="bold",
        va="top",
    )
    draw_box(
        axis,
        9.25,
        3.55,
        1.6,
        0.88,
        labels["sensor"],
        edge=sensor_color,
        face="#fbf3e7",
        fontsize=11.5,
        family=family,
    )
    draw_box(
        axis,
        11.2,
        3.55,
        1.45,
        0.88,
        labels["sensor_mlp"],
        edge=sensor_color,
        face="#fbf3e7",
        fontsize=11.5,
        family=family,
    )
    draw_box(
        axis,
        12.95,
        3.55,
        1.05,
        0.88,
        labels["sensor_token"],
        edge=sensor_color,
        face="#fbf3e7",
        fontsize=10.5,
        family=family,
    )
    draw_arrow(axis, (10.85, 3.99), (11.2, 3.99), color=sensor_color)
    draw_arrow(axis, (12.65, 3.99), (12.95, 3.99), color=sensor_color)
    draw_arrow(axis, (13.45, 4.43), (12.65, 5.3), color=sensor_color, rad=0.12)
    axis.text(
        12.1,
        4.75,
        labels["concat"],
        fontsize=9.5,
        family=family,
        color=sensor_color,
        ha="center",
        va="bottom",
    )

    axis.text(
        0.35,
        2.86,
        labels["lane_d"],
        fontsize=10.5,
        family=family,
        color=query_color,
        fontweight="bold",
        va="top",
    )
    draw_box(
        axis,
        0.35,
        1.52,
        2.45,
        0.82,
        labels["query"],
        edge=query_color,
        face="#faf0eb",
        fontsize=11.3,
        family=family,
    )
    draw_box(
        axis,
        3.15,
        1.52,
        1.35,
        0.82,
        labels["query_mlp"],
        edge=query_color,
        face="#faf0eb",
        fontsize=11.5,
        family=family,
    )
    draw_box(
        axis,
        7.25,
        1.38,
        2.3,
        1.12,
        labels["attention"],
        edge=query_color,
        face="#f6e7df",
        fontsize=12.2,
        family=family,
        bold=True,
    )
    draw_box(
        axis,
        10.0,
        1.52,
        1.55,
        0.82,
        labels["head"],
        edge=query_color,
        face="#faf0eb",
        fontsize=11.5,
        family=family,
    )
    draw_box(
        axis,
        11.9,
        1.52,
        1.2,
        0.82,
        labels["output"],
        edge=query_color,
        face="#faf0eb",
        fontsize=11.5,
        family=family,
        bold=True,
    )
    draw_arrow(axis, (2.80, 1.93), (3.15, 1.93), color=query_color)
    draw_arrow(axis, (4.50, 1.93), (7.25, 1.93), color=query_color)
    draw_arrow(axis, (7.87, 3.55), (8.05, 2.50), color=latent_color, rad=-0.08)
    draw_arrow(axis, (11.25, 5.30), (8.85, 2.50), color=field_color, rad=0.08)
    draw_arrow(axis, (9.55, 1.93), (10.0, 1.93), color=query_color)
    draw_arrow(axis, (11.55, 1.93), (11.9, 1.93), color=query_color)

    loss_box = Rectangle(
        (0.35, 0.28),
        12.75,
        0.58,
        linewidth=1.1,
        edgecolor="#8b7d6b",
        facecolor="#f8f6f1",
    )
    axis.add_patch(loss_box)
    axis.text(
        0.55,
        0.76,
        labels["loss"],
        fontsize=11.5,
        family=family,
        color=ink,
        va="center",
    )
    axis.text(
        12.95,
        0.57,
        labels["main"],
        fontsize=10.5,
        family=family,
        color=query_color,
        ha="right",
        va="center",
        fontweight="bold",
    )

    for side in axis.spines.values():
        side.set_visible(False)

    figure.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    ASSETS.mkdir(parents=True, exist_ok=True)
    output = ASSETS / f"architecture_{language}.png"
    figure.savefig(output, dpi=260, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    for language in ("en", "zh"):
        print(make_figure(language))


if __name__ == "__main__":
    main()

