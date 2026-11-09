"""Generate figures for the current 128-wide, four-layer XL operator paper."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parent.parent
ASSETS = Path(__file__).resolve().parent / "assets"
XL_METRICS = ROOT / "experiments" / "v_aug3x_xl_d128_l4" / "metrics.json"
PHYSICS_METRICS = ROOT / "experiments" / "v_aug3x_xl_physics15" / "metrics.json"

INK = "#202124"
MUTED = "#666c72"
GRID = "#d9dde1"
BLUE = "#365f7d"
RED = "#8b4a4a"
ORANGE = "#b36b39"
GREEN = "#2f6d5d"
LIGHT_BLUE = "#edf3f7"
LIGHT_RED = "#f7eeee"
LIGHT_ORANGE = "#faf2e9"
LIGHT_GREEN = "#edf5f2"


def setup_fonts() -> str:
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
    return family


def strings(language: str) -> dict[str, str]:
    if language == "zh":
        return {
            "architecture_title": "当前 XL 多模态神经算子结构",
            "field": "初始场\n$u_0(x)$",
            "manifold": "MLP 流形编码器",
            "latent": "潜状态\n$z_0$（32 维）",
            "tokenizer": "CNN tokenizer",
            "field_tokens": "32 个场 token\n宽度 128",
            "sensors": "8 个传感器\n$(x_i,0.5,u_i)$",
            "sensor_mlp": "MLP + 类型嵌入",
            "sensor_tokens": "8 个传感器 token",
            "context": "Transformer 上下文编码器\n4 层 · 4 头 · 宽度 128",
            "query": "查询特征\n$[\\gamma(x,t),z_0,u_0^{local}]$",
            "query_mlp": "查询投影",
            "attention": "交叉注意力\n读取上下文",
            "head": "前馈网络 + 输出头",
            "output": "预测\n$\\hat{u}(x,t)$",
            "loss": "损失：数据 + 初值 + 周期边界 + PDE 残差（$\\lambda=0.1$）",
            "field_lane": "场编码",
            "latent_lane": "流形潜变量",
            "sensor_lane": "传感器模态",
            "query_lane": "查询解码",
            "baseline": "旧版 v11",
            "ours": "当前 XL",
            "test_l2": "测试相对 L2",
            "pde": "PDE RMS",
            "tradeoff_title": "当前模型以更高 L2 换取极低 PDE 残差",
            "tradeoff_note": "两组数据与架构不同，仅作描述性对照",
            "params_title": "XL 模型参数构成",
            "parameters": "参数量",
            "history_title": "XL 训练的验证集收敛曲线",
            "dev_l2": "验证相对 L2",
            "dev_pde": "验证 PDE RMS",
            "step": "训练步数",
            "sensitivity_title": "物理权重与 Dropout 的初步敏感性",
            "main_config": "physics=0.10\ndropout=0.10",
            "physics_config": "physics=0.15\ndropout=0.08",
            "relative_change": "相对变化",
        }
    return {
        "architecture_title": "Current XL multimodal neural operator",
        "field": "Initial field\n$u_0(x)$",
        "manifold": "MLP manifold encoder",
        "latent": "Latent state\n$z_0$ (32-d)",
        "tokenizer": "CNN tokenizer",
        "field_tokens": "32 field tokens\nwidth 128",
        "sensors": "8 sensors\n$(x_i,0.5,u_i)$",
        "sensor_mlp": "MLP + type embedding",
        "sensor_tokens": "8 sensor tokens",
        "context": "Transformer context encoder\n4 layers - 4 heads - width 128",
        "query": "Query features\n$[\\gamma(x,t),z_0,u_0^{local}]$",
        "query_mlp": "Query projection",
        "attention": "Cross-attention\nread context",
        "head": "Feed-forward + output head",
        "output": "Prediction\n$\\hat{u}(x,t)$",
        "loss": "Loss: data + initial + periodic boundary + PDE residual ($\\lambda=0.1$)",
        "field_lane": "FIELD ENCODING",
        "latent_lane": "MANIFOLD LATENT",
        "sensor_lane": "SENSOR MODALITY",
        "query_lane": "QUERY DECODER",
        "baseline": "Legacy v11",
        "ours": "Current XL",
        "test_l2": "Test relative L2",
        "pde": "PDE RMS",
        "tradeoff_title": "The XL model trades higher L2 for a very low PDE residual",
        "tradeoff_note": "Different data and architecture; descriptive comparison only",
        "params_title": "XL parameter composition",
        "parameters": "Parameters",
        "history_title": "Development convergence of the XL model",
        "dev_l2": "Development relative L2",
        "dev_pde": "Development PDE RMS",
        "step": "Training step",
        "sensitivity_title": "Preliminary physics-weight and dropout sensitivity",
        "main_config": "physics=0.10\ndropout=0.10",
        "physics_config": "physics=0.15\ndropout=0.08",
        "relative_change": "Relative change",
    }


def box(
    axis,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    edge: str,
    face: str,
    family: str,
    size: float,
    bold: bool = False,
) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.09",
            linewidth=1.2,
            edgecolor=edge,
            facecolor=face,
        )
    )
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        family=family,
        fontsize=size,
        fontweight="bold" if bold else "normal",
        linespacing=1.25,
    )


def arrow(axis, start, end, color="#66717a", rad=0.0) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2,
            shrinkB=2,
        )
    )


def architecture(language: str, family: str) -> None:
    text = strings(language)
    figure, axis = plt.subplots(figsize=(13.6, 6.8))
    axis.set_xlim(0, 13.6)
    axis.set_ylim(0, 6.8)
    axis.axis("off")
    figure.patch.set_facecolor("white")

    axis.text(0.3, 6.45, text["field_lane"], color=BLUE, fontsize=9.5, fontweight="bold", family=family)
    box(axis, 0.3, 5.25, 1.55, 0.78, text["field"], edge=BLUE, face=LIGHT_BLUE, family=family, size=10.5)
    box(axis, 2.2, 5.25, 1.8, 0.78, text["tokenizer"], edge=BLUE, face=LIGHT_BLUE, family=family, size=10.5)
    box(axis, 4.35, 5.25, 1.7, 0.78, text["field_tokens"], edge=BLUE, face=LIGHT_BLUE, family=family, size=10.5)
    box(axis, 9.25, 5.05, 3.65, 1.08, text["context"], edge=BLUE, face=LIGHT_BLUE, family=family, size=11.5, bold=True)
    arrow(axis, (1.85, 5.64), (2.2, 5.64), BLUE)
    arrow(axis, (4.0, 5.64), (4.35, 5.64), BLUE)
    arrow(axis, (6.05, 5.64), (9.25, 5.58), BLUE)

    axis.text(0.3, 4.66, text["latent_lane"], color=GREEN, fontsize=9.5, fontweight="bold", family=family)
    box(axis, 0.3, 3.5, 1.8, 0.78, text["manifold"], edge=GREEN, face=LIGHT_GREEN, family=family, size=10.5)
    box(axis, 2.45, 3.5, 1.35, 0.78, text["latent"], edge=GREEN, face=LIGHT_GREEN, family=family, size=10.5)
    arrow(axis, (1.15, 5.25), (1.15, 4.28), GREEN)
    arrow(axis, (2.1, 3.89), (2.45, 3.89), GREEN)
    arrow(axis, (3.8, 3.89), (10.2, 5.05), GREEN, rad=-0.08)

    axis.text(6.45, 4.66, text["sensor_lane"], color=ORANGE, fontsize=9.5, fontweight="bold", family=family)
    box(axis, 6.45, 3.5, 1.55, 0.78, text["sensors"], edge=ORANGE, face=LIGHT_ORANGE, family=family, size=10.5)
    box(axis, 8.25, 3.5, 1.65, 0.78, text["sensor_mlp"], edge=ORANGE, face=LIGHT_ORANGE, family=family, size=10.5)
    box(axis, 10.2, 3.5, 1.55, 0.78, text["sensor_tokens"], edge=ORANGE, face=LIGHT_ORANGE, family=family, size=10.5)
    arrow(axis, (8.0, 3.89), (8.25, 3.89), ORANGE)
    arrow(axis, (9.9, 3.89), (10.2, 3.89), ORANGE)
    arrow(axis, (11.0, 4.28), (11.4, 5.05), ORANGE, rad=0.05)

    axis.text(0.3, 2.83, text["query_lane"], color=RED, fontsize=9.5, fontweight="bold", family=family)
    box(axis, 0.3, 1.55, 2.35, 0.8, text["query"], edge=RED, face=LIGHT_RED, family=family, size=10)
    box(axis, 2.95, 1.55, 1.35, 0.8, text["query_mlp"], edge=RED, face=LIGHT_RED, family=family, size=10.5)
    box(axis, 6.95, 1.42, 2.2, 1.06, text["attention"], edge=RED, face=LIGHT_RED, family=family, size=11, bold=True)
    box(axis, 9.55, 1.55, 1.55, 0.8, text["head"], edge=RED, face=LIGHT_RED, family=family, size=10.5)
    box(axis, 11.45, 1.55, 1.35, 0.8, text["output"], edge=RED, face=LIGHT_RED, family=family, size=10.5, bold=True)
    arrow(axis, (2.65, 1.95), (2.95, 1.95), RED)
    arrow(axis, (4.3, 1.95), (6.95, 1.95), RED)
    arrow(axis, (3.35, 3.5), (8.0, 2.48), GREEN, rad=-0.05)
    arrow(axis, (11.0, 5.05), (8.55, 2.48), BLUE, rad=0.07)
    arrow(axis, (9.15, 1.95), (9.55, 1.95), RED)
    arrow(axis, (11.1, 1.95), (11.45, 1.95), RED)

    axis.add_patch(
        Rectangle(
            (0.3, 0.38),
            12.55,
            0.55,
            facecolor="#f8f8f8",
            edgecolor="#c7ccd1",
            linewidth=0.9,
        )
    )
    axis.text(0.5, 0.65, text["loss"], fontsize=10.2, family=family, va="center")
    axis.text(13.1, 6.7, text["architecture_title"], fontsize=14, fontweight="bold", family=family, ha="right")
    figure.tight_layout()
    figure.savefig(ASSETS / f"xl_architecture_{language}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def load_metrics() -> tuple[dict, dict]:
    return (
        json.loads(XL_METRICS.read_text(encoding="utf-8")),
        json.loads(PHYSICS_METRICS.read_text(encoding="utf-8")),
    )


def tradeoff(language: str, metrics: dict) -> None:
    text = strings(language)
    baseline = [0.115318924, 1.038709]
    current = [metrics["test"]["relative_l2"], metrics["test"]["pde_rms"]]
    figure, axes = plt.subplots(1, 2, figsize=(9.8, 4.35))
    for axis, before, after, ylabel in [
        (axes[0], baseline[0], current[0], text["test_l2"]),
        (axes[1], baseline[1], current[1], text["pde"]),
    ]:
        bars = axis.bar(
            [0, 1],
            [before, after],
            width=0.55,
            color=[RED, BLUE],
            edgecolor="white",
            linewidth=1.0,
        )
        for bar, value in zip(bars, [before, after]):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.4g}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )
        change = (after - before) / before * 100
        axis.text(
            0.5,
            max(before, after) * 1.08,
            f"{change:+.1f}%",
            ha="center",
            va="bottom",
            color=GREEN if change < 0 else ORANGE,
            fontsize=10.5,
            fontweight="bold",
        )
        axis.set_xticks([0, 1], [text["baseline"], text["ours"]])
        axis.set_ylabel(ylabel)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_axisbelow(True)
    axes[1].set_yscale("log")
    figure.suptitle(text["tradeoff_title"], x=0.02, ha="left", fontweight="bold")
    figure.text(0.98, 0.02, text["tradeoff_note"], ha="right", va="bottom", color=MUTED, fontsize=8.5)
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    figure.savefig(ASSETS / f"xl_result_tradeoff_{language}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def parameter_breakdown(language: str) -> None:
    text = strings(language)
    groups = [
        ("Context Transformer", 529920),
        ("Manifold encoder", 148256),
        ("Cross-attention", 66048),
        ("Feed-forward block", 65920),
        ("Query projection", 27008),
        ("Sensor projection", 17024),
        ("Tokenizer / token projections", 11136),
        ("Normalization / output head", 4673),
        ("Embeddings", 4224),
    ]
    if language == "zh":
        groups = [
            ("上下文 Transformer", 529920),
            ("流形编码器", 148256),
            ("交叉注意力", 66048),
            ("前馈网络", 65920),
            ("查询投影", 27008),
            ("传感器投影", 17024),
            ("Tokenizer / token 投影", 11136),
            ("归一化 / 输出头", 4673),
            ("嵌入参数", 4224),
        ]
    labels = [item[0] for item in groups]
    values = [item[1] for item in groups]
    figure, axis = plt.subplots(figsize=(8.7, 4.8))
    y = np.arange(len(labels))
    axis.barh(y, values, color=[BLUE, GREEN, ORANGE, "#7f8b98", "#9a6a4f", "#c08a3c", "#6f8390", "#8d9297", "#a7adb2"])
    for index, value in enumerate(values):
        axis.text(value + 5000, index, f"{value:,}", va="center", fontsize=8.9)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel(text["parameters"])
    axis.set_title(text["params_title"], loc="left", fontweight="bold")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(ASSETS / f"xl_parameter_breakdown_{language}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def training_curves(language: str, metrics: dict) -> None:
    text = strings(language)
    history = metrics["training_history"]
    steps = [row["step"] for row in history if row["step"] > 0]
    dev_l2 = [row["dev_relative_l2"] for row in history if row["step"] > 0]
    dev_pde = [row["dev_pde_rms"] for row in history if row["step"] > 0]
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.25))
    axes[0].plot(steps, dev_l2, color=BLUE, linewidth=1.8)
    axes[0].set_ylabel(text["dev_l2"])
    axes[0].set_xlabel(text["step"])
    axes[1].semilogy(steps, dev_pde, color=ORANGE, linewidth=1.8)
    axes[1].set_ylabel(text["dev_pde"])
    axes[1].set_xlabel(text["step"])
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_axisbelow(True)
    figure.suptitle(text["history_title"], x=0.02, ha="left", fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(ASSETS / f"xl_training_curves_{language}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def physics_sensitivity(language: str, main_metrics: dict, physics_metrics: dict) -> None:
    text = strings(language)
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
    series = [
        (text["test_l2"], [main_metrics["test"]["relative_l2"], physics_metrics["test"]["relative_l2"]]),
        (
            f"{text['pde']} ($\\times 10^{{-3}}$)",
            [
                main_metrics["test"]["pde_rms"] * 1000,
                physics_metrics["test"]["pde_rms"] * 1000,
            ],
        ),
    ]
    for axis, (ylabel, values) in zip(axes, series):
        bars = axis.bar([0, 1], values, width=0.55, color=[BLUE, ORANGE], edgecolor="white", linewidth=1.0)
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
            f"{change:+.1f}%",
            ha="center",
            va="bottom",
            color=GREEN if change < 0 else ORANGE,
            fontsize=10.5,
            fontweight="bold",
        )
        axis.set_xticks([0, 1], [text["main_config"], text["physics_config"]])
        axis.tick_params(axis="x", labelsize=8.5)
        axis.set_ylabel(ylabel)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_axisbelow(True)
    figure.suptitle(text["sensitivity_title"], x=0.02, ha="left", fontweight="bold")
    figure.text(
        0.98,
        0.035,
        text["tradeoff_note"],
        ha="right",
        va="bottom",
        color=MUTED,
        fontsize=8.5,
    )
    figure.subplots_adjust(
        left=0.10,
        right=0.98,
        top=0.82,
        bottom=0.29,
        wspace=0.30,
    )
    figure.savefig(
        ASSETS / f"xl_physics_sensitivity_{language}.png",
        dpi=300,
        bbox_inches=None,
    )
    plt.close(figure)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for language in ("zh", "en"):
        family = setup_fonts()
        architecture(language, family)
        main_metrics, physics_metrics = load_metrics()
        tradeoff(language, main_metrics)
        parameter_breakdown(language)
        training_curves(language, main_metrics)
        physics_sensitivity(language, main_metrics, physics_metrics)


if __name__ == "__main__":
    main()

