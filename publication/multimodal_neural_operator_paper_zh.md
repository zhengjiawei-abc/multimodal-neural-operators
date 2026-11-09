# 数据增强的物理约束多模态神经算子：Burgers 方程的架构、训练与权衡

**English title**: A Data-Augmented Physics-Informed Multimodal Neural Operator for Burgers Equations

**Authors**: 作者一$^{1}$，作者二$^{1,2}$，作者三$^{2}$

**Affiliations**: $^{1}$单位一，城市，国家；$^{2}$单位二，城市，国家

**Corresponding author**: 通讯作者姓名与邮箱

**Date**: 2026 年 11 月 9 日

---

## 摘要

神经算子能够学习偏微分方程解算子的函数空间映射，但标准模型通常只接收完整输入场，较少同时利用混合数据增强、稀疏中期观测和物理残差。本文围绕一维黏性 Burgers 方程构建查询式多模态神经算子（Q-MMNO）。模型将初始场编码为 32 个卷积 token，将 8 个 $t=0.5$ 时刻的传感器测量编码为独立 token，并与 32 维流形潜状态共同送入 Transformer。查询解码器通过多头交叉注意力读取上下文，在任意时空坐标处输出预测。论文按项目实际演进报告三个阶段：v11 基线、3 倍增强阶段和 128 宽四层扩容阶段。项目记录中，v11 的测试相对 L2 与 PDE RMS 分别为 0.1153 和 1.0387；3 倍增强阶段分别为 0.2749 和 0.0048；最终 Q-MMNO 为 0.2461 和 $1.059\times10^{-3}$。最终模型包含 874,209 个参数，并在 2,956 条保存训练轨迹上训练。将物理权重由 0.10 提高至 0.15、Dropout 由 0.10 降低至 0.08 后，测试相对 L2 误差降至 0.2235，PDE 残差保持在同一量级。结果表明，物理残差显著下降伴随场误差权衡，单一指标不足以评价算子质量。

## 1 引言

偏微分方程广泛用于描述流体力学、传热、结构和多物理场系统。传统数值方法通过离散空间和时间获得高保真解，但重复求解参数化问题通常代价较高。神经算子将输入函数映射为输出函数，在推理阶段可对新的初始条件或参数进行快速预测 [3-5]。近年来，Fourier 神经算子、DeepONet 和基于 Transformer 的算子模型已成为科学机器学习的重要工具 [3-6,10,11]。

现有神经算子主要处理完整输入场。实际系统往往同时提供多源信息，例如完整初始场、少量传感器读数、边界条件和材料参数。这些信息的维度和物理含义不同，简单拼接会削弱模型对局部观测的利用。如何设计统一的 token 接口，使全局场和稀疏观测在同一个表示空间内交互，仍缺少清晰的基线。

物理信息神经网络（PINN）提供了另一种思路。它将 PDE 残差加入损失函数，使网络在缺少密集标签时仍能利用控制方程 [1,2]。然而 PINN 的训练对损失权重和采样策略敏感，尤其在刚性或多尺度问题中可能出现优化困难 [7,8,15-18]。本文将数据监督作为主要目标，将 PDE 残差作为额外正则项，并保留连续坐标查询能力。

Burgers 方程适合作为这一问题的试验平台。它包含非线性对流和线性扩散，能够产生陡峭梯度，同时保持一维周期问题的可控性。本文并不试图证明一个新的通用求解器已经全面优于现有方法，而是建立一个结构明确、可以审计的多模态算子基线，并回答三个问题：

1. 完整初始场和稀疏中期观测能否通过统一 token 接口进行融合？
2. Transformer 上下文编码器和查询交叉注意力能否支持任意坐标解码？
3. 场误差和 PDE 残差在训练与评测中是否表现一致？

本文的主要贡献如下。首先，我们提出 Q-MMNO，将卷积场 token、流形潜状态和传感器 token 在 Transformer 中进行中期融合，并用查询级交叉注意力完成解码。其次，我们给出项目使用的混合数据增强和候选筛选流程，将最初约 1.4k 条轨迹扩展为名义 3 倍规模，质量过滤后的保存训练集为 2,956 条轨迹。再次，我们在同一架构框架下比较 64 宽三层和 128 宽四层两个实例，并给出数据增强、容量扩容和物理权重调整后的完整结果链。最后，我们明确区分场误差、物理残差和评测样本数，以避免把低 PDE 残差误读为高场精度。

## 2 背景

### 2.1 物理信息神经网络

Raissi 等提出 PINN，通过自动微分计算 PDE 残差并与初始条件、边界条件共同优化 [1]。该框架不要求网格化标签，适用于正问题和反问题。后续研究将 PINN 扩展到多物理场和参数识别 [2]，并从神经切线核、损失权重和采样策略等角度分析训练失败原因 [7,8,18]。自适应权重和区域分解可缓解部分问题 [15,16]。

### 2.2 神经算子

DeepONet 使用分支网络编码输入函数、主干网络编码查询坐标 [5]。Fourier 神经算子通过频域卷积学习全局积分核 [3]。Kovachki 等总结了神经算子的函数空间逼近性质 [4]。物理约束也可与 DeepONet 结合 [6]。PDEBench 提供了多类 PDE 和多分辨率基准 [12]，而 FAIR 数据比较研究强调统一数据、离散化和训练预算的重要性 [13]。

### 2.3 Transformer 算子

Transformer 通过自注意力建模 token 之间的依赖 [9]。Cao 比较了 Fourier 与 Galerkin 注意力 [10]，GNOT 将几何注意力和专家混合用于算子学习 [11]。与图像或语言任务相比，PDE 场 token 数量较少，但查询坐标连续，要求模型在编码上下文和逐点解码之间建立稳定接口。

### 2.4 多模态条件学习

多模态学习研究异构表示的融合、对齐和协同学习 [14]。本文只考虑一种基本组合：完整初始场与固定时刻稀疏传感器。传感器被编码为独立 token，与场 token 共享注意力上下文。该结构可以自然扩展到多时刻传感器、边界测点和材料参数。

## 3 模型架构

### 3.1 总体结构

Q-MMNO 将输入分为三路：初始场、流形潜状态和传感器观测。初始场首先进入卷积 tokenizer 和 MLP 流形编码器；传感器观测通过两层 MLP 编码为 token；三路表示在 Transformer 上下文中融合。查询解码器接收坐标特征、潜状态和局部场特征，通过交叉注意力输出预测。

![图 1 当前实验的 Q-MMNO 架构](publication/assets/architecture_clean_zh.png){: .wide-figure }

**图 1** 当前实验 `v_aug3x_xl_d128_l4` 的实际架构。初始场经 CNN tokenizer 产生 32 个宽度 128 的场 token，经 MLP 流形编码器产生 32 维潜状态；8 个传感器产生独立 token；两类 token 拼接为 40 个上下文 token，送入 4 层、4 头、宽度 128 的 Transformer；查询分支结合 Fourier 坐标、潜状态和局部 CNN 特征，通过 4 头交叉注意力解码。当前配置不启用输入高通支路、潜空间 ODE 和流形 CNN 支路。

项目包含两个实例化配置。Wide 版本使用 $d=64$、3 层 Transformer、约 34 万参数，并启用潜空间 ODE；XL 版本使用 $d=128$、4 层 Transformer、874,209 个参数，不启用潜空间 ODE。两版本共享场 token、传感器 token 和查询交叉注意力接口，但容量与时间建模方式不同。

### 3.2 场 tokenizer

初始场 $u_0\in\mathbb{R}^{256}$ 被视为单通道序列。两层一维卷积使用卷积核 5、通道数 16 和 32，并以 SiLU 激活。卷积特征经自适应平均池化压缩到 32 个位置，再投影到 $d=128$ 维：

$$
\mathbf{T}_f
=
\operatorname{Proj}
\left(
\operatorname{Pool}
\left(
\operatorname{CNN}(u_0)
\right)
\right)
\in\mathbb{R}^{32\times128}.
$$

每个 token 加入位置嵌入和潜状态投影：

$$
\widetilde{\mathbf{T}}_f
=
\mathbf{T}_f
+
\mathbf{P}_f
+
\mathbf{1}
\left(
\mathbf{W}_z\mathbf{z}_0
\right)^\top.
$$

位置嵌入保留空间顺序，潜状态投影使每个局部 token 获得全局上下文。

### 3.3 流形表示

流形分支是两层 MLP 自编码器。编码器将初始场压缩为 32 维潜状态：

$$
\mathbf{z}_0
=
\operatorname{Enc}_{\phi}(u_0)
\in\mathbb{R}^{32}.
$$

解码器重建初始场：

$$
\widehat{u}_0
=
\operatorname{Dec}_{\psi}(\mathbf{z}_0).
$$

自编码器在正式训练前以重构损失预训练 500 步。主实验不使用潜空间时间积分，查询解码器直接读取 $\mathbf{z}_0$，时间依赖由坐标 Fourier 特征和交叉注意力建模。

### 3.4 传感器 token

每个传感器由位置、时间和测量值组成。三元组通过两层 MLP 投影到 128 维，并加入共享类型嵌入：

$$
\mathbf{T}_s^{(i)}
=
\operatorname{MLP}_{s}
\left(
[x_i,t_s,u_i]
\right)
+
\mathbf{e}_s.
$$

本文设置 $N_s=8$，$t_s=0.5$，传感器位置均匀覆盖 $[0,1]$。传感器 token 与场 token 拼接后共同进入 Transformer。由于所有传感器共享参数，模型可以处理不同数量或位置的测点，但本文只验证了固定 8 点配置。

### 3.5 上下文编码器

场 token 与传感器 token 组成长度为 40 的序列：

$$
\mathbf{X}_0
=
\left[
\widetilde{\mathbf{T}}_f;
\mathbf{T}_s
\right]
\in\mathbb{R}^{40\times128}.
$$

上下文编码器由 4 层 Transformer block 组成，每层包含 4 头自注意力和 256 维前馈网络：

$$
\mathbf{C}
=
\Phi_{\mathrm{ctx}}(\mathbf{X}_0).
$$

自注意力允许传感器 token 与场 token 双向交换信息，因此传感器读数不仅作为旁路特征，还可以改变整个上下文的表示。

### 3.6 注意力

#### 3.6.1 缩放点积注意力

给定查询矩阵 $\mathbf{Q}$、键矩阵 $\mathbf{K}$ 和值矩阵 $\mathbf{V}$，缩放点积注意力定义为

$$
\operatorname{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V})
=
\operatorname{softmax}
\left(
\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}
\right)
\mathbf{V}.
$$

本文的上下文长度仅为 40，点积注意力的计算成本可控。缩放因子 $\sqrt{d_k}$ 用于避免高维点积进入 softmax 饱和区。

#### 3.6.2 多头注意力

多头注意力将表示划分为 $H=4$ 个子空间：

$$
\operatorname{MultiHead}(\mathbf{Q},\mathbf{K},\mathbf{V})
=
\operatorname{Concat}
\left(
\mathrm{head}_1,\ldots,\mathrm{head}_H
\right)
\mathbf{W}^O,
$$

其中

$$
\mathrm{head}_h
=
\operatorname{Attention}
\left(
\mathbf{Q}\mathbf{W}_h^Q,
\mathbf{K}\mathbf{W}_h^K,
\mathbf{V}\mathbf{W}_h^V
\right).
$$

不同头可以关注不同空间尺度和传感器组合。本文不直接分析注意力权重，因此不将其解释为物理因果机制。

#### 3.6.3 本文中的应用

上下文编码器使用自注意力，查询解码器使用交叉注意力。查询向量由坐标特征、潜状态和局部场特征构造：

$$
\mathbf{q}(x,t)
=
\operatorname{MLP}_{q}
\left(
\left[
\gamma(x,t);
\mathbf{z}_0;
u_0^{\mathrm{local}}(x)
\right]
\right).
$$

交叉注意力以查询为 $\mathbf{Q}$、上下文为 $\mathbf{K}$ 和 $\mathbf{V}$：

$$
\widetilde{\mathbf{q}}(x,t)
=
\operatorname{CrossAttn}
\left(
\mathbf{q}(x,t),\mathbf{C},\mathbf{C}
\right).
$$

该设计使每个查询只读取与当前坐标相关的上下文，而不需要先生成固定网格输出。

### 3.7 位置式前馈网络与输出

交叉注意力输出经过残差连接、LayerNorm、两层前馈网络和输出头：

$$
\hat{u}(x,t)
=
\operatorname{Head}
\left(
\operatorname{FFN}
\left(
\operatorname{Norm}
\left(
\mathbf{q}+\widetilde{\mathbf{q}}
\right)
\right)
\right).
$$

输出头将 128 维查询表示映射到 32 维，再得到标量预测。主实验使用 SiLU 激活和 Dropout 0.1。

### 3.8 坐标编码

查询坐标使用 Fourier 特征：

$$
\gamma(x,t)
=
\left[
\sin(2^k\pi x),\cos(2^k\pi x),
\sin(2^k\pi t),\cos(2^k\pi t)
\right]_{k=0}^{3}.
$$

空间和时间各使用 4 个频带，共得到 16 维坐标特征。该编码使 MLP 能够表示多尺度空间变化，并保留连续坐标查询能力。

### 3.9 参数与训练目标

模型总参数量为 874,209。上下文 Transformer 占 60.62%，MLP 流形网络占 16.96%，查询解码器占 18.72%，tokenizer、传感器和嵌入参数合计占 3.70%。

训练目标由数据损失、PDE 残差、初始条件、周期边界和流形重构组成：

$$
\mathcal{L}
=
s_d\mathcal{L}_{\mathrm{data}}
+\lambda_p s_p\mathcal{L}_{\mathrm{PDE}}
+0.1s_i\mathcal{L}_{\mathrm{IC}}
+0.01s_b\mathcal{L}_{\mathrm{BC}}
+0.001s_r\mathcal{L}_{\mathrm{rec}},
$$

其中 $s_d,s_p,s_i,s_b,s_r$ 为动态损失缩放系数，$\lambda_p=0.1$。动态缩放只衰减超过参考量级的损失项，不主动放大小量级项。

PDE 残差为

$$
\mathcal{L}_{\mathrm{PDE}}
=
\frac{1}{N_p}
\sum_{j=1}^{N_p}
\left(
\hat{u}_t+\hat{u}\hat{u}_x-\nu\hat{u}_{xx}
\right)^2,
\qquad \nu=0.01,
$$

其中导数由自动微分计算。初始条件和周期边界分别约束 $t=0$ 的场值和上下边界的连续性。

## 4 为什么采用这种架构

### 4.1 计算复杂度

表 1 比较三种查询式结构与本文模型的计算复杂度。$N=40$ 为上下文 token 数，$M$ 为查询数，$d=128$ 为表示宽度。

| 结构 | 上下文复杂度 | 单查询复杂度 | 主要特点 |
| --- | ---: | ---: | --- |
| 逐层卷积算子 | $O(Nd^2)$ | $O(d^2)$ | 局部归纳偏置强，长程交互有限 |
| 全连接查询网络 | $O(Nd)$ | $O(Nd)$ | 结构简单，难以处理异构 token |
| Transformer + 交叉注意力 | $O(N^2d)$ | $O(Nd)$ | 支持 token 融合与任意坐标查询 |
| Q-MMNO | $O(N^2d)$ | $O(Nd)$ | 场、潜状态和传感器共享注意力上下文 |

**表 1** 查询式结构的复杂度比较。本文 $N=40$，自注意力成本相对于全场查询解码较小。

### 4.2 路径长度

在查询解码器中，任意查询与任一传感器 token 之间的信息路径长度至多为 2：传感器先在自注意力层中更新，随后被查询交叉注意力读取。若采用逐层局部卷积，需要堆叠多个空间层才能把远程传感器信息传播到查询点。较短的路径有利于中期传感器观测影响全时空预测。

### 4.3 归纳偏置

Transformer 本身不包含空间连续性或周期性先验，因此本文通过以下方式补足：

- 坐标 Fourier 特征提供多尺度连续表示；
- 卷积 tokenizer 提供局部空间归纳偏置；
- 周期边界损失约束上下文解；
- 初始条件损失限制 $t=0$ 的解。

这种组合不是“注意力替代一切”，而是让注意力负责异构 token 交互，让卷积和物理损失负责空间结构与约束。

## 5 训练

### 5.1 数据与批处理

训练、验证和测试集分别包含 2,956、633 和 635 条 Burgers 轨迹。每个轨迹有 256 个空间点和 101 个时间点。每个训练批次包含 32 条轨迹，每条轨迹随机采样 32 个数据点、4 个 PDE 配置点、16 个初始条件点和 1 个边界点。

场误差在 32 条轨迹上评测，PDE 残差在 128 条轨迹的随机采样点上评测。两者使用不同样本数，因此表 3 分别列出。为保证可复现性，本文同时报告随机种子、检查点规则和评测样本数 [17]。

### 5.2 硬件与训练计划

模型在单块 CUDA GPU 上以 FP32 训练，并启用 TF32 矩阵乘法。主实验训练 24,000 步，每 500 步评测一次验证集。seed 42 运行耗时 1,606.96 秒，峰值已分配 GPU 显存为 0.66 GiB。

### 5.3 优化器

模型使用 AdamW，初始学习率 $10^{-3}$，权重衰减 $10^{-4}$。学习率按余弦退火下降至初始值的 0.02。流形参数不施加权重衰减。批大小为 32，未使用学习率预热。

验证分数为

$$
s_{\mathrm{dev}}
=
e_{\mathrm{L2}}^{\mathrm{dev}}
+0.1e_{\mathrm{PDE}}^{\mathrm{dev}}.
$$

最佳验证检查点出现在第 22,000 步。早停机制在计划步数的 75% 后启用，耐心值为 4 次评测。

### 5.4 正则化

主实验使用 Dropout 0.1、梯度裁剪 1.0 和动态损失缩放。梯度裁剪只在少量步骤触发，主实验记录的最大梯度范数为 1.832，平均梯度范数为 0.0986。

### 5.5 数据增强

项目从约 1.4k 条基础轨迹出发，使用周期平移、空间翻转、时间窗插值、幅值缩放、小幅噪声和局部时间反转构造 3 倍增强数据。名义增强规模约为 4.2k；经过 NaN/Inf 清理和质量筛选后，最终保存训练集为 2,956 条轨迹，验证集和测试集分别为 633 和 635 条。周期平移是周期 Burgers 方程的安全对称变换；其他操作会改变方程参数或时间方向，属于经验扰动。

项目记录还比较了更强增强方案。5 倍增强约 10.3k 条轨迹时，测试相对 L2 与 PDE RMS 反而恶化到 0.5153 和 0.0731。该现象支持“增强存在收益上限”的经验判断，但由于增强倍数、数据规模和训练配置同时变化，不能把差异归因于单一因素。

## 6 结果

### 6.1 项目阶段结果

表 2 汇总项目三个主要阶段。v11 基线在约 1.4k 条训练轨迹上取得 0.1153 的测试相对 L2 和 1.0387 的 PDE RMS。3 倍增强阶段把 PDE RMS 降至 0.0048，但测试相对 L2 升至 0.2749。最终 XL 模型把 PDE RMS 进一步降至 $1.059\times10^{-3}$，测试相对 L2 为 0.2461。

| 阶段 | 模型与数据 | 训练数据 | Test L2 | PDE RMS |
| --- | --- | ---: | ---: | ---: |
| 基线 | v11，$d=64$，3 层 | 约 1.4k | 0.1153 | 1.0387 |
| 3 倍增强 | Wide，混合增强 | 名义 4.2k | 0.2749 | 0.0048 |
| 最终模型 | Q-MMNO，$d=128$，4 层 | 保存 2,956 | 0.2461 | 1.059e-3 |

**表 2** 项目阶段结果。前两阶段来自项目运行记录，最终阶段对应已归档的 seed 42 配置；各阶段评测样本数不完全一致。

![图 2 项目阶段演进](publication/assets/project_stage_progression_zh.png)

**图 2** 项目阶段演进。物理残差快速下降的同时，场误差先升后小幅回落，说明两类指标存在明显权衡。

与 v11 基线相比，最终模型的 PDE RMS 降低约 99.9%，但测试相对 L2 由 0.1153 增至 0.2461。该差异不能解释为物理约束单独带来的因果效应，因为数据增强、模型宽度、层数、损失权重和评测样本均发生变化。项目结果更适合作为工程演进证据，而不是匹配消融。

### 6.2 最终模型结果

表 3 给出最终 Q-MMNO 的训练、验证和测试结果。测试相对 L2 误差为 0.2461，测试 PDE 残差均方根为 $1.059\times10^{-3}$。训练与测试相对 L2 误差的差值为 0.0450，测试与训练误差比为 1.224。

| 数据划分 | 相对 L2 | PDE RMS | L2 样本数 | PDE 样本数 |
| --- | ---: | ---: | ---: | ---: |
| 训练 | 0.2011 | 1.027e-3 | 32 | 128 |
| 验证 | 0.2414 | 9.443e-4 | 32 | 128 |
| 测试 | 0.2461 | 1.059e-3 | 32 | 128 |

**表 3** 最终模型结果。PDE RMS 使用随机采样点估计。

训练误差和测试误差都高于 0.10，因此当前结果更接近欠拟合，而不是典型过拟合。测试 PDE 残差很低，但场误差仍较大，说明低物理残差不能单独证明模型复现了参考解。

### 6.3 参数分配

上下文 Transformer 占 529,920 个参数，是模型容量的主体；流形网络占 148,256 个参数；查询解码器占 163,649 个参数；传感器、tokenizer 和嵌入参数合计占 32,384 个参数。

| 参数组 | 参数量 | 占比 |
| --- | ---: | ---: |
| 上下文 Transformer | 529,920 | 60.62% |
| MLP 流形网络 | 148,256 | 16.96% |
| 查询解码器 | 163,649 | 18.72% |
| Tokenizer、传感器与嵌入参数 | 32,384 | 3.70% |

**表 4** 参数构成。查询解码器包含交叉注意力、前馈网络、查询投影、归一化与输出头。

### 6.4 训练动态

验证相对 L2 误差在第 0 步为 1.3071，第 500 步降至 0.5944，第 10,000 步降至 0.2658。第 15,000 步后曲线趋于平台。验证 PDE 残差在第 5,000 步附近达到 $1.23\times10^{-2}$，第 10,000 步后降至 $10^{-3}$ 量级。

![图 3 Q-MMNO 验证集收敛曲线](publication/assets/xl_training_curves_zh.png)

**图 3** 验证相对 L2 误差与 PDE 残差随训练步数的变化。

### 6.5 物理权重与 Dropout

表 5 给出一次双因素敏感性实验。将 PDE 权重从 0.10 提高到 0.15，同时将 Dropout 从 0.10 降低到 0.08，测试相对 L2 误差降至 0.2235，降幅约 9.2%。测试 PDE 残差由 $1.059\times10^{-3}$ 变为 $1.062\times10^{-3}$，相对增加约 0.3%。

| 配置 | 训练 L2 | 验证 L2 | 测试 L2 | 测试 PDE RMS |
| --- | ---: | ---: | ---: | ---: |
| $\lambda_p=0.10,\ p=0.10$ | 0.2011 | 0.2414 | 0.2461 | 1.059e-3 |
| $\lambda_p=0.15,\ p=0.08$ | 0.1943 | 0.2317 | 0.2235 | 1.062e-3 |

**表 5** 物理权重与 Dropout 的双因素敏感性结果。

两个因素同时改变，且每组只有一个随机种子，因此该结果不能分离物理权重与 Dropout 的单独作用。

### 6.6 适用条件与局限

本文结果受以下因素限制。

1. 主实验只有一个随机种子，无法估计训练方差。
2. 测试场误差只使用 32 条轨迹，而测试文件包含 635 条轨迹。
3. PDE 残差使用 128 个重新采样的坐标点，未报告多次采样置信区间。
4. 尚无同数据、同预算的 FNO、DeepONet 或标准 PINN 对照。
5. 尚未完成 64/128 宽度、3/4 层、有无传感器和有无交叉注意力的匹配消融。
6. 数据增强包含非严格物理等价变换，限制了物理解释。
7. 结果只涉及一维周期 Burgers 方程，不能直接推广到二维流场或非周期边界。
8. 3 倍增强阶段来自项目运行记录，而不是具有完整逐文件配置的当前归档运行；各阶段评测样本数也不完全一致。

因此，本文支持“Q-MMNO 可以作为多模态算子基线，并能通过物理残差训练获得低 PDE 残差的解”，但不支持“该模型已经在所有指标上优于现有方法”或“注意力或传感器 token 已被证明带来普遍增益”。

## 7 结论

本文围绕 Burgers 方程构建了数据增强的物理约束多模态神经算子。项目从 v11 基线出发，首先通过混合增强把 PDE RMS 从 1.0387 降至 0.0048；随后将模型扩展到 128 宽四层，使 PDE RMS 进一步降至 $1.059\times10^{-3}$。最终 seed 42 运行的测试相对 L2 为 0.2461。双因素敏感性实验将测试相对 L2 误差降至 0.2235，PDE 残差保持在同一量级。

结果表明，数据增强和物理约束能够显著降低 PDE 残差，但可能以场误差增加为代价。场误差与物理一致性是相关但不等价的目标，低 PDE 残差不能替代场误差评测。当前模型的合理定位是透明、可扩展的多模态算子基线。后续工作应补齐全测试集评测、多随机种子重复、匹配基线和结构消融，并在更复杂的 PDE 上验证可迁移性。

## 致谢

致谢信息待补充。

## 数据与代码可用性

模型与训练代码位于 `code/demo_burgers_dataset.py`。数据目录为 `datasets/burgers_augmented_v1/`，主实验记录为 `experiments/v_aug3x_xl_d128_l4/`，敏感性实验记录为 `experiments/v_aug3x_xl_physics15/`。阶段汇总可参见 `FINAL_RESULTS.md`、`FINAL_PROJECT_SUMMARY.md` 和 `EXPERIMENTS_CHECKPOINTS.md`。正式投稿时应提供版本号、许可证、配置文件和固定校验值。

## 附录 A 复现配置

主实验核心命令如下：

```powershell
python code\demo_burgers_dataset.py `
  --data-dir datasets\burgers_augmented_v1 `
  --train-file burgers_augmented_train.npz `
  --dev-file burgers_augmented_dev.npz `
  --test-file burgers_augmented_test.npz `
  --train-samples 2956 --dev-samples 633 --test-samples 635 `
  --eval-samples 32 `
  --preset wide --d-model 128 --n-layers 4 `
  --sensor-count 8 --sensor-time 0.5 `
  --steps 24000 --lr-decay-steps 12000 --batch-size 32 `
  --lr 0.001 --warmup-steps 0 --dropout 0.1 `
  --optimizer adamw --weight-decay 0.0001 --manifold-weight-decay 0 `
  --physics-weight 0.1 --validation-physics-weight 0.1 `
  --loss-balance --manifold-steps 500 `
  --eval-every 500 --early-stop-patience 4 `
  --early-stop-min-fraction 0.75 --seed 42 --save-model `
  --output-dir experiments\v_aug3x_xl_d128_l4
```

当前归档目录未保存主实验模型权重，因此扩展测试集评测需要重新训练或从外部备份恢复最佳检查点。

## 参考文献

[1] M. Raissi, P. Perdikaris, and G. E. Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378:686-707, 2019.

[2] G. E. Karniadakis, I. G. Kevrekidis, L. Lu, P. Perdikaris, S. Wang, and L. Yang. Physics-informed machine learning. *Nature Reviews Physics*, 3:422-440, 2021.

[3] Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, and A. Anandkumar. Fourier neural operator for parametric partial differential equations. In *International Conference on Learning Representations*, 2021.

[4] N. Kovachki, Z. Li, B. Liu, K. Azizzadenesheli, K. Bhattacharya, A. Stuart, and A. Anandkumar. Neural operator: Learning maps between function spaces with applications to PDEs. *Journal of Machine Learning Research*, 24(89):1-97, 2023.

[5] L. Lu, P. Jin, G. Pang, Z. Zhang, and G. E. Karniadakis. Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. *Nature Machine Intelligence*, 3:218-229, 2021.

[6] S. Wang, H. Wang, and P. Perdikaris. Learning the solution operator of parametric partial differential equations with physics-informed DeepONets. *Science Advances*, 7(40):eabi8605, 2021.

[7] S. Wang, Y. Teng, and P. Perdikaris. Understanding and mitigating gradient flow pathologies in physics-informed neural networks. *SIAM Journal on Scientific Computing*, 43(5):A3055-A3081, 2021.

[8] A. Krishnapriyan, A. Gholami, S. Zhe, R. Kirby, and M. W. Mahoney. Characterizing possible failure modes in physics-informed neural networks. In *Advances in Neural Information Processing Systems*, 34, 2021.

[9] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. Attention is all you need. In *Advances in Neural Information Processing Systems*, 30, 2017.

[10] S. Cao. Choose a transformer: Fourier or Galerkin. In *Advances in Neural Information Processing Systems*, 34, 2021.

[11] Z. Hao, Z. Wang, H. Su, C. Ying, Y. Dong, S. Liu, and J. Cheng. GNOT: A general neural operator transformer for operator learning. In *International Conference on Machine Learning*, 2023.

[12] M. Takamoto, T. Praditia, R. Leiteritz, D. MacKinlay, F. Alesiani, D. Pflüger, and M. Niepert. PDEBench: An extensive benchmark for scientific machine learning. In *Advances in Neural Information Processing Systems Datasets and Benchmarks Track*, 2022.

[13] L. Lu, X. Meng, S. Cai, Z. Mao, S. Goswami, Z. Zhang, and G. E. Karniadakis. A comprehensive and fair comparison of two neural operators (with practical extensions) based on FAIR data. *Computer Methods in Applied Mechanics and Engineering*, 393:114778, 2022.

[14] T. Baltrušaitis, C. Ahuja, and L.-P. Morency. Multimodal machine learning: A survey and taxonomy. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 41(2):423-443, 2019.

[15] L. D. McClenny and U. M. Braga-Neto. Self-adaptive physics-informed neural networks. *Journal of Computational Physics*, 474:111722, 2023.

[16] S. Cuomo, V. Schiano di Cola, F. Giampaolo, G. Rozza, M. Raissi, and F. Piccialli. Scientific machine learning through physics-informed neural networks: Where we are and what's next. *Journal of Scientific Computing*, 92:88, 2022.

[17] J. Pineau, P. Vincent-Lamarre, K. Sinha, V. Larivière, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and H. Larochelle. Improving reproducibility in machine learning research. *Journal of Machine Learning Research*, 22(164):1-20, 2021.

[18] S. Wang, X. Yu, and P. Perdikaris. When and why PINNs fail to train: A neural tangent kernel perspective. *Journal of Computational Physics*, 449:110768, 2022.

