# 数据增强的物理约束多模态神经算子：研究方法和实验结果

**English title**: A Data-Augmented Physics-Informed Multimodal Neural Operator for Burgers Equations

**Subtitle**: 场编码、MLP 流形表示、8 传感器 token、交叉注意力与容量扩展

**Version**: V2.0 · 2026 年 11 月 10 日

---

## 摘要

多模态神经算子将密集初始场、低维潜状态和稀疏传感器观测统一表示为可组合的 token，用于构建参数化偏微分方程的查询式代理模型。本文以周期一维黏性 Burgers 方程为对象，给出项目三个阶段的研究方法与实验结果：v11 基线、3 倍增强 Wide 模型和 128 宽四层 Q-MMNO 模型。当前主实验使用 32 个 CNN 场 token、32 维 MLP 流形潜状态和 8 个固定时刻传感器 token，上下文总长度为 40；Transformer 编码器为 4 层、4 个注意力头、宽度 128，查询解码器通过 4 头交叉注意力输出任意 $(x,t)$ 坐标处的预测。模型总参数量为 874,209。项目记录中，3 倍增强阶段把测试 PDE RMS 从 1.0387 降至 0.0048，最终 Q-MMNO 进一步降至 $1.059\times10^{-3}$；测试相对 L2 分别为 0.1153、0.2749 和 0.2461。将 PDE 权重由 0.10 提高至 0.15，同时将 Dropout 由 0.10 降低至 0.08 后，测试相对 L2 降至 0.2235，PDE RMS 保持在同一量级。本文同时说明数据增强的名义规模与实际保存规模、场误差与物理残差的不同评测口径，以及当前实验的直接适用边界。

**关键词**：神经算子；Burgers 方程；物理约束；交叉注意力；数据增强；多模态融合

## 1 研究背景

### 1.1 研究对象与项目目标

偏微分方程的数值求解通常需要精细网格和稳定的时间推进算法，重复求解参数化问题代价较高。神经算子以函数到函数的映射为学习对象，在初始条件或参数变化后可直接给出新的解场预测 [1-4]。物理信息神经网络则通过自动微分将控制方程残差加入训练目标 [5,6]，为数据稀疏或约束不足的场景提供额外监督。

本项目以一维周期黏性 Burgers 方程为统一实验对象，研究密集初始场、低维潜状态、中期传感器观测和物理残差如何在同一模型中组合。项目的最终目标不是只降低单一测试集误差，而是形成一个可复现的查询式算子框架，并明确不同阶段结果之间的比较边界。

### 1.2 与现有工作的关系

Fourier 神经算子通过频域卷积学习全局积分核 [1]，DeepONet 通过分支与主干网络编码输入函数和查询坐标 [2]，通用神经算子理论进一步讨论了函数空间之间的逼近能力 [3]。PDEBench 为数据驱动 PDE 求解提供统一评测基础 [4]，物理约束可与 DeepONet 等算子结构结合 [6]。Transformer 通过自注意力建模 token 依赖 [8]，多模态学习关注异构输入的对齐与融合 [7]。

本文沿用这些工作的基本思想，但不把注意力解释为物理因果机制。Transformer 在当前项目中主要负责场 token、传感器 token 和查询向量之间的信息交互；空间局部性由 CNN tokenizer 提供，物理一致性由初始条件、周期边界和 PDE 残差约束提供。

### 1.3 本文结构

第 2 节给出当前实验的实际模型结构与训练目标；第 3 节说明数据划分、增强方式和评价协议；第 4 节报告 v11、3 倍增强和最终 Q-MMNO 的阶段结果，并讨论场误差与物理残差之间的权衡。所有结论均以项目保存的配置、日志和汇总记录为依据。

## 2 当前实验方法

### 2.1 控制方程与学习目标

一维周期黏性 Burgers 方程为

$$
\frac{\partial u}{\partial t}
+u\frac{\partial u}{\partial x}
=\nu\frac{\partial^2 u}{\partial x^2},
\qquad x\in[0,1),\quad t\in[0,1],
$$

其中 $\nu=0.01$。给定初始场 $u_0(x)$，目标是学习算子

$$
\hat{u}(x,t)
=
\mathcal{G}_{\theta}
\left(
u_0,
\{(x_i,t_s,u_i)\}_{i=1}^{8}
\right)(x,t),
$$

其中 $t_s=0.5$，传感器位置在空间域内均匀分布。

### 2.2 当前实验的实际结构

![图 1 当前实验的 Q-MMNO 架构](publication/assets/architecture_clean_zh.png)

**图 1** 主实验 `v_aug3x_xl_d128_l4` 的实际计算图。模型采用 CNN 场 tokenizer、MLP 流形编码器、8 个传感器 token、4 层宽度 128 的 Transformer 和查询交叉注意力解码器。当前配置不启用输入高通支路、潜空间 ODE 和流形 CNN 支路。

图 1 对应已归档主实验配置，而不是早期原型。与旧版宽 64、3 层并使用潜空间 ODE 的架构相比，当前模型将宽度提高到 128，将上下文层数提高到 4，并关闭潜空间时间积分。参数总量由约 34 万增加到 874,209。

### 2.3 CNN 场 tokenizer

初始场包含 256 个空间点。两层一维卷积依次使用 16 和 32 个通道，卷积核大小为 5，激活函数为 SiLU。卷积特征经自适应平均池化压缩为 32 个位置，再通过线性层投影到 128 维：

$$
\mathbf{T}_f
=
\operatorname{Proj}_{128}
\left(
\operatorname{Pool}_{32}
\left(
\operatorname{CNN}(u_0)
\right)
\right)
\in\mathbb{R}^{32\times128}.
$$

场 token 加入可学习位置嵌入和潜状态投影。位置嵌入保留空间顺序，潜状态投影使每个局部 token 获得全局上下文。

### 2.4 MLP 流形表示

流形分支使用两层 MLP 自编码器。编码器结构为

$$
256\rightarrow256\rightarrow32,
$$

得到潜状态

$$
\mathbf{z}_0
=
\operatorname{Enc}_{\phi}(u_0)
\in\mathbb{R}^{32}.
$$

解码器重建初始场。自编码器在正式训练前以重构损失预训练 500 步。主实验不执行潜空间 ODE 积分，查询解码器直接使用静态潜状态 $\mathbf{z}_0$。

### 2.5 传感器 token

每个传感器观测由空间位置、采样时刻和测量值组成。三元组通过两层 MLP 映射到 128 维，并加入共享类型嵌入：

$$
\mathbf{T}_s^{(i)}
=
\operatorname{MLP}_{s}
\left(
[x_i,0.5,u_i]
\right)
+
\mathbf{e}_s,
\qquad i=1,\ldots,8.
$$

32 个场 token 与 8 个传感器 token 拼接，形成长度为 40 的上下文序列：

$$
\mathbf{X}_0
=
\left[
\mathbf{T}_f;
\mathbf{T}_s
\right]
\in\mathbb{R}^{40\times128}.
$$

### 2.6 Transformer 上下文编码器

当前编码器包含 4 个 Transformer block，每个 block 使用 4 头自注意力、残差连接、LayerNorm 和 256 维前馈网络：

$$
\mathbf{C}
=
\Phi_{\mathrm{ctx}}(\mathbf{X}_0).
$$

缩放点积注意力为

$$
\operatorname{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V})
=
\operatorname{softmax}
\left(
\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}
\right)
\mathbf{V}.
$$

上下文长度为 40，因此自注意力的二次复杂度在当前模型中可控。上下文编码器同时处理场 token 和传感器 token，使两类模态在查询解码前完成交互。

### 2.7 查询交叉注意力解码器

查询坐标使用 4 个空间频带和 4 个时间频带的 Fourier 特征：

$$
\gamma(x,t)
=
\left[
\sin(2^k\pi x),\cos(2^k\pi x),
\sin(2^k\pi t),\cos(2^k\pi t)
\right]_{k=0}^{3}.
$$

查询向量由坐标特征、潜状态和局部 CNN 特征组成：

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

其中 $u_0^{\mathrm{local}}(x)$ 由查询点处的初始场插值值 1 维和局部 CNN 特征 32 维组成，共 33 维。Fourier 坐标特征 16 维、潜状态 32 维和局部特征 33 维相加，因此查询投影的输入维度为 81。

查询通过 4 头交叉注意力读取上下文：

$$
\widetilde{\mathbf{q}}(x,t)
=
\operatorname{CrossAttn}
\left(
\mathbf{q}(x,t),\mathbf{C},\mathbf{C}
\right).
$$

交叉注意力输出经过残差连接、LayerNorm、两层前馈网络和输出头，得到 $\hat{u}(x,t)$。全场预测按 4,096 个查询点分块处理。

### 2.8 训练目标

训练损失由数据拟合、初始条件、周期边界、PDE 残差和流形重构组成：

$$
\mathcal{L}
=
s_d\mathcal{L}_{\mathrm{data}}
+\lambda_p s_p\mathcal{L}_{\mathrm{PDE}}
+0.1s_i\mathcal{L}_{\mathrm{IC}}
+0.01s_b\mathcal{L}_{\mathrm{BC}}
+0.001s_r\mathcal{L}_{\mathrm{rec}},
$$

其中 $\lambda_p=0.1$，$s_d,s_p,s_i,s_b,s_r$ 为动态损失缩放系数。PDE 残差为

$$
\mathcal{L}_{\mathrm{PDE}}
=
\frac{1}{N_p}
\sum_{j=1}^{N_p}
\left(
\hat{u}_t+\hat{u}\hat{u}_x-\nu\hat{u}_{xx}
\right)^2.
$$

验证分数为

$$
s_{\mathrm{dev}}
=
e_{\mathrm{L2}}^{\mathrm{dev}}
+0.1e_{\mathrm{PDE}}^{\mathrm{dev}}.
$$

训练保留验证分数最低的检查点。主实验在第 22,000 步取得最佳验证分数。

### 2.9 优化与运行配置

模型使用 AdamW，初始学习率 $10^{-3}$，权重衰减 $10^{-4}$，余弦退火至初始学习率的 0.02。批大小为 32，训练 24,000 步，每 500 步验证一次。主实验耗时 1,606.96 秒，峰值已分配 GPU 显存为 0.66 GiB。主要配置见表 1。

| 配置项 | 当前实验 |
| --- | ---: |
| 模型宽度 $d$ | 128 |
| Transformer 层数 $L$ | 4 |
| 注意力头数 $H$ | 4 |
| 场 token 数 | 32 |
| 流形潜维度 | 32 |
| 流形编码器 | MLP |
| 传感器数 | 8 |
| 传感器时刻 | 0.5 |
| 输入高通支路 | 关闭 |
| 潜空间 ODE | 关闭 |
| 流形 CNN 支路 | 关闭 |
| 参数总量 | 874,209 |
| 训练步数 | 24,000 |
| 最佳检查点 | 22,000 |
| 随机种子 | 42 |

**表 1** 当前主实验配置。表中各项与 `experiments/v_aug3x_xl_d128_l4/metrics.json` 一致。

## 3 数据与实验设计

### 3.1 数据划分

项目最初包含约 1.4k 条基础 Burgers 轨迹。训练集使用 3 倍周期平移和经验增强构造候选样本。名义增强规模约为 4.2k；经过 NaN/Inf 清理和质量筛选后，最终保存训练集为 2,956 条轨迹。验证集和测试集分别包含 633 和 635 条轨迹。

| 数据划分 | 保存轨迹数 | 用途 |
| --- | ---: | --- |
| 训练集 | 2,956 | 参数优化 |
| 验证集 | 633 | 检查点选择与早停 |
| 测试集 | 635 | 最终评测 |

**表 2** 当前数据划分。

### 3.2 数据增强

增强流程主要使用周期空间平移。对于周期 Burgers 方程，若 $u(x,t)$ 为解，则

$$
u_\delta(x,t)=u(x-\delta,t)
$$

仍满足原控制方程，因此周期平移不改变 PDE 形式。项目另外试验了空间翻转、幅值缩放、时间窗插值和局部时间反转等经验扰动。这些操作可能改变时间方向或方程参数，应当作为数据扰动而不是严格物理等价增强。

项目记录显示，3 倍增强阶段取得最优物理残差；增强规模扩大到约 10.3k 条时，测试相对 L2 和 PDE RMS 反而恶化到 0.5153 和 0.0731。该比较说明增强收益存在上限，但不同增强倍数的训练配置并不完全一致，不能把差异归因于单一因素。

### 3.3 评价指标

逐轨迹平均相对 L2 误差定义为

$$
e_{\mathrm{L2}}
=
\frac{1}{M}
\sum_{m=1}^{M}
\frac{
\left\|
\hat{u}_m-u_m
\right\|_2
}{
\left\|
u_m
\right\|_2
}.
$$

PDE 残差 RMS 定义为

$$
e_{\mathrm{PDE}}
=
\sqrt{
\frac{1}{N}
\sum_{j=1}^{N}
\left(
\hat{u}_t+\hat{u}\hat{u}_x-\nu\hat{u}_{xx}
\right)^2
}.
$$

主实验的相对 L2 使用 32 条轨迹，PDE RMS 使用 128 个随机采样点。两者样本数不同，不应合并为同一“测试集指标”。

### 3.4 对照阶段

项目保留三个主要阶段。v11 基线用于提供早期数据驱动参考；3 倍增强阶段用于检验周期平移和经验扰动；最终 Q-MMNO 用于检验容量扩展和当前结构。三阶段的模型、数据和评测口径并不完全一致，因此结果用于描述项目演进，不作为单一因素的匹配消融。

## 4 实验结果

### 4.1 项目阶段演进

表 3 和图 2 汇总三个阶段的主要结果。

| 阶段 | 模型与数据 | 训练数据 | 测试相对 L2 | PDE RMS |
| --- | --- | ---: | ---: | ---: |
| 基线 | v11，宽 64，3 层 | 约 1.4k | 0.1153 | 1.0387 |
| 3 倍增强 | Wide，增强数据 | 名义 4.2k | 0.2749 | 0.0048 |
| 最终模型 | Q-MMNO，宽 128，4 层 | 保存 2,956 | 0.2461 | 1.059e-3 |

**表 3** 项目阶段结果。前两阶段来自项目运行记录，最终阶段使用已归档 seed 42 配置；各阶段评测样本数不完全一致。

![图 2 项目阶段演进](publication/assets/project_stage_progression_zh.png)

**图 2** 项目阶段演进。物理残差快速下降的同时，场误差先升后小幅回落，说明两类指标存在明显权衡。

与 v11 基线相比，最终模型的 PDE RMS 降低约 99.9%，但测试相对 L2 由 0.1153 增至 0.2461。该变化不能解释为单一模块的因果效应，因为数据增强、模型宽度、层数、损失权重和评测样本均发生变化。

### 4.2 最终模型结果

| 数据划分 | 相对 L2 | PDE RMS | L2 样本数 | PDE 样本数 |
| --- | ---: | ---: | ---: | ---: |
| 训练 | 0.2011 | 1.027e-3 | 32 | 128 |
| 验证 | 0.2414 | 9.443e-4 | 32 | 128 |
| 测试 | 0.2461 | 1.059e-3 | 32 | 128 |

**表 4** 最终 Q-MMNO 结果。

训练误差和测试误差均高于 0.10，当前结果更接近欠拟合而非典型过拟合。测试 PDE 残差很低，但场误差仍较大，说明物理残差不能单独证明预测解与参考轨迹一致。

### 4.3 收敛过程

验证相对 L2 在第 0 步为 1.3071，第 500 步降至 0.5944，第 10,000 步降至 0.2658。验证 PDE RMS 在第 5,000 步附近达到 $1.23\times10^{-2}$，第 10,000 步后降至 $10^{-3}$ 量级。图 3 标出了第 22,000 步的最佳检查点。

![图 3 当前实验训练曲线](publication/assets/xl_training_curves_zh.png)

**图 3** 当前实验的验证相对 L2 与 PDE RMS。红色虚线为第 22,000 步最佳检查点。

### 4.4 物理权重与 Dropout 敏感性

| 配置 | 训练 L2 | 验证 L2 | 测试 L2 | 测试 PDE RMS |
| --- | ---: | ---: | ---: | ---: |
| $\lambda_p=0.10$, $p=0.10$ | 0.2011 | 0.2414 | 0.2461 | 1.059e-3 |
| $\lambda_p=0.15$, $p=0.08$ | 0.1943 | 0.2317 | 0.2235 | 1.062e-3 |

**表 5** 物理权重与 Dropout 的双因素敏感性结果。

第二组配置的测试相对 L2 降低约 9.2%，PDE RMS 增加约 0.3%。由于两个超参数同时改变，且每组只有一个随机种子，该结果不能分离物理权重和 Dropout 的单独作用。

### 4.5 参数分配

| 参数组 | 参数量 | 占比 |
| --- | ---: | ---: |
| 上下文 Transformer | 529,920 | 60.62% |
| MLP 流形网络 | 148,256 | 16.96% |
| 查询解码器 | 163,649 | 18.72% |
| Tokenizer、传感器与嵌入参数 | 32,384 | 3.70% |

**表 6** 当前模型的参数构成。

上下文 Transformer 是参数容量主体。流形编码器占约 17%，但由于潜空间 ODE 关闭，其主要作用是提供紧凑的全局表示。传感器和 tokenizer 相关参数仅占约 3.7%。

### 4.6 结论与适用边界

当前结果支持以下结论：

1. 项目已经实现从 CNN 场 token、MLP 流形状态、传感器 token 到 Transformer 上下文和查询交叉注意力的完整计算路径。
2. 3 倍增强阶段显著降低了 PDE RMS，但测试场误差同时增加。
3. 128 宽四层模型进一步降低 PDE RMS，并在双因素敏感性配置下取得 0.2235 的测试相对 L2。
4. 场误差与 PDE 残差是相关但不等价的目标，必须分别报告。

当前结果不支持以下结论：

1. Q-MMNO 已在所有指标上优于现有神经算子。
2. 传感器 token 或交叉注意力已被证明带来普遍增益。
3. 最终模型可以在二维流场、非周期边界或高雷诺数问题上直接复用。

主要限制包括：主实验只有 seed 42；测试相对 L2 只使用 32 条轨迹；PDE RMS 使用 128 个随机采样点；尚无同数据、同预算的 FNO、DeepONet 和标准 PINN 对照；3 倍增强阶段缺少当前目录中的完整逐文件配置。

## 数据与代码可用性

模型与训练代码位于 `code/demo_burgers_dataset.py`。当前数据目录为 `datasets/burgers_augmented_v1/`，主实验记录为 `experiments/v_aug3x_xl_d128_l4/`，敏感性实验记录为 `experiments/v_aug3x_xl_physics15/`。阶段汇总可参见 `FINAL_RESULTS.md`、`FINAL_PROJECT_SUMMARY.md` 和 `EXPERIMENTS_CHECKPOINTS.md`。

## 参考文献

[1] Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, and A. Anandkumar. Fourier neural operator for parametric partial differential equations. ICLR, 2021.

[2] L. Lu, P. Jin, G. Pang, Z. Zhang, and G. E. Karniadakis. Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. Nature Machine Intelligence, 3:218-229, 2021.

[3] N. Kovachki, Z. Li, B. Liu, K. Azizzadenesheli, K. Bhattacharya, A. Stuart, and A. Anandkumar. Neural operator: Learning maps between function spaces with applications to PDEs. Journal of Machine Learning Research, 24(89):1-97, 2023.

[4] M. Takamoto, T. Praditia, R. Leiteritz, D. MacKinlay, F. Alesiani, D. Pflüger, and M. Niepert. PDEBench: An extensive benchmark for scientific machine learning. NeurIPS Datasets and Benchmarks, 2022.

[5] M. Raissi, P. Perdikaris, and G. E. Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378:686-707, 2019.

[6] S. Wang, H. Wang, and P. Perdikaris. Learning the solution operator of parametric partial differential equations with physics-informed DeepONets. Science Advances, 7(40):eabi8605, 2021.

[7] T. Baltrušaitis, C. Ahuja, and L.-P. Morency. Multimodal machine learning: A survey and taxonomy. IEEE TPAMI, 41(2):423-443, 2019.

[8] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. Attention is all you need. NeurIPS, 2017.

[9] A. Krishnapriyan, A. Gholami, S. Zhe, R. Kirby, and M. W. Mahoney. Characterizing possible failure modes in physics-informed neural networks. NeurIPS, 2021.

[10] J. Pineau, P. Vincent-Lamarre, K. Sinha, V. Larivière, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and H. Larochelle. Improving reproducibility in machine learning research. Journal of Machine Learning Research, 22(164):1-20, 2021.

