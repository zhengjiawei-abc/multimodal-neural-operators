# 面向 Burgers 方程的 128 宽四层多模态神经算子：结构、训练与证据边界

**文档类型**: 研究论文，预印本草稿
**版本**: 2.0
**日期**: 2026 年 11 月 9 日
**作者**: 作者信息待补充
**副标题**: 当前 XL 配置的架构说明、结果核验与可复现性边界

---

## 摘要

本文描述当前交付版本的多模态神经算子，而不是早期 64 宽三层原型。当前主模型采用 MLP 流形编码器、CNN tokenizer、8 个固定时刻传感器 token、4 层 128 宽 Transformer 上下文编码器和查询式交叉注意力解码器，总参数量为 874,209。训练数据为 `burgers_augmented_v1`，保存划分为 2,956 条训练轨迹、633 条验证轨迹和 635 条测试轨迹；现有评测配置实际从前 32 条轨迹计算报告指标。

在 seed 42 存档运行中，模型在第 22,000 步取得最佳验证分数，完成 24,000 步训练，训练耗时 1,606.96 秒。报告测试相对 L2 为 0.2461，测试 PDE RMS 为 0.001059。与旧版 v11 参考点相比，PDE 残差下降 99.9%，但测试 L2 从 0.1153 增至 0.2461。两者使用不同数据和不同架构，因此该比较只能作为描述性权衡，不能解释为架构升级的因果效应。

本文同时纠正了早期文档中的两处口径问题。第一，当前模型不是 `d=64`、3 层、336,546 参数的旧版 `wide` 模型，而是 `d=128`、4 层、874,209 参数的 XL 模型。第二，`burgers_augmented_v1` 的实际训练集为 2,956 条，而不是文档中反复出现的 4.2k；当前测试指标也不是在全部 635 条测试轨迹上计算，而是在 32 条评测轨迹上计算。

**关键词**: 神经算子；多模态学习；Burgers 方程；交叉注意力；物理约束；可复现性

## 1 引言

神经算子学习函数空间之间的映射，可以在推理阶段对任意空间和时间坐标给出预测 [1-4]。当前项目进一步接收稀疏传感器观测，并使用物理残差约束输出，因此属于多模态算子与物理约束训练的结合。早期原型主要验证了场编码、注意力上下文和潜变量表示是否能够工作。当前 XL 版本把模型宽度从 64 提升到 128，把上下文层数从 3 提升到 4，并显式接收 8 个传感器。

项目文档中同时存在多个历史版本和相互冲突的指标。为避免把旧模型配置写进当前方法，本文以保存的 `experiments/v_aug3x_xl_d128_l4/metrics.json` 和对应配置为第一证据源。凡是保存文件与二次总结冲突之处，均采用保存文件并在文中说明差异。

本文的贡献有四点：

1. 给出当前 XL 多模态神经算子的完整结构、数据接口和参数规模。
2. 报告 seed 42 主实验的指标、训练过程、最佳检查点和耗时。
3. 给出物理权重与 Dropout 的初步敏感性结果，并指出其混杂因素。
4. 修正旧文档中的架构、样本数和评测样本数口径。

本文不声称当前模型已经在所有指标上优于旧版基线。现有结果是一个明确的精度与物理残差权衡，而不是全面胜出。

## 2 问题设置

一维黏性 Burgers 方程为

$$
\frac{\partial u}{\partial t}
+u\frac{\partial u}{\partial x}
=\nu\frac{\partial^2 u}{\partial x^2},
\qquad x\in[0,1),\quad t\in[0,1].
$$

空间采用周期边界。当前数据为 256 个空间点和 101 个时间点。模型输入为初始场 \(u_0(x)\) 与 8 个固定时刻 \(t_s=0.5\) 的传感器测点，输出为任意查询坐标 \((x,t)\) 处的解：

$$
\hat{u}(x,t)
=G_\theta\left(u_0,\{(x_i,t_s,u_i)\}_{i=1}^{8}\right)(x,t).
$$

保存文件中的名义黏性系数为 \(\nu=0.01\)。当前主数据目录为 `datasets/burgers_augmented_v1`，不再使用早期 `burgers_custom_v1` 的三 seed 周期平移实验作为主结果。

## 3 当前 XL 模型

### 3.1 总体结构

![图 1 当前 XL 多模态神经算子结构](publication/assets/xl_architecture_zh.png)

图 1 当前 XL 模型。主路径由 MLP 流形编码器、CNN tokenizer、传感器 token、4 层宽 128 的 Transformer 上下文编码器与查询式交叉注意力解码器组成。

### 3.2 场编码与流形分支

当前配置不启用输入高通支路，也不启用流形 CNN 支路。初始场首先进入两层一维卷积 tokenizer，得到局部特征，然后自适应平均池化到 32 个位置并投影到 128 维：

$$
T_f=\operatorname{Tokenize}(u_0),
\qquad
T_f\in\mathbb{R}^{32\times128}.
$$

同一初始场通过 MLP 流形自编码器编码为 32 维潜状态：

$$
z_0=\operatorname{Enc}_{\mathrm{MLP}}(u_0),
\qquad
z_0\in\mathbb{R}^{32}.
$$

潜状态的线性投影加到场 token 上。当前主实验的 `latent_ode=False`，因此没有潜空间时间积分；查询解码器直接读取静态潜状态 \(z_0\)。这一点与早期架构图不同。

### 3.3 传感器 token

8 个传感器测点由位置、采样时刻和测值组成。每个三元组通过两层 MLP 投影到 128 维，并加入传感器类型嵌入：

$$
T_s^{(i)}
=\operatorname{MLP}([x_i,0.5,u_i])+e_{\mathrm{sensor}},
\qquad i=1,\ldots,8.
$$

传感器 token 与场 token 在上下文编码器之前拼接。当前实现使用固定等间隔传感器位置和单一采样时刻。

### 3.4 上下文编码器

上下文编码器为 4 层、4 个注意力头、宽度 128 的 Transformer：

$$
C=\Phi_{\mathrm{context}}
\left([T_f;T_{z_0};T_s]\right).
$$

当前实现属于早期 token 拼接与后期查询注意力的混合融合，而不是纯预融合或纯晚期融合。

### 3.5 查询解码器

查询向量由 Fourier 坐标特征、静态潜状态和查询点局部场特征组成：

$$
q(x,t)=\operatorname{MLP}
\left[
\gamma(x,t),z_0,u_0^{\mathrm{local}}(x)
\right].
$$

查询通过交叉注意力读取上下文，再经过前馈网络和输出头：

$$
\tilde{q}(x,t)=\operatorname{Attn}(q,C,C),
\qquad
\hat{u}(x,t)=\operatorname{Head}(\operatorname{FFN}(\tilde{q})).
$$

评测时全场查询按 4,096 个点分块处理。

### 3.6 参数规模

当前 XL 模型总参数量为 874,209。主要参数构成如下：

| 参数组 | 参数量 |
| --- | ---: |
| 上下文 Transformer | 529,920 |
| MLP 流形编码器 | 148,256 |
| 交叉注意力 | 66,048 |
| 前馈网络 | 65,920 |
| 查询投影 | 27,008 |
| 传感器投影 | 17,024 |
| Tokenizer 与 token 投影 | 11,136 |
| 归一化与输出头 | 4,673 |
| 嵌入参数 | 4,224 |
| 总计 | 874,209 |

![图 2 XL 模型参数构成](publication/assets/xl_parameter_breakdown_zh.png)

图 2 当前 XL 模型的参数构成。上下文 Transformer 占主要容量，流形编码器和交叉注意力次之。

## 4 数据、增强与训练

### 4.1 保存的数据划分

| 数据划分 | 文件 | 轨迹数 |
| --- | --- | ---: |
| 训练 | `burgers_augmented_train.npz` | 2,956 |
| 验证 | `burgers_augmented_dev.npz` | 633 |
| 测试 | `burgers_augmented_test.npz` | 635 |

虽然测试文件加载 635 条轨迹，但配置中的 `eval_samples=32`，因此保存的 train/dev/test 指标只使用前 32 条轨迹。论文不能把 635 写成实际评测样本数。

### 4.2 增强策略

`burgers_augmented_v1` 的生成脚本组合了空间翻转、周期循环平移、时间窗口截取与插值、幅值缩放、小噪声注入和后半段时间反转。

其中，周期循环平移对周期 Burgers 方程是精确对称变换。空间翻转、幅值缩放、时间反转和时间窗插值会改变方程或解的物理含义，属于启发式数据扰动，而不是严格保持 PDE 的增强。当前数据集因此只能称为混合增强集。

### 4.3 目标函数

训练目标包含数据损失、初始条件损失、周期边界损失和 PDE 残差损失：

$$
\mathcal{L}
=\mathcal{L}_{\mathrm{data}}
+\mathcal{L}_{\mathrm{IC}}
+\mathcal{L}_{\mathrm{BC}}
+\lambda_{\mathrm{PDE}}\mathcal{L}_{\mathrm{PDE}},
\qquad
\lambda_{\mathrm{PDE}}=0.1.
$$

验证分数使用

$$
s_{\mathrm{dev}}
=L_2^{\mathrm{dev}}
+0.1\,\mathrm{PDE}_{\mathrm{RMS}}^{\mathrm{dev}}.
$$

### 4.4 优化配置

| 配置项 | 当前值 |
| --- | ---: |
| Preset | wide |
| `d_model` | 128 |
| `n_layers` | 4 |
| `n_heads` | 4 |
| Field tokens | 32 |
| Latent dimension | 32 |
| Manifold hidden | 256 |
| Sensors | 8 |
| Batch size | 32 |
| Steps | 24,000 |
| Learning rate | 1e-3 |
| LR decay steps | 12,000 |
| Warmup | 0 |
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| Dropout | 0.1 |
| Physics weight | 0.1 |
| Validation physics weight | 0.1 |
| Gradient clip | 1.0 |
| Eval interval | 500 |
| Early-stop patience | 4 |
| Early-stop minimum fraction | 0.75 |
| Seed | 42 |

学习率在第 12,000 步到达最低比例，而训练继续到 24,000 步。最佳验证检查点出现在第 22,000 步，训练结束前触发早停。

### 4.5 评价指标

相对 L2 为

$$
L_2^{\mathrm{rel}}
=\frac{1}{M}\sum_{m=1}^{M}
\frac{\lVert\hat{u}_m-u_m\rVert_2}
{\lVert u_m\rVert_2}.
$$

PDE RMS 在随机采样点上通过自动微分计算。由于查询点和评测样本集都需要固定后才能严格比较，本文把它作为当前运行的项目内诊断量。

## 5 结果

### 5.1 主实验结果

| 指标 | 保存值 |
| --- | ---: |
| 参数量 | 874,209 |
| 训练相对 L2 | 0.201059 |
| 验证相对 L2 | 0.241361 |
| 测试相对 L2 | 0.246087 |
| 测试 PDE RMS | 0.001059 |
| 最佳验证步数 | 22,000 |
| 完成步数 | 24,000 |
| 训练耗时 | 1,606.96 秒 |

测试 L2 高于训练 L2，差值为 0.0450，当前诊断仍为欠拟合。PDE 残差非常低，但场误差并不优于旧版数据驱动原型。

### 5.2 与旧版 v11 的描述性对照

| 指标 | 旧版 v11 | 当前 XL |
| --- | ---: | ---: |
| 参数量 | 341,026 | 874,209 |
| 测试相对 L2 | 0.115319 | 0.246087 |
| 测试 PDE RMS | 1.038709 | 0.001059 |

当前 XL 的 PDE 残差比旧版降低约 99.9%，但测试 L2 增加约 113%。两组使用不同数据和不同模型，因此不能把差异归因于模型扩容。

![图 3 当前模型以更高 L2 换取极低 PDE 残差](publication/assets/xl_result_tradeoff_zh.png)

图 3 旧版 v11 与当前 XL 的描述性对照。左图为测试 L2，右图为 PDE RMS。两组数据和架构不同，不构成匹配消融。

### 5.3 训练过程

![图 4 XL 训练收敛曲线](publication/assets/xl_training_curves_zh.png)

图 4 seed 42 XL 运行的验证相对 L2 和验证 PDE RMS。曲线来自保存的 500 步间隔训练历史。

验证 L2 在前期快速下降，约 10,000 步后进入低变化区间。PDE 残差在训练中期波动明显，说明检查点选择对物理诊断仍然敏感。

### 5.4 测试预测

![图 5 XL 模型代表性测试预测](experiments/v_aug3x_xl_d128_l4/test_prediction.png)

图 5 XL 模型 seed 42 的代表性测试场预测。该图仅展示第一条可视轨迹。

### 5.5 物理权重与 Dropout 的初步敏感性

另一次保存运行把训练物理权重从 0.1 提高到 0.15，同时把 Dropout 从 0.1 降低到 0.08：

| 配置 | Test L2 | Test PDE RMS | 最佳步数 |
| --- | ---: | ---: | ---: |
| physics=0.10, dropout=0.10 | 0.246087 | 0.001059 | 22,000 |
| physics=0.15, dropout=0.08 | 0.223471 | 0.001062 | 22,500 |

第二个配置的测试 L2 相对降低约 9.2%，PDE RMS 仅增加约 0.3%。但该比较同时改变两个超参数，并且每个配置只有一次存档运行，因此只能作为后续实验线索。

![图 6 物理权重与 Dropout 的初步敏感性](publication/assets/xl_physics_sensitivity_zh.png)

图 6 两个已保存 XL 配置的测试指标。变化同时涉及物理权重和 Dropout，不能分离单一因素。

### 5.6 与旧版架构的区别

| 项目 | 旧版 d64 原型 | 当前 XL |
| --- | ---: | ---: |
| 模型宽度 | 64 | 128 |
| Transformer 层数 | 3 | 4 |
| 参数数量 | 336,546 或 341,026 | 874,209 |
| 潜空间 ODE | 启用 | 未启用 |
| 流形 CNN 支路 | 启用 | 未启用 |
| 输入高通支路 | 启用 | 未启用 |
| 传感器 | 0 或实验性 8 | 固定 8 |
| 主数据 | `burgers_custom_v1` | `burgers_augmented_v1` |
| 物理权重 | 0.0 | 0.1 |

旧版信息只作为历史对照保留，不再描述当前方法。

## 6 可复现性与证据边界

当前主实验入口为

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

当前证据边界包括：

1. 主实验只有一个随机种子。
2. 测试指标只在 32 条轨迹上计算，而不是全部 635 条。
3. 混合增强包含非精确物理变换。
4. 与旧版 v11 的数据和架构均不匹配。
5. 物理权重与 Dropout 的敏感性实验同时改变两个因素。
6. 尚未完成同数据、同预算的 64/128 宽和 3/4 层匹配消融。

## 7 结论

当前论文的主模型应描述为 128 宽、4 层、874,209 参数的 XL 多模态神经算子。该模型在 seed 42 保存运行中取得测试相对 L2 0.2461 和 PDE RMS 0.001059，并表现出明显的精度与物理残差权衡。旧版 64 宽三层模型只应作为历史参照，不能继续出现在当前方法章节。

现有证据支持“当前 XL 配置能够稳定训练并显著降低 PDE 残差”，但不支持“在全部指标上优于旧版”或“架构扩容已被匹配实验证明为唯一原因”。后续最重要的实验不是继续增加容量，而是补齐同数据、同评测样本、同优化预算的宽度和层数消融。

## 数据与代码可用性

- 主训练入口：`code/demo_burgers_dataset.py`
- 当前数据：`datasets/burgers_augmented_v1/`
- 主运行：`experiments/v_aug3x_xl_d128_l4/`
- 敏感性运行：`experiments/v_aug3x_xl_physics15/`
- 当前指标：`experiments/v_aug3x_xl_d128_l4/metrics.json`

## 参考文献

[1] Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, and A. Anandkumar. Fourier Neural Operator for Parametric Partial Differential Equations. International Conference on Learning Representations, 2021.

[2] L. Lu, P. Jin, G. Pang, Z. Zhang, and G. E. Karniadakis. Learning Nonlinear Operators via DeepONet Based on the Universal Approximation Theorem of Operators. Nature Machine Intelligence, 3:218-229, 2021.

[3] N. Kovachki, Z. Li, B. Liu, K. Azizzadenesheli, K. Bhattacharya, A. Stuart, and A. Anandkumar. Neural Operator: Learning Maps Between Function Spaces with Applications to PDEs. Journal of Machine Learning Research, 24(89):1-97, 2023.

[4] M. Takamoto, T. Praditia, R. Leiteritz, D. MacKinlay, F. Alesiani, D. Pfluger, and M. Niepert. PDEBench: An Extensive Benchmark for Scientific Machine Learning. NeurIPS Datasets and Benchmarks Track, 2022.

[5] M. Raissi, P. Perdikaris, and G. E. Karniadakis. Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations. Journal of Computational Physics, 378:686-707, 2019.

[6] S. Wang, H. Wang, and P. Perdikaris. Learning the Solution Operator of Parametric Partial Differential Equations with Physics-Informed DeepONets. Science Advances, 7(40):eabi8605, 2021.

[7] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. Attention Is All You Need. Advances in Neural Information Processing Systems, 2017.

[8] A. Baltrušaitis, C. Ahuja, and L.-P. Morency. Multimodal Machine Learning: A Survey and Taxonomy. IEEE Transactions on Pattern Analysis and Machine Intelligence, 41(2):423-443, 2019.

[9] A. Krishnapriyan, A. Gholami, S. Zhe, R. Kirby, and M. W. Mahoney. Characterizing Possible Failure Modes in Physics-Informed Neural Networks. Advances in Neural Information Processing Systems, 2021.

[10] J. Pineau, P. Vincent-Lamarre, K. Sinha, V. Lariviere, A. Beygelzimer, F. d'Alche-Buc, E. Fox, and H. Larochelle. Improving Reproducibility in Machine Learning Research. Journal of Machine Learning Research, 22(164):1-20, 2021.

