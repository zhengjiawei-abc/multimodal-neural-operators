# 多模态神经算子 (Multimodal Neural Operators)

用混合 PINN（物理信息神经网络 + 神经算子）学习一维 Burgers 方程的解算子：输入初值 u0(x)，输出整个时空场 u(x, t)。

```
∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²,   ν = 0.01,   网格 256 (x) × 101 (t)
```

## 模型

主模型是 [code/demo_burgers_dataset.py](code/demo_burgers_dataset.py) 中的 `BurgersHybridPINN`：

- **编码**：流形自编码器（先单独预训练）+ CNN tokenizer，把 u0 压成 latent 和一组 context token
- **时间演化**：latent ODE（`--latent-ode --temporal ode`）
- **解码**：查询点 (x, t) 经 Fourier 特征后与 context 做交叉注意力，再经 FFN 输出 u(x, t)
- **损失**：数据 MSE + 初值 + 边界 + PDE 残差（自动微分），各项由 loss balancer 自动缩放
- **正则化与优化**：dropout、AdamW 权重衰减、梯度裁剪、warmup + 余弦退火、按 dev 集早停

## 主要结果

L2 均为相对 L2 误差（越小越好）。

**分布内（`burgers_custom_v1`，独立 dev/test）**

| 实验 | 参数量 | Test L2 | Test PDE RMS |
|---|---|---|---|
| `custom_v14_shift3_drop005_24k`（d64, l3, dropout 0.05） | 336K | **0.0873** | 0.856 |

**分布外（`burgers_custom_pb_ood`，2026-11-11 修正划分后）**

| 实验 | Test L2（前 32 个） | Test L2（全部 128 个） | 说明 |
|---|---|---|---|
| d64, l3 | 0.3438 | — | 353K 参数 |
| d96, l4 | 0.2717 ~ 0.3048 | 0.2987 ± 0.0097 | 3 个 seed（7, 42, 123） |
| d96, l4 + 流态加权采样 | 0.2746 ~ 0.3018 | 0.3141 ± 0.0141 | 3 个 seed，L2 更差 |

分布外的误差约为分布内的 3 倍，这是目前的主要问题。

**PDEBench（`pdebench_burgers`，自带独立 dev 集）与 FNO 对比**

最好的配置是 `lowdata/opt5_n512_drop10_steps24k`（d64, l3, dropout 0.1, conv 流形自编码器, manifold-cnn, latent ODE, 训练时不加 physics loss, 24k 步, 336K 参数），用 `code/demo_burgers_resnet.py` 训练。

| 训练样本数 | FNO（87K 参数） | 混合 PINN | 混合 PINN 低多少 |
|---|---|---|---|
| 32 | 0.577 | 0.501 | 13% |
| 64 | 0.479 | 0.388 | 19% |
| 128 | 0.344 | 0.272 | 21% |
| 256 | 0.230 | 0.177 ± 0.001（3 个 seed） | 23% |
| 512 | 0.150（24k 步） | **0.130 ± 0.009**（8 个 seed） | 13% |

512 样本的 seed 7 模型在和 FNO 相同的前 128 个 test 样本上是 0.1167（`eval_pdebench_vs_fno.py`），与只评估前 32 个时的 0.1162 几乎一样。FNO 每档只有 1 个 seed，256 及以下只训练了 12k 步。更多样本（1024、2375）的对比正在跑。

`custom_v14_multi_dataset_pdebench_24k`（0.2138）等 v14 系列配置在 PDEBench 上明显更差，不是这个数据集上的最好结果。

**PB_ID（`burgers_custom_pb_id`）**

之前 PB_ID 的结果（Test L2 0.160 ~ 0.245）都是把 test 当 dev 选出来的，已作废。下面是干净划分（train/dev/test = 851/173/128）下重跑的结果，seed 123，配置与 PB_OOD 的干净对照相同：

| 配置 | 参数量 | Dev L2（全部 173 个） | Test L2（前 32 个） | Test L2（全部 128 个） | Test PDE RMS |
|---|---|---|---|---|---|
| d64, l3 | 353K | 0.2729 | 0.2319 | 0.2480 | 0.438 |
| d96, l4 | 606K | 0.1935 | 0.1763 | **0.1824** | 0.725 |

PB_ID 上 d96/l4 的 Test L2 比 d64/l3 低约 26%，和 PB_OOD 上的趋势一致（约 19%）。同样是 d96/l4，PB_ID 的 Test L2（0.18）明显低于 PB_OOD（0.30），这个差距就是分布外带来的误差。各只跑了一个 seed。

## 2026-11-11 的修正和结论

- **dev/test 划分修正**：PB_ID 和 PB_OOD 原来用 test 文件同时当 dev，checkpoint 是在 test 上挑的，之前这两个数据集的结果都偏乐观，已作废。现在从训练集切出独立 dev（train/dev/test = 851/173/128），三份之间没有重复样本，训练脚本也会拒绝 dev 与 test 相同的配置。
- **容量**：在干净划分下，d96/l4 的 L2 比 d64/l3 低约 19%，但过拟合更明显（gap 70% 对 41%）。
- **随机森林（u0 → 场的 PCA 系数）**：Test L2 0.70，比网络差很多，预剪枝也几乎没有改善过拟合。
- **LDA + K-means**：在 u0 上，LDA 方向比 PCA 更能区分输出场流态（ARI 0.825 对 0.262）；分布外 test 在 LDA 空间到最近簇中心的距离是 train 的 2.44 倍。按流态加权采样做了 3 个 seed 的配对比较：L2 在 3 个 seed 上都变差（平均 +0.015），PDE 残差中位数在 3 个 seed 上都下降（0.522 → 0.450，约 −14%，远大于基线跨 seed 标准差 0.005）。也就是说，加权用约 5% 的 L2 换来了更好的物理一致性，不是单纯的改进。PDE RMS 被少数激波点主导，跨 seed 波动很大，不能据此下结论。

## 已知限制

- 训练脚本默认 `eval_samples=32`，打印出的 L2 只用了每个划分的前 32 个样本；完整评估请用 `nn_predict_cache.py`。
- `--preset wide` 只读取 512 个训练样本，没有用满 851 个。
- 同一配置重跑，Test L2 波动约 0.006，PDE RMS 波动可达一倍，单次结果的小差异不可靠。
- 有限差分算的 PDE 残差在激波处很大（真值场自身约 5.5），不能拿来和模型自动微分的 PDE RMS 比较。

## 安装

```bash
git clone https://github.com/zhengjiawei-abc/multimodal-neural-operators.git
cd multimodal-neural-operators
pip install -r code/requirements.txt
pip install torch scikit-learn   # PyTorch 请按自己的 CUDA 版本安装
```

数据集文件（`datasets/` 下的 parquet/npz）和模型权重没有放进仓库，需要自行准备。

## 运行

所有命令都在仓库根目录执行。

```bash
# 1. 为 PB_ID / PB_OOD 生成独立的 dev 划分（只需一次）
python datasets_make_dev_split.py

# 2. 训练（示例：PB_OOD 上的 d96/l4，并保存模型）
python code/demo_burgers_dataset.py --preset wide \
  --data-dir datasets/burgers_custom_pb_ood \
  --train-file burgers_nu0.01_train_split.parquet \
  --dev-file burgers_nu0.01_dev_split.parquet \
  --test-file burgers_nu0.01_test.parquet \
  --d-model 96 --n-layers 4 --dropout 0.05 --physics-weight 0.05 \
  --latent-ode --temporal ode --cross-attention --input-highpass \
  --manifold-cnn --query-local-features \
  --steps 24000 --seed 123 --save-model \
  --output-dir experiments/outputs_burgers_dataset/my_run

# 3. 用保存的模型在全部样本上评估
python nn_predict_cache.py experiments/outputs_burgers_dataset/my_run/model.pt
```

加 `--quick` 可以只跑 60 步，用来检查环境。`--sample-weights path.npy` 可以按样本权重采样训练数据。

## 目录

| 路径 | 内容 |
|---|---|
| `code/demo_burgers_dataset.py` | 主模型、训练和评估 |
| `code/baselines_1d_burgers.py`、`code/train_fno_baseline.py` | FNO、DeepONet 等基线 |
| `code/burgers_solver.py` | Burgers 参考求解器 |
| `datasets_make_dev_split.py` | 生成独立 dev 划分并查重 |
| `pb_ood_clean_split_runs.py` | 干净划分下 d64 对 d96 的对照实验 |
| `lda_kmeans.py`、`make_regime_weights.py` | LDA + K-means 流态分析与采样权重 |
| `compare_regime_weighted.py`、`compare_regime_seeds.py` | 基线与加权模型的对比（单 seed / 多 seed） |
| `rf_common.py`、`rf_standalone.py` | 随机森林基线 |
| `experiments/` | 各实验的 `metrics.json` 和图（只提交了小文件） |
| `publication/` | 论文草稿和图 |

## 联系

zhengjiawei-abc · 1103493759@qq.com

最后更新：2026-11-11
