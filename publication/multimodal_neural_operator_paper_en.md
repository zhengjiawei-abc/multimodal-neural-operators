# A 128-Wide Four-Layer Multimodal Neural Operator for the Burgers Equation

**Document type**: Research manuscript, preprint draft
**Version**: 2.0
**Date**: 9 November 2026
**Author**: Project author information to be completed
**Running Title**: Current XL architecture, verified results, and evidence boundaries

---

## Abstract

This paper documents the current XL multimodal neural operator rather than the earlier 64-wide, three-layer prototype. The current model uses an MLP manifold encoder, a CNN tokenizer, eight fixed-time sensor tokens, a four-layer 128-wide Transformer context encoder, and a query decoder with cross-attention. It contains 874,209 parameters. Training uses `burgers_augmented_v1`, with 2,956 training trajectories, 633 development trajectories, and 635 test trajectories. The saved evaluation configuration computes the reported metrics on the first 32 trajectories.

In the archived seed-42 run, the best development checkpoint occurs at step 22,000. Training completes 24,000 steps in 1,606.96 seconds. The reported test relative L2 is 0.2461 and the test PDE RMS is 0.001059. Relative to the legacy v11 reference, PDE residual decreases by 99.9%, while test L2 increases from 0.1153 to 0.2461. The two runs use different data and architectures, so this is a descriptive tradeoff rather than a causal architecture comparison.

The paper also corrects two stale claims in earlier project documents. First, the current model is not the `d=64`, three-layer, 336,546-parameter prototype; it is the `d=128`, four-layer, 874,209-parameter XL model. Second, the actual augmented training split contains 2,956 trajectories rather than 4.2k, and the reported test metric uses 32 evaluated trajectories rather than all 635.

**Keywords**: neural operator; multimodal learning; Burgers equation; cross-attention; physics-informed learning; reproducibility

## 1 Introduction

Neural operators learn mappings between function spaces and can predict solutions at arbitrary space-time coordinates [1-4]. The present system additionally receives sparse sensor observations and applies a PDE residual objective, combining multimodal conditioning with physics-informed training. Earlier prototypes established the field encoder, latent manifold representation, and attention context. The current XL model increases width from 64 to 128, increases the context encoder from three to four layers, and uses eight sensors explicitly.

Project documents contain several historical configurations and conflicting metric summaries. To prevent legacy model details from being presented as the current method, this paper treats the saved `experiments/v_aug3x_xl_d128_l4/metrics.json` and its configuration as the primary evidence source. When a secondary summary conflicts with the saved artifact, the artifact takes precedence and the discrepancy is stated explicitly.

The contributions are fourfold:

1. A precise specification of the current XL multimodal operator and its parameter composition.
2. A report of the seed-42 metrics, training history, checkpoint rule, and runtime.
3. A preliminary sensitivity comparison involving physics weight and dropout.
4. A correction of stale architecture, dataset-size, and evaluation-size claims.

The paper does not claim that the current model dominates the legacy baseline on every metric. The evidence supports a tradeoff between field accuracy and PDE residual.

## 2 Problem Setting

The one-dimensional viscous Burgers equation is

$$
\frac{\partial u}{\partial t}
+u\frac{\partial u}{\partial x}
=\nu\frac{\partial^2 u}{\partial x^2},
\qquad x\in[0,1),\quad t\in[0,1],
$$

with periodic spatial boundaries. The stored fields contain 256 spatial points and 101 time points. The model receives an initial field \(u_0(x)\) and eight sensor observations sampled at \(t_s=0.5\). It predicts the solution at arbitrary query coordinates:

$$
\hat{u}(x,t)
=G_\theta\left(u_0,\{(x_i,t_s,u_i)\}_{i=1}^{8}\right)(x,t).
$$

The saved configuration records \(\nu=0.01\). The main data directory is `datasets/burgers_augmented_v1`.

## 3 Current XL Model

### 3.1 Overview

![Figure 1. Current XL multimodal neural operator](publication/assets/xl_architecture_en.png)

Figure 1. Current XL model. The paths comprise an MLP manifold encoder, a CNN tokenizer, sensor tokens, a four-layer 128-wide Transformer context encoder, and a query decoder with cross-attention.

### 3.2 Field encoder and manifold branch

The current configuration disables the input high-pass branch and the manifold CNN branch. A two-layer one-dimensional CNN tokenizer first produces local features, which are adaptively pooled to 32 positions and projected to width 128:

$$
T_f=\operatorname{Tokenize}(u_0),
\qquad
T_f\in\mathbb{R}^{32\times128}.
$$

The initial field is also encoded by an MLP manifold autoencoder into a 32-dimensional latent state:

$$
z_0=\operatorname{Enc}_{\mathrm{MLP}}(u_0),
\qquad
z_0\in\mathbb{R}^{32}.
$$

The latent projection is added to the field tokens. The current run uses `latent_ode=False`, so there is no latent-space time integration. The query decoder reads the static latent state \(z_0\). This differs from the earlier prototype architecture.

### 3.3 Sensor tokens

Each of the eight sensors is represented by position, sampling time, and value. A two-layer MLP projects the triple to width 128 and adds a learned type embedding:

$$
T_s^{(i)}
=\operatorname{MLP}([x_i,0.5,u_i])+e_{\mathrm{sensor}},
\qquad i=1,\ldots,8.
$$

Sensor tokens are concatenated with field tokens before the context encoder. The current implementation uses fixed equispaced positions and a single sampling time.

### 3.4 Context encoder

The context encoder is a four-layer Transformer with four heads and width 128:

$$
C=\Phi_{\mathrm{context}}
\left([T_f;T_{z_0};T_s]\right).
$$

This is hybrid token fusion: token-level concatenation followed by query-level attention.

### 3.5 Query decoder

The query vector combines Fourier coordinate features, the static latent state, and local initial-field features:

$$
q(x,t)=\operatorname{MLP}
\left[
\gamma(x,t),z_0,u_0^{\mathrm{local}}(x)
\right].
$$

The query attends to the context and is passed through a feed-forward block and output head:

$$
\tilde{q}(x,t)=\operatorname{Attn}(q,C,C),
\qquad
\hat{u}(x,t)=\operatorname{Head}(\operatorname{FFN}(\tilde{q})).
$$

Evaluation processes full-field queries in chunks of 4,096 points.

### 3.6 Parameter scale

The current XL model contains 874,209 parameters.

| Parameter group | Parameters |
| --- | ---: |
| Context Transformer | 529,920 |
| MLP manifold encoder | 148,256 |
| Cross-attention | 66,048 |
| Feed-forward block | 65,920 |
| Query projection | 27,008 |
| Sensor projection | 17,024 |
| Tokenizer and token projections | 11,136 |
| Normalization and output head | 4,673 |
| Embedding parameters | 4,224 |
| Total | 874,209 |

![Figure 2. XL parameter composition](publication/assets/xl_parameter_breakdown_en.png)

Figure 2. Parameter composition of the current XL model.

## 4 Data, Augmentation, and Training

### 4.1 Saved split sizes

| Split | File | Trajectories |
| --- | --- | ---: |
| Training | `burgers_augmented_train.npz` | 2,956 |
| Development | `burgers_augmented_dev.npz` | 633 |
| Test | `burgers_augmented_test.npz` | 635 |

Although the test file contains 635 trajectories, `eval_samples=32`. The reported train, development, and test metrics therefore use the first 32 trajectories. The paper does not claim that 635 trajectories were evaluated.

### 4.2 Augmentation

`burgers_augmented_v1` combines spatial flips, periodic rolls, time-window interpolation, amplitude scaling, small noise, and partial time reversal.

Periodic rolling is an exact symmetry for periodic Burgers data. Spatial flips, amplitude scaling, time reversal, and time-window interpolation alter the governing equation or the physical interpretation of a solution. They are heuristic perturbations, not exact PDE-preserving transformations. The data are therefore described as a mixed augmented set.

### 4.3 Objective

The objective combines data, initial-condition, periodic-boundary, and PDE residual losses:

$$
\mathcal{L}
=\mathcal{L}_{\mathrm{data}}
+\mathcal{L}_{\mathrm{IC}}
+\mathcal{L}_{\mathrm{BC}}
+\lambda_{\mathrm{PDE}}\mathcal{L}_{\mathrm{PDE}},
\qquad
\lambda_{\mathrm{PDE}}=0.1.
$$

The development score is

$$
s_{\mathrm{dev}}
=L_2^{\mathrm{dev}}
+0.1\,\mathrm{PDE}_{\mathrm{RMS}}^{\mathrm{dev}}.
$$

### 4.4 Optimization

| Setting | Value |
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
| Evaluation interval | 500 |
| Early-stop patience | 4 |
| Early-stop minimum fraction | 0.75 |
| Seed | 42 |

The learning rate reaches its minimum ratio at step 12,000 while training continues to 24,000. The best development checkpoint occurs at step 22,000.

### 4.5 Metrics

Relative L2 is

$$
L_2^{\mathrm{rel}}
=\frac{1}{M}\sum_{m=1}^{M}
\frac{\lVert\hat{u}_m-u_m\rVert_2}
{\lVert u_m\rVert_2}.
$$

PDE RMS is computed by automatic differentiation on sampled queries. Because the evaluation samples and query set must be fixed for strict comparison, it is treated as a project-internal diagnostic.

## 5 Results

### 5.1 Main run

| Metric | Saved value |
| --- | ---: |
| Parameters | 874,209 |
| Training relative L2 | 0.201059 |
| Development relative L2 | 0.241361 |
| Test relative L2 | 0.246087 |
| Test PDE RMS | 0.001059 |
| Best development step | 22,000 |
| Completed steps | 24,000 |
| Training time | 1,606.96 s |

The test L2 is higher than the training L2 by 0.0450, so the run is still diagnosed as underfit. The PDE residual is low, but field accuracy does not exceed the legacy data-driven prototype.

### 5.2 Descriptive comparison with legacy v11

| Metric | Legacy v11 | Current XL |
| --- | ---: | ---: |
| Parameters | 341,026 | 874,209 |
| Test relative L2 | 0.115319 | 0.246087 |
| Test PDE RMS | 1.038709 | 0.001059 |

PDE residual decreases by about 99.9%, while test L2 increases by about 113%. The data and architecture differ, so the comparison does not isolate the effect of model scale.

![Figure 3. Accuracy and PDE-residual tradeoff](publication/assets/xl_result_tradeoff_en.png)

Figure 3. Descriptive comparison between legacy v11 and current XL. The two runs are not a matched ablation.

### 5.3 Training behavior

![Figure 4. Development convergence of the XL model](publication/assets/xl_training_curves_en.png)

Figure 4. Development L2 and PDE RMS for the seed-42 XL run. Curves use the saved 500-step history.

Development L2 decreases quickly and then enters a low-change regime after about 10,000 steps. PDE residual fluctuates during training, so checkpoint selection remains sensitive to the physical diagnostic.

### 5.4 Test prediction

![Figure 5. Representative XL test prediction](experiments/v_aug3x_xl_d128_l4/test_prediction.png)

Figure 5. Representative test prediction from the seed-42 XL run.

### 5.5 Preliminary physics-weight and dropout sensitivity

A second saved run increases physics weight from 0.1 to 0.15 and decreases dropout from 0.1 to 0.08:

| Configuration | Test L2 | Test PDE RMS | Best step |
| --- | ---: | ---: | ---: |
| physics=0.10, dropout=0.10 | 0.246087 | 0.001059 | 22,000 |
| physics=0.15, dropout=0.08 | 0.223471 | 0.001062 | 22,500 |

The second configuration reduces test L2 by about 9.2% and changes PDE RMS by only about 0.3%. Because two hyperparameters change together and each configuration has one run, this is a hypothesis rather than an isolated ablation.

![Figure 6. Physics-weight and dropout sensitivity](publication/assets/xl_physics_sensitivity_en.png)

Figure 6. Two saved XL configurations. The comparison changes both physics weight and dropout.

### 5.6 Difference from the legacy architecture

| Item | Legacy d64 prototype | Current XL |
| --- | ---: | ---: |
| Model width | 64 | 128 |
| Transformer layers | 3 | 4 |
| Parameters | 336,546 or 341,026 | 874,209 |
| Latent ODE | enabled | disabled |
| Manifold CNN branch | enabled | disabled |
| Input high-pass branch | enabled | disabled |
| Sensors | 0 or exploratory 8 | fixed 8 |
| Main data | `burgers_custom_v1` | `burgers_augmented_v1` |
| Physics weight | 0.0 | 0.1 |

Legacy information is retained only as historical context, not as the current method.

## 6 Reproducibility and Evidence Boundaries

The main reproduction entry is

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

Evidence boundaries:

1. The main result uses one seed.
2. Metrics use 32 evaluated trajectories, not all 635.
3. The mixed augmentation includes non-exact physical transforms.
4. The legacy comparison changes data and architecture.
5. The physics/dropout sensitivity changes two factors at once.
6. Matched 64/128-width and 3/4-layer ablations are still missing.

## 7 Conclusion

The current model should be described as a 128-wide, four-layer multimodal neural operator with 874,209 parameters. In the saved seed-42 run it reaches a test relative L2 of 0.2461 and a PDE RMS of 0.001059, revealing a tradeoff between field accuracy and physical residual. The 64-wide, three-layer model is only a historical reference and must not be presented as the current architecture.

The evidence supports stable training and a large PDE-residual reduction. It does not support a claim that the current model dominates the legacy baseline on every metric or that width alone caused the change. The next priority is a matched width and depth ablation under identical data, evaluation samples, and optimization budget.

## Data and Code Availability

- Main training entry: `code/demo_burgers_dataset.py`
- Current data: `datasets/burgers_augmented_v1/`
- Main run: `experiments/v_aug3x_xl_d128_l4/`
- Sensitivity run: `experiments/v_aug3x_xl_physics15/`
- Current metrics: `experiments/v_aug3x_xl_d128_l4/metrics.json`

## References

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

