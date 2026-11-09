# Multimodal Neural Operator Technical Document

**Document type**: Engineering and reproducibility manual
**Version**: 1.0
**Date**: 8 November 2026
**Author**: Project maintainers
**Running Title**: Burgers operator implementation, evaluation, and handoff

---

## 1 Purpose and Scope

This document describes the implemented multimodal neural operator for the one-dimensional viscous Burgers equation. It is written for developers who need to run, inspect, extend, or audit the model. It covers the computation graph, data format, augmentation, training configuration, evaluation logic, retained results, known failure modes, and reproduction commands.

The system has two levels of maturity.

The field-only operator is supported by a three-seed, matched augmentation experiment. The original 1,408-trajectory training set and the 4,224-trajectory exact-shift set are compared under the same model and optimization budget.

The sensor path is implemented but not validated. A preliminary eight-sensor run underperforms its field-only reference. The current sensor objective contains no cross-modal alignment term and no modality dropout.

This document does not describe the structural-health-monitoring classifier, the video pipeline, or unrelated experiments in the workspace.

## 2 System Summary

| Item | Current value |
| --- | --- |
| PDE | 1D viscous Burgers |
| Domain | periodic \(x\in[0,1)\) |
| Resolution | 256 spatial points, 101 time points |
| Viscosities | 0.004, 0.006, 0.008, 0.01, 0.012 |
| Initial-field families | Fourier, mixture, weak high-frequency, weak spike, weak step |
| Main training file | `burgers_custom_train_curriculum2_shift3.parquet` |
| Main training samples | 4,224 |
| Development samples | 128 |
| In-distribution test samples | 128 |
| Backbone | CNN tokenizer, convolutional manifold, latent ODE, Transformer context, query decoder |
| Field-only parameters | 336,546 |
| Eight-sensor parameters | 341,026 |
| Main seeds | 11, 7, 23 |
| Main metric | full-field relative L2 |
| Diagnostic metric | random-query PDE RMS |

## 3 Repository Map

| Path | Role |
| --- | --- |
| `code/demo_burgers_dataset.py` | Model, training, evaluation, and CLI |
| `code/make_custom_shift_augment.py` | Exact periodic-shift data generator |
| `code/make_custom_burgers.py` | Base dataset generation utilities |
| `datasets/burgers_custom_v1/` | Parquet trajectories and metadata |
| `experiments/outputs_burgers_dataset/custom_v7_hp05_24k*` | Original-data three-seed controls |
| `experiments/outputs_burgers_dataset/custom_v13_shift3_24k*` | Exact-shift three-seed runs |
| `experiments/outputs_burgers_dataset/custom_v11_multimodal_sensors8_24k/` | Eight-sensor trial |
| `experiments/outputs_burgers_dataset/custom_v12_multimodal_prefusion_24k/` | Archived fusion-labelled run |
| `code/figures/fig1_architecture.png` | Architecture overview used in the paper |
| `publication/` | Current paper and technical documentation sources |

## 4 Data Contract

### 4.1 Trajectory shape

Each sample is a full space-time field with logical shape

$$
u\in\mathbb{R}^{256\times101}.
$$

The loader returns a tensor with shape `[batch, nx, nt, 1]`. The first time slice is used as the initial field \(u_0\), and the full field is the prediction target.

### 4.2 Split sizes

| File | Trajectories | Role |
| --- | ---: | --- |
| `burgers_custom_train_curriculum2.parquet` | 1,408 | Original training set |
| `burgers_custom_train_curriculum2_shift3.parquet` | 4,224 | Exact periodic-shift training set |
| `burgers_custom_dev.parquet` | 128 | Development and checkpoint selection |
| `burgers_custom_test_id.parquet` | 128 | In-distribution test |
| `burgers_custom_test_ood.parquet` | 128 | Step OOD file |
| `burgers_custom_test_ood2.parquet` | 128 | Spike OOD file |
| `burgers_custom_test_ood3.parquet` | 128 | High-frequency OOD file |

The OOD files exist, but the current `metrics.json` files used in the core paper table do not contain OOD metrics. OOD values should not be quoted from memory or copied from a secondary summary without re-running evaluation.

### 4.3 Metadata

Each Parquet file has a companion `.meta.json` array. Every record includes at least:

- `family`: initial-field family;
- `nu`: viscosity coefficient;
- augmentation fields such as `augmentation`, `shift`, and `copy` for shifted samples.

The original training split contains:

| Family | Count |
| --- | ---: |
| Fourier | 512 |
| Mixture | 512 |
| Weak high-frequency | 128 |
| Weak spike | 128 |
| Weak step | 128 |

The viscosity distribution is:

| Viscosity | Training count |
| ---: | ---: |
| 0.004 | 295 |
| 0.006 | 330 |
| 0.008 | 251 |
| 0.010 | 267 |
| 0.012 | 265 |

### 4.4 Data integrity checks

Before training, verify:

1. The selected train file exists under `--data-dir`.
2. The metadata array length equals the number of trajectories.
3. All fields are finite.
4. The first time slice is nonconstant for every trajectory.
5. Shifted metadata contains a valid integer `shift` in `[1, nx-1]`.

The current loader checks file loading and tensor shape. It does not perform a complete metadata-to-row audit or finite-value audit on every run.

## 5 Model Specification

![Figure 1. Implemented multimodal computation graph](publication/assets/architecture_en.png)

Figure 1. Implemented field, latent, sensor, context, and query paths.

### 5.1 Field path

The model constructs a high-pass channel

$$
h(u_0)=u_0-\operatorname{AvgPool}_{15}(u_0),
$$

where circular padding is used before the average pool. The CNN input is

$$
[u_0,\,0.5h(u_0)].
$$

The tokenizer contains:

1. `Conv1d(2, 16, kernel=5)` with circular padding;
2. SiLU activation;
3. `Conv1d(16, 32, kernel=5)` with circular padding;
4. SiLU activation;
5. adaptive average pooling to 32 positions;
6. linear projection to width 64.

### 5.2 Manifold path

The convolutional manifold autoencoder maps the initial field to a 32-dimensional latent state. It is pretrained for 1,500 reconstruction steps before the main operator objective. The main model adds manifold information in two ways:

- a linear projection of the latent state is added to every spatial token;
- a manifold CNN branch reshapes the latent state into a token field and injects the resulting features.

The manifold CNN branch contributes 8,320 parameters.

### 5.3 Context encoder

The context encoder is a standard pre-normalized Transformer encoder with:

| Setting | Value |
| --- | ---: |
| Width | 64 |
| Layers | 3 |
| Heads | 4 |
| Token positions | 32 field tokens |
| Extra tokens | manifold and optional sensor tokens |

The exact number of context tokens depends on whether the manifold CNN tokens are added as a residual and whether sensors are present. The encoder accepts variable token counts because no fixed positional table is applied after the initial field positions.

### 5.4 Temporal path

The latent state is integrated through four explicit Euler steps:

$$
z_{k+1}=z_k+\Delta t\,v_\theta(z_k),
\qquad
\Delta t=t/4.
$$

The vector field is a two-layer MLP with hidden width 64. It contributes 8,352 parameters.

### 5.5 Query decoder

The query representation concatenates:

- Fourier features of \(x\) and \(t\);
- the evolved latent state \(z(t)\);
- local initial-field features.

The decoder applies:

1. query projection to width 64;
2. cross-attention over context tokens;
3. residual normalization and feed-forward block;
4. output head `64 -> 32 -> 1`.

Evaluation processes full fields in chunks of 4,096 queries to avoid excessive memory use.

### 5.6 Sensor path

When `--sensor-count N` is positive, the model selects `N` equispaced spatial indices and a fixed time index determined by `--sensor-time`. Each sensor token is

$$
T_s^{(i)}=\operatorname{MLP}([x_i,t_s,u_i])+e_{\mathrm{sensor}}.
$$

The MLP is `3 -> 64 -> SiLU -> 64 -> 64`. Sensor tokens are concatenated with field tokens before the context Transformer.

The current implementation uses a fixed sensor layout for every sample. It does not randomize sensor positions, does not use modality dropout, and does not include a loss that explicitly aligns sensor values with the predicted field.

## 6 Loss and Optimization

### 6.1 Loss terms

| Term | Purpose | Main-run status |
| --- | --- | --- |
| Data | Sample predicted values at random field points | Active |
| Initial condition | Match prediction at \(t=0\) | Active |
| Periodic boundary | Match value and optional derivative across \(x=0,1\) | Active |
| PDE residual | Enforce Burgers residual under automatic differentiation | Weight 0.0 |
| Reconstruction | Preserve manifold reconstruction | Active through pretraining and regularizer |

The loss balancer scales active terms using a running reference. The core runs use:

- `loss_balance=True`;
- `loss_floor=1e-3`;
- `loss_balance_warmup=500`;
- `physics_weight=0.0`;
- `validation_physics_weight=0.1`.

### 6.2 Core configuration

| Parameter | Value |
| --- | ---: |
| `preset` | `wide` |
| `train_samples` | 4,224 |
| `dev_samples` | 128 |
| `test_samples` | 128 |
| `eval_samples` | 128 |
| `nx` | 256 |
| `nt` | 101 |
| `latent_dim` | 32 |
| `manifold_hidden` | 256 |
| `conv_ae_channels` | 16 |
| `cnn_channels` | 32 |
| `cnn_kernel_size` | 5 |
| `token_count` | 32 |
| `d_model` | 64 |
| `n_heads` | 4 |
| `n_layers` | 3 |
| `head_width` | 32 |
| `ode_steps` | 4 |
| `batch_size` | 32 |
| `steps` | 24,000 |
| `lr` | 1e-3 |
| `lr_decay_steps` | 24,000 |
| `warmup_steps` | 200 |
| `min_lr_ratio` | 0.02 |
| `dropout` | 0.1 |
| `grad_clip_norm` | 1.0 |
| `eval_every` | 1,000 |
| `early_stop_patience` | 100 |
| `manifold_steps` | 1,500 |

The stored configurations for the retained runs predate explicit optimizer and weight-decay fields. Re-running with the current `wide` preset defaults would use AdamW and weight decay. To reproduce the retained recipe, pass the optimizer and weight-decay settings explicitly as shown in Section 9.

### 6.3 Checkpoint rule

The training loop stores the state with the lowest validation score

$$
s_{\mathrm{dev}}=L_2^{\mathrm{dev}}
+0.1\,\mathrm{PDE}_{\mathrm{RMS}}^{\mathrm{dev}}.
$$

After training, the best state is restored before train, development, and test evaluation. The saved `model.pt` contains the same best state.

| Seed | Best step | Completed steps |
| ---: | ---: | ---: |
| 11 | 22,000 | 24,000 |
| 7 | 20,000 | 24,000 |
| 23 | 18,000 | 24,000 |

## 7 Exact Periodic-Shift Augmentation

### 7.1 Mathematical basis

For periodic Burgers dynamics, a spatial translation of an exact solution is another exact solution:

$$
u(x,t)\longrightarrow u(x-\delta,t).
$$

No amplitude, viscosity, boundary condition, or derivative operator changes. This makes the operation a symmetry transform rather than a heuristic perturbation.

### 7.2 Generator behavior

`code/make_custom_shift_augment.py` performs the following steps:

1. Load `burgers_custom_train_curriculum2.parquet`.
2. Read the companion metadata.
3. Preserve the original 1,408 fields.
4. For each of two extra copies, draw one random integer shift per trajectory.
5. Apply `numpy.roll(field, shift, axis=0)`.
6. Copy the metadata and add `augmentation`, `shift`, and `copy`.
7. Save the combined 4,224 trajectories and metadata.

The generator uses `numpy.random.default_rng(101)`.

### 7.3 Design constraints

- The domain must be periodic.
- Shift is applied to the full space-time field, not only the initial slice.
- Shift must remain an integer number of grid cells.
- The operation changes position coverage but not trajectory identity.
- Duplicate augmented samples are not independent physical simulations.

## 8 Evaluation

### 8.1 Relative L2

For each evaluated trajectory,

$$
e_m=\frac{\lVert\hat{u}_m-u_m\rVert_2}
{\lVert u_m\rVert_2}.
$$

The reported metric is `mean(e_m)` over the evaluated test trajectories.

### 8.2 PDE residual

The evaluator samples 128 random \((x,t)\) pairs, obtains \(\hat{u}\), computes first and second derivatives with automatic differentiation, and evaluates

$$
r=\hat{u}_t+\hat{u}\hat{u}_x-\nu\hat{u}_{xx}.
$$

The stored residual is \(\sqrt{\mathbb{E}[r^2]}\).

Two implementation issues require attention:

1. The evaluation query coordinates are not fixed by a saved seed.
2. The residual uses `cfg.nu`, currently 0.01, while the dataset contains multiple viscosities.

The residual is therefore a stability and sanity diagnostic, not a calibrated physical error.

### 8.3 Training diagnostics

`metrics.json` also contains:

- maximum, mean, and clipped gradient norms;
- loss-balancing references;
- best development score and step;
- stale-evaluation count;
- test-minus-training separation;
- manifold pretraining history;
- full training history.

## 9 Reproduction Commands

Run commands from the repository root.

### 9.1 Regenerate the exact-shift dataset

```powershell
python code\make_custom_shift_augment.py --extra-copies 2 --output-name burgers_custom_train_curriculum2_shift3.parquet
```

Expected summary:

```json
{
  "base": "burgers_custom_train_curriculum2.parquet",
  "augmentation": "exact periodic shift along x",
  "extra_copies": 2,
  "total": 4224
}
```

### 9.2 Train the field-only shifted model

```powershell
python code\demo_burgers_dataset.py `
  --data-dir datasets\burgers_custom_v1 `
  --train-file burgers_custom_train_curriculum2_shift3.parquet `
  --dev-file burgers_custom_dev.parquet `
  --test-file burgers_custom_test_id.parquet `
  --train-samples 4224 --dev-samples 128 --test-samples 128 `
  --preset wide --steps 24000 --lr-decay-steps 24000 `
  --batch-size 32 --lr 0.001 --warmup-steps 200 `
  --dropout 0.1 --input-highpass --highpass-scale 0.5 --highpass-kernel 15 `
  --manifold-cnn --latent-ode --query-local-features `
  --optimizer adam --weight-decay 0 --manifold-weight-decay 0 `
  --physics-weight 0 --validation-physics-weight 0.1 `
  --loss-balance --loss-balance-warmup 500 `
  --manifold-steps 1500 --eval-every 1000 --early-stop-patience 100 `
  --save-model --seed 11 `
  --output-dir experiments\outputs_burgers_dataset\repro_shift3_seed11
```

Repeat with `--seed 7` and `--seed 23`, changing the output directory.

### 9.3 Train the original-data control

Use the same command with:

```text
--train-file burgers_custom_train_curriculum2.parquet
--train-samples 1408
```

Keep the seed, model, and optimization settings matched. Do not compare an original-data run trained with one optimizer against a shifted run trained with another.

### 9.4 Train with eight sensors

Add:

```text
--sensor-count 8
--sensor-time 0.5
```

Because online shift and sensors are currently incompatible, do not pass `--online-shift-augment`.

## 10 Retained Results

### 10.1 Three-seed matched comparison

| Seed | Original L2 | Shifted L2 | Relative change |
| ---: | ---: | ---: | ---: |
| 11 | 0.097000 | 0.090097 | -7.12% |
| 7 | 0.102857 | 0.095846 | -6.82% |
| 23 | 0.110597 | 0.105622 | -4.50% |
| Mean | 0.103485 | 0.097188 | -6.09% |

The shifted mean standard deviation is 0.007849 across seeds.

### 10.2 PDE residual

| Seed | Original PDE | Shifted PDE |
| ---: | ---: | ---: |
| 11 | 0.838240 | 0.855308 |
| 7 | 1.354726 | 1.738279 |
| 23 | 1.524887 | 1.471190 |
| Mean | 1.239284 | 1.354926 |

The residual does not improve consistently. This outcome should remain visible in future reports.

### 10.3 Sensor trial

| Seed 11 | Field only | Eight sensors |
| --- | ---: | ---: |
| Test L2 | 0.097000 | 0.115319 |
| Test PDE RMS | 0.838240 | 1.038709 |
| Parameters | 336,546 | 341,026 |

The sensor branch adds 4,480 parameters and worsens both recorded metrics in this run.

### 10.4 Archived fusion-labelled runs

The saved configurations for `custom_v11_multimodal_sensors8_24k` and `custom_v12_multimodal_prefusion_24k` are identical field by field. Both contain 341,026 parameters and matching checkpoint signatures. The recorded outputs differ:

| Metric | `custom_v11` | `custom_v12` |
| --- | ---: | ---: |
| Test L2 | 0.115319 | 0.114495 |
| Test PDE | 1.038709 | 1.795859 |
| Training time (s) | 1,597.907 | 1,374.561 |

These differences cannot be attributed to fusion mode from the retained evidence.

## 11 Known Issues and Risks

### 11.1 Reproduction risk from historical configuration drift

The current `DatasetConfig` defaults and `wide` preset have changed since some runs were produced. A command that relies only on `--preset wide` will not reproduce an archived run. Always record and pass explicit optimizer, dropout, warmup, and loss settings.

### 11.2 Missing architecture manifest

Runs save `metrics.json` and optionally `model.pt`, but they do not save a manifest containing module names, parameter-shape hashes, code hash, and fusion mode. Directory names are currently used as labels, which is insufficient for causal comparison.

### 11.3 Random PDE evaluation

The residual changes with random query locations. It should not be compared across runs unless the query set is fixed or the sampling variance is estimated.

### 11.4 Nominal viscosity in residual

The PDE evaluator uses a single `cfg.nu` value. The dataset spans multiple viscosities. Use the per-sample metadata before treating the residual as a physical error.

### 11.5 Fixed sensor geometry

The sensor branch uses deterministic equispaced locations and one fixed sampling time. A model can learn location-specific shortcuts. The branch should be tested with randomized layouts, multiple sampling times, missing sensors, and noise.

### 11.6 No multimodal alignment

The sensor token is concatenated but not explicitly trained to agree with the field prediction. Add the alignment objective from the paper before evaluating whether sensors provide a benefit.

### 11.7 Single PDE

The code contains grid and spectral decoder variants, but the retained evidence concerns the query decoder on one-dimensional periodic Burgers data. Do not transfer the result to two-dimensional flow, irregular domains, or non-periodic boundaries without a new experiment.

## 12 Handoff Checklist

### Before a run

- [ ] Select the exact Parquet file and sample count.
- [ ] Verify metadata length and augmentation fields.
- [ ] Record the source hash.
- [ ] Record all non-default CLI options.
- [ ] Confirm train, development, and test files are distinct.
- [ ] Confirm sensor count and sampling time.
- [ ] Confirm physics weight and validation physics weight separately.

### During a run

- [ ] Check finite losses after the first 100 steps.
- [ ] Check gradient clipping frequency.
- [ ] Check that best development step is not at step zero.
- [ ] Monitor GPU memory.

### After a run

- [ ] Save `metrics.json`.
- [ ] Save `model.pt` if the run may be analyzed later.
- [ ] Verify parameter count.
- [ ] Save the exact command line.
- [ ] Save an architecture fingerprint.
- [ ] Compare against matched seeds and budgets.
- [ ] Report negative metrics, not only the best metric.

## Appendix A Data Provenance Fields

Each shifted metadata record contains the original base record plus:

```json
{
  "augmentation": "periodic_shift",
  "shift": 123,
  "copy": 1
}
```

The `copy` index identifies the augmented copy. The `shift` value is an integer grid offset. Neither field changes the underlying physical coefficient.

## Appendix B Output Files

| File | Content |
| --- | --- |
| `metrics.json` | Configuration, metrics, stability, histories |
| `model.pt` | Best state dict, config, parameter count |
| `training_curve.png` | Training and development curves |
| `test_prediction.png` | First test trajectory prediction |

The prediction figure is a visual diagnostic, not a complete error report. Keep `metrics.json` with every figure.

