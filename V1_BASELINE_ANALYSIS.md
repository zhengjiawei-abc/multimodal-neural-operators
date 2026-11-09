# V1 Baseline Analysis & V2 Expectations

## V1 Final Results (Baseline)

### Test Metrics
```
Train L2 Error:    0.2011
Test L2 Error:     0.2461  ← Target metric
PDE RMS Error:     0.001059
Generalization Gap: 0.245 (4.5%)
```

### Model Architecture
- d_model: 128
- n_layers: 4
- dropout: 0.1
- physics_weight: 0.1
- Total params: ~520K

### Training Config
- Optimizer: AdamW (lr=0.001)
- Batch size: 32
- Total steps: 24,000
- Best step: ~18,000
- Convergence: Smooth, no oscillations

### Dataset Used (v1)
- Train: 2,956 samples
- Dev: 633 samples
- Test: 635 samples
- **Single IC type:** smooth sine wave
- **Fixed ν:** 0.01 only
- Format: Parquet

---

## V2 Extended Dataset

### Enhanced Coverage

**IC Types** (6 variants, 20% each + 20% sine)
```
1. sine(2πx)           - baseline, smooth
2. gaussian(x)         - smooth, localized
3. sine(2πx) + sine(4πx) - multi-frequency
4. smooth_step(x)      - smooth boundary
5. step(x)             - sharp boundary
6. triangle(x)         - periodic sawtooth
```

**Viscosity Range** (6 values)
```
ν = 0.005  → Low viscosity (17%)   - weak diffusion
ν = 0.01   → Standard (50%)         - baseline
ν = 0.015  → Low-mid (11%)
ν = 0.02   → Mid (11%)
ν = 0.03   → Mid-high (6%)
ν = 0.05   → High viscosity (5%)   - strong diffusion
```

### V2 Dataset Stats
- Train: 3,544 (+20%)
- Dev: 472
- Test: 708 (+12%)
- Format: NPZ (compressed)

---

## Expected Improvements

### Conservative Estimate (5% gain)
```
Test L2: 0.2461 → 0.2337 ✓
Improvement: 0.0124 error reduction
```

### Baseline Estimate (7% gain)
```
Test L2: 0.2461 → 0.2288 ✓
Improvement: 0.0173 error reduction
```

### Optimistic Estimate (10% gain)
```
Test L2: 0.2461 → 0.2215 ✓✓
Improvement: 0.0246 error reduction
```

### Why V2 Should Improve

1. **20% More Data**
   - Better statistical coverage
   - More gradient signal
   - Reduced overfitting risk

2. **6× IC Diversity**
   - Models learns richer feature representations
   - Better feature extraction from initial conditions
   - Improved transfer to unseen patterns

3. **6× Parameter Coverage (ν)**
   - Learns viscosity-dependent dynamics
   - Better extrapolation to new ν values
   - More robust PDE residual behavior

4. **Implicit Regularization**
   - Higher IC variety forces generalizable patterns
   - Reduces memorization of dataset-specific features
   - Better inductive bias for Burgers equation

---

## Success Criteria

### Tier 1: Confirmed Success (must achieve)
- [ ] Test L2 < 0.246 (any improvement)

### Tier 2: Meaningful Improvement
- [ ] Test L2 < 0.234 (5% gain) ✓ Likely
- [ ] Train/Test gap < 4.5%

### Tier 3: Strong Validation
- [ ] Test L2 < 0.225 (8% gain) ✓✓ Good sign
- [ ] PDE RMS ≤ 0.001000

### Tier 4: Breakthrough
- [ ] Test L2 < 0.215 (12% gain) ✓✓✓ Excellent
- [ ] Consistent convergence

---

## Key Questions to Answer

1. **Does IC diversity help?**
   - Compare error patterns across IC types
   - Check if model performs better on seen vs unseen IC types

2. **Does ν coverage improve generalization?**
   - Measure error correlation with ν value
   - Test extrapolation to ν not in training set

3. **Is there a scale effect?**
   - Would even more data help further?
   - What's the saturation point?

4. **Which factors matter most?**
   - IC diversity vs data volume vs ν range
   - Design next experiment accordingly

---

## V2 Training Configuration

**Same as V1:**
- d_model: 128
- n_layers: 4
- dropout: 0.1
- physics_weight: 0.1
- optimizer: AdamW
- batch_size: 32

**Adjusted for efficiency:**
- steps: 18,000 (75% of v1)
  - Reasoning: More data, less overfitting risk
  - Will exit early if validation plateaus
- data_dir: burgers_augmented_test (v2 copy)
- format: NPZ (same as generation)

---

## Prediction Summary

| Metric | V1 Baseline | V2 Prediction | Confidence |
|--------|-------------|---------------|-----------|
| Test L2 | 0.2461 | 0.225-0.235 | High ✓ |
| Train L2 | 0.2011 | 0.185-0.195 | High ✓ |
| Gap | 4.5% | 3.5-4.5% | Medium |
| PDE RMS | 0.001059 | ~0.001000 | Medium |
| Convergence Step | 18,000 | 14,000-16,000 | Low |

**Overall Success Probability:** 75-85%

