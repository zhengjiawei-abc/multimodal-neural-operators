# V2 Training Analysis & Findings

## Issue: V2 Dataset NaN Loading

### v_test_bs16 Results (FAILED)
```
train relative L2: nan ← NaN across all metrics
dev   relative L2: nan
test  relative L2: nan
u range: train [nan, nan] ← Data loading failed
```

**Diagnosis:** Data format incompatibility
- v2 dataset contains mixed IC types and viscosity values (ν ∈ [0.005, 0.05])
- Model evaluation expects fixed ν=0.01 (baseline)
- NPZ format handling may have loading issue

**Root Cause:** 
When model loads v2 data, it tries to read as PDEBench format with fixed ν=0.01, but v2 has variable ν. The data loading pipeline assumes homogeneous dataset structure.

---

## Dataset Format Analysis

### v1 (Working)
```
Format: Parquet
Structure: Each file indexed by ν value (burgers_nu0.01_*.parquet)
Shapes: train(512, 256, 101, 1) dev(64, 256, 101, 1) test(128, 256, 101, 1)
Size: ~450 MB
Compatibility: Direct with demo_burgers_dataset.py
```

### v2 (Failed)
```
Format: NPZ (NumPy compressed)
Structure: Single files with mixed data
- burgers_augmented_train.npz (287 MB)
- burgers_augmented_dev.npz (39 MB)
- burgers_augmented_test.npz (57 MB)
Issue: Not recognized by standard PDEBench loader
```

---

## Path Forward

### Option A: Convert v2 to Parquet Format
- Split v2 data by ν value into separate parquet files
- Create ν-indexed structure matching v1
- Requires data restructuring

### Option B: Use v1 Dataset (Current)
- Repeat v1 training with reduced steps (18k)
- Establish clean baseline for future comparison
- Faster iteration

### Option C: Custom Data Loader
- Modify demo_burgers_dataset.py to handle NPZ format
- Support mixed-ν datasets
- Requires code changes + testing

---

## Immediate Action: v1_repeat

**Config:**
- Data: burgers_augmented_v1 (3 parquet files, ν=0.01)
- Steps: 18,000 (vs 24,000 original)
- Batch size: 32 (default)
- Output: v1_repeat

**Expected Result:**
- Train L2: ~0.20
- Test L2: ~0.24-0.246 (should match v1 baseline ± 5%)

This establishes a working baseline before attempting v2 integration.

---

## Lessons Learned

1. **Data Format Matters:** NPZ is not directly compatible with PDEBench loader
2. **Dataset Homogeneity:** Mixed-parameter datasets need special handling
3. **Validation First:** Check data loading before full training

## Next Steps (Post v1_repeat)

1. Verify v1_repeat produces expected metrics
2. If successful: Commit v1_repeat results
3. Plan v2 integration:
   - Option A: Convert to parquet format
   - Option B: Extend data loader for NPZ
   - Option C: Create parameter-aware training pipeline

