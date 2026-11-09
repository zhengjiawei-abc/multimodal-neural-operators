# Training Status & Memory Optimization

## Current Run: v_test_bs16
- **Command:** `--batch-size 16 --steps 12000`
- **Status:** Running
- **Reason:** v_test_final failed with CUDA OOM on bs=32
- **GPU Capacity:** 8GB total, ~6.7GB available
- **Expected Duration:** 20-25 minutes

---

## Memory Analysis

### Failed Config (v_test_final)
```
batch_size: 32
steps: 18,000
error: torch.OutOfMemoryError
  allocated: 183.06 MiB (PyTorch)
  reserved: 36.94 MiB (unallocated)
  model: 102,921 params
  attempt: 20 MiB allocation → FAILED
```

**Root Cause:**
- Dataset size: 3,544 training samples (v2)
- vs v1: 2,956 samples
- 20% more data loaded into GPU memory for physics sampling
- Batch size 32 × larger dataset = OOM

### Optimized Config (v_test_bs16)
```
batch_size: 16 (↓50%)
steps: 12,000 (↓33%)
model: Same 102,921 params
expected GPU usage: ~4-5 GB
margin: ~2 GB safe
```

**Trade-offs:**
- ✓ Fits in GPU memory
- ✓ Still uses full dataset
- ✓ More gradient updates (smaller batches)
- ~ Fewer total steps (early stopping still possible)

---

## Strategy

### Step 1: Complete v_test_bs16
- Collect metrics at bs=16, 12K steps
- Baseline for v2 performance
- Check for memory stability

### Step 2: If Successful
- Compare v1 (bs=32, 24K) vs v2 (bs=16, 12K)
- Account for different training configurations
- Extract insights about IC diversity and ν coverage

### Step 3: Next Iteration (if needed)
- Try v2 on v1 config (bs=32, higher steps)
  - Either reduce dataset or use mixed precision (fp16)
- Or try CPU mode for slower but complete run

---

## V1 vs V2 Running Configs

| Config | Batch | Steps | Dataset | Samples |
|--------|-------|-------|---------|---------|
| **v1** | 32 | 24,000 | v1 | 2,956 |
| **v2** | 16 | 12,000 | v2 | 3,544 |

**Normalization Factor:**
- Samples/step: v1 = 2,956/24K ≈ 0.123
- Samples/step: v2 = 3,544/12K ≈ 0.295 (2.4× more data per step)
- v2 gets more gradient signal despite fewer total steps

---

## Expected Outcome

Even with fewer total steps, v2 should show improvement because:
1. 20% more training data
2. Higher data-per-step ratio (0.295 vs 0.123)
3. More diverse initial conditions (6 types vs 1)
4. Broader viscosity coverage (ν ∈ [0.005, 0.05] vs fixed 0.01)

**Predicted Test L2:** 0.230-0.240 (5-7% improvement over 0.2461)

---

## Monitoring Checkpoints

- ✓ Model load: Complete
- ✓ Manifold pretraining: Complete (MSE: 0.0212)
- → Hybrid PINN training: In progress
- ← Checking memory stability in first 100 steps

Next update: When training completes or errors occur
