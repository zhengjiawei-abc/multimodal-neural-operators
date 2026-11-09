# V2 数据集兼容性修复

## 问题识别

### 数据质量问题
原V2数据集包含大量NaN/Inf值：
- NaN: 8.1% (7.46M / 91.6M 值)
- Inf: 0.35% (320K / 91.6M 值)
- 损坏样本: 10.5% (373 / 3544 train样本)

**根本原因：** 数据生成过程中，某些参数组合导致Burgers方程数值求解器发散

### 症状
```
train relative L2: nan ← 所有指标都是NaN
dev   relative L2: nan
test  relative L2: nan
u range: train [nan, nan]
```

---

## 解决方案

### 步骤1: 清理数据集 ✅
移除包含NaN/Inf的样本

**清理结果：**
```
Train: 3544 → 3171 (移除373, -10.5%)
Dev:   472  → 423  (移除49,  -10.4%)
Test:  708  → 630  (移除78,  -11.0%)

输出文件:
- burgers_augmented_train_clean.npz (295 MB)
- burgers_augmented_dev_clean.npz (39 MB)
- burgers_augmented_test_clean.npz (59 MB)
```

### 步骤2: 修改数据加载器 ✅
更新 `demo_burgers_dataset.py` 中的 `load_split()` 函数

**改进：**
- 自动检测 `_clean` 版本文件
- 如果存在clean版本，优先加载
- 保持向后兼容性

**代码片段：**
```python
# 尝试加载_clean版本（如果存在）
clean_path = path.parent / (path.stem + '_clean.npz')
if clean_path.exists():
    path = clean_path
    print(f"  using cleaned version: {path.name}")
```

### 步骤3: 重新训练 🔄
两条并行线：

**配置A: v2_clean_fixed**
- Data: burgers_augmented_test (clean版本)
- Steps: 18,000
- Batch: 32
- Status: 训练中...

**配置B: v1_repeat** (基准对照)
- Data: burgers_augmented_v1
- Steps: 18,000
- Batch: 32
- Status: 已启动

---

## 数据清理详情

### 损坏模式分析
损坏样本均匀分布在三个数据集中（10-11%比例）

**可能原因：**
1. 特定的初始条件类型 (IC) 在某些ν值下不稳定
2. 时间步长太大，导致求解器发散
3. 边界条件处理不当

### 清理策略
- ✅ 简单保守策略：移除所有包含NaN/Inf的样本
- ✗ 不使用插值/填充（可能引入伪数据）
- 最终数据集：89-90%的原始样本保留

---

## 预期效果

### V2 Clean数据集优势
尽管样本减少（-10.5%），但质量提升：

1. **数据完整性** ✓
   - 0% NaN/Inf
   - 所有样本可用于训练

2. **IC多样性保留** ✓
   - 6种IC类型保留
   - 多ν值覆盖保留

3. **有效样本量**
   - Train: 3171 vs v1的 2956 (+7%)
   - Dev: 423 vs v1的 633 (-33%)
   - Test: 630 vs v1的 635 (-1%)

### 预期性能对比

| 指标 | V1基线 | V2Clean预期 | 改善 |
|------|--------|------------|------|
| Test L2 | 0.2461 | 0.23-0.24 | ✓ 5-8% |
| Train L2 | 0.2011 | 0.18-0.20 | ✓ 0-10% |
| PDE RMS | 0.00106 | ~0.001 | ≈ 稳定 |
| 收敛 | 快速 | 稳定 | ✓ 改善 |

---

## 技术细节

### 修改文件
- `code/demo_burgers_dataset.py` - load_split()函数
- `clean_v2_dataset.py` - 清理脚本

### 生成clean文件
```bash
python clean_v2_dataset.py
# 输出3个_clean.npz文件
```

### 自动检测机制
```python
# 在load_split()中自动执行
clean_path = path.parent / (path.stem + '_clean.npz')
if clean_path.exists():
    path = clean_path  # 自动切换到clean版本
```

---

## 状态

| 任务 | 状态 | 时间 |
|------|------|------|
| 数据质量分析 | ✅ 完成 | 已识别NaN/Inf |
| 数据清理脚本 | ✅ 完成 | clean文件已生成 |
| 代码修改 | ✅ 完成 | load_split() 已更新 |
| v2_clean_fixed训练 | 🔄 运行中 | 18K步，预计25min |
| v1_repeat对照 | ⚠️ 检查中 | 需要验证 |

---

## 下一步

1. ✓ 等待v2_clean_fixed完成
2. 对比v1_repeat和v2_clean_fixed结果
3. 验证IC多样性带来的改善
4. 分析改进因素 (样本数 vs IC多样性 vs ν覆盖)

