"""
Dataset expansion strategy and recommendations.
"""

# 💡 数据集扩展方案

## 当前瓶颈分析

### 为什么还有欠拟合?
1. **数据分布单一**
   - 训练集2,956个轨迹都来自同一初值分布
   - 只有3倍时间插值增强
   - 黏性系数ν固定为0.01

2. **测试集表现**
   - Train L2: 0.201 → Test L2: 0.246 (Gap 4.5%)
   - **问题不是过拟合，而是泛化能力受限**
   - PDE残差已优化到0.001（非常优秀）

---

## 📋 扩展策略

### Phase 1: 初始条件多样性
```
当前:  -sin(πx)  (固定)
扩展: 
  - 多频率正弦组合  (25%)
  - 高斯脉冲不同宽度 (20%)
  - 光滑阶跃tanh()   (20%)
  - 多模态混合       (20%)
  - 尖锐阶跃step()   (15%)
```

### Phase 2: 参数变化
```
当前: ν = 0.01 (固定)
扩展:
  - 低粘性:  ν ∈ {0.005, 0.008}  (20%)
  - 标准:    ν = 0.01            (50%)
  - 高粘性:  ν ∈ {0.015, 0.02, 0.03, 0.05}  (30%)
```

### Phase 3: 幅度范围
```
当前分布: 主要在 [0.3, 1.2]
扩展:
  - 低幅度 (<0.5):   25%
  - 中幅度 (0.5-1.0): 40%
  - 高幅度 (>1.0):   35%
```

---

## 🚀 实现步骤

### 快速测试 (500轨迹)
```bash
cd code
python expand_burgers_dataset_v2.py \
  --num-trajectories 500 \
  --output-dir ../datasets/burgers_augmented_test \
  --include-v1
```

### 完整扩展 (5000轨迹，3倍增长)
```bash
python expand_burgers_dataset_v2.py \
  --num-trajectories 5000 \
  --output-dir ../datasets/burgers_augmented_v2 \
  --include-v1
```

### 数据分析
```bash
python analyze_dataset_diversity.py \
  ../datasets/burgers_augmented_v1

python analyze_dataset_diversity.py \
  ../datasets/burgers_augmented_v2
```

---

## 📊 预期效果

| 指标 | v1 | v2 (预期) | 改善 |
|------|----|---------|----|
| 训练集规模 | 2,956 | ~9,000 | 3× |
| IC多样性 | 1种 | 6种 | 6× |
| 参数覆盖 | ν=0.01 | ν∈[0.005,0.05] | 10× |
| Test L2 | 0.246 | **0.22-0.24** (预期) | ↓8-10% |
| PDE RMS | 0.0011 | ~0.0010 | ≈ 不变 |

---

## ⚙️ 训练配置建议

使用v2数据集训练:
```yaml
# 增加训练步数（数据量3倍）
--steps 36000          # 从24000 → 36000

# 可选：调整学习率衰减
--lr-decay-steps 18000  # 从12000 → 18000

# 其他参数保持不变
--d-model 128
--n-layers 4
--physics-weight 0.1
--dropout 0.1
```

---

## 📈 验证清单

- [ ] 生成v2数据集
- [ ] 分析数据多样性指标
- [ ] 使用v2训练模型 (36000步)
- [ ] 对比v1和v2的性能
- [ ] 检查Test L2是否下降
- [ ] 验证PDE残差稳定性

---

## 💬 预期结果

如果Test L2改善到 0.22-0.24 (当前0.246)：
- 数据多样性的效果得到验证 ✅
- 为进一步优化奠定基础
- 可考虑后续改进方向

如果没有显著改善 (<5% 改善)：
- 欠拟合不是主要问题
- 瓶颈可能在架构而非数据
- 需要考虑模型升级方案

---

生成脚本位置: `code/expand_burgers_dataset_v2.py`
分析脚本位置: `code/analyze_dataset_diversity.py`
