# V2 Clean Dataset Training Results

## 第一次运行: v2_clean_fixed

**配置：**
- 数据：清理后的V2数据集
- Train样本：3171 (从3544，移除NaN/Inf后)
- Steps：18,000
- Batch Size：默认(512 train)
- 启用early stopping

**结果：**
```
Train L2: 0.5507
Dev L2:   0.5581
Test L2:  0.5614
PDE RMS:  0.3836
Gap:      1.07%
诊断：underfit
早停：第15001步停止
总耗时：777.6秒（13分钟）
```

**分析：**
- ❌ 性能远低于预期 (期望~0.24，实际~0.56)
- 原因分析：
  1. 数据加载可能有问题 (只加载512样本而非3171)
  2. 或者模型未正确训练
  3. 或者clean数据对这个任务不兼容

**问题识别：**
- cfg.train_samples = 512 (默认值)
- 实际可用的train样本为3171，但只用了512个
- 这导致欠拟合(underfit)

## 第二次运行: v2_clean_working (当前)

**改进：**
- 明确指定train/dev/test文件为NPZ格式
- 应该自动检测并使用_clean版本
- 期望能正确加载所有3171个样本

**状态：** 🔄 运行中...

## 下一步调整

需要考虑：
1. 增加cfg.train_samples数值
2. 或者在命令行参数中指定样本数
3. 或者修改默认配置以支持更多样本
