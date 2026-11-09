# 依赖安装指南

## 错误原因

```
ModuleNotFoundError: No module named 'pandas'
```

您的Python环境缺少必要的数据处理库。

---

## 快速修复

### 方法1：安装所有依赖（推荐）

```bash
# 在您的终端运行：
pip install pandas pyarrow numpy matplotlib seaborn tqdm
```

### 方法2：使用 requirements.txt

创建 `requirements.txt` 文件（我已为您创建，见下文），然后运行：

```bash
pip install -r requirements.txt
```

---

## 完整依赖列表

```
# 核心依赖
pandas>=2.0.0           # 数据处理
pyarrow>=12.0.0         # Parquet文件读写
numpy>=1.24.0           # 数值计算

# 可视化
matplotlib>=3.7.0       # 绘图
seaborn>=0.12.0        # 统计可视化

# 工具
tqdm>=4.65.0           # 进度条

# 深度学习（如果还没装）
torch>=2.0.0           # PyTorch
```

---

## 验证安装

安装完成后，运行以下命令验证：

```bash
python -c "import pandas; print('pandas:', pandas.__version__)"
python -c "import pyarrow; print('pyarrow:', pyarrow.__version__)"
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "import matplotlib; print('matplotlib:', matplotlib.__version__)"
python -c "import torch; print('torch:', torch.__version__)"
```

---

## 安装后重新运行

```bash
cd C:\Users\郑炜锋\Desktop\pinn_hybrid_demo_20261001\code
python quickstart_augment_train.py --quick-test
```

---

## 如果安装失败

### 使用国内镜像加速

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas pyarrow numpy matplotlib seaborn tqdm
```

### 如果pip版本过旧

```bash
python -m pip install --upgrade pip
pip install pandas pyarrow numpy matplotlib seaborn tqdm
```

### 如果权限不足

```bash
pip install --user pandas pyarrow numpy matplotlib seaborn tqdm
```

---

## 常见问题

### Q: 我有多个Python环境，应该装在哪个？

**A**: 检查当前使用的Python：

```bash
python --version
python -c "import sys; print(sys.executable)"
```

然后对该Python安装依赖：

```bash
# 使用完整路径安装
C:\Python312\python.exe -m pip install pandas pyarrow numpy matplotlib seaborn tqdm
```

### Q: conda环境怎么安装？

**A**: 如果使用conda：

```bash
conda install pandas pyarrow numpy matplotlib seaborn tqdm
```

或：

```bash
conda install -c conda-forge pandas pyarrow numpy matplotlib seaborn tqdm
```

---

## 检查清单

- [ ] 已运行 `pip install pandas pyarrow numpy matplotlib seaborn tqdm`
- [ ] 运行验证命令确认安装成功
- [ ] 重新运行 `python quickstart_augment_train.py --quick-test`
- [ ] 如果仍有错误，检查Python环境路径

---

安装完成后，再次运行快速测试即可！
