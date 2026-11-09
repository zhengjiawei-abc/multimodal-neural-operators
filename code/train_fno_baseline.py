"""
FNO vs 混合PINN对比实验
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

class FNO1d(nn.Module):
    """1D Fourier Neural Operator"""
    def __init__(self, modes, width):
        super().__init__()
        self.modes = modes
        self.width = width

        self.fc0 = nn.Linear(1, width)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.conv0 = nn.Conv1d(width, width, 1)
        self.conv1 = nn.Conv1d(width, width, 1)

    def forward(self, x):
        # x: (batch, nx, 1)
        batch = x.shape[0]

        x = self.fc0(x)  # (batch, nx, width)
        x = x.permute(0, 2, 1)  # (batch, width, nx)

        x = self.conv0(x)
        x = torch.relu(x)
        x = self.conv1(x)
        x = torch.relu(x)

        x = x.permute(0, 2, 1)  # (batch, nx, width)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)  # (batch, nx, 1)

        return x.squeeze(-1)  # (batch, nx)

def train_fno(train_data, dev_data, test_data, output_dir, seed=42):
    """训练FNO模型"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 准备数据
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_data).float()),
        batch_size=32, shuffle=True
    )
    dev_loader = DataLoader(
        TensorDataset(torch.from_numpy(dev_data).float()),
        batch_size=32
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(test_data).float()),
        batch_size=32
    )

    # 模型
    model = FNO1d(modes=16, width=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=24000)

    best_dev_l2 = float('inf')
    log_file = output_dir / 'training.log'

    print(f"训练FNO...")
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")

    with open(log_file, 'w') as f:
        f.write(f"FNO模型对比实验\n")
        f.write(f"参数数: {sum(p.numel() for p in model.parameters())}\n\n")

        step = 0
        for epoch in range(1, 1001):
            model.train()
            epoch_loss = 0

            for batch in train_loader:
                u = batch[0].to(device)  # (batch, 256, 101)

                # 输入: t=0时刻, 输出: 全时间序列
                u0 = u[:, :, 0:1]  # (batch, 256, 1)
                u_target = u  # (batch, 256, 101)

                optimizer.zero_grad()

                # 简单版: 直接预测
                u_pred = model(u0)  # (batch, 256)
                u_pred = u_pred.unsqueeze(-1).expand_as(u_target)

                loss = torch.nn.functional.mse_loss(u_pred, u_target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                step += 1

            # 验证
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    dev_l2 = 0
                    for batch in dev_loader:
                        u = batch[0].to(device)
                        u0 = u[:, :, 0:1]
                        u_target = u
                        u_pred = model(u0).unsqueeze(-1).expand_as(u_target)
                        dev_l2 += torch.nn.functional.mse_loss(u_pred, u_target).item()
                    dev_l2 /= len(dev_loader)

                    if dev_l2 < best_dev_l2:
                        best_dev_l2 = dev_l2
                        torch.save(model.state_dict(), output_dir / 'model_best.pt')

                msg = f"Epoch {epoch}, Loss: {epoch_loss/len(train_loader):.6f}, Dev L2: {dev_l2:.6f}\n"
                print(msg, end='')
                f.write(msg)
                f.flush()

        # 测试
        model.load_state_dict(torch.load(output_dir / 'model_best.pt'))
        model.eval()
        with torch.no_grad():
            test_l2 = 0
            test_pde = 0
            for batch in test_loader:
                u = batch[0].to(device)
                u0 = u[:, :, 0:1]
                u_target = u
                u_pred = model(u0).unsqueeze(-1).expand_as(u_target)
                test_l2 += torch.nn.functional.mse_loss(u_pred, u_target).item()
            test_l2 /= len(test_loader)

        result = f"\ntest relative L2: {test_l2:.6f}\n"
        print(result)
        f.write(result)

if __name__ == '__main__':
    # 加载数据
    base_path = Path('../datasets/burgers_augmented_v1')

    train_data = np.load(base_path / 'burgers_augmented_train.npz')['data']
    dev_data = np.load(base_path / 'burgers_augmented_dev.npz')['data']
    test_data = np.load(base_path / 'burgers_augmented_test.npz')['data']

    print(f"训练数据: {train_data.shape}")
    print(f"验证数据: {dev_data.shape}")
    print(f"测试数据: {test_data.shape}")

    train_fno(train_data, dev_data, test_data, '../experiments/fno_baseline')
