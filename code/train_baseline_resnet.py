"""
简化版基线对比: 标准ResNet (无物理约束)
"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import time

class SimpleResNet(nn.Module):
    """简单ResNet作为数据驱动基线"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(64, 32, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 1, 5, padding=2),
        )

    def forward(self, x):
        # x: (batch, 256, 101)
        x = x.unsqueeze(1)  # (batch, 1, 256, 101)
        b, c, h, w = x.shape
        x = x.reshape(b*h, c, w)  # (batch*256, 1, 101)

        enc = self.encoder(x)
        dec = self.decoder(enc)

        x = dec.reshape(b, h, w)  # (batch, 256, 101)
        return x

def train_baseline(train_data, dev_data, test_data, output_dir):
    """训练纯数据驱动基线"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = SimpleResNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {params:,}")

    best_dev_l2 = float('inf')
    log_file = output_dir / 'training.log'

    with open(log_file, 'w') as f:
        f.write("ResNet数据驱动基线\n")
        f.write(f"参数数: {params}\n")
        f.write(f"无物理约束(对比用)\n\n")

        start = time.time()
        step = 0

        for epoch in range(1, 501):
            model.train()
            epoch_loss = 0

            # 随机小批次
            for i in range(0, len(train_data), 32):
                batch = train_data[i:i+32]
                u = torch.from_numpy(batch).float().to(device)

                optimizer.zero_grad()
                u_pred = model(u)
                loss = torch.nn.functional.mse_loss(u_pred, u)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                step += 1

            # 每50 epoch验证
            if epoch % 50 == 0:
                model.eval()
                with torch.no_grad():
                    dev_u = torch.from_numpy(dev_data).float().to(device)
                    dev_pred = model(dev_u)
                    dev_l2 = torch.nn.functional.mse_loss(dev_pred, dev_u).item()

                    if dev_l2 < best_dev_l2:
                        best_dev_l2 = dev_l2
                        torch.save(model.state_dict(), output_dir / 'model_best.pt')

                msg = f"Epoch {epoch}, Loss: {epoch_loss/max(len(train_data)//32, 1):.6f}, Dev L2: {dev_l2:.6f}\n"
                print(msg, end='')
                f.write(msg)
                f.flush()

        # 测试
        elapsed = time.time() - start
        model.load_state_dict(torch.load(output_dir / 'model_best.pt'))
        model.eval()
        with torch.no_grad():
            test_u = torch.from_numpy(test_data).float().to(device)
            test_pred = model(test_u)
            test_l2 = torch.nn.functional.mse_loss(test_pred, test_u).item()

            f.write(f"\ntest relative L2: {test_l2:.6f}\n")
            f.write(f"training seconds: {elapsed:.2f}\n")

        print(f"\nTest L2: {test_l2:.6f}")
        print(f"Time: {elapsed:.0f}s")

if __name__ == '__main__':
    base_path = Path('../datasets/burgers_augmented_v1')
    train_data = np.load(base_path / 'burgers_augmented_train.npz')['data']
    dev_data = np.load(base_path / 'burgers_augmented_dev.npz')['data']
    test_data = np.load(base_path / 'burgers_augmented_test.npz')['data']

    print(f"训练数据: {train_data.shape}")
    train_baseline(train_data, dev_data, test_data, '../experiments/baseline_resnet_pure_data')
