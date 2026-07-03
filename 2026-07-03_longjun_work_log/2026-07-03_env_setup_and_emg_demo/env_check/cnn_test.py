"""CNN 全链路验证：用随机数据（不依赖联网下载）训练 3 层小 CNN 2 个 epoch，
确认 前向->反向->优化器->GPU 全链路正常。
用法: python cnn_test.py
注意: 用随机标签，loss 不会真正收敛，目的是验证链路跑通，不是精度。
"""
import sys, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

print(f"torch {torch.__version__}")
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")

# 随机数据代替 MNIST/CIFAR（服务器不一定能联网下载公开数据集）
N = 2000
X = torch.randn(N, 1, 28, 28)
y = torch.randint(0, 10, (N,))
dl = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28->14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14->7
            nn.Flatten(), nn.Linear(32 * 7 * 7, 10),
        )
    def forward(self, x): return self.net(x)

model = Net().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
lossf = nn.CrossEntropyLoss()
print("=== 训练开始（随机数据，验证链路；loss 不代表精度）===")
for epoch in range(2):
    losses = []
    for i, (xb, yb) in enumerate(dl):
        xb, yb = xb.to(dev), yb.to(dev)
        opt.zero_grad()
        out = model(xb)
        loss = lossf(out, yb)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if i % 10 == 0:
            print(f"  epoch{epoch} batch{i:02d} loss={loss.item():.4f}")
    print(f"epoch{epoch} 平均loss={sum(losses)/len(losses):.4f}")
print("CNN_TEST_DONE: 前向+反向+优化器+GPU 全链路正常 ✅")
