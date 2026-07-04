"""
预实验共享工具：数据加载 / SimpleCNN1D / 评估指标
被 pre_exp_train_ce.py (第四步) 与 pre_exp_train_robust.py (第五步) 复用。
"""
import os
import math
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

EXP = os.path.expanduser('~/aa_workspace/experiments/pre_exp')
N_CLS = 34


def load_split():
    d = np.load(f'{EXP}/split_data.npz', allow_pickle=True)
    return {k: d[k] for k in ['X_norm', 'Y', 'train_idx', 'val_idx', 'test_idx']}


def get_train_labels(setting):
    """setting: clean/noise20/noise40 -> 对应的 train 标签 (np.int64, len=len(train_idx))"""
    s = load_split()
    if setting == 'clean':
        return s['Y'][s['train_idx']].astype(np.int64)
    elif setting == 'noise20':
        return np.load(f'{EXP}/noise_20/noisy_train_labels.npy').astype(np.int64)
    elif setting == 'noise40':
        return np.load(f'{EXP}/noise_40/noisy_train_labels.npy').astype(np.int64)
    raise ValueError(setting)


def make_loaders(batch=128, setting='clean'):
    s = load_split()
    X = torch.from_numpy(s['X_norm']).float()
    Y = torch.from_numpy(s['Y']).long()
    ytr = torch.from_numpy(get_train_labels(setting)).long()
    tr = TensorDataset(X[s['train_idx']], ytr)
    va = TensorDataset(X[s['val_idx']], Y[s['val_idx']])
    te = TensorDataset(X[s['test_idx']], Y[s['test_idx']])
    return (DataLoader(tr, batch_size=batch, shuffle=True),
            DataLoader(va, batch_size=256),
            DataLoader(te, batch_size=256))


class SimpleCNN1D(nn.Module):
    """输入 (B, 4096) -> unsqueeze (B,1,4096) -> 4 层 Conv1d 降采样 -> 34 类。
    不依赖 4096 的空间结构假设（把它当单通道长序列）。"""
    def __init__(self, n_classes=N_CLS, p_drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 9, stride=2, padding=4), nn.BatchNorm1d(32), nn.ReLU(),   # 4096->2048
            nn.Conv1d(32, 64, 7, stride=2, padding=3), nn.BatchNorm1d(64), nn.ReLU(),  # ->1024
            nn.Conv1d(64, 128, 7, stride=4, padding=3), nn.BatchNorm1d(128), nn.ReLU(),# ->256
            nn.Conv1d(128, 128, 5, stride=4, padding=2), nn.BatchNorm1d(128), nn.ReLU(),# ->64
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(p_drop), nn.Linear(128, n_classes))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.net(x)


class ResBlock(nn.Module):
    """1D 残差块：Conv-BN-ReLU-Conv-BN + shortcut。"""
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.shortcut = (nn.Sequential() if (c_in == c_out and stride == 1)
                         else nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False),
                                            nn.BatchNorm1d(c_out)))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet1D(nn.Module):
    """1D 残差网络：stem + 3 个 stage(各 2 残差块) -> 34 类。输入 (B, 4096) 单通道序列。
    比 SimpleCNN1D 更深更强，用于抬高 clean 上限。"""
    def __init__(self, n_classes=N_CLS, p_drop=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(32), nn.ReLU())   # 4096->2048
        self.layer1 = nn.Sequential(ResBlock(32, 64, 2), ResBlock(64, 64, 1))     # ->1024
        self.layer2 = nn.Sequential(ResBlock(64, 128, 2), ResBlock(128, 128, 1))  # ->512
        self.layer3 = nn.Sequential(ResBlock(128, 256, 2), ResBlock(256, 256, 1)) # ->256
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(p_drop)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.drop(x))


@torch.no_grad()
def evaluate(model, loader, dev):
    from sklearn.metrics import balanced_accuracy_score, f1_score
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        out = model(xb.to(dev))
        ps.append(out.argmax(1).cpu().numpy())
        ys.append(yb.numpy())
    ys = np.concatenate(ys); ps = np.concatenate(ps)
    return {
        'accuracy': float((ps == ys).mean()),
        'balanced_accuracy': float(balanced_accuracy_score(ys, ps)),
        'macro_f1': float(f1_score(ys, ps, average='macro', zero_division=0)),
    }


# ===== 鲁棒损失函数（第二阶段） =====

def sce_loss(logits, target, alpha=1.0, beta=1.0, num_classes=N_CLS):
    """Symmetric Cross Entropy (Wang et al. 2019): alpha*CE + beta*RCE。
    RCE = -sum_i p_i log(q_i)，q=onehot 经 clamp(exp(-A)) 平滑防 log0；A=4。
    RCE ~ A*(1-p_y)，对噪声标签鲁棒。"""
    ce = F.cross_entropy(logits, target)
    p = F.softmax(logits, dim=1)
    q = F.one_hot(target, num_classes).float().clamp(min=math.exp(-4))
    rce = -(p * torch.log(q)).sum(dim=1).mean()
    return alpha * ce + beta * rce


def gce_loss(logits, target, q=0.7):
    """Generalized Cross Entropy (Zhang & Sabuncu 2018): (1 - p_y^q)/q。
    q->0 退化为 MAE(抗噪但难收敛)，q->1 退化为 CE(易收敛但不抗噪)，q=0.7 折中。"""
    p = F.softmax(logits, dim=1)
    p_y = p.gather(1, target.unsqueeze(1)).squeeze(1).clamp(min=1e-7)
    return ((1.0 - p_y.pow(q)) / q).mean()
