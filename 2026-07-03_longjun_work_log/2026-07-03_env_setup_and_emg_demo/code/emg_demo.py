#!/usr/bin/env python
"""
HD-EMG 分类 demo 范式（高密度肌电）
- 数据：/data/data/EMG/**/features_all.npz  （真实预处理特征 X:(N,4,256), Y:(N,)）
- 模型：轻量 1D-CNN 分类器
- 设计：在 tmux 后台长跑，关掉 SSH 继续运行；日志同时写屏幕和 ~/aa_workspace/logs/emg_demo.log
- 用法（在 tmux 里）：  python code/emg_demo.py
"""
import os, glob, time, argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.expanduser("~/aa_workspace")
LOG_FILE = os.path.join(ROOT, "logs", "emg_demo.log")
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 42
_parser = argparse.ArgumentParser(description="HD-EMG 分类 demo")
_parser.add_argument("--epochs", type=int, default=30, help="训练轮数（想长跑演示可设 200）")
_parser.add_argument("--batch", type=int, default=128, help="batch size（越小每轮越慢）")
_args = _parser.parse_args()
EPOCHS = _args.epochs
BATCH = _args.batch


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def main():
    open(LOG_FILE, "w").close()
    log("=" * 60)
    log("HD-EMG 分类 demo 开始（在 tmux 后台运行，关掉 SSH 也会继续）")
    log("=" * 60)
    log(f"PyTorch {torch.__version__} | CUDA available={torch.cuda.is_available()}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)} | 含sm_120={'sm_120' in str(torch.cuda.get_arch_list())}")

    # 1. 加载真实数据
    cands = glob.glob("/data/data/EMG/**/features_all.npz", recursive=True)
    if not cands:
        log("❌ 找不到 features_all.npz，退出"); return
    log(f"加载数据: {cands[0]}")
    z = np.load(cands[0], allow_pickle=True)
    X = np.asarray(z["data"], dtype=np.float32)
    Y = np.asarray(z["label"], dtype=np.int64).ravel()
    classes = np.unique(Y); n_cls = len(classes)
    remap = {c: i for i, c in enumerate(classes)}; Y = np.array([remap[v] for v in Y])
    log(f"X shape={X.shape} dtype={X.dtype} | Y shape={Y.shape} | 类别数={n_cls}（原标签 {list(classes)}）")

    # 2. 通道级 z-score 归一化
    mu = X.mean(axis=(0, 2), keepdims=True); sd = X.std(axis=(0, 2), keepdims=True) + 1e-6
    X = (X - mu) / sd

    # 3. 划分 train/val（80/20，固定种子，可复现）
    rng = np.random.RandomState(SEED); perm = rng.permutation(len(X))
    n_tr = int(len(X) * 0.8)
    tr, va = perm[:n_tr], perm[n_tr:]
    Xtr, Ytr = torch.from_numpy(X[tr]), torch.from_numpy(Y[tr])
    Xva, Yva = torch.from_numpy(X[va]), torch.from_numpy(Y[va])
    log(f"train={len(Xtr)}  val={len(Xva)}")
    tr_dl = DataLoader(TensorDataset(Xtr, Ytr), batch_size=BATCH, shuffle=True)
    va_dl = DataLoader(TensorDataset(Xva, Yva), batch_size=256)

    # 4. 1D-CNN（输入通道=X.shape[1]，长度=256）
    class EMGNet(nn.Module):
        def __init__(self, c_in, n_cls):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(c_in, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
                nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_cls))
        def forward(self, x): return self.net(x)

    c_in = X.shape[1]
    model = EMGNet(c_in, n_cls).to(dev)
    log(f"模型 EMGNet：输入通道={c_in} 类别={n_cls} 参数量={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lossf = nn.CrossEntropyLoss()

    # 5. 训练循环
    best_acc = 0.0; t0 = time.time()
    log(f"开始训练 {EPOCHS} epoch ...")
    for ep in range(EPOCHS):
        model.train(); tl = 0; correct = 0
        for xb, yb in tr_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); out = model(xb); loss = lossf(out, yb); loss.backward(); opt.step()
            tl += loss.item() * len(yb); correct += (out.argmax(1) == yb).sum().item()
        sched.step()
        tr_loss, tr_acc = tl / len(Xtr), correct / len(Xtr)
        model.eval(); vc = 0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(dev), yb.to(dev)
                vc += (model(xb).argmax(1) == yb).sum().item()
        val_acc = vc / len(Xva)
        star = "★" if val_acc > best_acc else " "
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "acc": val_acc, "epoch": ep,
                        "classes": list(classes), "in_channels": c_in},
                       os.path.join(OUT_DIR, "emg_demo_best.pt"))
        log(f"{star} epoch {ep:02d}/{EPOCHS} | loss {tr_loss:.4f} | train_acc {tr_acc:.3f} | val_acc {val_acc:.3f} | best {best_acc:.3f}")
    log("=" * 60)
    log(f"训练完成！用时 {time.time()-t0:.1f}s | 最佳 val_acc={best_acc:.3f}")
    log(f"最佳模型已存：{os.path.join(OUT_DIR, 'emg_demo_best.pt')}")
    log(f"日志已存：{LOG_FILE}")
    log("进程自然结束。若你在 tmux 里看到这里，可 Ctrl+a d 安全离开。")
    log("=" * 60)


if __name__ == "__main__":
    main()
