"""epilepsy 发作检测 — 最小 baseline：滑窗 + 手工特征 + 逻辑回归。

设计要点：
- 按病人留出划分（train chb01-20 / val chb21-22 / test chb23-24），杜绝同病人泄漏。
- 2s 窗(512@256Hz)、stride 1s；窗内 ≥50% 发作 -> 正样本。
- 每通道手工特征：方差 / 平均绝对值 / line length + δ/θ/α/β 相对功率。
- 不平衡：class_weight=balanced + 训练集每个记录负窗下采样到 NEG_PER_REC。
- 指标：sens / spec / prec / f1 / auc（accuracy 仅参考，因 1.66% 极不平衡）。
只读原始 npy，不改动数据。
"""
import numpy as np, glob, os, re, json, time
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             roc_curve, confusion_matrix, accuracy_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/data/data/epilepsy/chb-mit/chb-mit_18ch"
OUT = "/home/longjun/aa_workspace/outputs"
FS, WIN, STRIDE, POS_TH = 256, 512, 256, 0.5
BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30)]   # δ θ α β
SEED, NEG_PER_REC = 0, 300


def split_of(p):
    return "train" if p <= 20 else ("val" if p <= 22 else "test")


def extract(x, y):
    """x:(T,C) float32, y:(T,) -> feats:(n, nfeat*C), wlab:(n,)."""
    T, C = x.shape
    n = (T - WIN) // STRIDE + 1
    idx = np.arange(n)[:, None] * STRIDE + np.arange(WIN)[None, :]   # (n,WIN)
    xw = x[idx]                          # (n,WIN,C)
    wlab = (y[idx].mean(axis=1) >= POS_TH).astype(np.int8)

    var = xw.var(axis=1)                 # (n,C)
    meanabs = np.abs(xw).mean(axis=1)
    line_len = np.abs(np.diff(xw, axis=1)).sum(axis=1)

    spec = np.abs(np.fft.rfft(xw, axis=1)) ** 2          # (n, 257, C)
    freqs = np.fft.rfftfreq(WIN, 1 / FS)
    total = spec.sum(axis=1) + 1e-12
    feats = [var, meanabs, line_len]
    for lo, hi in BANDS:
        m = (freqs >= lo) & (freqs < hi)
        feats.append(spec[:, m, :].sum(axis=1) / total)  # 相对功率 (n,C)

    F = np.stack(feats, axis=1)          # (n, nfeat, C)
    return F.reshape(n, -1).astype(np.float32), wlab


def collect(files, subsample, rng):
    Xs, ys = [], []
    for i, f in enumerate(files):
        rec = f.replace("_label.npy", ".npy")
        x = np.load(rec, mmap_mode="r").astype(np.float32)
        y = np.load(f).astype(np.float32)
        F, lab = extract(x, y)
        del x
        if subsample:
            pos = np.where(lab == 1)[0]
            neg = np.where(lab == 0)[0]
            if len(neg) > NEG_PER_REC:
                neg = rng.choice(neg, NEG_PER_REC, replace=False)
            keep = np.concatenate([pos, neg])
            F, lab = F[keep], lab[keep]
        Xs.append(F); ys.append(lab)
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(files)} …")
    return np.concatenate(Xs), np.concatenate(ys)


def score(y, pred, prob):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        n=int(len(y)), pos=int(y.sum()),
        sens=tp / (tp + fn) if tp + fn else 0.0,
        spec=tn / (tn + fp) if tn + fp else 0.0,
        prec=precision_score(y, pred, zero_division=0),
        f1=f1_score(y, pred, zero_division=0),
        auc=roc_auc_score(y, prob) if len(np.unique(y)) > 1 else float("nan"),
        acc=accuracy_score(y, pred),
    )


def main():
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    labs = sorted(glob.glob(f"{D}/*_label.npy"))
    splits = defaultdict(list)
    for f in labs:
        p = int(re.match(r"chb(\d+)", os.path.basename(f)).group(1))
        splits[split_of(p)].append(f)
    print(f"记录数: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    print("[train] 提取特征（负样本下采样）…")
    Xtr, ytr = collect(splits["train"], True, rng)
    print("[val]   提取 …")
    Xva, yva = collect(splits["val"], False, rng)
    print("[test]  提取 …")
    Xte, yte = collect(splits["test"], False, rng)
    print(f"train {Xtr.shape} pos={ytr.mean()*100:.2f}% | "
          f"val pos={yva.mean()*100:.2f}% | test pos={yte.mean()*100:.2f}%")

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
    clf.fit(sc.transform(Xtr), ytr)

    res = {}
    fig, ax = plt.subplots(figsize=(4, 4))
    for name, X, y in [("val", Xva, yva), ("test", Xte, yte)]:
        prob = clf.predict_proba(sc.transform(X))[:, 1]
        pred = (prob >= 0.5).astype(int)
        res[name] = score(y, pred, prob)
        fpr, tpr, _ = roc_curve(y, prob)
        ax.plot(fpr, tpr, label=f"{name} AUC={res[name]['auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", lw=1)
    ax.set_xlabel("FPR (1 − specificity)")
    ax.set_ylabel("TPR (sensitivity)")
    ax.set_title("ROC — seizure vs interictal (patient-level split)")
    ax.legend()
    fig.tight_layout()

    print("\n=== 结果 ===")
    for k in ("val", "test"):
        m = res[k]
        print(f"[{k}] sens={m['sens']:.3f} spec={m['spec']:.3f} prec={m['prec']:.3f} "
              f"f1={m['f1']:.3f} auc={m['auc']:.3f} acc={m['acc']:.3f} "
              f"(n={m['n']}, pos={m['pos']})")
    print(f"耗时 {time.time()-t0:.1f}s")

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/baseline_metrics.json", "w") as fp:
        json.dump(res, fp, indent=2)
    fig.savefig(f"{OUT}/roc_baseline.png", dpi=110)
    print(f"已保存: {OUT}/baseline_metrics.json, {OUT}/roc_baseline.png")


if __name__ == "__main__":
    main()
