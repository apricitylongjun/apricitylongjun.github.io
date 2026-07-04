"""
预实验 · 第二步：数据划分
数据源 = 候选 B：/data/data/EMG/PR_raw/preprocess_features/*_dynamic_{data,label}.npy

做什么：
  1. 加载 39 个 session（跳过缺 label 的 subject20_session1）的 dynamic 数据/标签，合并
  2. 标签 remap 1..34 -> 0..33（避免 CE 与 logit 索引错位）
  3. 样本级随机划分 train/val/test = 80/10/10，seed=42（可复现）
  4. per-feature z-score 归一化，mu/sigma 只用 train（防 val/test 泄漏）
  5. 保存 split_info.json（元信息）+ split_data.npz（indices / Y / mu / sigma / X_norm）

不改原始数据；test 标签保持干净；噪声只施加于 train（第三步）。
"""
import glob, os, json
import numpy as np

SEED = 42
SRC = '/data/data/EMG/PR_raw/preprocess_features'
OUT = os.path.expanduser('~/aa_workspace/experiments/pre_exp')
os.makedirs(OUT, exist_ok=True)


def main():
    # 1. 加载 39 session
    data_files = sorted(glob.glob(f'{SRC}/*_dynamic_data.npy'))
    Xs, Ys, sess_ids = [], [], []
    missing = []
    for f in data_files:
        lf = f.replace('_dynamic_data.npy', '_dynamic_label.npy')
        if not os.path.exists(lf):
            missing.append(os.path.basename(f).replace('_dynamic_data.npy', ''))
            continue
        x = np.load(f); y = np.load(lf)
        name = os.path.basename(f).replace('_dynamic_data.npy', '')
        Xs.append(x.astype(np.float64)); Ys.append(y.astype(np.int64))
        sess_ids += [name] * len(y)
    X = np.concatenate(Xs, 0); Y = np.concatenate(Ys, 0)
    sess_ids = np.array(sess_ids)
    print(f"[1] 加载: X{X.shape} Y{Y.shape}  来自 {len(Xs)} session  (缺label跳过: {missing})")

    # 2. 标签 remap
    orig_classes = sorted(np.unique(Y).tolist())
    remap = {o: i for i, o in enumerate(orig_classes)}
    Yr = np.array([remap[v] for v in Y], dtype=np.int64)
    n_cls = len(orig_classes)
    print(f"[2] 类别 {n_cls} 类, remap {orig_classes[0]}..{orig_classes[-1]} -> 0..{n_cls-1}")

    # 3. 划分 80/10/10
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(X))
    n = len(X)
    n_tr = int(n * 0.8); n_va = int(n * 0.1)
    tr_idx = np.sort(perm[:n_tr])
    va_idx = np.sort(perm[n_tr:n_tr + n_va])
    te_idx = np.sort(perm[n_tr + n_va:])
    print(f"[3] 划分: train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}  (seed={SEED})")

    # 4. 归一化 per-feature, mu/sigma 只用 train
    mu = X[tr_idx].mean(0)
    sigma = X[tr_idx].std(0)
    sigma[sigma < 1e-8] = 1e-6   # 防全零特征除零
    Xn = (X - mu) / sigma
    print(f"[4] 归一化(mu/sigma仅train): mu[{mu.min():.1f},{mu.max():.1f}] "
          f"sigma[{sigma.min():.4f},{sigma.max():.1f}] -> Xn mean={Xn.mean():.3f} std={Xn.std():.3f}")

    # 5. 类别分布
    def dist(idx):
        return np.bincount(Yr[idx], minlength=n_cls).tolist()
    d_tr, d_va, d_te = dist(tr_idx), dist(va_idx), dist(te_idx)
    print(f"[5] train类分布 min/max={min(d_tr)}/{max(d_tr)}; test类分布 min/max={min(d_te)}/{max(d_te)}")

    # 6. 跨 split 的 session 重叠（随机划分预期，记录）
    tr_sess, te_sess = set(sess_ids[tr_idx]), set(sess_ids[te_idx])
    overlap = len(tr_sess & te_sess)
    print(f"[6] 跨train/test的session重叠: {overlap}/{len(te_sess)} (随机划分预期,本实验聚焦标签噪声)")

    # 7. 保存
    np.savez(os.path.join(OUT, 'split_data.npz'),
             X_norm=Xn.astype(np.float32), Y=Yr.astype(np.int64),
             sess_ids=sess_ids,
             train_idx=tr_idx.astype(np.int64), val_idx=va_idx.astype(np.int64),
             test_idx=te_idx.astype(np.int64),
             mu=mu, sigma=sigma)
    info = {
        'step': 2, 'date': '2026-07-04', 'seed': SEED,
        'data_source': 'B: preprocess_features/*_dynamic_{data,label}.npy',
        'data_root': SRC,
        'sessions_used': len(Xs),
        'sessions_missing_label': missing,
        'n_samples': int(n), 'n_features': int(X.shape[1]), 'n_classes': n_cls,
        'orig_label_range': [int(orig_classes[0]), int(orig_classes[-1])],
        'label_remap': {str(o): i for o, i in remap.items()},
        'split': {'train': len(tr_idx), 'val': len(va_idx), 'test': len(te_idx),
                  'ratios': [0.8, 0.1, 0.1]},
        'normalization': 'per-feature z-score; mu/sigma computed on TRAIN only',
        'test_label_clean': True,
        'noise_applied_to': 'train only (step 3, pending)',
        'class_distribution': {'train': d_tr, 'val': d_va, 'test': d_te},
        'cross_split_session_overlap': overlap,
        'note_R2': '随机样本级划分;同被试可能跨split(预实验聚焦标签噪声,可接受;正式实验应做被试独立划分)',
        'files': ['split_info.json', 'split_data.npz'],
    }
    with open(os.path.join(OUT, 'split_info.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"[7] 已保存: {OUT}/split_info.json , split_data.npz")
    print("DONE")


if __name__ == '__main__':
    main()
