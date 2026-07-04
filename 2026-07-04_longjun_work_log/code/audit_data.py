#!/usr/bin/env python
"""只读数据审计脚本 —— 第一步：确认数据 shape / 类别 / 被试数。
不修改任何原始数据，不训练，只统计并打印。
"""
import os, glob, json
from collections import OrderedDict, Counter
import numpy as np

BASE = "/data/data/EMG/PR_raw/preprocess_npy"
SEP = "=" * 70


def section(t): print(f"\n{SEP}\n{t}\n{SEP}")


# ---------- 1. 聚合特征文件 features_all.npz ----------
section("1. 聚合特征 features_all.npz")
feat_paths = glob.glob("/data/data/EMG/**/features_all.npz", recursive=True)
print("找到:", feat_paths)
z = np.load(feat_paths[0], allow_pickle=True)
print("npz keys:", list(z.keys()))
X = np.asarray(z["data"])
Y = np.asarray(z["label"]).ravel()
print(f"X shape={X.shape}  dtype={X.dtype}")
print(f"Y shape={Y.shape}  dtype={Y.dtype}")
print(f"X 内存≈{X.nbytes/1e6:.1f} MB")
print(f"X min/max={X.min():.4f}/{X.max():.4f}  mean/std={X.mean():.4f}/{X.std():.4f}")
print(f"NaN 数={np.isnan(X).sum()}  Inf 数={np.isinf(X).sum()}")
classes, counts = np.unique(Y, return_counts=True)
print("Y 类别与样本数:")
for c, n in zip(classes, counts):
    print(f"   类 {c}: {n} 样本  ({n/len(Y)*100:.2f}%)")
print(f"类别数 = {len(classes)}  总样本 N = {len(Y)}")


# ---------- 2. per-subject_session 标签与维度 ----------
section("2. per-subject_session 目录统计")
subj_dirs = sorted(glob.glob(f"{BASE}/subject*_session*"))
print(f"subject_session 目录数 = {len(subj_dirs)}")

subjects = OrderedDict()
per_subj_summary = []
for d in subj_dirs:
    name = os.path.basename(d)  # subject01_session1
    subj, sess = name.replace("subject", "").split("_session")
    subjects.setdefault(subj, []).append(sess)
    lbl_path = os.path.join(d, "label_dynamic.npy")
    if not os.path.exists(lbl_path):
        print(f"  [警告] {name} 无 label_dynamic.npy"); continue
    lab = np.load(lbl_path)
    per_subj_summary.append((name, lab.shape, lab.dtype, np.unique(lab).tolist(),
                             (int(lab.min()), int(lab.max())) if lab.size else None))
print(f"被试数 = {len(subjects)}  每被试 session 数 = "
      f"{ {s: len(v) for s, v in subjects.items()} }")
print("\n各 subject_session 的 label_dynamic.npy：")
for row in per_subj_summary[:8]:
    print("  ", row)
print(f"  ... 共 {len(per_subj_summary)} 行")

# ---------- 3. 抽样看 raw 64 通道与 preprocessed 维度 ----------
section("3. 抽样 64channel_dynamic.npy / preprocessed_dynamic.npy (mmap 只读头)")
s1 = os.path.join(BASE, "subject01_session1")
ch64 = np.load(os.path.join(s1, "64channel_dynamic.npy"), mmap_mode="r")
prep = np.load(os.path.join(s1, "preprocessed_dynamic.npy"), mmap_mode="r")
print(f"64channel_dynamic.npy   shape={ch64.shape}  dtype={ch64.dtype}")
print(f"preprocessed_dynamic.npy shape={prep.shape} dtype={prep.dtype}")
print("(注：这些是连续 recordings，不是滑窗样本；features_all.npz 才是滑窗特征版)")


# ---------- 4. 适合性小结 ----------
section("4. 适合性判断（机器自动汇总）")
summary = {
    "features_npz_shape": list(X.shape),
    "features_npz_dtype": str(X.dtype),
    "n_classes": int(len(classes)),
    "class_values": [int(c) for c in classes],
    "n_total_samples": int(len(Y)),
    "class_distribution": {int(c): int(n) for c, n in zip(classes, counts)},
    "n_subjects": int(len(subjects)),
    "n_sessions_per_subject": {s: len(v) for s, v in subjects.items()},
    "has_nan": bool(np.isnan(X).any()),
    "has_inf": bool(np.isinf(X).any()),
}
print(json.dumps(summary, indent=2, ensure_ascii=False))
