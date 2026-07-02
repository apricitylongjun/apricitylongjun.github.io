"""最小 demo：读取 CHB-MIT 18ch npy，验证数据链路（只读不训）。

用法:
    ~/aa_workspace/.venv/bin/python ~/aa_workspace/code/demo_chb_npy.py
"""
import numpy as np

BASE = "/data/data/epilepsy/chb-mit/chb-mit_18ch"
REC = "chb01_03"   # 已知含发作的记录（chb01 的发作段在 03/04/15/16...）

x = np.load(f"{BASE}/{REC}.npy", mmap_mode="r")   # mmap：不真正读入内存
y = np.load(f"{BASE}/{REC}_label.npy")

print(f"=== {REC} ===")
print(f"X  shape={x.shape}  dtype={x.dtype}  (推测: [通道, 时间窗*样本] 或 [窗, 通道, 时间])")
print(f"Y  shape={y.shape}  dtype={y.dtype}")

vals, cnts = np.unique(y, return_counts=True)
print("标签分布:")
total = y.size
for v, c in zip(vals, cnts):
    print(f"  label {v}: {c:>8d}  ({100*c/total:6.2f}%)")

# 发作样本占比 → 反映类别不平衡程度
print(f"\n发作(非0)样本占比: {100*(total-cnts[0])/total:.4f}% （CHB-MIT 典型极不平衡）")
