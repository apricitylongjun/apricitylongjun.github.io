"""
预实验 · 第三步：标签噪声构造（symmetric label noise）
依赖：第二步产物 experiments/pre_exp/split_data.npz

做什么：
  1. 只读取 train 的干净标签（val/test 完全不动）
  2. 构造 symmetric noise，rate = 20% 和 40%
     - 每个样本以概率 rate 被翻转；翻转时均匀随机选一个【不同于原类】的新类
  3. 保存 noisy_train_labels.npy / noise_mask.npy / noise_config.json
     到 experiments/pre_exp/noise_20/ 和 noise_40/

不变式（脚本内 assert 验证）：
  - 被污染样本的新标签必然 != 原标签
  - 未被污染样本标签 == 原标签
  - test/val 标签完全不变
"""
import os, json
import numpy as np

SEED = 42
OUT = os.path.expanduser('~/aa_workspace/experiments/pre_exp')
N_CLS = 34
RATES = [0.20, 0.40]


def make_symmetric_noise(y_clean, rate, seed, n_cls):
    """向量化 symmetric noise：被选中样本翻转为一个不同的类（均匀）。"""
    rng = np.random.RandomState(seed)
    n = len(y_clean)
    mask = rng.rand(n) < rate                       # 哪些样本被污染
    y_noisy = y_clean.copy()
    # 经典技巧：randint(0, n_cls-1) 得 0..n_cls-2；若 < 原标签则取它，否则 +1，从而跳过原标签
    r = rng.randint(0, n_cls - 1, size=n)
    new_labels = np.where(r < y_clean, r, r + 1)    # 必然 != y_clean
    y_noisy[mask] = new_labels[mask]
    return y_noisy, mask


def main():
    d = np.load(os.path.join(OUT, 'split_data.npz'), allow_pickle=True)
    Y = d['Y']
    train_idx, val_idx, test_idx = d['train_idx'], d['val_idx'], d['test_idx']
    y_train_clean = Y[train_idx].astype(np.int64)   # (6293,)
    print(f"[load] train 标签 {y_train_clean.shape}, 类别 0..{N_CLS-1}")
    print(f"       (val/test 标签不动: val={len(val_idx)} test={len(test_idx)})")

    for rate in RATES:
        y_noisy, mask = make_symmetric_noise(y_train_clean, rate, SEED, N_CLS)
        # —— 不变式验证 ——
        assert np.all(y_noisy[mask] != y_train_clean[mask]), "被污染样本新标签必须!=原标签"
        assert np.all(y_noisy[~mask] == y_train_clean[~mask]), "未污染样本标签必须不变"
        assert len(np.unique(y_noisy)) <= N_CLS
        # 对比：干净标签下 train 的"标签被改"比例应等于 mask.mean()
        flip_rate_emp = float((y_noisy != y_train_clean).mean())

        sub = f'noise_{int(rate * 100)}'
        sub_dir = os.path.join(OUT, sub)
        os.makedirs(sub_dir, exist_ok=True)
        np.save(os.path.join(sub_dir, 'noisy_train_labels.npy'), y_noisy)
        np.save(os.path.join(sub_dir, 'noise_mask.npy'), mask)
        cfg = {
            'step': 3, 'date': '2026-07-04', 'seed': SEED,
            'noise_type': 'symmetric (class-independent)',
            'noise_rate_setting': rate,
            'n_classes': N_CLS,
            'n_train': int(len(y_train_clean)),
            'n_corrupted': int(mask.sum()),
            'actual_corrupt_ratio': float(mask.mean()),
            'empirical_flip_ratio': flip_rate_emp,
            'applied_to': 'train labels ONLY; val/test labels unchanged',
            'label_space': '0..33 (remapped from original 1..34)',
            'depends_on': 'experiments/pre_exp/split_data.npz',
        }
        with open(os.path.join(sub_dir, 'noise_config.json'), 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        print(f"[{sub}] 设定rate={rate:.0%} | 实际污染 {mask.sum()}/{len(mask)} = {mask.mean():.4f} "
              f"| 翻转比例 {flip_rate_emp:.4f} | 已保存 3 个文件")

    print("DONE")


if __name__ == '__main__':
    main()
