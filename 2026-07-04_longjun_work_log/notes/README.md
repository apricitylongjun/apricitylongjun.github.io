# 预实验：噪声标签下 sEMG 手势识别自修正学习

## 1. 实验日期与主线
- 日期：2026-07-04
- 主线：用最小预实验验证"噪声标签下 sEMG 手势识别自修正学习"方向是否值得深挖
- 流程：数据审计 → 划分 → 注入 symmetric noise → CE baseline vs 简单鲁棒方法 → 干净 test 比较

## 2. 数据集信息
- **数据源 = 候选 B**：`preprocess_features/*_dynamic_{data,label}.npy`
  - 候选 A `features_all.npz` 经 `np.array_equal` 核实有 **2/4 切片损坏**（切片[0]==[1]、切片[2]全0），弃用，见 `notes/data_audit_findings.md`
- 39 session（缺 subject20_session1 label）、20 受试者、**34 类手势**、样本 7867、每样本 4096 维 HD-EMG 特征
- 原始幅值未归一化 → 训练前 per-feature z-score

## 3. 数据划分
- train/val/test = 80/10/10 = **6293 / 786 / 788**，seed=42，样本级随机
- **test 标签干净**；归一化 μ/σ **仅用 train**（防泄漏）
- 已知风险 R2：随机划分有同被试跨 split（39/39 重叠）——本实验聚焦标签噪声可接受；正式实验须做被试独立划分
- 详见 `split_info.json`、`notes/data_audit.md`、`notes/data_audit_findings.md`

## 4. 噪声构造
- symmetric (class-independent) noise，rate 20% / 40%，仅施加于 train
- 实际污染：noise_20 = 20.66%，noise_40 = 40.58%；val/test 不动（已 assert 验证）
- 每目录含：`noisy_train_labels.npy` / `noise_mask.npy` / `noise_config.json`

## 5. 训练命令（可复制）
```bash
cd ~/aa_workspace && source .venv/bin/activate && export CUDA_VISIBLE_DEVICES=0
python code/pre_exp_train_ce.py     --setting clean  --epochs 30
python code/pre_exp_train_ce.py     --setting noise20 --epochs 30
python code/pre_exp_train_ce.py     --setting noise40 --epochs 30
python code/pre_exp_train_robust.py --setting noise20 --epochs 30 --warmup 10 --lam_cons 0.5 --noise_std 0.1
python code/pre_exp_train_robust.py --setting noise40 --epochs 30 --warmup 10 --lam_cons 0.5 --noise_std 0.1
```
- backbone：SimpleCNN1D（4096 当单通道序列 → 4 层 Conv1d+BN 降采样 → 34 类，~0.1M 参数）
- 模型选择用 val_acc；**test 仅末尾用 best-val 模型评一次**（防 test 调参）

## 6. 关键结果（干净 test）
| setting | method | noise | acc | bal_acc | macro_f1 |
|---|---|---|---|---|---|
| clean  | CE   | 0%  | **0.477** | 0.476 | 0.475 |
| noise20| CE   | 20% | **0.348** | 0.353 | 0.340 |
| noise20| ours | 20% | 0.321 | 0.327 | 0.311 |
| noise40| CE   | 40% | **0.207** | 0.214 | 0.201 |
| noise40| ours | 40% | 0.216 | 0.228 | 0.209 |

（详见 `results_summary.csv`）

## 7. 当前结论（诚实）

**✅ 方向动机成立**：噪声标签显著击垮 CE —— clean 0.477 → noise20 0.348 → noise40 0.207，趋势单调、幅度大（noise40 比 clean 低 27pp）。证明 sEMG 手势识别在噪声标签下确有"自修正"的研究价值。

**⚠️ 但本次简单鲁棒方法未取得明显改进**：
- noise20：ours(0.321) **低于** CE(0.348) — 反而更差
- noise40：ours(0.216) 仅**微弱高于** CE(0.207)，+0.9pp 统计上不显著

**可能原因（待验证）**：
1. backbone 偏弱：clean 仅 0.477，模型容量不足 → loss-based 可靠性估计信噪比低
2. 鲁棒超参未调（warmup/lam_cons/noise_std 用默认值）
3. 权重用 batch 内 min-max 归一化，偏粗糙（应改 epoch 级全局 loss / GMM 拟合）
4. 4096 维特征结构未充分利用（当单通道序列处理）

**这不是"方向不可行"**，而是"最简方案不够"。预实验成功定位了问题。

## 8. 下一步建议（按优先级）
1. **换更强 backbone**：ResNet1D（残差块）/ Transformer encoder，抬高 clean 上限，给噪声鲁棒更大改进空间
2. **上成熟 noisy-label 方法**：DivideMix（GMM 分干净/噪声 + 半监督）、Co-teaching、Confident Learning(cleanlab) —— 这些有公开 SOTA 表现
3. **调鲁棒超参 + 改权重策略**：warmup、λ_cons、增强强度；权重改用 epoch 级全局 loss 排名或 GMM 拟合
4. **利用数据维度结构**：4096 reshape (64,64) 用 2D-CNN（若确认是通道×时间），或回原始 `64channel_dynamic.npy (N,2048,64)`
5. **被试独立划分**（正式实验）：消除 R2 泄漏，结论更真实可信

## 9. 安全合规（第七步）
- ✅ 未修改 `/data` 原始数据（全程只读）
- ✅ 未覆盖已有结果（`data_audit.md` 保留；新发现写入 `data_audit_findings.md`）
- ✅ 新代码/产物均在 `code/` 与 `experiments/pre_exp/`
- ✅ 写代码前已在对话中说明新建文件清单
- ✅ aa_workspace 非 git 仓库 → 用文件清单代替 `git diff`
- ✅ 所有训练命令可复制（各目录 `run_command.txt` + 本文件第 5 节）
- ✅ 数据适合性：候选 A 不适合（特征损坏）→ 改用 B（已在 findings 文档说明并决策）

## 文件结构
```
experiments/pre_exp/
├── split_info.json , split_data.npz        # 第二步
├── noise_20/ , noise_40/                    # 第三步
├── ce_clean/ , ce_noise20/ , ce_noise40/    # 第四步
├── ours_noise20/ , ours_noise40/            # 第五步
├── results_summary.csv                       # 第六步
└── README.md                                 # 本文件
code/  pre_exp_common.py · pre_exp_split.py · pre_exp_noise.py · pre_exp_train_ce.py · pre_exp_train_robust.py
notes/ data_audit.md(已有) · data_audit_findings.md(补充核实)
```
