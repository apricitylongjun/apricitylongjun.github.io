# 第一阶段预实验 · 检查报告（stage1_review）

- 生成日期：2026-07-04
- 范围：`experiments/pre_exp/`（第一阶段，SimpleCNN1D，seed=42）——**冻结，不覆盖**
- 目的：进入第二阶段前的结果回顾、风险评估与计划

## 1. 已有结果（干净 test，来自 results_summary.csv + metrics.json）

| setting | method | noise | acc | bal_acc | macro_f1 |
|---|---|---|---|---|---|
| clean  | CE   | 0%  | 0.477 | 0.476 | 0.475 |
| noise20| CE   | 20% | 0.348 | 0.353 | 0.340 |
| noise40| CE   | 40% | 0.207 | 0.214 | 0.201 |
| noise20| ours(简单鲁棒) | 20% | 0.321 | 0.327 | 0.311 |
| noise40| ours(简单鲁棒) | 40% | 0.216 | 0.228 | 0.209 |

stage1 文件齐全：ce_{clean,noise20,noise40}/、ours_{noise20,noise40}/ 各 5 文件（config.yaml/train.log/metrics.json/best_model.pth/run_command.txt），noise_{20,40}/ 各 3 文件，外加 results_summary.csv / split_info.json / split_data.npz / README.md。

## 2. 主要结论
1. **方向动机成立**：噪声单调击垮 CE（0.477→0.348→0.207，noise40 近失效）。sEMG 手势识别在噪声标签下确有"自修正"研究价值。
2. **简单鲁棒方法不够**：warmup-CE+loss 加权+一致性，noise20 反而更差(0.321<0.348)，noise40 仅微弱 +0.9pp(0.216 vs 0.207)。

## 3. 主要风险
- **R1（backbone）**：SimpleCNN1D 偏弱。但临时探索（ResNet1D，在 `pre_exp2/`）显示 clean 仅 0.472、noise 下更差 → **瓶颈很可能不在 backbone**。
- **R2（划分）**：sample-level 随机划分，同被试跨 train/test（39/39 session 重叠）。本阶段聚焦标签噪声可接受；正式实验需被试独立划分。
- **R3（4096 表示）**：4096 维当单通道长序列，1D 卷积在展平向量上物理意义不明确（4096 真实结构=通道×时间？未确认）。**可能是 clean ~0.47 天花板的主因**。
- **R4（鲁棒实现粗糙）**：batch 内 min-max 权重归一化偏粗糙。
- **R5（单 seed）**：仅 seed=42，无 mean/std，结果可能受随机波动影响。**第二阶段用 3 seeds 解决。**

## 4. 4096 维是否适合 ResNet1D？
**判断：技术上适合，不触发停止条件，继续。**
- 已验证：ResNet1D 在 (B,1,4096) 上正常训练、收敛（clean 0.472，见 `pre_exp2/`）。
- 但 R3 风险仍在：4096 展平结构未利用，可能限制上限。
- 本阶段按计划用 ResNet1D（单通道 4096）跑通；reshape(64,64) 等留作后续探索。

## 5. 第二阶段计划（`experiments/pre_exp_stage2/`，不覆盖 stage1）
- **backbone**：ResNet1D（已在 `code/pre_exp_common.py`，复用）
- **方法**（按用户要求，**不上 Transformer / 不上 DivideMix**）：
  - CE baseline
  - SCE（Symmetric Cross Entropy）
  - GCE（Generalized Cross Entropy）
  - Co-teaching（双网络选小 loss 样本）
- **seeds**：42, 2024, 2026（多 seed → mean/std，解 R5）
- **noise**：0%, 20%, 40%；test 保持干净；模型选择只用 val；test 仅 best-val 末尾评一次
- **每实验目录 5 文件**：config.yaml / train.log / metrics.json / best_model.pth / run_command.txt
- **汇总**：results_summary.csv（seed,backbone,method,noise_rate,accuracy,balanced_accuracy,macro_f1）+ results_mean_std.csv + README.md
- **规模**：4 方法 × 3 噪声 × 3 seed = 36 实验（Co-teaching 双网络计 1 实验）
