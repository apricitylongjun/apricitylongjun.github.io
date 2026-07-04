# 2026-07-04 · 噪声标签 sEMG 预实验：第二阶段启动 + 第一阶段复盘 + 数据质量核查

## 1. 当日目标
在第一阶段（SimpleCNN1D）证实"噪声标签会击垮 CE、但简单鲁棒方法不够"之后，进入第二阶段：换 ResNet1D backbone，跑 CE/SCE/GCE/Co-teaching × {0%,20%,40%} × 3 seeds 的网格，拿到 mean±std；同时核查数据源质量，定位 clean 准确率 ~0.47 的天花板成因。

## 2. 文件清单
### code/（今日脚本，复制自 ~/aa_workspace/code/）
| 文件 | 作用 | 运行命令 |
|---|---|---|
| `audit_data.py` | 数据审计：shape/类别/被试/切片质量核查 | `python code/audit_data.py` |
| `pre_exp_split.py` | 80/10/10 划分，seed=42，保存 split_info.json + split_data.npz | `python code/pre_exp_split.py` |
| `pre_exp_noise.py` | 对称标签噪声构造（只污染 train，存 noise_mask） | `python code/pre_exp_noise.py` |
| `pre_exp_train_ce.py` | 第一阶段 CE baseline 训练（SimpleCNN1D） | `python code/pre_exp_train_ce.py --setting clean` |
| `pre_exp_train_robust.py` | 第一阶段简单鲁棒版（warmup-CE + loss 加权 + 一致性） | `python code/pre_exp_train_robust.py --setting noise20` |
| `pre_exp2_train.py` | ResNet1D 探索训练（验证换 backbone 是否有用） | `python code/pre_exp2_train.py --setting clean` |
| `pre_exp_common.py` | 共用模块：ResNet1D、数据加载、指标（stage2 复用） | （被 import，不单独跑） |
| `pre_exp_stage2_train.py` | **第二阶段单实验训练入口** | `python code/pre_exp_stage2_train.py --method sce --setting noise20 --seed 42 --epochs 40 --batch 128 --lr 1e-3` |
| `pre_exp_stage2_summarize.py` | 汇总 stage2 各 run 的 metrics → mean/std 表 | `python code/pre_exp_stage2_summarize.py` |
| `run_all_stage2.sh` | **第二阶段批量脚本：4方法×3噪声×3seed=36实验** | `bash code/run_all_stage2.sh` |

### notes/（今日笔记与复盘）
| 文件 | 内容 |
|---|---|
| `stage1_review.md` | **今日核心判断文档**：第一阶段结论 + R1–R5 风险评估 + 第二阶段计划 |
| `data_audit_findings.md` | **关键发现**：features_all.npz 4 切片中 [1]冗余/[2]全零，有效仅 2/4 |
| `data_audit.md` | 数据审计主报告（被 findings 补充，不覆盖） |
| `README.md` | 第一阶段自述（来自 experiments/pre_exp/README.md） |

### results/（指标/配置/日志，按 experiments/ 原结构）
- `pre_exp/` — 第一阶段 5 个 run（ce_{clean,noise20,noise40}、ours_{noise20,noise40}）+ noise_{20,40}/ 噪声产物 + results_summary.csv + split_info.json
- `pre_exp2/` — ResNet1D 探索 3 个 run
- `pre_exp_stage2/` — 第二阶段已完成的 11 个 run + run_all.log（完整训练日志）
- **未复制**（原路径见 `~/aa_workspace/experiments/`）：各 run 的 `best_model.pth`（模型权重）、`pre_exp/split_data.npz`（129MB）

## 3. 关键结果

### 第一阶段（SimpleCNN1D, seed=42, 干净 test）
| setting | method | acc | bal_acc | macro_f1 |
|---|---|---|---|---|
| clean   | CE            | 0.477 | 0.476 | 0.475 |
| noise20 | CE            | 0.348 | 0.353 | 0.340 |
| noise40 | CE            | 0.207 | 0.214 | 0.201 |
| noise20 | ours(简单鲁棒) | 0.321 | 0.327 | 0.311 |
| noise40 | ours(简单鲁棒) | 0.216 | 0.228 | 0.209 |

→ 噪声单调击垮 CE（动机成立）；但简单鲁棒 noise20 反而更差，**简单方法不够**。

### 第二阶段（ResNet1D，**进行中/中断**）
CE 已跑完 3seed×3noise=9 组（mean）：
| setting | CE acc (seed42/2024/2026) | mean |
|---|---|---|
| clean   | 0.487 / 0.509 / 0.520 | ≈0.505 |
| noise20 | 0.269 / 0.258 / 0.258 | ≈0.262 |
| noise40 | 0.165 / 0.197 / 0.199 | ≈0.187 |

其余方法状态：
- **SCE**：仅 clean×{42,2024} 跑完（acc 0.392 / 未出 metrics），noise 与 seed2026 未跑
- **GCE clean/seed42**：acc **0.132**，best_epoch=2 → **疑似 bug，未收敛**
- **Co-teaching clean/seed42**：acc **0.152**，best_epoch=2 → **疑似 bug，未收敛**
- **run_all.log 停在 `sce clean seed2024` ep37，无 ALL_DONE** → 非报错中断，剩余未跑

### 两条今日最重要的判断
1. **瓶颈很可能不在 backbone、不在方法，而在数据表示**：ResNet1D 的 clean（0.472）与 SimpleCNN1D（0.477）几乎相同 → 换 backbone 无效（证伪 R1）。
2. **数据源有硬伤**：features_all.npz 的 (4,256) 里只有切片 [0]与[3] 有效，[1]与[0]完全相同、[2]全零 → 名义 4 通道实际 ~2 通道，信号偏弱，**可能是 clean ~0.47 天花板的主因（R3）**。

## 4. 复现环境
- venv：`~/aa_workspace/.venv`（`source ~/aa_workspace/.venv/bin/activate`）
- 数据：`/data/data/EMG/**/features_all.npz`，shape (4030,4,256)，label 1..34 → remap 0..33（34 类）
- GPU：`CUDA_VISIBLE_DEVICES=0`（**GPU3 同事在用，禁碰**；今日归档时 GPU0 空闲）
- 关键依赖：torch（venv 内，版本见 venv）
- 主流程：`bash code/run_all_stage2.sh`（stage2 全网格）；单实验命令见上表

## 5. 遇到的问题
1. **GCE / Co-teaching 未收敛**（best_epoch=2，clean acc 0.13/0.15）—— 现象：训练 2 个 epoch 即触发早停；原因疑似损失实现或 early-stopping/超参 bug；状态：**未修，明日优先处理**。
2. **features_all 切片损坏**（[1]冗余、[2]全零，有效 2/4）—— 已用 `np.array_equal` 精确核实；状态：已定位，待决策是否换数据源 B（preprocess_features (7867,4096) 全有效）。
3. **stage2 中断**（停在 sce clean seed2024，非报错）—— 原因疑为执行会话被打断；状态：剩余 ~25 组待续跑。
4. **划分泄漏**（sample-level 随机划分，同被试跨 train/test）—— 预实验聚焦标签噪声可接受；正式实验需被试独立划分。

## 6. 关联笔记
- Notion：今日未记（GitHub/Notion 已降级手动）
- 相关记忆：`pre-exp-noisy-emg`（第一阶段结论）、`research-workflow-project`

## 7. 下一步
- **代码层**：修 GCE / Co-teaching 早停 bug → 续跑 stage2 剩余网格 → 出 mean±std 汇总表
- **方法层**：决策数据源（A 当前 / B 推荐 4096 维 / C 最原始 64ch），先解 R3 天花板再谈方法对比
- **研究层（用户）**：明日看 10 篇 noisy-label 论文（中午 5 + 下午 5），晚上汇报；下次会议定「结果分类论证」，下下次会议定「整体范式 + 网络架构 + 数据集方案」
