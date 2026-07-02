# 2026-07-02 工作记录：环境搭建 + 数据勘察 + epilepsy 发作检测 baseline

> 本文件夹是**当日工作的完整归档**（代码 + 结果 + 说明 + 三篇 Notion 笔记全文），可脱离服务器/Notion 独立阅读。
> 原文件仍保留在 `~/aa_workspace/{code,outputs}` 原位，**此处为副本**，未移动、未删除任何原始数据。

---

## 0. 当日时间线（CST）
| 时间 | 事件 |
|---|---|
| 17:21 | 建工作区 `~/aa_workspace`（code/data/outputs/logs） |
| ~19:50 | 配置 `.claude`；`hello.py` VSCode Remote 连通性自测 |
| 20:11 | 数据勘察首版写入 `dataset_notes.md`（/data/data 三套数据 + 文件类型统计） |
| 20:26 | 注册 Notion MCP（`--scope user`，hosted OAuth） |
| ~20:30 | Notion 写入"数据集初步检查"小节 |
| 20:57 | 用 uv 建 `.venv`，装 numpy/scipy/mne/matplotlib |
| ~21:05 | 跑最小 demo：`chb01_03.npy` → 256Hz、18 通道、发作 1.11% |
| ~21:10 | 装 scikit-learn 1.9.0 |
| 21:18 | 跑通 baseline：test AUC 0.841 |
| 21:00 / 21:22 / 21:40 | 三篇 Notion 时间戳笔记（见 `notes/`） |
| 21:58 | 整理本归档文件夹 |
| 22:05 | 配置 `/daily` 斜杠命令（`~/.claude/commands/daily.md`） |

## 1. 逻辑链（为什么这么做）
1. **卡点：无 sudo** → 系统 Python 无 pip 且 ensurepip 缺失。→ 解法：用 **uv**（不依赖 ensurepip）建隔离 venv。
2. **先验证可读** → 写最小 demo 读一个 npy，确认格式/采样率/标签语义，再谈建模。
3. **选最易上手的数据** → epilepsy 18ch 已预处理为 npy、标签现成、文件小，优先做 baseline。
4. **方法论站得住** → **按病人留出**（杜绝同病人泄漏）、class_weight + 负下采样处理 1.66% 极不平衡、主指标用 AUC/F1 而非 accuracy。
5. **诚实读数** → AUC 0.84 说明特征有信号；precision 低是跨病人检测的典型痛点，下一步用阈值调优 + 时域后处理压假阳。

## 2. 当日目标
在无 sudo 的服务器上完成 Claude Code 工作环境搭建，勘察 `/data/data` 数据集，并跑通癫痫发作检测的最小 baseline，验证整条"切窗→特征→训练→评估"管线。

## 3. 文件清单

### `code/`
| 文件 | 作用 | 运行方式 |
|---|---|---|
| `demo_chb_npy.py` | 读 CHB-MIT 18ch npy，打印形状/标签分布（最小数据验证） | `.venv/bin/python code/demo_chb_npy.py` |
| `baseline_seizure_lr.py` | 滑窗 + 126 维手工特征 + 逻辑回归，**病人级划分**训练评估（核心脚本） | `.venv/bin/python code/baseline_seizure_lr.py` |
| `hello.py` | VSCode Remote 连通性自测（bootstrap，非实验代码） | `python code/hello.py` |

### `results/`
- `baseline_metrics.json` — val/test 的 sens / spec / prec / F1 / AUC
- `roc_baseline.png` — ROC 曲线（val/test）

### `notes/`（说明文件，含三篇 Notion 笔记全文）
| 文件 | 内容 |
|---|---|
| `dataset_notes.md` | 数据勘察 + baseline 细节（本地 source of truth） |
| `notion_202607022100_env_and_data_check.md` | Notion 笔记：环境搭建 + 数据检查 + 最小 demo |
| `notion_202607022122_epilepsy_baseline.md` | Notion 笔记：epilepsy baseline 实验 |
| `notion_202607022140_daily_summary.md` | Notion 笔记：今日汇总（最全面的一篇） |

> 建议阅读顺序：先 `notion_202607022140_daily_summary.md`（全景），再看 `dataset_notes.md`（技术细节）。

## 4. 关键结果（baseline，跨病人划分）
划分：train chb01–20(115 记录) / val chb21–22(7) / test chb23–24(15)；2s 窗、stride 1s、窗内 ≥50% 发作判正；class_weight=balanced + 负样本下采样。

| | sens | spec | prec | F1 | AUC |
|---|---|---|---|---|---|
| val  | 0.593 | 0.950 | 0.168 | 0.261 | **0.904** |
| test | 0.685 | 0.868 | 0.063 | 0.115 | **0.841** |

**判断**：AUC 0.84（跨病人）说明手工特征有判别力，是合格的第一条 baseline；precision 低（假阳多）是 CHB-MIT 跨病人检测的典型痛点，下一步做阈值调优 + 时域后处理。accuracy 因正样本仅 ~1.6% 不可信。

## 5. 复现环境
- Python venv：`~/aa_workspace/.venv`（uv 管理；numpy 2.5 / scipy 1.18 / mne 1.12 / matplotlib 3.11 / scikit-learn 1.9）
- 数据：`/data/data/epilepsy/chb-mit/chb-mit_18ch/*.npy`（只读，勿改动）
- 运行：`source ~/aa_workspace/.venv/bin/activate` 后执行上述命令；baseline 耗时约 260s

## 6. 遇到的问题
- 系统 Python 无 pip 且 ensurepip 缺失、无 sudo → 用 uv 绕开（✅ 已解决）
- Sleep-EDF 文件后缀配对（PSG `E0` / hypnogram `EC`）→ 按前 9 位 `SC4xxx` 配对（✅）
- 18ch 目录 138 数据 vs 137 标签，1 条缺标签 → 训练前过滤（⚠️ 待处理）
- baseline 假阳过多（precision 6.3%）→ 阈值调优 + 时域后处理（⚠️ 下一步）

## 7. 下一步计划
- [ ] v2：阈值调优（val 上选工作点）+ 时域后处理（连续窗 hysteresis/多数投票）压假阳
- [ ] 换小 1D-CNN（原始窗，需 `uv pip install torch`）与 LR 对比
- [ ] sleep 分期：hypnogram 对齐 PSG 的 30s epoch，5 分类 toy
- [ ] 装 torch 验证 EMG 的 `.pt`（2.5G/个）可读

## 8. 关联外部
- Notion 页面：「🛠️ longjun 服务器 Claude Code 配置日志」下的三篇时间戳笔记（正文同 `notes/` 内三篇 md）
- `/daily` 斜杠命令：`~/.claude/commands/daily.md`（用户级，三端通用）
