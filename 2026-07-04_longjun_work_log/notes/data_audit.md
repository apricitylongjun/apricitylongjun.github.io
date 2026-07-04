# 数据审计报告 — EMG 手势识别 / 噪声标签自修正预实验

> 第一步产出（只读检查，**未修改任何原始数据，未训练**）。
> 检查时间：2026-07-04 ｜ 检查者：Claude Code ｜ 脚本：`code/audit_data.py`

---

## 1. 工作目录与项目结构

- `pwd` = `/home/longjun/aa_workspace`
- 顶层结构：

| 路径 | 内容 |
|------|------|
| `code/` | `emg_demo.py`(已有 1D-CNN demo)、`baseline_seizure_lr.py`、`demo_chb_npy.py`、`audit_data.py`(本次新增,只读) |
| `data/` | 空（真实数据挂载在 `/data/data`，只读） |
| `outputs/` | `emg_demo_best.pt` 等历史产物 |
| `logs/` | 日志 |
| `daily/` | 每日归档（含 2026-07-03 env+emg demo） |
| `notes/` | 本次新增，存放审计/笔记 |
| `experiments/` | 本次预实验将建在 `experiments/pre_exp/` |

- venv：`~/aa_workspace/.venv`（Python 3.12，torch 2.12.1+cu130，已可用）
- GPU：4× RTX 5080，**GPU3 被同事占用，只用 GPU 0/1/2**（后续训练前 `export CUDA_VISIBLE_DEVICES=0,1,2`）

---

## 2. 数据路径与三种存储形式

数据根：`/data/data/EMG/PR_raw/`（公共只读，属主 guoyao）

| 子目录 | 内容 | 大小 | 用途 |
|--------|------|------|------|
| `preprocess_npy/features_all.npz` | 聚合**滑窗特征版** `data:(4030,4,256) label:(4030,)` | 32 MB | **本次预实验直接用它** |
| `preprocess_npy/data_all.npz` | 全量聚合原始 | 7.9 GB | 暂不用 |
| `preprocess_npy/subjectXX_sessionY/` | per-session：`64channel_dynamic.npy (N,2048,64)`、`preprocessed_dynamic.npy (N,2048,256)`、`label_dynamic.npy (N,)` | ~500 MB/sess | 被试级重建时用 |
| `preprocess/` | 40 个 `.pt`（subject1–20 × {dynamic,maintenance}，2.5–3.5 GB/个） | ~118 GB | 暂不用 |
| `preprocess_features/` | 159 个 `.npy`（data/label 成对，按 subj×sess×task） | — | 备选 |
| `subjectXX_sessionY/` | 原始 `.mat`（raw/preprocessed/rest + label） | ~3 GB/sess | 原始，不碰 |

---

## 3. 选定数据 `features_all.npz` 详情（本次预实验数据源）

```
npz keys: ['data', 'label']
X shape = (4030, 4, 256)   dtype = float64      # 4030 样本 × 4 通道 × 256 时间点
Y shape = (4030,)          dtype = uint8        # 标签取值 1..34
X min/max = 0.0000 / 25.0000   mean/std = 0.1340 / 0.6668
NaN = 0   Inf = 0   (干净)
内存 ≈ 33 MB
```

> 注：X 非负、稀疏（mean 0.13），明显是**特征表示**而非原始肌电（原始 HD-EMG 应是 64 通道、零均值波形，见 `64channel_dynamic.npy`）。`emg_demo.py` 与 `AI_TASK.md` 均沿用此 (4,256) 特征版，本预实验保持一致。

---

## 4. 类别数与每类样本数

- **类别数 = 34**（手势标签取值 **1..34**，非 0 起）
- **类别高度均衡**：每类 109–120 样本，各占 ≈ 2.9%

| 指标 | 值 |
|------|----|
| 类别数 | 34 |
| 总样本 N | 4030 |
| 每类样本范围 | 109（类34） ~ 120（多数类） |
| 是否均衡 | ✅ 近似均匀（最多/最少 ≈ 1.10） |

完整逐类分布见 `code/audit_data.py` 输出。代表性：类1=116, 类4=111, 类34=109, 其余多为 118–120。

---

## 5. 被试 / Session 结构

- `preprocess_npy/` 下 **40 个 `subjectXX_sessionY` 目录** = **20 被试 × 2 session**
- 每 session 的 `label_dynamic.npy` 长度 ≈ 195–204（即每 session 约 200 次试验，每次试验 = 1 个手势重复）
- **全 40 session 试验总数 = 8068**
- `features_all.npz` 的 **4030 = 8068 × 0.500** → 即 features_all 约为**每被试取 1 个 session**（dynamic 任务），另一半 session 未纳入

> 推断：features_all ≈ 20 被试 × ~201 试验/被试 ≈ 4030。该聚合文件**已丢失被试 ID**。

原始 HD-EMG 维度（参考）：
- `64channel_dynamic.npy`: `(N_trials≈202, 2048, 64)` — 64 通道、2048 时间采样/试验
- `preprocessed_dynamic.npy`: `(N_trials≈202, 2048, 256)` — 疑为 256 个运动单元分解结果

---

## 6. 适合性判断

### ✅ 结论：适合做 "sEMG 手势识别 + 噪声标签自修正学习" 最小预实验

理由：
1. **标准多分类**：34 类、均衡、N=4030，样本量足够支撑 80/10/10 划分（train≈3224 / val≈403 / test≈403）。
2. **标签干净可作为 ground truth**：原始标签来自 `label_dynamic`（每试验一个手势编号），可控注入 symmetric noise 后，test 集保留干净标签即可严格评估。
3. **输入即用**：(4, 256) 可直接喂 1D-CNN / ResNet1D，无需额外预处理。
4. **类别多 (34)**：噪声标签的危害在多分类下更显著（随机猜对概率仅 1/34），便于放大 CE vs 鲁棒方法的差异，适合验证方向。

### ⚠️ 注意事项 / 风险（写进后续设计）

| # | 风险 | 处理 |
|---|------|------|
| R1 | 标签取值 **1..34**，直接做 CE 会与 logit 索引错位 | 训练前 **remap 到 0..33**；保存映射表 |
| R2 | `features_all` 已聚合、**无被试 ID** → 随机 80/10/10 会有**同被试试验跨 train/test 的轻微泄漏** | 本预实验聚焦"标签噪声鲁棒性"（方法学验证），非"跨被试泛化"，按用户 spec 用随机 split 可接受；**正式实验需回 per-subject 文件做被试独立 split** |
| R3 | 通道级 z-score 归一化的 μ/σ 若在全集上算会泄漏 val/test 信息 | μ/σ **只在 train 上拟合**，再 transform val/test |
| R4 | features_all 仅含 **dynamic 任务、约 1 session/被试**；maintenance 任务与第 2 session 未用 | 预实验范围足够；扩大规模时再纳入 |
| R5 | 原始 64 通道 HD-EMG 未用 | 预实验先用 (4,256) 特征版跑通流程；后续可换 64 通道 backbone |

---

## 7. 下一步衔接（→ 第二步）

- 数据源锁定：`/data/data/EMG/PR_raw/preprocess_npy/features_all.npz`
- 标签 remap：`{1..34} → {0..33}`，映射表随 split 一并保存
- 划分：**train/val/test = 80/10/10**，`seed=42`，`np.random.RandomState(42).permutation`
- 归一化：通道级 z-score，**μ/σ 只用 train**
- 产物：`experiments/pre_exp/split_info.json`（含索引、remap、归一化统计量、随机种子）
- test 标签保持干净；噪声只施加于 train（第三步）
