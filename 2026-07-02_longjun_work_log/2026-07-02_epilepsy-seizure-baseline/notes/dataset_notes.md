# Dataset Notes

> 服务器项目数据检查记录。本次为只读检查：**未删除、未移动、未运行任何训练任务**。
> 检查时间：2026-07-02 ｜ 检查者：Claude Code

---

## 0. 项目工作区 `~/aa_workspace`

四个关键目录全部存在（来自之前的 bootstrap）：

| 目录 | 状态 | 内容 |
|------|------|------|
| `code/` | ✅ 存在 | 仅 `hello.py`（打印测试脚本，99B） |
| `data/` | ✅ 存在 | 空（数据实际在 `/data/data`） |
| `outputs/` | ✅ 存在 | 空 |
| `logs/` | ✅ 存在 | 空 |

> 注意：`data/` 是空的——真实数据集挂载在 `/data/data`，不在工作区内。
> 建议后续在 `code/` 里用**绝对路径** `/data/data/...` 引用数据，或建软链，避免把 330GB 复制进工作区。

---

## 1. 数据根目录 `/data/data` 一级结构

共 3 个数据集目录，总占用约 **330 GB**：

```
/data/data/
├── EMG/              255 G   肌电（EMG）原始/预处理数据，guoyao 用户
├── epilepsy/          68 G   CHB-MIT 癫痫脑电数据集
└── sleep_stages/      7.1G   Sleep-EDF 睡眠分期数据集
```

全量统计：**2267 个文件，121 个目录**。

---

## 2. 二级目录结构

### 2.1 `EMG/PR_raw/`（肌电）
```
PR_raw/
├── preprocess/             40 个 .pt 文件（PyTorch 张量，2.5~3.5 GB/个）
│                             命名 subject{10..}_dynamic.pt / _maintenance.pt
├── preprocess_features/   159 个 .npy（data/label 成对，按 subject×session×任务）
├── preprocess_npy/         42 项：data_all.npz + features_all.npz + 24 个 subject_session 子目录
└── subject01_session1/ ... subject12_session2/   （12 受试 × 2 session = 24 个目录）
```
每个 `subjectXX_sessionY/` 内含 12 个 `.mat` 文件（raw_ / preprocessed_ / pre_pre_*_rest，以及 `label_dynamic.mat`、`label_maintenance.mat`）。

### 2.2 `epilepsy/`（癫痫 EEG）
```
epilepsy/
├── chbmit/                       原始 CHB-MIT Scalp EEG Database（PhysioNet）
│   ├── chb01/ ... chb24/         24 名病人，每个含 chbXX_YY.edf（每段 1 小时，23 导联）
│   │                               + chbXX_YY.edf.seizures（癫痫发作标注，141 个）
│   └── RECORDS / ANNOTATORS / index.html
└── chb-mit/chb-mit_18ch/        已预处理为 18 通道的 npy 版本（275 个文件：
                                  chbXX_YY.npy + chbXX_YY_label.npy 成对）
```

### 2.3 `sleep_stages/`（睡眠分期）
```
sleep_stages/sleep_edf_expanded/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/
├── SC4xxx_xx-PSG.edf            153 个 PSG（多导睡眠图）信号文件
└── SC4xxx_xx-Hypnogram.edf      153 个睡眠分期标注文件（W/N1/N2/N3/R）
```
> 这是标准的 **Sleep-EDF Dataset Expanded（sleep-cassette 子集）**。
> （注：仅含 sleep-cassette，未见 sleep-telemetry。）

---

## 3. 文件类型统计（全 `/data/data`，按扩展名）

| 扩展名 | 数量 | 主要分布 | 说明 |
|--------|------|----------|------|
| `.edf`  | 992 | sleep 306（153 PSG+153 Hypnogram）+ epilepsy ~686 | 欧洲数据格式，生物信号标准格式 |
| `.npy`  | 554 | epilepsy 18ch 275 + EMG preprocess_features 159 + EMG preprocess_npy ~120 | NumPy 单数组 |
| `.mat`  | 480 | EMG 各 subject_session 目录（12/目录） | MATLAB 矩阵（含 raw/preprocessed/label） |
| `.seizures` | 141 | epilepsy chbXX/*.edf.seizures | CHB-MIT 发作时段标注 |
| `.pt`   | 40  | EMG preprocess/ | PyTorch 张量（大文件，2.5~3.5 GB） |
| `.txt`  | 27  | 杂项 | — |
| `.html` | 26  | epilepsy 各 chb 目录索引页 | — |
| `.npz`  | 2   | EMG preprocess_npy/{data_all,features_all}.npz | NumPy 多数组打包 |
| `.pdf`  | 1   | — | — |
| 无扩展名 | 4 | RECORDS / ANNOTATORS / robots.txt 等 | — |

> 用户提到的 jpg/png/csv/json 全部为 **0**——本服务器是纯**一维时序/生物电信号**数据，没有图像或表格数据集。

---

## 4. Python 环境现状（关键约束）

| 项 | 状态 |
|----|------|
| Python | ✅ 3.12.3（`/usr/bin/python3`） |
| pip / conda / uv / venv | ❌ 全部未安装 |
| numpy / scipy / torch / mne / h5py / matplotlib / pandas | ❌ 全部缺失 |

> **结论**：当前无法直接 `import numpy`，连读取 `.npy` 都做不到。
> 跑任何 demo 之前，第一步必须先搭一个 Python 科学计算环境（见第 6 节）。
>
> ✅ **已于 2026-07-02 用 uv 搭好** `~/aa_workspace/.venv`（numpy/scipy/mne/matplotlib 齐全），详见第 7 节。

---

## 5. 数据集类型判断

| 数据集 | 信号类型 | 典型任务 | 标签 | 单样本量级 |
|--------|----------|----------|------|------------|
| **EMG (PR_raw)** | 表面肌电（多通道） | 运动意图/力估计回归 或 动作分类；分 `dynamic`/`maintenance` 两类任务 | label_*.mat（小，~250B） | 极大（.pt 单个 2.5~3.5 GB） |
| **epilepsy (CHB-MIT)** | 头皮 EEG（23 导联，1 小时/段） | 癫痫发作检测（二分类：发作期 vs 间期） | .edf.seizures / 18ch 的 _label.npy | 中（.edf 42 MB/段） |
| **sleep_stages (Sleep-EDF)** | PSG 多导睡眠图 | 5 类睡眠分期（W/N1/N2/N3/R） | Hypnogram.edf（30s epoch） | 中（.edf 数十 MB） |

三者都是**时间序列 / 生物电信号**，可走同一类技术栈：滑窗切分 → CNN/Transformer/TCN。

---

## 6. 最小 demo 推荐（从易到难，均不训练）

> 目标：验证环境能读到数据，**只读不训**。

**推荐首选：epilepsy 的 18ch npy 版本**——已预处理、文件小、可直接 numpy 加载、标签现成。

- **Step 0（必须先做）搭环境**
  ```bash
  sudo apt-get install -y python3-pip python3-venv   # 或装 miniconda
  python3 -m venv ~/aa_workspace/.venv
  source ~/aa_workspace/.venv/bin/activate
  pip install numpy scipy matplotlib mne              # 读 edf 需要 mne
  # 若要碰 .pt：pip install torch --index-url ...cpu
  ```

- **Step 1（最小 demo）读一个 18ch npy，打印形状 + 标签分布**
  ```python
  import numpy as np
  b = "/data/data/epilepsy/chb-mit/chb-mit_18ch"
  x = np.load(f"{b}/chb01_03.npy", mmap_mode="r")     # mmap，不占内存
  y = np.load(f"{b}/chb01_03_label.npy")
  print(x.shape, x.dtype, y.shape, np.unique(y, return_counts=True))
  ```
  预期：通道×时间窗形状 + `{0: 间期, 1: 发作期}` 分布（发作样本极少，类别极度不平衡）。

- **Step 2（可选）读一段 sleep hypnogram**，验证 mne 读 EDF 链路。
- **Step 3（更重，先别碰）** EMG 的 `.pt`（2.5 GB/个）需 torch，留到后续。

---

## 7. 已跑通的 demo ✅

> 2026-07-02 实跑。环境：uv + `~/aa_workspace/.venv`（见下）。**只读不训**。

### 环境搭建（免 sudo，已落地）
系统 Python 无 pip/venv（ensurepip 缺失），改用 **uv** 绕开：
```bash
# uv 装在 ~/.local/bin/uv（0.11.26）
~/.local/bin/uv venv ~/aa_workspace/.venv
~/.local/bin/uv pip install --python ~/aa_workspace/.venv/bin/python \
    numpy scipy mne matplotlib
# 版本：numpy 2.5.0 / scipy 1.18.0 / mne 1.12.1 / matplotlib 3.11.0
```
以后跑脚本统一用：`~/aa_workspace/.venv/bin/python <脚本>`（或 `source .venv/bin/activate`）。
脚本见 [code/demo_chb_npy.py](code/demo_chb_npy.py)。

### demo 1：epilepsy 18ch npy（`chb01_03`）
```
X  shape=(921600, 18)  dtype=float64   # 18 通道 × 921600 采样 = 3600s × 256Hz（1 小时）
Y  shape=(921600,)     dtype=float64   # 逐采样标签
  label 0.0: 911360 (98.89%)   # 间期
  label 1.0:  10240 ( 1.11%)   # 发作期 = 10240/256 = 40 秒发作段
```
**结论**：布局 `(时间采样, 通道)`，采样率 **256 Hz**，逐采样二分类标签，类别**极不平衡**（发作 ~1%）。
⚠️ 配对瑕疵：18ch 目录有 **138 个数据 .npy 但只有 137 个 _label.npy**（1 个记录缺标签，用前注意过滤）。

### demo 2：sleep PSG（mne 读 EDF，preload=False 只读头）
```
SC4001E0-PSG.edf:  sfreq=100Hz  通道=7  时长=1325min
通道: EEG Fpz-Cz, EEG Pz-Oz, EOG horizontal, Resp oro-nasal, EMG submental, Temp rectal, Event marker
```
**结论**：Sleep-EDF cassette 采样率 **100 Hz**，2 路 EEG + EOG/呼吸/EMG/体温/事件。

### demo 3：sleep hypnogram 标注（mne.read_annotations）
```
SC4001EC-Hypnogram.edf:  154 个标注
睡眠期: W / 1 / 2 / 3 / 4 / R / ?    # 标准 5 类 + 旧版 stage4 + 未知(?)
```
注意：PSG 后缀 `E0`，hypnogram 后缀 `EC`（同 subject 不同设备后缀，配对靠前 9 位 `SC4xxx`）。

### Baseline 实验：epilepsy 发作检测（LR）✅ 2026-07-02
脚本 [code/baseline_seizure_lr.py](code/baseline_seizure_lr.py)，产物 `outputs/baseline_metrics.json` + `outputs/roc_baseline.png`。
- 划分：**按病人** train chb01–20(115 记录) / val chb21–22(7) / test chb23–24(15)；2s 窗 stride 1s，窗内≥50% 发作为正。
- 特征：每通道 方差/平均绝对值/line length + δθαβ 相对功率（126 维）。负样本下采样 + class_weight=balanced。
- 结果（**accuracy 不可信**，看 AUC/F1）：

| | sens | spec | prec | F1 | AUC | acc |
|---|---|---|---|---|---|---|
| val  | 0.593 | 0.950 | 0.168 | 0.261 | **0.904** | 0.944 |
| test | 0.685 | 0.868 | 0.063 | 0.115 | **0.841** | 0.866 |

**判断**：AUC 0.84（跨病人）说明手工特征确有信号，是合格的第一个 baseline；precision/F1 低是**典型癫痫检测痛点**——13.2% FP × 7.4 万负窗 ≈ 9800 个 FP 淹没了 958 个正样本。耗时 259s。

### 下一步（按性价比排序）
- [ ] **阈值调优 + 时域后处理**（最便宜、F1 提升最大）：在 val 上选阈值，并对连续窗做 hysteresis/多数投票压掉孤立 FP
- [ ] 换小 1D-CNN（原始窗，需 `uv pip install torch`）
- [ ] sleep 分期：hypnogram 对齐 PSG 的 30s epoch，5 分类 toy
- [ ] EMG 的 `.pt`（2.5G/个）装 torch 后验证可读

---

## 附：历史检查记录（上一轮会话）

- `/data` 存在；`/data/data` 存在，约 329G（本次实测 330G，吻合）
- `/data/lost+found` 存在，约 16K
- `/datasets` 不存在或无权限
- `/mnt` 存在，基本为空，约 4.0K
- 初判：服务器主要数据在 `/data/data` —— **本次已确认**
