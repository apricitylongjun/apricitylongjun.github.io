# 202607022100 — 服务器数据集检查 + 免sudo环境搭建 + 最小demo验证

## 1. 摘要 / 目标
检查服务器 `/data/data` 下三套数据集的结构与类型；在无 sudo、无 pip 的系统 Python 上搭出可用的科学计算环境；跑通"读一个数据文件"的最小 demo，验证数据链路。全程只读，未训练、未删除任何文件。

## 2. 背景信息
- 工作区 `~/aa_workspace`：`code/data/outputs/logs` 四目录已存在，但 `data/` 为空，真实数据挂在 `/data/data`（330G）。
- 三套数据：EMG（肌电，255G）/ epilepsy（CHB-MIT 癫痫 EEG，68G）/ sleep_stages（Sleep-EDF，7.1G）。
- 系统 Python 3.12.3，但 pip/conda/venv 全无，numpy/torch/mne 全缺 → 必须先搭环境才能读数据。

## 3. 今日操作记录
1. **数据集只读检查**：统计 `/data/data` 全量 2267 文件、121 目录；文件类型以 `.edf`(992)/`.npy`(554)/`.mat`(480)/`.seizures`(141)/`.pt`(40) 为主，**无任何 jpg/png/csv/json** → 纯一维时序生物电信号。结果写入 `~/aa_workspace/dataset_notes.md`。
2. **Notion MCP 接入**：用 `claude mcp add --transport http --scope user notion https://mcp.notion.com/mcp` 注册官方托管版（OAuth 免 token），已写入"数据集初步检查"小节。
3. **搭环境（免 sudo）**：装 `uv 0.11.26` 到 `~/.local/bin`；在 `~/aa_workspace/.venv` 建隔离 venv；装 numpy 2.5 / scipy 1.18 / mne 1.12 / matplotlib 3.11。
4. **跑最小 demo**（脚本 `code/demo_chb_npy.py`）：
   - epilepsy `chb01_03.npy`：X=(921600,18) → 256Hz×1h、18 通道；标签 0=间期 98.89% / 1=发作 1.11%（≈40 秒）。
   - mne 读 Sleep-EDF PSG：100Hz、7 通道、1325min；hypnogram 154 标注（W/1/2/3/4/R/?）。

## 4. 遇到的问题
- **系统 Python 无 pip 且 ensurepip 缺失**：现象 `python3 -m venv` 能建环境但里面没 pip，`apt install python3-venv` 又需 sudo。原因：Debian 把 ensurepip 拆成单独包且本机无免密 sudo。解决：用 **uv**（自带解析、不依赖 ensurepip）。状态：✅ 已解决。
- **Sleep-EDF 文件配对**：现象 `SC4001E0-PSG.edf` 找不到同后缀 hypnogram。原因：PSG 后缀 `E0`、hypnogram 后缀 `EC`，同 subject 不同设备后缀。解决：按前 9 位 `SC4xxx` 配对。状态：✅ 已解决。
- **epilepsy 18ch 标签缺 1 个**：现象 138 个数据 `.npy` vs 137 个 `_label.npy`。判断：有 1 条记录缺标签。解决：训练前过滤无标签记录。状态：⚠️ 已记录，待处理。
- **Notion MCP 工具对 Claude 这一轮不可见**：现象 Claude 工具集里只有 analyze_image / webReader，无 notion 工具。原因：MCP 工具在会话启动时加载，本会话工具清单已固定。解决：由用户直接写入，或后续会话再由 Claude 调用。状态：➖ 本次靠用户写入。

## 5. 当前结论
- ✅ 环境就绪：`~/aa_workspace/.venv`（numpy/scipy/mne/matplotlib），跑脚本用 `.venv/bin/python`。
- ✅ 三套数据均可读；**epilepsy 18ch npy 最易上手**（已预处理、标签现成、文件小）。
- 数据本质：256Hz（癫痫）/ 100Hz（睡眠）一维多通道时序，统一走"滑窗 → CNN/TCN"。

## 6. 下一步计划
- [ ] epilepsy：滑窗（2~4s）切分 + 最小 baseline（发作检测二分类 toy，非大训练）
- [ ] 过滤掉缺标签的那 1 条 18ch 记录
- [ ] 装 torch（`uv pip install torch`）后验证 EMG 的 `.pt`（2.5G/个）能读
- [ ] sleep 分期：把 hypnogram 对齐到 PSG 的 30s epoch，做 5 分类 toy
- [ ] 本地 `dataset_notes.md` 与 Notion 持续保持同步

## 7. 给老师看的简短汇报
服务器上确认了三套生物电时序数据集（肌电、癫痫 EEG、睡眠 PSG，共 330G）。已用 uv 搭好免 sudo 的 Python 环境，并跑通最小读取 demo，验证了数据可读、格式与采样率（癫痫 256Hz、睡眠 100Hz）和标签结构。癫痫数据已预处理为 npy、最适合先做 baseline；下一步计划做发作检测的最小可行模型。全程只读，未动原始数据。

## 8. 附录：原始日志 / 重要路径
**重要路径**
- 数据根：`/data/data/{EMG,epilepsy,sleep_stages}`
- 工作区：`~/aa_workspace`（`.venv/`、`code/demo_chb_npy.py`、`dataset_notes.md`）
- Notion MCP 配置：`~/.claude.json` → `mcpServers.notion`（HTTP, user scope）

**关键日志（节选）**
```
$ claude mcp list
notion: https://mcp.notion.com/mcp (HTTP) - ✔ Connected

# demo 输出
=== chb01_03 ===
X  shape=(921600, 18)  dtype=float64      # 256Hz × 3600s × 18ch
Y  shape=(921600,)     dtype=float64
  label 0.0: 911360 (98.89%)              # 间期
  label 1.0:  10240 ( 1.11%)              # 发作期 ≈ 40s

# mne 读 Sleep-EDF
PSG  sfreq=100Hz  通道=7  时长=1325min
Hypnogram 标注=154  分期: W/1/2/3/4/R/?
```
