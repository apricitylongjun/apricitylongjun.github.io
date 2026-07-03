# 2026-07-03 · 深度学习环境从零搭建 + HD-EMG 分类 demo 跑通

## 1. 当日主线
从一台"裸"的共享服务器（只有 OS + GPU 驱动）出发，完成 **体检 → 修复 → 补全 → 跑通第一个真实数据训练**，并把整套科研环境固化。这是后续所有 HD-EMG（高密度肌电）实验的地基。

## 2. 当日目标（1-2 句）
搭好可用的深度学习环境（torch + GPU + 常用库 + tmux/claude/git），并用真实 HD-EMG 数据跑通一个分类 demo，验证"数据→GPU→训练→存档"全链路。

## 3. 时间线（逻辑链）⭐
| 时间 | 事件 | 产出 |
|---|---|---|
| 20:23 | 写 tmux 一键启动脚本 | `code/start-research.sh` |
| 21:04 | 体检服务器：发现 4×RTX5080，GPU3 被同事 guoyao 占用 | — |
| 21:10 | 写环境测试脚本 | `env_check/gpu_test.py`, `cnn_test.py` |
| 21:13 | 发现 venv 无 pip，首次装 torch 失败 | `env_check/install_torch.log` |
| 21:18 | get-pip 修复 + torch 2.12.1+cu130 装好（含 sm_120，支持 5080） | `env_check/install_torch.sh` |
| 21:19 | GPU 矩阵乘法(40 TFLOPS) + CNN 全链路验证通过 | `env_check/gpu_test.log`, `cnn_test.log` |
| 21:22 | 出完整体检报告 | `env_check/REPORT.md` |
| 21:37 | 补齐科研库（pandas/cv2/timm/transformers/einops/mne…） | — |
| 21:40 | 装 miniconda + git/htop/nvtop（conda-forge 绕过 ToS） | — |
| 21:47 | 写项目指南 | `notes/CLAUDE.md` |
| 21:54 | 写 HD-EMG 分类 demo（1D-CNN） | `code/emg_demo.py` |
| 21:51 | demo 首跑（30 epoch，batch=128）→ val_acc 0.233 | — |
| 21:55 | 改长跑版（200 epoch, batch=16）挂 tmux 后台 | — |
| 21:56 | **长跑完成，val_acc 0.799** | `results/emg_demo.log`, `emg_demo_best.pt` |
| 22:14 | tmux 配置（history-limit 50000→100000，mouse 已有） | `config_snapshot/key_settings.md` |
| 22:19 | 写 AI 自主任务文档（headless 执行被安全机制拦截） | `notes/AI_TASK.md` |

## 4. 文件清单 + 运行命令
### env_check/（环境体检）
- `REPORT.md` — 完整体检报告（系统/GPU/Python/数据集/网络 + 总结表 + 修复方案）
- `gpu_test.py` — GPU 矩阵乘法测试
- `cnn_test.py` — 3 层 CNN 全链路测试（随机数据）
- `install_torch.sh` — torch 安装脚本（含 get-pip 修复 + 验证）
- `*.log` — 对应运行日志
- 运行示例：`source ~/aa_workspace/.venv/bin/activate && CUDA_VISIBLE_DEVICES=0 python env_check/gpu_test.py`

### code/
- `emg_demo.py` — HD-EMG 分类 demo（1D-CNN，支持 `--epochs` `--batch`）
  - 运行：`source ~/aa_workspace/.venv/bin/activate && CUDA_VISIBLE_DEVICES=0 python code/emg_demo.py --epochs 200 --batch 16`
- `start-research.sh` — 一键进 tmux 科研会话（自动激活 venv）
  - 运行：`bash ~/start-research.sh`

### results/
- `emg_demo.log` — 训练完整日志（200 epoch）
- `emg_demo_best.pt` — 最佳模型权重（val_acc 0.799，175KB）

### notes/
- `CLAUDE.md` — 项目指南（GPU 约定 / 数据集 / 网络 / 科研诚信红线）
- `AI_TASK.md` — 给 Claude Code 的自主任务（3 模型×5 折对比实验，**未执行**，留作下次）
- `tmux-notes.md` — tmux 学习笔记

### config_snapshot/
- `key_settings.md` — 关键配置摘要（**已脱敏**，无 token/密钥）

## 5. 关键结果
### 环境体检结论
- 硬件：4×RTX 5080(16GB) + 64 核 + 251G 内存 + 7.3T 数据盘（/data，6.6T 空闲）
- torch 2.12.1+cu130，`cuda.is_available()=True`，4 卡全可见，arch 含 sm_120（原生支持 5080）
- 数据集（/data/data，全可读）：**EMG 255G（高密度肌电，PR_raw，.mat/.npy/.pt）**、epilepsy(CHB-MIT) 68G、sleep_stages 7.1G
- 网络：PyPI/清华源通；HuggingFace/GitHub 不通 → 已配 hf-mirror + ghproxy

### HD-EMG demo 训练结果
| 配置 | epoch | batch | val_acc |
|---|---|---|---|
| 首跑 | 30 | 128 | 0.233 |
| 长跑 | 200 | 16 | **0.799** |
- 数据：`features_all.npz`（4030 样本，4×256 特征）
- 模型：1D-CNN（Conv1d 32→64→128 + AvgPool + Dropout + Linear）
- 结论：batch 小 + epoch 多，精度从 23% 提升到 80%（此为 demo 验证链路，非最优模型）

## 6. 复现环境
- venv：`~/aa_workspace/.venv`（Python 3.12, torch 2.12.1+cu130）
- 激活：`source ~/aa_workspace/.venv/bin/activate`
- 数据：`/data/data/EMG/**/features_all.npz`
- GPU：`export CUDA_VISIBLE_DEVICES=0`（**只用 GPU0/1/2，GPU3 同事在用**）
- 关键依赖：torch torchvision pandas opencv-python-headless timm transformers einops mne scikit-learn h5py tensorboard

## 7. 遇到的问题（现象 + 原因 + 状态）
1. **venv 无 pip**：创建 venv 时未带 pip 且 ensurepip 被剥离 → 用 `get-pip.py` 修复（pip 26.1.2）。✅ 已解决
2. **conda ToS 拦截**：新版 conda 要求接受 anaconda 官方源服务条款，git/htop 装不上 → 改用 conda-forge 源（`--override-channels`）绕过，不替用户接受条款。✅ 已解决
3. **headless Claude Code 被安全机制拦**：试图用 `--dangerously-skip-permissions` 启动无人值守 agent 被系统拒绝（共享服务器 + 无限制执行风险）→ 改为交互式 claude + Shift+Tab auto 模式（用户亲自监督）。⚠️ 已规避，待用户实操

## 8. 关联笔记
- `notes/tmux-notes.md`（tmux 学习要点）
- 长期方向：高密度肌电 HD-EMG；环境已就绪，下一步回归"看论文 + 找具体目标"
