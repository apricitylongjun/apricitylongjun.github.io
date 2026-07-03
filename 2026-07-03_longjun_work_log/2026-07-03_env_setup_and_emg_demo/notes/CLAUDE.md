# aa_workspace — 科研工作区指南（给 Claude Code 看）

> 本文件让 AI 助手一进入项目就懂环境、数据、约定。更新于 2026-07-03。

## 我是谁 / 在做什么
研究生，方向 **高密度肌电 (HD-EMG)** 信号处理 + 深度学习（已确认，2026-07-03）。
当前在跑基线算法、尝试把图像/时序模型移植到 EMG 领域；入门 demo 用 CHB-MIT 癫痫数据。

## 运行环境（关键）
- Python 虚拟环境：`~/aa_workspace/.venv`（Python 3.12，torch 2.12.1+cu130，原生支持 RTX 5080 / sm_120）
- **每次跑代码前先激活**：`source ~/aa_workspace/.venv/bin/activate`
- 装包默认已走清华源（见 ~/.config/pip/pip.conf）：`pip install <包>` 即可

## GPU 调度约定（重要！多用户共享服务器）
- 4× RTX 5080，**GPU3 被同事 guoyao 长期占用**，**只能用 GPU 0/1/2**
- 跑训练/推理前**必须**先：`export CUDA_VISIBLE_DEVICES=0,1,2`
- 跑前看一眼占用：`nvidia-smi`

## 数据集（/data/data，公共只读，全部可直接读）
| 数据集 | 路径 | 格式 | 读取 |
|---|---|---|---|
| **EMG（你的方向）** | /data/data/EMG/PR_raw | .mat/.npy/.pt/.npz | scipy.io.loadmat / numpy.load / torch.load |
| 癫痫 CHB-MIT | /data/data/epilepsy/chb-mit | .edf/.npy/.seizures | mne 或 pyedflib 读 .edf |
| 睡眠 sleep_edf | /data/data/sleep_stages | .edf | mne |

> .mat 若为 v7.3 格式用 `h5py` 读；旧版用 `scipy.io.loadmat`。

## 目录约定
`code/` 代码 · `data/` 自己的小数据 · `outputs/` 实验结果 · `logs/` 日志 · `daily/` 每日归档。
重要结果写进 `outputs/` 并在 `logs/` 留文字记录。

## 网络
- pip 清华源 ✅；**HuggingFace 走镜像**：`HF_ENDPOINT=https://hf-mirror.com`（已写入 .bashrc）
- GitHub 不通：clone 走 ghproxy（git 已全局配 insteadOf），或下载 zip 离线上传
- 下模型权重优先 hf-mirror，其次离线传入

## 红线（科研诚信）
- 跑出的指标必须能解释；警惕**数据泄漏 / 评估作弊 / 过拟合**导致的虚高数字
- 不确定的结果先标记存疑，不要当结论写进笔记
