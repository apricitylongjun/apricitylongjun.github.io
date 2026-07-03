# 关键配置摘要（2026-07-03）

> ⚠️ **已脱敏**：本文件只记录配置项，不含任何 token / API key / 密码。
> 原始的 `~/.claude.json`（API 配置）、`~/.conda/*`（anaconda token）、`~/.condarc`、`~/.bash_history` 等**未纳入归档**，防止泄露。

## 网络 / 源
- pip 默认清华源：`~/.config/pip/pip.conf`
  - `index-url = https://pypi.tuna.tsinghua.edu.cn/simple`
- HuggingFace 镜像：`~/.bashrc` → `export HF_ENDPOINT=https://hf-mirror.com`（HF 官方不通）
- conda 源：`~/miniconda`，channel = 清华 conda-forge 镜像，`auto_activate=false`

## git
- `user.name = longjun`（user.email 待用户设置）
- GitHub 代理（不通 github.com）：`git config --global url."https://ghproxy.com/https://github.com/".insteadOf "https://github.com/"`
- 软链：git/htop/nvtop → `~/.local/bin/`（全局可用）

## tmux（~/.tmux.conf 关键项）
- 前缀键：`Ctrl+a`（非默认 Ctrl+b）
- `set -g mouse on`（鼠标点击/滚轮/拖拽）
- `set -g history-limit 100000`（今日从 50000 调高）
- 分屏：`|` 左右、`-` 上下；切面板 vim 风格 `hjkl`；vi 复制模式（`v` 选 `y` 复制）

## GPU 调度约定（多用户共享，重要）
- 4×RTX 5080，**只用 GPU0/1/2**：`export CUDA_VISIBLE_DEVICES=0,1,2`
- **GPU3 被同事 guoyao 长期占用**（跑癫痫 anomaly_transformer），勿碰

## 环境位置
- 科研 venv：`~/aa_workspace/.venv`
- 一键进 tmux 会话：`bash ~/start-research.sh`
- 体检报告：`~/env_check/REPORT.md`
