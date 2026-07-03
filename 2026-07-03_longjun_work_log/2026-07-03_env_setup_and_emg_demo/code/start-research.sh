#!/usr/bin/env bash
# =====================================================================
# 一键科研会话  (~/start-research.sh)
# 用途: 在 tmux 里打开你的科研工作区 aa_workspace，自动激活 Python 环境
# 场景: 在 MobaXterm / VSCode 终端里运行  bash ~/start-research.sh
#       关掉电脑前按 Ctrl+a d 脱离，任务继续在服务器后台跑；
#       回来再跑一次同样命令即可重新连上看结果。
# =====================================================================
export PATH="$HOME/.local/bin:$PATH"      # 确保能找到 tmux

SESSION="research"
WORKDIR="$HOME/aa_workspace"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "→ 已有 [$SESSION] 会话，直接连入（你之前的任务还在后台跑）"
    exec tmux attach -t "$SESSION"
else
    echo "→ 新建 [$SESSION] 会话，进入 $WORKDIR ..."
    tmux new-session -d -s "$SESSION" -c "$WORKDIR"
    tmux send-keys -t "$SESSION" "source .venv/bin/activate" C-m
    tmux send-keys -t "$SESSION" "clear && echo '✅ 已就绪：aa_workspace + venv。输入 claude 启动 AI（进 claude 后按 Shift+Tab 切自动模式）'" C-m
    exec tmux attach -t "$SESSION"
fi
