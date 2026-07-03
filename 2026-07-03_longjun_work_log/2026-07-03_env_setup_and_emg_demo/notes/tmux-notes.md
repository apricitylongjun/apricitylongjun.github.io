# tmux 速查笔记

> tmux = terminal multiplexer。核心价值:**让程序在后台持续运行**,断开 SSH / 关掉终端都不会死。适合跑长任务(模型训练、服务、监控)。
> 所有快捷键都是「**先按前缀键 `Ctrl+b`,松开,再按功能键**」这个两步动作。

---

## 0. 三个层级(先搞清楚这个)

| 层级 | 类比 | 说明 |
|------|------|------|
| **会话 session** | 工作台 | 最大的单位,一个会话里可以有多个窗口 |
| **窗口 window** | 浏览器标签页 | 一个窗口占满整个屏幕,在底部显示标签栏 |
| **面板 pane** | 分屏 | 把一个窗口切成几块,同时看几个东西 |

---

## 1. 会话管理(最常用)

```bash
tmux                    # 新建匿名会话(不推荐,名字随机)
tmux new -s 名字        # 新建并命名(推荐!)
tmux ls                 # 查看现有会话
tmux a -t 名字          # 进入(attach)指定会话
tmux a                  # 只有一个会话时直接进入
tmux kill-session -t 名字   # 杀掉指定会话
tmux kill-server        # 杀掉所有会话(慎用)
```

**脱离会话(保留运行,退出登录也不死):**

```
Ctrl+b  然后 d
```

> ⭐ 最重要的一条:跑长任务时,**永远用 detach 退出**,不要直接关终端(关了其实也不会死,但 detach 更干净)。

---

## 2. 窗口管理(标签页式多任务)

进入会话后,用快捷键操作:

| 操作 | 快捷键 |
|------|--------|
| 新建窗口 | `Ctrl+b` → `c` |
| 下一个窗口 | `Ctrl+b` → `n` |
| 上一个窗口 | `Ctrl+b` → `p` |
| 列出/选择窗口 | `Ctrl+b` → `w` |
| 重命名当前窗口 | `Ctrl+b` → `,` |
| 关闭当前窗口 | `Ctrl+b` → `&` |
| 按编号切窗口 | `Ctrl+b` → `数字` |

**会话间切换**(有多个 session 时):`Ctrl+b` → `s`

---

## 3. 面板管理(分屏)

| 操作 | 快捷键 |
|------|--------|
| 左右分屏 | `Ctrl+b` → `%` |
| 上下分屏 | `Ctrl+b` → `"` |
| 在面板间移动 | `Ctrl+b` → `方向键` |
| 关闭当前面板 | `Ctrl+b` → `x`(确认 y) 或直接 `exit` |
| 当前面板全屏/还原 | `Ctrl+b` → `z` |
| 调整面板大小 | `Ctrl+b` 长按 `方向键`(或先按住不放) |
| 面板布局自动重排 | `Ctrl+b` → `空格` |

---

## 4. 实用进阶

**复制模式(向上翻页 / 查看历史输出):**
```
Ctrl+b → [        进入复制模式(可用方向键/PageUp 翻页)
q                 退出复制模式
```

**滚屏:** 如果配置了鼠标,直接滚轮即可;否则进复制模式翻页。

**使能鼠标**(临时,重启失效):
```
tmux 内输入:    tmux set -g mouse on
```

---

## 5. 我(用户)的实际场景示例

当前正在跑的会话(2026-07-03):

```bash
tmux ls
# emg_demo:  1 windows   (EMG 肌电 demo)
# emg_train: 1 windows   (EMG 肌电训练)
```

典型工作流:

```bash
# 1. 开训练会话,起名字
tmux new -s emg_train

# 2. 在里面启动训练脚本(如 python train.py)
python train.py

# 3. 训练中想干别的事 → 新开一个窗口跑 demo
Ctrl+b → c        # 新窗口
python demo.py

# 4. 切回训练窗口看进度
Ctrl+b → 0        # 切到 0 号窗口(训练)

# 5. 断开但不杀进程(关电脑、断网都不会停训练)
Ctrl+b → d

# 6. 下次接回来继续看
tmux a -t emg_train
```

---

## 6. 推荐配置(~/.tmux.conf)

把下面内容写进 `~/.tmux.conf`,然后 `tmux source ~/.tmux.conf` 生效:

```bash
# 把前缀键从 Ctrl+b 改成 Ctrl+a(更顺手,可选)
set -g prefix C-a
unbind C-b
bind C-a send-prefix

# 开启鼠标(滚轮翻页、点窗口标签、拖动面板边框)
set -g mouse on

# 窗口编号从 1 开始(0 在键盘最左边,不顺手)
set -g base-index 1
setw -g pane-base-index 1

# 关掉窗口后自动重新编号
set -g renumber-windows on

# 历史记录加大(默认 2000 行太少)
set -g history-limit 50000

# 状态栏显示会话名和窗口名(方便多会话时认)
set -g status-style bg=default
```

---

## 7. 速记口诀

- **会话**:`tmux new -s 名` / `tmux a -t 名` / `Ctrl+b d` 脱离
- **窗口**:新 `c` / 切 `n p 数字` / 选 `w` / 重名 `,`
- **面板**:竖 `%` / 横 `"` / 移 `方向键` / 全屏 `z` / 关 `x`
- **救命三连**:`Ctrl+b d`(脱离)`Ctrl+b s`(切会话)`Ctrl+b [`(翻页)

---
