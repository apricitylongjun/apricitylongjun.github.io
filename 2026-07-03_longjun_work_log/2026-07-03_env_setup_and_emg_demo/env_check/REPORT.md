# 🩺 深度学习科研环境体检报告

- 服务器：主机名 `ubuntu` · Ubuntu 24.04.4 LTS · 内核 6.17.0-35-generic x86_64
- 体检日期：2026-07-03 · 受检用户：`longjun`
- 全程**只读**，未修改任何系统文件；测试脚本与日志存于 `~/env_check/`

---

## 一、系统与权限

| 项 | 结果 | 状态 |
|---|---|---|
| 操作系统 | Ubuntu 24.04.4 LTS (Noble) | ✅ |
| 内核 | 6.17.0-35-generic x86_64 | ✅ |
| 当前用户 | longjun (uid 1003)，仅属 longjun 组 | ✅ |
| home 目录 | /home/longjun | ✅ |
| sudo 权限 | 非免密（需密码），不在 sudo/wheel 组 | ⚠️ |
| home 所在盘 | 877G 总 / **710G 可用**（15% 已用） | ✅ |
| 数据盘 /data | 7.3T 总 / **6.6T 可用**（5% 已用） | ✅ |
| 磁盘配额 | 无 | ✅ |

> **sudo 说明**：跑深度学习算法**用不到 sudo**（pip 装包、训练都在你的 home 内）。本报告所有补全方案均**无需 sudo**。

## 二、GPU 与驱动

| 项 | 结果 | 状态 |
|---|---|---|
| GPU | **4 × NVIDIA RTX 5080（每张 16GB 显存）** | ✅ |
| 驱动版本 | 595.71.05 | ✅ |
| nvidia-smi 显示 CUDA | 13.2（驱动支持的最高版，非已装版） | ✅ |
| nvcc（CUDA Toolkit） | 未安装 | ⚠️ 不影响 |
| cuDNN | torch 自带 nvidia-cudnn-cu13 **9.20.0** | ✅ |
| GPU 占用 | GPU0/1/2 空闲；**GPU3 被 `guoyao` 占用（2.9GB，跑癫痫 anomaly_transformer）** | ⚠️ |
| /dev/nvidia* 权限 | 所有用户可读写 | ✅ |

> - **nvcc/cuDNN**：没装独立 CUDA Toolkit 很正常——pip 装的 torch **自带完整 CUDA Runtime + cuDNN**，跑模型足够。nvcc 只有自己编译 C++/CUDA 算子时才需要。
> - **多用户共享**：guoyao 在用 GPU3，你跑任务用 `CUDA_VISIBLE_DEVICES=0,1,2`（空闲 3 张），避免互抢。

## 三、Python 与深度学习框架

| 项 | 结果 | 状态 |
|---|---|---|
| 系统 Python | 3.12.3 (/usr/bin/python3) | ✅ |
| 科研环境 venv | `~/aa_workspace/.venv`（Python 3.12.3） | ✅ |
| conda/mamba | 未安装 | ⚠️ 可选 |
| **PyTorch** | **2.12.1+cu130（本次新装，含 sm_120，完美支持 5080）** | ✅ |
| torchvision | 0.27.1 | ✅ |
| `torch.cuda.is_available()` | **True**，可见 **4 张卡** | ✅ |
| numpy / scipy / sklearn / matplotlib | 2.5.0 / 1.18 / 1.9 / 3.11 | ✅ |
| pandas / opencv(cv2) | 未装 | ❌ |
| timm / transformers / einops | 未装 | ❌ |
| tensorboard / wandb | 未装 | ❌ |
| tensorflow / jax | 未装 | ❌（用 torch 即可，无需装） |

> 体检中发现 venv **原本连 pip 都没有**（创建时未带），已用 `get-pip.py` 修复，现 pip 26.1.2 正常。

## 四、实际跑通验证 ✅（关键）

| 验证项 | 结果 |
|---|---|
| GPU 矩阵乘法 | 4096×4096 ×20 次，**平均 3.41 ms/次，≈40 TFLOPS**（RTX 5080） ✅ |
| 3 层 CNN 训练 2 epoch | 前向+反向+优化器+GPU **全链路正常**，loss 正常回传 ✅ |
| 多卡识别 | 4 张 RTX 5080 全部可见 ✅ |

脚本：`gpu_test.py`、`cnn_test.py`；日志：`gpu_test.log`、`cnn_test.log`。

> **结论：环境现在可以跑深度学习了。** CNN 用随机数据故 loss 不收敛（≈2.30，10 类随机基线），属预期——验证链路而非精度。

## 五、数据集盘点（/data/data，7.3T 数据盘）

| 数据集 | 大小 | 格式 | 说明 | 状态 |
|---|---|---|---|---|
| **EMG** | **255G** | .mat×480 + .npy×159 + .pt×40 + .npz×2 | 🎯 **高密度肌电（`PR_raw/`），真实方向数据，已预处理可直接读** | ✅ 可读 |
| epilepsy | 68G | .edf×686 + .npy×275 + .seizures | CHB-MIT 癫痫（入门 demo，guoyao 也在用） | ✅ 可读 |
| sleep_stages | 7.1G | sleep_edf | 睡眠分期 | ✅ 可读 |

> 三个数据集权限均为 `rwxrwxrwx`，**你都能直接读**。EMG 的 .mat/.npy/.pt 用 `scipy.io.loadmat` / `numpy.load` / `torch.load` 即可加载。

## 六、网络与工具

| 项 | 结果 | 状态 |
|---|---|---|
| PyPI 官方 / 清华 PyPI 源 | 可达 (200 / 302) | ✅ |
| 清华镜像站 | 可达 (200) | ✅ |
| **HuggingFace** | **不通** | ❌ |
| **GitHub** | **不通** | ❌ |
| tmux / gcc / make / wget / curl / python3 | 已装 | ✅ |
| **git** | **未装** | ❌ |
| htop / nvtop | 未装 | ⚠️ 可选 |
| CPU / 内存 | **64 核 / 251G 内存（空闲 141G）** | ✅ |

> **HF/GitHub 不通的影响**：下 HuggingFace 预训练权重、`git clone` GitHub 仓库会失败。**对策**：HF 走 `hf-mirror.com` 镜像，GitHub 走代理/离线传。

## 七、总结：要实现无人值守自动跑实验，还缺什么（全部无 sudo）

**总体**：硬件豪华（4×5080 + 64 核 + 251G 内存 + 7T 数据盘），torch 已跑通，**主干道已通**。剩余是补常用库 + 配镜像 + 养成 GPU 调度习惯。

| 优先级 | 待补项 | 怎么补（无 sudo） |
|---|---|---|
| 🔴 高 | 补常用科研库（pandas/opencv/timm/transformers/einops/tensorboard/h5py…） | `~/aa_workspace/.venv/bin/python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas opencv-python timm transformers einops tensorboard scikit-image h5py tqdm` |
| 🔴 高 | HuggingFace 不通（下权重会失败） | `echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc`（用国内镜像） |
| 🔴 高 | GPU 调度（避免和 guoyao 抢卡） | 跑任务前 `export CUDA_VISIBLE_DEVICES=0,1,2`；写进 CLAUDE.md 让 AI 自动遵守 |
| 🟡 中 | git（本地版本管理） | 装 miniconda 到 home：`wget 清华/.../Miniconda3...sh && bash ... -b -p ~/miniconda`，再 `conda install git htop` |
| 🟡 中 | GitHub 不通（clone 仓库） | 用 ghproxy 镜像，或下载 zip 离线上传 |
| 🟢 低 | htop/nvtop（GPU 监控） | 同上 conda 装；或临时用 `nvidia-smi -l 2` |
| 🟢 低 | tensorflow/jax | 不需要，torch 已够 |

## 附录：本次产出文件
```
~/env_check/
├── gpu_test.py         # GPU 矩阵乘法测试脚本
├── cnn_test.py         # CNN 全链路测试脚本
├── gpu_test.log        # GPU 测试输出
├── cnn_test.log        # CNN 测试输出
├── install_torch.sh    # 装 torch 的脚本（含 get-pip 修复）
├── install_torch.log   # 安装日志
└── REPORT.md           # 本报告
```
