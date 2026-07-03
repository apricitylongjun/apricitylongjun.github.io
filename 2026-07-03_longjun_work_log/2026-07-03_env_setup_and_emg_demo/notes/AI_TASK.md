# 任务：HD-EMG 分类模型对比实验（自主完成，不要询问，直接动手）

你在 ~/aa_workspace 科研工作区（高密度肌电 HD-EMG 分类）。请自主完成以下实验。

## 数据
- 路径：`/data/data/EMG/**/features_all.npz`（用 glob 找，约 1 个）
- 格式：`data` 形状 (N, 4, 256) float，`label` 形状 (N,) int
- 参考 `~/aa_workspace/code/emg_demo.py` 的加载、标签重映射、通道级 z-score 归一化方式

## 要求
1. 实现 **3 个模型**并对比：
   - **M1 1D-CNN**（类似 emg_demo.py 的 EMGNet）
   - **M2 CNN + MultiHead Self-Attention**（CNN 提特征后接 1 层多头注意力再分类）
   - **M3 小型 Transformer Encoder**（把 (4,256) 当序列，2 层 transformer encoder + 分类头）
2. 用 **5 折交叉验证**（KFold, seed=42），每个模型每折训练 **30 epoch**
3. GPU：只用 **GPU0**（`CUDA_VISIBLE_DEVICES=0`，进程内应已设置；**GPU3 是同事在用，绝对不要碰**）
4. 每折记录最佳 val_acc；全部追加写进 `~/aa_workspace/logs/model_comparison.log`（带时间戳）
5. 训练代码写到 `~/aa_workspace/code/model_comparison.py`
6. 全部跑完后：
   - 在日志末尾打印**对比表**（各模型 5 折平均 val_acc ± 标准差）
   - 把 val_acc 最高的模型存到 `~/aa_workspace/outputs/best_model.pt`
   - 打印一句总结：哪个模型最好、平均 val_acc 多少

## 约束
- 在 venv 跑：先 `source ~/aa_workspace/.venv/bin/activate`
- 不要修改 /data 下原始数据（只读）
- 不要用 sudo
- 完成后简短总结（不超过 10 行）
