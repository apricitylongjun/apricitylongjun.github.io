# 202607022122 — epilepsy 发作检测最小 baseline（滑窗+手工特征+LR）

## 1. 摘要 / 目标
在上一篇（202607022100）搭好的 uv 环境上，跑通 epilepsy 发作检测的**第一个建模实验**：2s 滑窗 + 126 维手工特征 + 逻辑回归，按病人留出划分。目标是验证"切窗 → 特征 → 划分 → 训练 → 评估"整条管线能跑通，并拿到一个诚实的跨病人 baseline 数字，定位下一步改进方向。全程未改动原始数据。

## 2. 背景信息
- 数据：`/data/data/epilepsy/chb-mit/chb-mit_18ch`，137 个记录（已预筛为含发作段），每条 `(921600, 18)` = 256Hz × 1h × 18 通道，逐采样二分类标签。
- 环境：`~/aa_workspace/.venv`（numpy/scipy/mne/matplotlib），本阶段新增 scikit-learn 1.9.0。
- 已知难点：正样本仅 ~1.66%，极度不平衡 → accuracy 不可信，需看 AUC/F1。

## 3. 今日操作记录
1. **摸正样本分布**（决定 toy 可行性）：扫全部 137 个标签文件 → 137/137 含发作、总发作 ~3 小时（11116 秒）、全局正样本 1.66%；24 个病人都有发作记录（chb16 最少 75s，chb15 最多 1992s）。
2. **定方案**（方法论）：按病人留出 train chb01–20 / val chb21–22 / test chb23–24，杜绝同病人泄漏；2s 窗(512 样本)、stride 1s、窗内 ≥50% 发作判正；每通道 7 特征(方差/均值绝对值/line length + δθαβ 相对功率)×18 = 126 维；`class_weight=balanced` + 训练集每记录负窗下采样到 300。
3. **装 sklearn**：`uv pip install scikit-learn`（1.9.0，含 joblib/narwhals/threadpoolctl）。
4. **写脚本** `code/baseline_seizure_lr.py`：特征提取用 mmap 读信号、向量化 FFT，避免 Python 循环。
5. **跑通评估**（耗时 259s）：出 val/test 的 sens/spec/prec/F1/AUC + ROC 图。

## 4. 遇到的问题
- **precision 极低（test 0.063）/ F1 低（0.115）**：现象——AUC 不差(0.84)但 precision 只有 6.3%。原因判断——基率 1.27%，spec 86.8% 对应 7.4 万负窗里冒出 ~9800 个假阳，淹没了 958 个真阳；这是 CHB-MIT 跨病人检测的经典痛点，不是管线 bug。解决办法——阈值调优（在 val 上选工作点而非写死 0.5）+ 时域后处理（连续窗 hysteresis/多数投票压孤立 FP）。状态：⚠️ 已定位，列入下一步首选。
- **accuracy 误导**：现象 test acc=0.866 看着不差。原因——极度不平衡下"基本全判负 + 一堆 FP"也能拿高 acc。解决——明确以 AUC/F1/sens/spec 为主指标，acc 仅参考。状态：✅ 已在脚本输出与笔记中标注。
- **sklearn `n_jobs` 弃用告警**：现象 LR 传入 `n_jobs=-1` 触发 FutureWarning。原因——sklearn 1.8+ 的 LR 已内部并行，该参数将移除。解决：无害，下次去掉即可。状态：➖ 不影响结果。

## 5. 当前结论
- ✅ 整条训练管线跑通且可复现（脚本 + 固定 seed），产物落盘 `outputs/`。
- ✅ **test AUC 0.841（跨病人）** 说明 126 维手工特征确有判别力，是合格的第一个 baseline。
- ⚠️ 短板是**工作点与假阳抑制**，不是特征/模型本身——这是下一步最高性价比的改进点。

## 6. 下一步计划
- [ ] **阈值调优 + 时域后处理**（最便宜、F1 提升最大）：val 上扫阈值选工作点；连续窗做 hysteresis/多数投票压孤立 FP，预期 precision 翻倍
- [ ] 换小 1D-CNN（原始窗，需 `uv pip install torch`），对比 LR
- [ ] sleep 分期：hypnogram 对齐 PSG 的 30s epoch，5 分类 toy
- [ ] 装 torch 验证 EMG 的 `.pt`（2.5G/个）可读
- [ ] 本地 `dataset_notes.md` 与 Notion 保持同步

## 7. 给老师看的简短汇报
已完成癫痫发作检测的第一个最小 baseline：用 2 秒滑窗 + 手工时频特征 + 逻辑回归，按病人严格划分（训练 chb01–20，测试 chb23/24）。在未见过的病人上 AUC 达到 0.84，说明特征有判别力、管线可用；目前主要问题是假阳性偏多（精确率 6.3%），这是该数据集跨病人检测的典型难点。下一步计划做阈值优化和时域后处理来压低假阳，并尝试用小 CNN 替换逻辑回归作对比。全程只读，未改动原始数据。

## 8. 附录：原始日志 / 重要路径
**重要路径**
- 脚本：`~/aa_workspace/code/baseline_seizure_lr.py`
- 产物：`~/aa_workspace/outputs/baseline_metrics.json`、`outputs/roc_baseline.png`
- 数据：`/data/data/epilepsy/chb-mit/chb-mit_18ch/*.npy`

**关键日志（节选）**
```
记录数: train=115 val=7 test=15
train (44432, 126) pos=22.35% | val pos=1.67% | test pos=1.27%

=== 结果 ===
[val]  sens=0.593 spec=0.950 prec=0.168 f1=0.261 auc=0.904 acc=0.944
[test] sens=0.685 spec=0.868 prec=0.063 f1=0.115 auc=0.841 acc=0.866
耗时 259.4s
已保存: outputs/baseline_metrics.json, outputs/roc_baseline.png
```
