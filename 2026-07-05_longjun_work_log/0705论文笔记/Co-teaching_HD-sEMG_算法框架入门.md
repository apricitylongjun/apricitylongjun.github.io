# Co-teaching 算法框架入门：面向高密度肌电 HD-sEMG 噪声标签学习

> 目标：让你能把 **Co-teaching** 这一个 LNL 经典算法框架完整复述给老师。  
> 适用背景：高密度肌电 HD-sEMG 手势识别，数据中可能存在错误标签、动作边界不准、相似手势混淆、过渡段标签不确定等问题。

---

## 0. 文章参考

### 0.1 主论文

**论文标题**  
Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels

**作者**  
Bo Han, Quanming Yao, Xingrui Yu, Gang Niu, Miao Xu, Weihua Hu, Ivor W. Tsang, Masashi Sugiyama

**会议**  
NeurIPS 2018

**核心方法名**  
Co-teaching

**核心关键词**  
noisy labels, memorization effect, small-loss samples, dual networks, sample selection

**论文链接**  
- arXiv: https://arxiv.org/abs/1804.06872
- NeurIPS PDF: https://papers.neurips.cc/paper_files/paper/2018/file/a19744e268754fb0148b017647355b7b-Paper.pdf
- 官方代码: https://github.com/bhanML/Co-teaching

### 0.2 与 Co-teaching 相关的后续论文

| 论文 | 方法 | 和 Co-teaching 的关系 |
|---|---|---|
| How does Disagreement Help Generalization against Label Corruption? | Co-teaching+ | 在 Co-teaching 基础上加入 disagreement 策略 |
| Combating Noisy Labels by Agreement: A Joint Training Method with Co-Regularization | JoCoR | 反过来强调 agreement，而不是 disagreement |
| DivideMix: Learning with Noisy Labels as Semi-supervised Learning | DivideMix | 先分 clean/noisy，再把 noisy 当作 unlabeled 用半监督学习 |
| FINE Samples for Learning with Noisy Labels | FINE | 不只看 loss，而是看 feature geometry 选干净样本 |

---

## 1. 先弄懂任务：什么是 noisy label learning？

你的数据是高密度肌电 HD-sEMG。一个样本可以理解为：

```text
输入 x：一段高密度肌电信号
标签 y：这个信号对应的手势类别
```

例如：

| 样本 | 输入 | 标签 |
|---|---|---|
| sample 1 | 一段 HD-sEMG 矩阵 | 握拳 |
| sample 2 | 一段 HD-sEMG 矩阵 | 伸腕 |
| sample 3 | 一段 HD-sEMG 矩阵 | 捏合 |
| sample 4 | 一段 HD-sEMG 矩阵 | 放松 |

普通监督学习默认标签都是正确的。但在真实 HD-sEMG 数据里，标签可能不完全可信。

常见错误包括：

| 噪声类型 | 在 HD-sEMG 中的表现 |
|---|---|
| 类别错标 | 真实是握拳，却被标成伸腕 |
| 动作边界不准 | 动作刚开始的窗口已经被标成完整动作 |
| 过渡段不确定 | 从放松到握拳的中间状态被硬标成握拳 |
| 相似手势混淆 | 捏合、握拳、屈指等手势之间相互混淆 |
| 低幅值/弱激活 | 信号弱，模型难判断，但标签不一定错 |
| 通道质量问题 | 某些电极接触差，导致样本难分类 |

**Noisy Label Learning，简称 LNL，就是在标签不完全可信的情况下训练模型。**

---

## 2. 普通 CE 训练为什么会失败？

普通分类网络的训练流程是：

```text
HD-sEMG 输入 x
    ↓
神经网络 f(x)
    ↓
输出每个手势类别的概率
    ↓
和人工标签 y 计算交叉熵 loss
    ↓
反向传播更新网络
```

如果标签是对的，CE 训练没有问题。

例如，标签是“握拳”，模型预测也是“握拳”：

| 类别 | 模型预测概率 |
|---|---:|
| 握拳 | 0.85 |
| 伸腕 | 0.05 |
| 捏合 | 0.05 |
| 放松 | 0.05 |

这时 loss 很小。

但是如果标签错了，问题就出现了。

假设真实信号像“握拳”，但标签写成“伸腕”：

| 类别 | 模型预测概率 |
|---|---:|
| 握拳 | 0.85 |
| 伸腕 | 0.05 |
| 捏合 | 0.05 |
| 放松 | 0.05 |

人工标签要求模型把它学成“伸腕”。此时 CE 会强迫模型降低“握拳”概率，提高“伸腕”概率。

这意味着：

> 普通 CE 不会判断标签是否正确，它只会无条件拟合标签。

所以只要训练时间足够长，模型最终可能连错误标签也记住。

---

## 3. Co-teaching 的理论基础：memorization effect

Co-teaching 的核心基础是一个现象：

> 深度网络通常先学习简单、干净、规律明显的样本，后期才逐渐记住错误标签。

这叫：

**memorization effect，记忆效应。**

可以这样理解：

训练早期：

```text
模型还比较简单
更容易学到真实规律
干净样本 loss 下降快
错标样本 loss 仍然较大
```

训练后期：

```text
模型容量很强
会慢慢把错标样本也硬记住
错标样本 loss 也可能下降
```

因此，Co-teaching 利用这个现象：

> 在模型还没有彻底记住错误标签之前，优先选择 loss 小的样本来训练。

---

## 4. Co-teaching 的核心思想

Co-teaching 的一句话解释：

> 同时训练两个网络，每个网络从一个 mini-batch 里挑选自己认为 loss 小、可能干净的样本，然后把这些样本交给另一个网络学习。

它有两个关键词：

1. **small-loss selection**：loss 小的样本更可能是 clean sample；
2. **cross update / peer teaching**：不是自己选自己学，而是两个网络互相教。

---

## 5. Co-teaching 的完整框架图

```text
                      一个 mini-batch 的 HD-sEMG 样本
                                  │
                    ┌─────────────┴─────────────┐
                    ↓                           ↓
              网络 A / Model A             网络 B / Model B
                    ↓                           ↓
          计算每个样本的 loss           计算每个样本的 loss
                    ↓                           ↓
          选择 loss 最小的一部分        选择 loss 最小的一部分
                    ↓                           ↓
          A 选出的样本集合 S_A          B 选出的样本集合 S_B
                    ↓                           ↓
          用 S_A 更新网络 B             用 S_B 更新网络 A
                    ↓                           ↓
              网络 B 更新                 网络 A 更新
```

注意重点：

```text
A 选样本，不给 A 自己学，而是给 B 学。
B 选样本，不给 B 自己学，而是给 A 学。
```

---

## 6. Co-teaching 每一步具体做什么？

假设你有一个 batch，里面有 8 个 HD-sEMG 样本。

| 样本 | 人工标签 | 网络 A loss | 网络 B loss |
|---|---|---:|---:|
| 1 | 握拳 | 0.10 | 0.20 |
| 2 | 伸腕 | 0.15 | 0.12 |
| 3 | 捏合 | 0.30 | 0.25 |
| 4 | 放松 | 0.40 | 0.35 |
| 5 | 握拳 | 1.80 | 1.20 |
| 6 | 伸腕 | 2.10 | 1.90 |
| 7 | 捏合 | 2.60 | 2.40 |
| 8 | 放松 | 3.00 | 2.80 |

如果当前保留比例是 50%，那每个网络只选择 loss 最小的 4 个样本。

网络 A 选择：

```text
样本 1、2、3、4
```

网络 B 也可能选择：

```text
样本 1、2、3、4
```

然后：

```text
网络 A 选出的样本 1、2、3、4 → 用来更新网络 B
网络 B 选出的样本 1、2、3、4 → 用来更新网络 A
```

样本 5、6、7、8 loss 较大，暂时不参与这次训练。

这就是 Co-teaching 的核心训练方式。

---

## 7. small-loss 是什么意思？

loss 可以理解成：

> 模型预测结果和人工标签之间的差距。

loss 小，说明模型觉得这个样本和标签比较一致。

loss 大，说明模型觉得这个样本和标签不太一致。

Co-teaching 的假设是：

> 在训练前期，loss 小的样本更可能是标签正确的样本。

但这只是一个假设，不一定永远成立。

在 HD-sEMG 里，有些 loss 大的样本可能不是错标，而是：

- 手势本来很难；
- 信号幅值低；
- 被试动作不标准；
- 通道接触不好；
- 处于动作起始或结束阶段；
- 相似手势本来就容易混淆。

所以 Co-teaching 是重要 baseline，但不是完美方法。

---

## 8. Co-teaching 为什么需要两个网络？

如果只有一个网络，会出现这个问题：

```text
网络自己判断哪些样本干净
网络自己用这些样本训练自己
如果网络一开始判断错了，就会不断强化自己的错误判断
```

这叫自我确认偏差。

Co-teaching 用两个网络，是为了让它们互相提供样本选择结果。

```text
网络 A 的 small-loss 样本给网络 B 学
网络 B 的 small-loss 样本给网络 A 学
```

两个网络参数初始化不同，学习路径不同，它们的错误不一定完全一样。互相教学可以降低单个网络自己骗自己的风险。

---

## 9. Co-teaching 的训练过程

### 9.1 准备阶段

你需要：

```text
训练集 D = {(x_i, y_i)}
网络 A = f_A
网络 B = f_B
噪声率估计值 = τ
训练总轮数 = T
每个 epoch 的保留比例 = R(t)
```

其中：

- x_i 是 HD-sEMG 样本；
- y_i 是手势标签；
- τ 是估计的标签噪声比例；
- R(t) 是第 t 个 epoch 保留多少样本。

### 9.2 保留比例怎么变化？

训练刚开始时，模型还没有严重记忆噪声，可以多保留一些样本。

训练后期，模型更容易记住错标样本，所以需要丢掉更多 high-loss 样本。

常见策略：

```text
训练早期：保留比例接近 100%
训练中期：保留比例逐渐下降
训练后期：保留比例约等于 1 - 噪声率
```

比如你估计噪声率是 20%，后期就保留 80% small-loss 样本。

如果噪声率是 40%，后期就保留 60% small-loss 样本。

---

## 10. Co-teaching 伪代码

```python
# 输入：训练集 D，两个网络 model_A, model_B，保留比例 schedule R(t)

for epoch in range(num_epochs):
    remember_rate = R(epoch)

    for batch_x, batch_y in dataloader:
        # 1. 两个网络分别预测
        pred_A = model_A(batch_x)
        pred_B = model_B(batch_x)

        # 2. 分别计算每个样本的 loss
        loss_A_each = cross_entropy(pred_A, batch_y, reduction='none')
        loss_B_each = cross_entropy(pred_B, batch_y, reduction='none')

        # 3. 每个网络选出自己的 small-loss 样本
        num_remember = int(remember_rate * len(batch_x))
        idx_A = argsort(loss_A_each)[:num_remember]
        idx_B = argsort(loss_B_each)[:num_remember]

        # 4. 交叉更新
        # A 用 B 选出的样本更新
        loss_A_update = mean(loss_A_each[idx_B])

        # B 用 A 选出的样本更新
        loss_B_update = mean(loss_B_each[idx_A])

        # 5. 反向传播
        update(model_A, loss_A_update)
        update(model_B, loss_B_update)
```

最关键的两行是：

```python
loss_A_update = mean(loss_A_each[idx_B])
loss_B_update = mean(loss_B_each[idx_A])
```

也就是：

```text
网络 A 用网络 B 选择的样本更新
网络 B 用网络 A 选择的样本更新
```

---

## 11. Co-teaching 属于哪一类方法？

Co-teaching 属于：

```text
sample selection-based noisy label learning
```

中文可以说：

> 基于样本选择的噪声标签学习方法。

它不是：

| 类型 | Co-teaching 是否属于 |
|---|---|
| 改 loss 的方法 | 不是 |
| 标签修正方法 | 不是 |
| 半监督方法 | 不是 |
| 自监督方法 | 不是 |
| 样本选择方法 | 是 |
| 双网络互教方法 | 是 |

Co-teaching 的核心不是设计新 backbone，而是设计一种训练策略。

你可以用 ResNet1D、CNN、Transformer、TCN 等任何 backbone 做 Co-teaching。

---

## 12. Co-teaching 放到 HD-sEMG 里怎么用？

### 12.1 输入形式

HD-sEMG 可能有几种输入形式：

| 输入形式 | 说明 |
|---|---|
| channel × time | 把所有电极当成通道 |
| electrode_grid × time | 保留二维电极拓扑 |
| time-frequency map | 用 STFT/CWT 得到时频图 |
| trial-level sequence | 一个动作 trial 切成多个窗口 |

Co-teaching 不限制输入形式。只要你的网络能输出类别概率，就可以用。

### 12.2 两个网络可以是什么？

最简单：

```text
model_A = ResNet1D
model_B = ResNet1D
```

更适合 HD-sEMG：

```text
model_A = 2D CNN over electrode grid
model_B = 2D CNN over electrode grid
```

也可以更复杂：

```text
model_A = temporal CNN
model_B = CNN-Transformer
```

但入门阶段建议两个网络结构相同，只是初始化不同。

---

## 13. Co-teaching 在 HD-sEMG 中的优点

### 优点 1：容易实现

你只需要两个模型，训练时按 loss 排序选样本。

### 优点 2：适合做 baseline

Co-teaching 是 LNL 经典方法。后续你做任何新方法，都可以和它对比。

### 优点 3：能证明 noisy label 问题存在

如果 CE 在噪声标签下明显下降，而 Co-teaching 能提高性能，说明数据中确实存在标签噪声影响。

### 优点 4：便于向老师解释

它的逻辑非常直观：

> 错标样本更难学，loss 更大，所以用两个网络互相筛选 loss 小的样本进行训练。

---

## 14. Co-teaching 在 HD-sEMG 中的缺点

### 缺点 1：small-loss 不一定代表标签正确

HD-sEMG 中 loss 大可能有很多原因，不一定是标签错。

例如：

| loss 大的原因 | 是否一定是错标？ |
|---|---|
| 动作边界不准 | 不一定 |
| 低幅值肌电 | 不一定 |
| 被试动作不标准 | 不一定 |
| 通道接触差 | 不一定 |
| 相似手势难区分 | 不一定 |
| 真正标签错误 | 是可能的 |

所以 Co-teaching 可能误删 hard clean samples。

### 缺点 2：它不会修正标签

Co-teaching 对 high-loss 样本的处理是：

```text
暂时不用
```

但它不会判断：

```text
这个样本应该从握拳改成捏合
```

所以它只会筛选样本，不会修复标签。

### 缺点 3：它不利用 HD-sEMG 的空间拓扑

HD-sEMG 的重要优势是高密度电极阵列。

例如：

```text
8 × 8 electrode grid
16 × 8 electrode grid
```

不同手势应该有不同的肌肉激活空间分布。

但 Co-teaching 只看 loss，不看：

- 电极空间邻接；
- 肌肉激活区域；
- 相邻窗口预测一致性；
- 通道质量；
- 肌肉协同模式。

### 缺点 4：对噪声率估计敏感

Co-teaching 需要设置最终丢弃多少样本。如果你不知道真实噪声率，保留比例可能设错。

例如：

| 实际噪声率 | 你设定噪声率 | 可能问题 |
|---|---:|---|
| 20% | 40% | 丢太多干净样本 |
| 40% | 20% | 保留太多错标样本 |

---

## 15. 你应该怎么向老师复述 Co-teaching？

可以直接用下面这段：

> Co-teaching 是一种经典的 noisy label learning 方法，发表在 NeurIPS 2018。它基于 deep network 的 memorization effect，也就是神经网络通常先学习干净样本，后期才逐渐记住错误标签。因此，在训练早期，loss 小的样本更可能是标签正确的样本。Co-teaching 同时训练两个网络，每个网络在 mini-batch 内选择 small-loss 样本，但不是用这些样本更新自己，而是交给另一个网络更新。这样可以减少单网络自己选择样本、自己强化错误判断的问题。Co-teaching 本质上是 sample selection 方法，它不会主动修正标签，只是降低 high-loss 疑似错标样本对训练的影响。对于高密度肌电数据，Co-teaching 可以作为经典 baseline，但它只看 loss，不利用电极空间拓扑和相邻窗口时序一致性，因此可能误删低幅值、动作边界或困难手势样本。

---

## 16. Co-teaching 和你的 HD-sEMG 研究之间的关系

你现在可以把 Co-teaching 当成第一块基石。

它帮你理解：

```text
LNL 的核心问题不是只换网络，而是判断哪些样本标签可信。
```

但你的研究不能只停留在 Co-teaching。

因为 HD-sEMG 有自己的特点：

| HD-sEMG 特点 | Co-teaching 是否利用 |
|---|---|
| 多通道高密度空间拓扑 | 没有 |
| 相邻窗口时序连续性 | 没有 |
| 动作 onset/offset 边界噪声 | 没有 |
| 相似手势之间生理混淆 | 没有 |
| 通道接触质量差异 | 没有 |
| 肌肉协同空间模式 | 没有 |

所以你后续的改进方向可以是：

```text
Co-teaching 的 small-loss selection
+
HD-sEMG 的时序一致性
+
HD-sEMG 的电极空间拓扑
+
不确定性软标签修正
```

形成：

```text
HD-sEMG-specific reliable sample selection and label correction
```

---

## 17. 和 JoCoR 的关系：下一篇应该怎么学？

你学完 Co-teaching 后，下一篇建议学 JoCoR。

两者区别：

| 方法 | 核心思想 | 样本选择标准 | 是否双网络 | 是否标签修正 |
|---|---|---|---|---|
| Co-teaching | 两个网络互相选择 small-loss 样本 | 每个网络自己的 loss | 是 | 否 |
| JoCoR | 两个网络共同训练，并要求预测一致 | joint loss = classification loss + consistency loss | 是 | 否 |

Co-teaching 更像：

```text
你帮我选样本，我帮你选样本。
```

JoCoR 更像：

```text
我们一起判断样本，并且我们的预测要一致。
```

---

## 18. 一页纸记忆版

### Co-teaching 是什么？

两个网络互相挑选 small-loss 样本训练对方。

### 为什么这样做？

因为 DNN 先学 clean samples，后记 noisy samples。训练早期 loss 小的样本更可能干净。

### 算法流程

```text
同一批样本 → 网络 A 和 B 分别计算 loss
A 选 small-loss 样本 → 给 B 更新
B 选 small-loss 样本 → 给 A 更新
重复训练
```

### 它判断样本可靠的标准

```text
small loss = 更可能 clean
```

### 它怎么处理疑似错标？

```text
不修正标签，只是不使用 high-loss 样本训练。
```

### 对 HD-sEMG 的意义

可以作为经典 baseline，但不能直接解决 HD-sEMG 的时序边界噪声和空间拓扑问题。

---

## 19. 你现在应该会回答的 8 个问题

### Q1：Co-teaching 是不是一种网络结构？

不是。它主要是一种训练策略。你可以用 CNN、ResNet、Transformer 作为 backbone。

### Q2：Co-teaching 为什么用两个网络？

为了让两个网络互相选择样本，降低单网络自我强化错误的风险。

### Q3：Co-teaching 怎么判断样本是不是干净？

用 small-loss 标准。loss 小的样本更可能标签正确。

### Q4：Co-teaching 会不会改标签？

不会。它只选择样本，不主动修正标签。

### Q5：Co-teaching 和 CE 有什么区别？

CE 用所有样本训练；Co-teaching 只用被两个网络选择出来的 small-loss 样本训练。

### Q6：Co-teaching 对 HD-sEMG 有什么问题？

HD-sEMG 的 loss 大不一定是错标，可能是动作难、信号弱、边界不准或通道质量差。

### Q7：Co-teaching 后面可以怎么改进？

加入时序一致性、电极空间拓扑、通道质量评估、不确定性建模和软标签修正。

### Q8：Co-teaching 适合放在论文哪里？

适合作为 baseline，也适合作为引出你方法的经典 LNL 框架。

---

## 20. 给你后续写论文时的一句定位

> Co-teaching provides a simple and effective sample-selection strategy for learning with noisy labels. However, its small-loss criterion ignores the spatio-temporal structure of high-density sEMG signals, where high-loss samples may correspond to hard but clean gestures, weak muscle activation, or ambiguous movement transitions. Therefore, HD-sEMG noisy-label learning requires reliability estimation beyond loss values, incorporating temporal consistency and spatial electrode topology.

中文：

> Co-teaching 为噪声标签学习提供了一种简单有效的样本选择策略，但它的小损失准则忽略了高密度肌电信号的时空结构。在 HD-sEMG 中，高 loss 样本不一定是错标，也可能是困难手势、弱肌肉激活或动作过渡段。因此，高密度肌电噪声标签学习需要超越 loss 的可靠性估计，结合时序一致性和电极空间拓扑。

---

## 21. 最后总结

Co-teaching 是你入门 LNL 最适合先学的算法。

你只需要牢牢记住：

```text
两个网络
small-loss 样本
互相选择
交叉更新
不修标签
适合 baseline
不够适合直接解决 HD-sEMG 的时空噪声
```

学懂 Co-teaching 后，你再去看 JoCoR、DivideMix、DISC，就会顺很多。

