# JoCoR 算法框架入门：面向高密度肌电 HD-sEMG 噪声标签学习

> 适合阶段：刚学完 Co-teaching，准备学习第二个 LNL 经典算法。  
> 目标：不是背公式，而是能给老师清楚复述 JoCoR 的算法框架、训练流程、样本可靠性判断方式，以及它如何迁移到高密度肌电数据。

---

## 0. 论文参考

### 主论文

**Combating Noisy Labels by Agreement: A Joint Training Method with Co-Regularization**  
作者：Hongxin Wei, Lei Feng, Xiangyu Chen, Bo An  
会议：CVPR 2020  
方法名：**JoCoR: Joint Training with Co-Regularization**  
任务：Learning with Noisy Labels, LNL

### 官方/常用资源

- CVPR Open Access 论文页面：`Combating Noisy Labels by Agreement: A Joint Training Method with Co-Regularization`
- arXiv：`arXiv:2003.02752`
- 官方代码：`hongxin001/JoCoR`

---

## 1. 先一句话理解 JoCoR

**JoCoR 是 Co-teaching 的一个重要改进：它也训练两个网络，也选择 small-loss 样本，但它额外要求两个网络的预测结果保持一致。**

Co-teaching 的核心是：

> 两个网络互相选择 small-loss 样本。

JoCoR 的核心是：

> 两个网络不仅要选 small-loss 样本，还要通过 co-regularization 让两个网络在预测上达成 agreement。

所以 JoCoR 的中文理解可以是：

> **联合训练 + 一致性正则化。**

---

## 2. JoCoR 要解决什么问题？

在 noisy label learning 里，训练数据的标签可能是错的。

比如你的高密度肌电 HD-sEMG 数据中：

| 肌电信号真实情况 | 人工标签 | 问题 |
|---|---|---|
| 真实是握拳 | 标成伸腕 | 类别错标 |
| 动作刚开始，肌电还没稳定 | 标成完整握拳 | 动作边界噪声 |
| 捏合和抓握很像 | 标成另一个相似手势 | 相似手势错标 |
| 电极接触不好，局部通道弱 | 标签仍然硬标为某手势 | 低质量样本噪声 |

普通 CE 训练会强迫模型拟合这些标签。Co-teaching 通过 small-loss 样本选择缓解这个问题，但 JoCoR 认为：

> 仅仅让两个网络互相选择样本还不够，两个网络的预测还应该互相约束，避免它们各自学出不稳定、分歧很大的结果。

---

## 3. JoCoR 和 Co-teaching 的关系

你刚学过 Co-teaching，所以先从二者对比理解。

| 对比点 | Co-teaching | JoCoR |
|---|---|---|
| 网络数量 | 两个网络 | 两个网络 |
| 是否使用 small-loss | 使用 | 使用 |
| 谁选样本 | 网络 A 给 B 选，网络 B 给 A 选 | 两个网络共同计算 joint loss 后选样本 |
| 更新方式 | A 用 B 选的样本更新，B 用 A 选的样本更新 | 两个网络用同一批 joint small-loss 样本同时更新 |
| 是否要求两个网络一致 | 不显式要求 | 显式要求，通过 co-regularization |
| 核心思想 | 互相教学 | 联合训练 + 预测一致性 |

一句话区别：

> **Co-teaching 是“互相选样本”，JoCoR 是“共同选样本，并让两个网络预测一致”。**

---

## 4. JoCoR 的整体算法流程

JoCoR 的输入和输出可以这样理解。

### 输入

一批训练样本：

```text
(x_i, y_i)
```

其中：

- `x_i`：一段高密度肌电 HD-sEMG 信号；
- `y_i`：人工给的手势标签，但这个标签可能是错的。

### 两个网络

JoCoR 同时训练两个网络：

```text
Network 1: f1(x)
Network 2: f2(x)
```

它们可以是两个同结构但不同初始化的网络，比如：

```text
f1 = ResNet1D / CNN / Transformer
f2 = ResNet1D / CNN / Transformer
```

对你的 HD-sEMG 来说，输入可能是：

```text
HD-sEMG grid × time
```

或者：

```text
electrode × time
```

两个网络分别输出每个手势类别的概率：

```text
p1 = f1(x)
p2 = f2(x)
```

---

## 5. JoCoR 的核心：joint loss

JoCoR 的关键不是换 backbone，而是设计一个 **joint loss**。

这个 joint loss 由两部分组成：

```text
joint loss = supervised classification loss + co-regularization loss
```

也可以理解成：

```text
joint loss = 看标签是否匹配 + 看两个网络是否一致
```

---

## 6. 第一部分：supervised classification loss

这部分和普通分类一样。

对于同一个样本 `x_i`，两个网络都会根据标签 `y_i` 计算 CE loss：

```text
Network 1: CE(p1, y_i)
Network 2: CE(p2, y_i)
```

两个加起来就是 supervised loss：

```text
L_sup = CE(p1, y_i) + CE(p2, y_i)
```

### 怎么理解？

如果标签是“握拳”，两个网络都预测“握拳”，那么 CE 小。

如果标签是“握拳”，但两个网络都觉得像“捏合”，那么 CE 大。

所以 supervised loss 仍然反映：

> 这个样本和人工标签是否匹配。

---

## 7. 第二部分：co-regularization loss

JoCoR 最重要的新增部分是 **co-regularization**。

它要求两个网络对同一个样本的预测结果尽量一致。

比如 Network 1 输出：

```text
握拳：0.80
伸腕：0.10
捏合：0.05
放松：0.05
```

Network 2 输出：

```text
握拳：0.75
伸腕：0.15
捏合：0.05
放松：0.05
```

这两个预测很接近，说明 agreement 高。

如果 Network 1 觉得是握拳，Network 2 觉得是伸腕，两个网络分歧很大，co-regularization loss 就大。

### 常用形式

JoCoR 用两个网络预测分布之间的距离来约束一致性，常见写法可以理解为：

```text
L_coreg = KL(p1 || p2) + KL(p2 || p1)
```

也就是让 `p1` 和 `p2` 互相接近。

---

## 8. 合起来：JoCoR 的 joint loss

JoCoR 对每个样本计算：

```text
L_joint = CE(p1, y_i) + CE(p2, y_i) + λ · L_coreg
```

其中：

- `CE(p1, y_i)`：网络 1 和标签的分类误差；
- `CE(p2, y_i)`：网络 2 和标签的分类误差；
- `L_coreg`：两个网络预测是否一致；
- `λ`：控制一致性约束强度的参数。

### 用人话理解

JoCoR 判断一个样本是否可靠，不只看：

> 这个样本 loss 小不小。

还看：

> 两个网络对这个样本的判断是否一致。

所以 JoCoR 的可靠样本标准可以理解为：

> **标签匹配程度高，并且两个网络预测一致的样本，更可能是 clean sample。**

---

## 9. JoCoR 怎么选择样本？

JoCoR 仍然使用 small-loss 选择，但它选的不是普通 CE loss 小的样本，而是：

```text
joint loss 小的样本
```

### 一个简单例子

假设一个 batch 里有 5 个 HD-sEMG 样本。

| 样本 | 人工标签 | 网络1预测 | 网络2预测 | joint loss | JoCoR 判断 |
|---|---|---|---|---:|---|
| 样本1 | 握拳 | 握拳 | 握拳 | 0.15 | 可靠 |
| 样本2 | 伸腕 | 伸腕 | 伸腕 | 0.20 | 可靠 |
| 样本3 | 捏合 | 捏合 | 抓握 | 1.20 | 分歧大，可疑 |
| 样本4 | 放松 | 握拳 | 握拳 | 2.30 | 标签可能错 |
| 样本5 | 抓握 | 抓握 | 抓握 | 0.25 | 可靠 |

JoCoR 会选择 joint loss 小的样本，比如样本 1、2、5，用它们同时更新两个网络。

样本 3 因为两个网络分歧大，所以可疑。  
样本 4 因为两个网络都不支持人工标签，所以也可疑。

---

## 10. JoCoR 的训练流程图

```text
                   一个 mini-batch 的 HD-sEMG 样本
                              │
              ┌───────────────┴───────────────┐
              ↓                               ↓
        Network 1                         Network 2
              ↓                               ↓
       输出预测 p1                       输出预测 p2
              │                               │
              └───────────────┬───────────────┘
                              ↓
             计算每个样本的 joint loss
                              │
         joint loss = CE1 + CE2 + λ · agreement loss
                              │
                              ↓
                   选出 joint loss 小的样本
                              │
                              ↓
             用这些样本同时更新 Network 1 和 Network 2
```

---

## 11. JoCoR 的伪代码

```python
# 初始化两个网络
model1 = Network()
model2 = Network()

for epoch in range(num_epochs):
    remember_rate = get_remember_rate(epoch)

    for x, y in dataloader:
        # 1. 两个网络分别预测
        p1 = model1(x)
        p2 = model2(x)

        # 2. 计算两个网络各自的分类 loss
        loss_ce_1 = CE(p1, y)   # 每个样本一个 loss
        loss_ce_2 = CE(p2, y)

        # 3. 计算两个网络预测之间的一致性 loss
        loss_coreg = KL(p1, p2) + KL(p2, p1)

        # 4. 计算 joint loss
        loss_joint = loss_ce_1 + loss_ce_2 + lambda_ * loss_coreg

        # 5. 选出 joint loss 最小的一部分样本
        selected_indices = select_small_loss(loss_joint, remember_rate)

        # 6. 用选中的样本同时更新两个网络
        final_loss = mean(loss_joint[selected_indices])
        final_loss.backward()
        optimizer.step()
```

注意：真实代码里会分别更新两个网络的参数，但理解上可以先记为：

> 两个网络共同计算 joint loss，然后用 joint small-loss 样本一起更新。

---

## 12. remember rate 是什么？

JoCoR 和 Co-teaching 一样，训练时不是一直保留全部样本，而是逐渐减少可用于训练的样本比例。

这个比例叫：

```text
remember rate
```

如果估计噪声率是 20%，最后可能只保留 80% 的 small-loss 样本。

比如：

| 训练阶段 | 记住比例 remember rate | 含义 |
|---|---:|---|
| 训练早期 | 100% | 模型还没开始记噪声，先充分学习 |
| 中期 | 90% | 开始排除一部分可疑样本 |
| 后期 | 80% | 如果估计 20% 标签有噪声，最终只保留 80% |

### 为什么要这样？

因为 DNN 有 memorization effect：

> 早期先学干净模式，后期才逐渐记住噪声标签。

所以后期更需要谨慎，只用更可靠的样本训练。

---

## 13. JoCoR 为什么强调 agreement？

之前的 Co-teaching+ 强调 disagreement，意思是：

> 两个网络意见不同的样本更有价值。

但 JoCoR 反过来认为：

> 在 noisy label 场景下，两个网络长期分歧可能会带来不稳定训练。让两个网络逐渐达成一致，反而有助于抵抗标签噪声。

尤其对 HD-sEMG 来说，这一点很重要。

因为 HD-sEMG 中两个网络分歧可能不是因为样本“有信息量”，而是因为：

- 动作边界本身模糊；
- 相似手势很难区分；
- 某些通道噪声大；
- 某些被试肌电模式特殊；
- 低幅值样本不稳定。

所以 disagreement 不一定是好事。

JoCoR 选择 agreement 的思想更容易解释为：

> 如果两个网络都认为一个 HD-sEMG 样本属于同一类，并且这个判断和标签一致，那么这个样本更可信。

---

## 14. JoCoR 在高密度肌电 HD-sEMG 中怎么用？

### 14.1 输入形式

你的 HD-sEMG 输入可能有几种：

| 输入形式 | 说明 |
|---|---|
| `channel × time` | 把高密度电极展平成多通道时间序列 |
| `height × width × time` | 保留二维电极空间拓扑 |
| `time × height × width` | 适合 3D CNN 或时空 Transformer |
| `feature × time` | 用 RMS、MAV、WL 等特征序列 |

JoCoR 本身不限制输入形状。它只是训练策略。

所以你可以把 JoCoR 接到任何 backbone 上：

```text
HD-sEMG 输入 → Backbone 1 → p1
HD-sEMG 输入 → Backbone 2 → p2
p1, p2 → joint loss → small-loss selection → update
```

---

### 14.2 HD-sEMG 中什么样本会被 JoCoR 认为可靠？

| 样本类型 | JoCoR 的可能判断 |
|---|---|
| 稳定、清晰的手势激活 | joint loss 小，保留 |
| 两个网络都预测同一类，且与标签一致 | 保留 |
| 网络1和网络2分歧很大 | 可疑，可能不选 |
| 两个网络都不支持人工标签 | 可疑，可能不选 |
| 动作边界窗口 | 可能 joint loss 较大，容易被排除 |
| 相似手势样本 | 如果分歧大，也可能被排除 |

---

## 15. JoCoR 的优点

### 优点 1：比 Co-teaching 多了 agreement 约束

Co-teaching 只靠 small-loss 选样本。JoCoR 多看两个网络是否一致。

所以 JoCoR 的可靠样本判断更强：

```text
small-loss + prediction agreement
```

---

### 优点 2：训练框架仍然清楚

它仍然是双网络结构，不需要特别复杂的额外模块。

这对你入门很友好。

---

### 优点 3：适合做强 baseline

你老师表格里 JoCoR 表现很强，所以它一定要作为重点 baseline 去理解。

---

## 16. JoCoR 的缺点，尤其对 HD-sEMG

### 缺点 1：仍然依赖 small-loss

JoCoR 虽然加入 agreement，但最终还是选 joint small-loss 样本。

HD-sEMG 中 loss 大不一定是错标，可能是：

- 困难手势；
- 低幅值肌电；
- 动作过渡段；
- 被试差异；
- 局部电极噪声；
- 相似类别边界模糊。

所以 JoCoR 仍然可能误删困难但干净的样本。

---

### 缺点 2：没有真正修正标签

JoCoR 主要是选择 reliable samples，并没有明确告诉你：

> 这个疑似错标样本应该改成哪个标签。

所以它属于：

```text
sample selection 方法
```

而不是完整的：

```text
label correction 方法
```

---

### 缺点 3：没有利用 HD-sEMG 的空间拓扑

高密度肌电有电极二维空间结构。

不同手势会激活不同的肌肉区域。

但 JoCoR 不知道这些，它只看：

```text
loss + 两个网络预测一致性
```

它没有显式利用：

- 电极邻接关系；
- 肌肉区域激活图；
- 相邻时间窗口一致性；
- 动作 onset/offset 信息；
- trial 内时序稳定性。

这正是后续改进空间。

---

## 17. JoCoR 可以怎么改成 HD-sEMG-specific 方法？

你现在不一定马上做新方法，但可以先理解可能的改进方向。

### 方向 1：加入时序一致性

JoCoR 现在只看单个窗口。

HD-sEMG 中，相邻窗口应该有连续性。

可以加入：

```text
相邻窗口预测一致性
```

比如：

```text
p(t-1), p(t), p(t+1)
```

如果一个窗口标签和前后窗口完全不一致，它可能是边界噪声或错标。

---

### 方向 2：加入空间拓扑一致性

HD-sEMG 有二维电极阵列。

可以判断：

> 当前样本的空间激活图是否符合该手势的典型激活模式。

比如某类手势通常激活前臂某一区域，但这个样本的激活区域完全不一致，那么它可能可疑。

---

### 方向 3：加入 soft label，而不是直接丢弃

JoCoR 对疑似样本主要是不选。

但 HD-sEMG 中很多样本不是绝对错误，而是边界模糊。

所以可以改成：

```text
可靠样本：hard label CE
不确定样本：soft label / consistency loss
高疑似错标样本：降权或 label correction
```

---

### 方向 4：类别均衡选择

某些手势容易，joint loss 小；某些手势难，joint loss 大。

如果不控制类别均衡，JoCoR 可能总是选简单手势。

HD-sEMG 中可以做：

```text
class-wise small-loss selection
```

即每个手势类别内部都保留一部分样本，而不是全局排序。

---

## 18. JoCoR 和 Co-teaching 的一句话复述对比

### Co-teaching

> Co-teaching 同时训练两个网络，让两个网络互相选择 small-loss 样本更新对方，从而减少错误标签对训练的影响。

### JoCoR

> JoCoR 同时训练两个网络，但它不只是互相选择样本，而是计算 supervised loss 和 co-regularization loss 组成的 joint loss，用 joint small-loss 样本同时更新两个网络，使两个网络在抗噪训练中逐渐达成预测一致。

---

## 19. 给老师复述 JoCoR 的标准版本

你可以直接背下面这段。

> JoCoR 是 CVPR 2020 提出的 noisy label learning 方法，全称是 Joint Training with Co-Regularization。它和 Co-teaching 一样使用两个网络，但区别在于 Co-teaching 是两个网络互相选择 small-loss 样本，而 JoCoR 是把两个网络联合起来训练。具体来说，两个网络对同一个 mini-batch 分别输出预测，JoCoR 对每个样本计算一个 joint loss，这个 loss 包括两个网络各自和标签之间的交叉熵损失，以及两个网络预测分布之间的一致性正则项。然后它选择 joint loss 较小的样本，同时更新两个网络。这样做的核心思想是，可靠样本不仅应该和人工标签匹配，还应该让两个网络产生一致预测。对于高密度肌电数据，JoCoR 可以作为强 baseline，因为它能减少错标样本对训练的影响；但它仍然主要依赖 small-loss 和模型一致性，没有显式利用肌电的时序连续性和电极空间拓扑，因此后续可以在 JoCoR 基础上加入时序一致性、空间激活一致性和 soft label correction。

---

## 20. 你今天需要掌握的 6 个关键词

| 关键词 | 中文理解 |
|---|---|
| JoCoR | Joint Training with Co-Regularization，联合训练加一致性正则 |
| two networks | 两个网络同时训练 |
| agreement | 两个网络预测结果一致 |
| co-regularization | 用正则项约束两个网络输出接近 |
| joint loss | 分类损失 + 一致性损失 |
| joint small-loss sample | joint loss 小、标签和模型都比较一致的可靠样本 |

---

## 21. 学完 JoCoR 后，下一篇建议学什么？

建议下一篇学：

> **DivideMix: Learning with Noisy Labels as Semi-supervised Learning**

原因：

| 方法 | 主要处理方式 |
|---|---|
| Co-teaching | 选择 small-loss 样本，不修正标签 |
| JoCoR | 选择 joint small-loss 样本，并让两个网络一致 |
| DivideMix | 把样本分成 clean labeled 和 noisy unlabeled，用半监督学习继续利用 noisy 样本 |

你学完 DivideMix 后，就会理解 LNL 从“丢掉/排除可疑样本”进一步发展到：

> **把可疑样本当成 unlabeled data 继续利用。**

这对 HD-sEMG 很重要，因为你的数据珍贵，不能轻易丢掉大量样本。

---

# 附录 A：JoCoR 的极简代码理解

```python
# x: HD-sEMG batch
# y: noisy labels

p1 = net1(x)
p2 = net2(x)

ce1 = cross_entropy(p1, y, reduction='none')
ce2 = cross_entropy(p2, y, reduction='none')

coreg = kl_divergence(p1, p2) + kl_divergence(p2, p1)

joint_loss = ce1 + ce2 + lambda_ * coreg

idx = select_small_loss(joint_loss, remember_rate)

loss = joint_loss[idx].mean()
loss.backward()
optimizer.step()
```

核心只记一句：

```text
JoCoR = CE1 + CE2 + 两个网络预测一致性，然后选 joint small-loss 样本训练。
```

---

# 附录 B：HD-sEMG 场景下的 JoCoR 流程

```text
HD-sEMG 高密度电极信号
        │
        ├──────────────→ Network 1 → p1
        │
        └──────────────→ Network 2 → p2
                              │
                              ↓
             CE(p1, label) + CE(p2, label)
                              │
                              ↓
                 prediction agreement loss
                              │
                              ↓
                        joint loss
                              │
                              ↓
                  选择 joint loss 小的样本
                              │
                              ↓
                    同时更新两个网络
```

---

# 附录 C：和 HD-sEMG 结合时你可以问自己的问题

每次读 JoCoR 或跑实验时，建议问这几个问题：

1. joint small-loss 选出来的样本是不是主要来自简单手势？
2. 被排除的 high-loss 样本是真错标，还是困难但干净？
3. 动作 onset/offset 窗口是不是经常被排除？
4. 某些被试是不是更容易被判为 noisy？
5. 某些电极接触差的样本是不是 loss 更高？
6. 两个网络 agreement 高，是否一定代表标签正确？
7. 是否需要加入 class-wise selection，避免简单类主导训练？
8. 是否需要加入 temporal smoothing，避免相邻窗口预测剧烈跳变？

这些问题就是你后面从 JoCoR baseline 发展出 HD-sEMG-specific 方法的入口。
