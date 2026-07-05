# 02｜JoCoR 算法框架详细学习：Joint Training with Co-Regularization

> 适用对象：刚入门 noisy label learning，希望能把算法框架复述给老师。  
> 应用背景：高密度肌电 HD-sEMG 手势识别中的噪声标签学习。  
> 前置算法：Co-teaching。  

---

## 1. 论文参考

### 主论文

**Combating Noisy Labels by Agreement: A Joint Training Method with Co-Regularization**  
作者：Hongxin Wei, Lei Feng, Xiangyu Chen, Bo An  
会议：CVPR 2020  
方法名：**JoCoR，Joint Training with Co-Regularization**

### 一句话概括

> JoCoR 是 Co-teaching 之后非常重要的双网络 LNL 方法。它不再让两个网络互相交换样本，而是让两个网络在同一个 joint loss 下共同训练，并通过 co-regularization 让两个网络的预测保持一致，再根据 joint loss 选择 small-loss 样本更新两个网络。

---

## 2. 为什么在 Co-teaching 后学习 JoCoR

Co-teaching 的核心是：

```text
两个网络分别计算 loss
↓
各自选择 small-loss 样本
↓
网络 A 选出的样本给网络 B 学
网络 B 选出的样本给网络 A 学
```

它的关键词是：

> **cross update，交叉更新。**

JoCoR 的核心是：

```text
两个网络同时预测
↓
同时计算分类损失 + 一致性损失
↓
形成 joint loss
↓
用 joint loss 选择 small-loss 样本
↓
两个网络一起更新
```

它的关键词是：

> **joint training + co-regularization，共同训练 + 共同正则化。**

所以 JoCoR 可以理解为：

> Co-teaching 关心“两个网络互相选样本”；JoCoR 关心“两个网络在预测上达成一致”。

---

## 3. JoCoR 想解决什么问题

Co-teaching 的问题是：两个网络虽然互相选样本，但它们各自计算 loss、各自判断 small-loss。

如果某个网络本身选错了样本，它可能把错误样本传给另一个网络。这个错误叫：

> **error flow，错误流动。**

JoCoR 认为，与其强调两个网络之间的分歧，不如让两个网络在训练过程中保持一致。

它的核心假设是：

> 对于干净样本，两个网络更容易都预测正确并达成一致；对于噪声样本，两个网络更难同时和人工标签保持一致。

所以 JoCoR 用两个条件来判断样本可靠：

1. 这个样本的分类 loss 要小；
2. 两个网络对这个样本的预测要一致。

---

## 4. 问题定义：带噪声标签的 HD-sEMG 分类

训练集记为：

\[
\widetilde{\mathcal{D}} = \{(x_i, \widetilde{y}_i)\}_{i=1}^{N}
\]

其中：

| 符号 | 含义 |
|---|---|
| \(x_i\) | 第 \(i\) 个 HD-sEMG 样本，比如 \(H \times W \times T\) 的高密度肌电片段 |
| \(y_i\) | 真实标签，但训练时未知 |
| \(\widetilde{y}_i\) | 人工标签，可能有错 |
| \(N\) | 样本总数 |
| \(C\) | 手势类别数 |

噪声标签意味着：

\[
\widetilde{y}_i \neq y_i
\]

JoCoR 的目标是：

> 在训练标签可能错误的情况下，训练一个对真实手势类别更鲁棒的分类器。

---

## 5. JoCoR 的整体框架

JoCoR 同时维护两个网络：

\[
f_1(x;\theta_1), \quad f_2(x;\theta_2)
\]

它们可以是同样的 backbone，例如：

```text
ResNet1D + ResNet1D
CNN + CNN
Transformer + Transformer
HD-sEMG 2D-CNN + HD-sEMG 2D-CNN
```

但是两者初始化参数不同：

\[
\theta_1^{(0)} \neq \theta_2^{(0)}
\]

对于同一个输入 \(x_i\)，两个网络分别输出概率：

\[
p_i^{(1)} = \text{softmax}(f_1(x_i;\theta_1))
\]

\[
p_i^{(2)} = \text{softmax}(f_2(x_i;\theta_2))
\]

其中：

\[
p_i^{(1)}, p_i^{(2)} \in \mathbb{R}^{C}
\]

表示两个网络对 \(C\) 个手势类别的预测概率。

---

## 6. JoCoR 的损失函数总览

JoCoR 的核心是一个 joint loss：

\[
\ell(x_i) = (1-\lambda)\ell_{sup}(x_i,\widetilde{y}_i) + \lambda \ell_{con}(x_i)
\]

其中：

| 符号 | 含义 |
|---|---|
| \(\ell_{sup}\) | supervised classification loss，两个网络和人工标签之间的分类损失 |
| \(\ell_{con}\) | consistency / co-regularization loss，两个网络预测之间的一致性损失 |
| \(\lambda\) | 平衡两部分损失的超参数 |

一句话：

> JoCoR 不只看模型和人工标签是否一致，也看两个模型彼此是否一致。

---

## 7. 第一部分：监督分类损失 \(\ell_{sup}\)

对于网络 1：

\[
\ell_{ce}^{(1)}(x_i,\widetilde{y}_i) = -\log p_{i,\widetilde{y}_i}^{(1)}
\]

对于网络 2：

\[
\ell_{ce}^{(2)}(x_i,\widetilde{y}_i) = -\log p_{i,\widetilde{y}_i}^{(2)}
\]

JoCoR 把两个网络的分类损失加起来：

\[
\ell_{sup}(x_i,\widetilde{y}_i)
=
\ell_{ce}^{(1)}(x_i,\widetilde{y}_i)
+
\ell_{ce}^{(2)}(x_i,\widetilde{y}_i)
\]

也就是：

\[
\ell_{sup}(x_i,\widetilde{y}_i)
=
-\log p_{i,\widetilde{y}_i}^{(1)}
-\log p_{i,\widetilde{y}_i}^{(2)}
\]

### 直观意义

如果人工标签是“握拳”，两个网络都给“握拳”高概率，那么：

\[
\ell_{sup} \text{ 小}
\]

如果两个网络都不相信人工标签，那么：

\[
\ell_{sup} \text{ 大}
\]

---

## 8. 第二部分：一致性损失 \(\ell_{con}\)

JoCoR 还希望两个网络的预测概率分布彼此接近。

比如网络 1 输出：

```text
握拳 0.80
伸腕 0.10
屈腕 0.05
放松 0.05
```

网络 2 输出：

```text
握拳 0.78
伸腕 0.12
屈腕 0.06
放松 0.04
```

两个网络预测很接近，说明它们对这个样本比较一致。

如果网络 1 认为是握拳，网络 2 认为是伸腕，二者分歧就很大。

JoCoR 用对称 KL 散度衡量两个概率分布的差异：

\[
\ell_{con}(x_i)
=
D_{KL}(p_i^{(1)}||p_i^{(2)})
+
D_{KL}(p_i^{(2)}||p_i^{(1)})
\]

其中：

\[
D_{KL}(p_i^{(1)}||p_i^{(2)})
=
\sum_{c=1}^{C} p_{ic}^{(1)} \log \frac{p_{ic}^{(1)}}{p_{ic}^{(2)}}
\]

\[
D_{KL}(p_i^{(2)}||p_i^{(1)})
=
\sum_{c=1}^{C} p_{ic}^{(2)} \log \frac{p_{ic}^{(2)}}{p_{ic}^{(1)}}
\]

### 直观意义

| 情况 | \(\ell_{con}\) |
|---|---:|
| 两个网络预测非常接近 | 小 |
| 两个网络预测差异很大 | 大 |

所以 \(\ell_{con}\) 的作用是：

> 让两个网络互相靠近，不要各学各的。

这就是 **co-regularization**。

---

## 9. JoCoR 的 joint loss 怎么判断样本可靠

对于每个样本：

\[
\ell(x_i) = (1-\lambda)\ell_{sup}(x_i,\widetilde{y}_i) + \lambda \ell_{con}(x_i)
\]

如果一个样本 joint loss 小，意味着：

1. 两个网络都比较相信人工标签；
2. 两个网络之间预测也比较一致。

所以 JoCoR 认为：

\[
\ell(x_i) \text{ small} \Rightarrow x_i \text{ 更可能是 clean sample}
\]

注意，这比 Co-teaching 更严格。

Co-teaching 主要看：

```text
某个网络自己的 CE loss 小不小
```

JoCoR 看：

```text
两个网络对标签是否都满意
两个网络之间是否一致
```

---

## 10. mini-batch 中的样本选择

给定一个 mini-batch：

\[
\mathcal{B}=\{(x_i,\widetilde{y}_i)\}_{i=1}^{B}
\]

对每个样本计算 joint loss：

\[
\ell(x_i), \quad i\in \mathcal{B}
\]

按 joint loss 从小到大排序，选择前 \(K(t)\) 个样本。

\[
K(t)=\lfloor R(t)B \rfloor
\]

其中：

| 符号 | 含义 |
|---|---|
| \(B\) | batch size |
| \(R(t)\) | 第 \(t\) 个 epoch 的 remember rate |
| \(K(t)\) | 当前 batch 保留的样本数 |

small-loss 集合为：

\[
\widetilde{\mathcal{B}}
=
\operatorname{arg\,min}_{\mathcal{S}\subset\mathcal{B}, |\mathcal{S}|=K(t)}
\sum_{i\in \mathcal{S}} \ell(x_i)
\]

意思是：

> 选出 joint loss 最小的 \(K(t)\) 个样本。

---

## 11. 用 small-loss 样本共同更新两个网络

选出 \(\widetilde{\mathcal{B}}\) 后，JoCoR 计算平均 joint loss：

\[
\mathcal{L}_{JoCoR}
=
\frac{1}{|\widetilde{\mathcal{B}}|}
\sum_{i\in \widetilde{\mathcal{B}}}
\ell(x_i)
\]

然后同时更新两个网络：

\[
\theta_1 \leftarrow \theta_1 - \eta \nabla_{\theta_1}\mathcal{L}_{JoCoR}
\]

\[
\theta_2 \leftarrow \theta_2 - \eta \nabla_{\theta_2}\mathcal{L}_{JoCoR}
\]

这里和 Co-teaching 的区别很重要。

Co-teaching 是：

```text
网络 A 选样本给网络 B
网络 B 选样本给网络 A
```

JoCoR 是：

```text
两个网络一起计算 joint loss
选出 joint small-loss 样本
两个网络一起更新
```

---

## 12. remember rate 的作用

JoCoR 也使用 remember rate。

如果估计噪声率是 \(\tau\)，则最终希望保留大约：

\[
1-\tau
\]

比例的样本。

常见形式是：

\[
R(t)=1-\min\left(\frac{t}{T_k}\tau,\tau\right)
\]

其中：

| 符号 | 含义 |
|---|---|
| \(t\) | 当前 epoch |
| \(T_k\) | 逐渐增加筛选强度的阶段长度 |
| \(\tau\) | 估计噪声率 |
| \(R(t)\) | 当前保留比例 |

训练初期：

\[
R(t)\approx 1
\]

说明多用样本。

训练后期：

\[
R(t)\approx 1-\tau
\]

说明只保留更可信的 small-loss 样本。

---

## 13. JoCoR 伪代码

```text
Input:
    noisy training set D = {(x_i, y~_i)}
    two networks f1(x; θ1), f2(x; θ2)
    noise rate τ
    co-regularization weight λ
    remember rate R(t)
    learning rate η

For epoch t = 1, 2, ..., T:

    For each mini-batch B:

        1. Network f1 predicts probabilities:
              p_i^(1) = softmax(f1(x_i; θ1))

        2. Network f2 predicts probabilities:
              p_i^(2) = softmax(f2(x_i; θ2))

        3. For each sample, compute supervised loss:
              l_sup = CE(p_i^(1), y~_i) + CE(p_i^(2), y~_i)

        4. Compute consistency loss:
              l_con = KL(p_i^(1)||p_i^(2)) + KL(p_i^(2)||p_i^(1))

        5. Compute joint loss:
              l_i = (1 - λ) l_sup + λ l_con

        6. Select K(t) samples with smallest joint loss:
              B_small = top small-loss samples

        7. Compute average joint loss on B_small:
              L = mean(l_i for i in B_small)

        8. Update both networks simultaneously:
              θ1 ← θ1 - η ∇θ1 L
              θ2 ← θ2 - η ∇θ2 L

Output:
    trained networks f1 and f2
```

---

## 14. JoCoR 和 Co-teaching 的核心区别

| 对比项 | Co-teaching | JoCoR |
|---|---|---|
| 网络数量 | 两个 | 两个 |
| 样本选择依据 | 每个网络自己的 CE loss | 两个网络的 joint loss |
| 是否交叉更新 | 是，A 选给 B，B 选给 A | 否，两个网络共同更新 |
| 是否要求两个网络一致 | 不显式要求 | 显式要求 |
| 是否有一致性正则 | 没有 | 有，co-regularization |
| 是否修正标签 | 不修正 | 不直接修正 |
| 本质 | sample selection + cross update | sample selection + agreement regularization |

一句话记忆：

> Co-teaching 是“互相选样本”；JoCoR 是“共同选样本，并让两个网络预测一致”。

---

## 15. JoCoR 是否能修正标签

原始 JoCoR **不直接修正标签**。

它和 Co-teaching 一样，主要还是选择 small-loss 样本。

区别是它选择样本时更严格：不仅看 CE loss，也看两个网络是否一致。

所以 JoCoR 属于：

> **sample selection + consistency regularization**

不是严格的：

> **label correction**

如果想修正标签，需要看：

- SELFIE
- PENCIL
- DivideMix
- DISC
- Scale-teaching
- sEMG ALR-CNN

---

## 16. 用 HD-sEMG 来理解 JoCoR

假设一个 HD-sEMG 样本是稳定握拳：

```text
真实动作：握拳
人工标签：握拳
信号模式：典型握拳空间激活
```

训练一段时间后：

```text
网络 1：握拳概率 0.90
网络 2：握拳概率 0.88
```

那么：

- 两个网络都相信人工标签；
- 两个网络之间也一致；
- supervised loss 小；
- consistency loss 小；
- joint loss 小；
- 样本被选中训练。

---

再看一个错标样本：

```text
真实动作：握拳
人工标签：伸腕
信号模式：典型握拳空间激活
```

训练一段时间后：

```text
网络 1：握拳概率 0.80，伸腕概率 0.10
网络 2：握拳概率 0.75，伸腕概率 0.15
```

人工标签是伸腕，但两个网络都不相信伸腕。

所以：

- supervised loss 大；
- 即使两个网络彼此一致，也和人工标签不一致；
- joint loss 仍然大；
- 样本可能不被选中。

---

再看一个困难但干净样本：

```text
真实动作：捏合
人工标签：捏合
信号模式：弱、边界模糊、类似握拳
```

训练中可能出现：

```text
网络 1：捏合 0.45，握拳 0.40
网络 2：捏合 0.35，握拳 0.50
```

这时候：

- 两个网络不够相信人工标签；
- 两个网络之间也不一致；
- joint loss 大；
- JoCoR 可能误判它为可疑样本。

这就是 JoCoR 在 HD-sEMG 中的局限。

---

## 17. JoCoR 对 HD-sEMG 的优点

### 优点 1：比 Co-teaching 更稳定

Co-teaching 只看单个网络自己的 loss，JoCoR 看两个网络共同 loss 和一致性。

对于 HD-sEMG，这可以减少单个网络由于初始化或局部偏差造成的错误选择。

### 优点 2：适合相似手势场景

如果两个网络都对某个样本预测一致，说明这个样本的模式比较清楚。

对于稳定手势，这能帮助保留可靠样本。

### 优点 3：容易作为强 baseline

JoCoR 不要求复杂标签修正，也不需要额外真实 clean set，所以适合作为你论文里的强基线。

---

## 18. JoCoR 对 HD-sEMG 的缺点

### 缺点 1：仍然依赖 small-loss

JoCoR 的本质仍然是：

\[
\text{small joint loss} \Rightarrow \text{likely clean}
\]

但 HD-sEMG 中 high-loss 不一定是错标，可能是困难但真实样本。

### 缺点 2：不修正标签

JoCoR 只是选择可信样本，不会把错误标签 A 改成 B。

### 缺点 3：没有利用 HD-sEMG 的空间电极拓扑

HD-sEMG 有二维电极阵列结构，但 JoCoR 的样本选择只看 loss 和预测一致性。

它没有显式利用：

- 相邻电极空间关系；
- 肌肉激活区域；
- 时间窗口连续性；
- 动作 onset / offset；
- 过渡段不确定性。

### 缺点 4：一致性可能带来错误共识

如果两个网络都错了，而且错得一样，那么 consistency loss 会很小。

也就是说：

\[
\text{agreement} \not\Rightarrow \text{correctness}
\]

两个网络一致，不代表一定正确。

---

## 19. HD-sEMG 中可以怎么改进 JoCoR

你未来可以把 JoCoR 改成 HD-sEMG-specific JoCoR。

### 改进方向 1：加入时间一致性

相邻窗口预测应该平滑：

\[
p_{t} \approx p_{t+1}
\]

如果一个窗口和前后窗口预测完全冲突，它可能是边界或噪声样本。

### 改进方向 2：加入空间拓扑一致性

HD-sEMG 电极图中，真实手势应有合理的空间激活模式。

可以检查：

```text
这个样本的电极空间激活图
是否符合该手势的典型激活区域
```

### 改进方向 3：加入类别平衡选择

不要只选 easy class。

每类手势都要保留一定比例的可靠样本。

### 改进方向 4：加入 soft label correction

对于可疑样本，不一定直接丢掉。

可以改成：

```text
高可靠样本：用 hard label CE
中等可靠样本：用 soft label
低可靠样本：只做一致性/自监督
```

---

## 20. 给老师复述版

> JoCoR 是 CVPR 2020 提出的 noisy label learning 方法，全称是 Joint Training with Co-Regularization。它和 Co-teaching 一样使用两个网络，但核心区别是 Co-teaching 通过两个网络交叉选择 small-loss 样本更新对方，而 JoCoR 把两个网络作为一个整体进行 joint training。具体来说，对于同一个 mini-batch，两个网络分别输出预测概率，然后计算两个部分的损失：第一部分是两个网络各自与人工标签之间的交叉熵分类损失，第二部分是两个网络预测分布之间的 co-regularization loss，通常用对称 KL 散度衡量。两部分组成 joint loss 后，JoCoR 根据 joint loss 选择 small-loss 样本，并用这些样本同时更新两个网络。因此，JoCoR 的可靠样本判断标准不仅是模型是否相信人工标签，还包括两个网络之间是否预测一致。它的优点是比单纯 Co-teaching 更强调双网络一致性，能减少一部分错误选择；但它仍然不直接修正标签，仍依赖 small-loss 假设。对于 HD-sEMG，它可以作为强 baseline，但需要注意 high-loss 样本可能是动作边界、低幅值手势或相似手势，并不一定是错标。因此后续可以结合高密度肌电的空间电极拓扑和相邻窗口时序一致性改进 JoCoR 的样本可靠性判断。

---

## 21. 最短记忆版

如果你只记一句：

> Co-teaching 是两个网络互相选 small-loss 样本；JoCoR 是两个网络共同训练，用分类损失加一致性损失形成 joint loss，再根据 joint loss 选 small-loss 样本。

如果你只记公式：

\[
\ell(x_i) = (1-\lambda)\ell_{sup}(x_i,\widetilde{y}_i) + \lambda \ell_{con}(x_i)
\]

其中：

\[
\ell_{sup}
=
CE(p_i^{(1)},\widetilde{y}_i)
+
CE(p_i^{(2)},\widetilde{y}_i)
\]

\[
\ell_{con}
=
D_{KL}(p_i^{(1)}||p_i^{(2)})
+
D_{KL}(p_i^{(2)}||p_i^{(1)})
\]

如果你只记和 HD-sEMG 的关系：

> JoCoR 可以帮助筛选两个网络都认为可靠的 HD-sEMG 样本，但它没有利用电极空间结构和动作时间边界，所以后续需要做 HD-sEMG-specific 改进。
