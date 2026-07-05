# 03｜DivideMix 算法框架详细学习：从“筛掉错标”到“把可疑样本当无标签继续利用”

> 适用对象：刚入门 noisy label learning，需要能向老师复述算法框架。  
> 结合任务：高密度肌电 HD-sEMG 手势识别，存在人工标签错误、动作边界不准、相似手势混淆、跨被试差异等问题。  
> 本文目标：弄懂 DivideMix 的算法流程、数学框架、它和 Co-teaching / JoCoR 的区别，以及如何迁移到 HD-sEMG。

---

## 1. 论文参考

### 主论文

**DivideMix: Learning with Noisy Labels as Semi-supervised Learning**  
Authors: Junnan Li, Richard Socher, Steven C. H. Hoi  
Conference: ICLR 2020  
核心关键词：noisy label learning, semi-supervised learning, Gaussian Mixture Model, MixMatch, co-training, label refinement, pseudo label

### 推荐引用格式

```bibtex
@inproceedings{li2020dividemix,
  title={DivideMix: Learning with Noisy Labels as Semi-supervised Learning},
  author={Li, Junnan and Socher, Richard and Hoi, Steven C. H.},
  booktitle={International Conference on Learning Representations},
  year={2020}
}
```

### 你需要记住的一句话

> DivideMix 的核心思想是：先根据每个样本的 loss 把训练集分成“可能干净的 labeled set”和“可能有噪声的 unlabeled set”，然后不要直接丢弃 noisy samples，而是把它们当成无标签数据，用半监督学习继续利用。

---

## 2. 为什么 Co-teaching 之后要学 DivideMix？

你已经学过 Co-teaching 和 JoCoR，它们主要做的是：

```text
判断哪些样本可靠
↓
只用可靠样本训练
↓
疑似错标样本暂时不用
```

也就是说，Co-teaching / JoCoR 对高 loss 样本的处理比较保守：

> 这个样本看起来可疑，所以我先不学。

但是 DivideMix 更进一步：

> 这个样本的人工标签可能不可信，但它的输入信号本身仍然有价值。既然标签不可信，那我就把它当成 unlabeled data，用半监督学习继续利用。

这就是 DivideMix 和 Co-teaching 最大的区别。

---

## 3. 用 HD-sEMG 的例子理解 DivideMix

你的 HD-sEMG 数据可以理解为：

```text
x_i：一段高密度肌电信号，例如 H × W × T
\tilde{y}_i：人工标签，例如握拳、伸腕、屈腕、捏合
```

问题是，人工标签可能错。

例如：

| 样本 | 真实动作 | 人工标签 | 信号模式 | 标签是否可信 |
|---|---|---|---|---|
| A | 握拳 | 握拳 | 稳定握拳模式 | 可信 |
| B | 伸腕 | 伸腕 | 稳定伸腕模式 | 可信 |
| C | 握拳 | 伸腕 | 像握拳 | 不可信 |
| D | 过渡段 | 握拳 | 放松到握拳之间 | 不确定 |
| E | 低幅值捏合 | 捏合 | 信号弱 | 不确定 |

Co-teaching 可能只用 A、B，暂时不用 C、D、E。

DivideMix 的处理更像：

| 样本 | DivideMix 怎么处理 |
|---|---|
| A、B | 当 labeled clean samples，用人工标签训练 |
| C、D、E | 当 unlabeled samples，不完全相信人工标签，但仍然用模型猜标签继续训练 |

所以 DivideMix 比 Co-teaching 更积极：

> 它不直接浪费疑似错标样本，而是把它们转成无标签样本，用半监督学习利用。

---

## 4. DivideMix 的整体算法流程

DivideMix 可以分成 5 个核心步骤：

```text
Step 1：Warm-up 预训练两个网络
Step 2：用每个样本的 loss 拟合 GMM
Step 3：把数据分成 clean labeled set 和 noisy unlabeled set
Step 4：用 MixMatch 风格的半监督学习训练
Step 5：两个网络互相划分数据，减少 confirmation bias
```

整体流程图：

```text
带噪声标签训练集 D = {(x_i, y~_i)}
        │
        ↓
   两个网络 warm-up
        │
        ↓
计算每个样本的 CE loss
        │
        ↓
用 GMM 拟合 loss 分布
        │
        ↓
┌──────────────────────────────┐
│ 低 loss：clean labeled set     │
│ 高 loss：noisy unlabeled set   │
└──────────────────────────────┘
        │
        ↓
用 MixMatch 做半监督训练
        │
        ↓
两个网络互相提供数据划分
        │
        ↓
得到抗噪声分类模型
```

---

## 5. Step 1：Warm-up 是什么？

DivideMix 一开始不能马上分 clean/noisy，因为模型刚初始化时 loss 排名没有意义。

所以先做一段普通 CE 训练，叫 warm-up。

设两个网络为：

\[
f_1(x;\theta_1), \quad f_2(x;\theta_2)
\]

输入一段 HD-sEMG 样本：

\[
x_i \in \mathbb{R}^{H\times W\times T}
\]

观测标签是：

\[
\widetilde{y}_i
\]

模型输出 logits：

\[
z_i=f(x_i;\theta)
\]

经过 softmax 得到类别概率：

\[
p_{ic}=\frac{\exp(z_{ic})}{\sum_{k=1}^{C}\exp(z_{ik})}
\]

交叉熵损失：

\[
\ell_i(\theta)=-\log p_{i,\widetilde{y}_i}
\]

warm-up 阶段就是普通训练：

\[
\min_{\theta}\frac{1}{N}\sum_{i=1}^{N}\ell_i(\theta)
\]

### 为什么要 warm-up？

因为模型需要先学到一些基本模式。

对 HD-sEMG 来说，模型需要先大致知道：

- 握拳的空间激活模式是什么；
- 伸腕的空间激活模式是什么；
- 放松和动作状态有什么区别；
- 哪些通道区域更有判别性。

只有模型开始学到这些模式后，loss 才能反映样本是否可疑。

---

## 6. Step 2：用 loss 拟合 GMM

DivideMix 的关键不是简单设置一个阈值，例如：

```text
loss < 1.0 就是 clean
loss > 1.0 就是 noisy
```

它使用 **Gaussian Mixture Model，GMM，高斯混合模型** 来拟合样本 loss 的分布。

### 6.1 为什么 loss 可以分成两个分布？

训练一段时间后，通常会出现：

```text
clean samples：loss 较小
noisy samples：loss 较大
```

所以 loss 分布大致可以看成两个高斯分布的混合：

\[
p(\ell)=\pi_1\mathcal{N}(\ell;\mu_1,\sigma_1^2)+\pi_2\mathcal{N}(\ell;\mu_2,\sigma_2^2)
\]

其中：

| 符号 | 含义 |
|---|---|
| \(\ell\) | 某个样本的 loss |
| \(\mathcal{N}(\ell;\mu_1,\sigma_1^2)\) | 低 loss 分布，通常对应 clean samples |
| \(\mathcal{N}(\ell;\mu_2,\sigma_2^2)\) | 高 loss 分布，通常对应 noisy samples |
| \(\pi_1,\pi_2\) | 两个分布的混合权重 |

如果：

\[
\mu_1 < \mu_2
\]

那么均值较小的那个高斯分布被认为是 clean component。

### 6.2 每个样本的 clean probability

GMM 不只是给出 clean/noisy 的硬划分，还会给每个样本一个概率：

\[
w_i=P(\text{clean}\mid \ell_i)
\]

可以理解为：

> 样本 \(i\) 有多大概率是干净标签样本。

如果 \(w_i\) 高，例如：

\[
w_i=0.95
\]

说明它很可能是 clean sample。

如果：

\[
w_i=0.10
\]

说明它很可能是 noisy sample。

---

## 7. Step 3：划分 labeled set 和 unlabeled set

DivideMix 根据 \(w_i\) 把数据分成两类：

```text
clean probability 高 → labeled set
clean probability 低 → unlabeled set
```

数学上可以写成：

\[
\mathcal{X}=\{(x_i,\widetilde{y}_i,w_i): w_i \geq \gamma\}
\]

\[
\mathcal{U}=\{x_i: w_i < \gamma\}
\]

其中：

| 符号 | 含义 |
|---|---|
| \(\mathcal{X}\) | 可能干净的 labeled set |
| \(\mathcal{U}\) | 可能有噪声的 unlabeled set |
| \(\gamma\) | 阈值，例如 0.5 |

注意：

> \(\mathcal{U}\) 不是被丢弃，而是把标签去掉，当无标签数据使用。

这一步是 DivideMix 的核心思想。

---

## 8. Step 4：Label co-refinement：修正 labeled set 的标签

对于 labeled set \(\mathcal{X}\)，虽然这些样本比较可信，但也不一定 100% 干净。

所以 DivideMix 不直接完全相信原始标签，而是把人工标签和模型预测混合，得到 refined label。

设人工标签 one-hot 为：

\[
\widetilde{y}_i
\]

模型预测为：

\[
p_i
\]

GMM 给出的 clean probability 为：

\[
w_i
\]

则 refined label 可以理解为：

\[
\bar{y}_i=w_i\widetilde{y}_i+(1-w_i)p_i
\]

含义是：

| \(w_i\) | refined label 更相信谁 |
|---|---|
| 接近 1 | 更相信人工标签 |
| 接近 0 | 更相信模型预测 |
| 中间值 | 人工标签和模型预测折中 |

这一步很重要。

Co-teaching 只说：

> 可信样本用原标签训练。

DivideMix 更细：

> 即使是可信样本，也用 clean probability 对标签进行软修正。

---

## 9. Step 5：Label co-guessing：给 unlabeled set 猜标签

对于 unlabeled set \(\mathcal{U}\)，人工标签不可信，所以不用原标签。

DivideMix 会让两个网络对它进行预测，然后平均，得到 pseudo label。

设两个网络预测为：

\[
p_i^{(1)}=\text{softmax}(f_1(x_i;\theta_1))
\]

\[
p_i^{(2)}=\text{softmax}(f_2(x_i;\theta_2))
\]

则伪标签可以写成：

\[
q_i=\frac{p_i^{(1)}+p_i^{(2)}}{2}
\]

然后通常会做 sharpening，让概率分布更尖锐：

\[
q_{ic}^{sharp}=\frac{q_{ic}^{1/T}}{\sum_{k=1}^{C}q_{ik}^{1/T}}
\]

其中 \(T\) 是温度参数。

当 \(T<1\) 时，最大概率类别会更突出。

例如：

```text
原始 q = [0.60, 0.30, 0.10]
sharpen 后可能变成 [0.78, 0.20, 0.02]
```

这一步可以理解为：

> 对疑似 noisy 样本，不相信人工标签，而是让模型自己根据信号模式猜一个 soft pseudo label。

---

## 10. Step 6：MixMatch / MixUp 半监督训练

DivideMix 借鉴了 MixMatch 思想。

它把 labeled samples 和 unlabeled samples 组成一个集合，然后做 MixUp。

### 10.1 MixUp 公式

给两个样本：

\[
(x_a,y_a), \quad (x_b,y_b)
\]

从 Beta 分布采样：

\[
\lambda \sim \text{Beta}(\alpha,\alpha)
\]

然后混合输入：

\[
x'=\lambda x_a+(1-\lambda)x_b
\]

混合标签：

\[
y'=\lambda y_a+(1-\lambda)y_b
\]

这就是 MixUp。

在 DivideMix 里，\(y_a,y_b\) 可能是：

- refined label；
- pseudo label；
- soft label。

### 10.2 对 HD-sEMG 要注意

对图像来说，MixUp 比较常见。

但对 HD-sEMG，要谨慎。

直接混合两个不同手势的原始肌电信号，可能生理意义不强。

所以 HD-sEMG 里可以考虑：

| 原始 DivideMix | HD-sEMG 改法 |
|---|---|
| raw input MixUp | embedding-level MixUp |
| 图像增强 | 时间平移、幅值扰动、通道 dropout、电极 shift |
| 随机混合不同类 | 优先在相似力度/相近动作阶段内混合 |

也就是说：

> DivideMix 的半监督思想可以迁移，但 MixUp 的具体形式需要符合肌电信号生理意义。

---

## 11. Step 7：最终损失函数

DivideMix 的损失通常包含两部分：

### 11.1 labeled loss

对 labeled set 使用交叉熵或 soft-label cross entropy：

\[
\mathcal{L}_x=-\frac{1}{|\mathcal{X}|}\sum_{x_i\in \mathcal{X}} \bar{y}_i^T\log p_i
\]

其中：

| 符号 | 含义 |
|---|---|
| \(\bar{y}_i\) | refined label |
| \(p_i\) | 模型预测概率 |

### 11.2 unlabeled loss

对 unlabeled set，让模型预测接近 pseudo label：

\[
\mathcal{L}_u=\frac{1}{|\mathcal{U}|}\sum_{x_i\in \mathcal{U}}\|p_i-q_i\|_2^2
\]

其中：

| 符号 | 含义 |
|---|---|
| \(q_i\) | pseudo label |
| \(p_i\) | 模型预测 |

### 11.3 总损失

\[
\mathcal{L}=\mathcal{L}_x+\lambda_u\mathcal{L}_u+\lambda_r\mathcal{L}_{reg}
\]

其中：

| 项 | 含义 |
|---|---|
| \(\mathcal{L}_x\) | 有标签 clean 样本监督损失 |
| \(\mathcal{L}_u\) | 无标签 noisy 样本一致性损失 |
| \(\mathcal{L}_{reg}\) | 正则项，防止预测类别分布坍缩 |
| \(\lambda_u, \lambda_r\) | 权重系数 |

---

## 12. 两个网络为什么还要存在？

DivideMix 也使用两个网络，原因和 Co-teaching 类似：减少 confirmation bias。

如果一个网络自己划分数据，再自己用这个划分训练自己，就可能出现：

```text
我自己觉得它是 clean
我用它训练自己
训练后我更相信自己的错误判断
```

这叫 confirmation bias，自我确认偏差。

DivideMix 的解决办法是：

```text
网络 1 根据自己的 loss 划分数据
但这个划分给网络 2 用

网络 2 根据自己的 loss 划分数据
但这个划分给网络 1 用
```

这和 Co-teaching 的“互教”思想一致，但 DivideMix 更复杂，因为它不仅交换 clean sample selection，还交换 labeled/unlabeled division。

---

## 13. DivideMix 和 Co-teaching / JoCoR 的核心区别

| 方法 | 是否两个网络 | 怎么判断样本可靠 | 高 loss 样本怎么处理 | 是否修正标签 | 是否半监督利用 |
|---|---|---|---|---|---|
| CE | 否 | 不判断 | 照样训练 | 否 | 否 |
| Co-teaching | 是 | small-loss | 暂时不用 | 否 | 否 |
| JoCoR | 是 | joint small-loss + agreement | 暂时不用 | 否 | 否 |
| DivideMix | 是 | GMM on loss | 当作 unlabeled | 是，soft refinement / pseudo label | 是 |

最重要的一句话：

> Co-teaching 是“筛掉可疑样本”，DivideMix 是“把可疑样本当无标签样本继续利用”。

---

## 14. 用 HD-sEMG 重新解释 DivideMix

在高密度肌电中，样本可以写成：

\[
x_i \in \mathbb{R}^{H\times W\times T}
\]

标签为：

\[
\widetilde{y}_i\in\{1,2,\dots,C\}
\]

模型学习：

\[
f: \mathbb{R}^{H\times W\times T}\rightarrow \mathbb{R}^{C}
\]

DivideMix 对 HD-sEMG 的含义是：

### 14.1 稳定手势中段

```text
信号稳定
标签可信
loss 小
GMM 判断为 clean
进入 labeled set
用 refined label 监督训练
```

### 14.2 错标样本

```text
信号像握拳
标签写成伸腕
loss 大
GMM 判断为 noisy
进入 unlabeled set
不用原标签
用模型 pseudo label 训练
```

### 14.3 动作边界样本

```text
信号处于放松到握拳过渡
标签可能是握拳
loss 中等或较大
进入 unlabeled 或低权重 labeled
用 soft label / consistency 训练
```

这对 HD-sEMG 很有意义，因为动作边界样本并不一定“完全错误”，而是“不确定”。

DivideMix 的 soft-label / pseudo-label 思想比 Co-teaching 的直接排除更适合这种情况。

---

## 15. DivideMix 的优点

### 优点 1：不浪费疑似 noisy 样本

Co-teaching 可能会把 high-loss 样本直接排除。

DivideMix 把它们变成 unlabeled data，继续利用输入信息。

### 优点 2：有标签修正思想

DivideMix 对 labeled set 做 label refinement，对 unlabeled set 做 label guessing。

这比只用原始人工标签更稳。

### 优点 3：是强基线

DivideMix 是 LNL 领域非常经典的强 baseline。很多后续方法都会和它比较。

### 优点 4：适合扩展到 HD-sEMG

高密度肌电里有大量边界模糊和标签不确定窗口，DivideMix 的 soft label 思路更自然。

---

## 16. DivideMix 的缺点，尤其对 HD-sEMG

### 缺点 1：依赖 loss 分布可分

DivideMix 假设 clean/noisy samples 在 loss 分布上能被 GMM 分开。

但 HD-sEMG 中可能出现：

```text
hard clean samples：loss 大，但标签是对的
simple noisy samples：loss 小，但标签是错的
```

这会导致 GMM 误分。

### 缺点 2：MixUp 不一定有生理意义

图像 MixUp 可以较自然，但 HD-sEMG 原始信号直接线性混合可能不符合真实肌肉激活机制。

所以你需要改造增强策略。

### 缺点 3：计算复杂度高

DivideMix 需要：

- 两个网络；
- warm-up；
- 每个 epoch 拟合 GMM；
- semi-supervised training；
- MixUp；
- label refinement 和 co-guessing。

比 Co-teaching 和 JoCoR 更复杂。

### 缺点 4：伪标签可能会错

如果模型一开始猜错 pseudo label，后面可能继续强化错误。

这也是为什么它需要两个网络和 co-training。

---

## 17. 怎么迁移成 HD-sEMG-DivideMix？

一个适合你方向的改造版可以写成：

```text
HD-sEMG 输入
↓
时空编码器：2D/3D CNN、ResNet、TCN、Transformer
↓
Warm-up
↓
每个样本计算 loss
↓
GMM 划分 clean / noisy
↓
clean 样本：用 refined label 监督训练
noisy 样本：去掉人工标签，用 pseudo label / consistency 训练
↓
加入 HD-sEMG 特异性增强
↓
输出鲁棒手势分类器
```

### 17.1 HD-sEMG 特异性可靠性指标

原始 DivideMix 只看 loss，你可以加入：

| 指标 | 含义 |
|---|---|
| loss | 模型是否相信人工标签 |
| temporal consistency | 相邻窗口预测是否稳定 |
| spatial activation consistency | 电极空间激活是否符合该手势模式 |
| channel quality | 是否存在坏通道、电极脱落 |
| confidence / entropy | 模型输出是否不确定 |
| subject-wise normalization | 是否受被试差异影响 |

最终可靠性可以是：

\[
r_i = \alpha w_i^{loss}+\beta w_i^{temp}+\gamma w_i^{spatial}
\]

其中：

| 项 | 含义 |
|---|---|
| \(w_i^{loss}\) | GMM 根据 loss 给出的 clean probability |
| \(w_i^{temp}\) | 时序一致性得分 |
| \(w_i^{spatial}\) | 空间肌电拓扑一致性得分 |

这就是从普通 DivideMix 到 HD-sEMG-specific DivideMix 的创新方向。

---

## 18. 你给老师复述的版本

可以直接这样讲：

> DivideMix 是 ICLR 2020 的经典 noisy label learning 方法。它和 Co-teaching、JoCoR 的区别是，Co-teaching 主要把 high-loss 样本暂时排除，而 DivideMix 会把训练集动态划分为 clean labeled set 和 noisy unlabeled set。具体来说，DivideMix 先 warm-up 两个网络，然后根据每个样本的交叉熵 loss 拟合 Gaussian Mixture Model，估计每个样本属于 clean component 的概率。clean probability 高的样本作为 labeled data，clean probability 低的样本去掉原始标签，作为 unlabeled data。随后它使用 MixMatch 式半监督学习：对 labeled data 进行 label refinement，对 unlabeled data 进行 label guessing，再通过 MixUp 和一致性损失共同训练。两个网络互相提供数据划分，以减少 confirmation bias。对于高密度肌电，DivideMix 的价值在于它不浪费动作边界、过渡段和疑似错标样本，而是把它们作为无标签数据继续利用；但它也有局限，因为 HD-sEMG 中 high-loss 不一定代表错标，MixUp 也需要改造成符合肌电生理意义的增强方式。

---

## 19. 一句话总结

> DivideMix = GMM 根据 loss 分 clean/noisy + noisy 当 unlabeled + MixMatch 半监督训练 + 两个网络互相划分数据。

再简化：

> Co-teaching 是“可疑样本先不学”；DivideMix 是“可疑样本不信它的标签，但仍然把信号当无标签数据利用”。

---

## 20. 学完 DivideMix 后你应该能回答的 6 个问题

1. DivideMix 为什么要先 warm-up？  
   因为刚初始化时 loss 没有区分度，需要先让模型学到基本模式。

2. DivideMix 怎么判断 clean / noisy？  
   用每个样本的 CE loss 拟合两个成分的 GMM，低 loss component 视为 clean。

3. 高 loss 样本被丢掉了吗？  
   没有。它们被当成 unlabeled samples 继续用于半监督学习。

4. DivideMix 是否修正标签？  
   是。它对 labeled samples 做 label refinement，对 unlabeled samples 做 pseudo label guessing。

5. 为什么要两个网络？  
   为了互相划分数据，减少一个网络自己判断自己训练造成的 confirmation bias。

6. 对 HD-sEMG 有什么启发？  
   HD-sEMG 中很多样本不是完全错误，而是不确定或边界模糊；DivideMix 的 soft-label 和 unlabeled 利用方式比简单丢弃更合适。

