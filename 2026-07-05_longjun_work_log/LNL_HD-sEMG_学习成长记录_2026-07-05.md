# LNL × 高密度肌电（HD-sEMG）学习成长记录

> 日期：2026-07-05  
> 主题：从“看不懂 LNL 算法”到初步理解 Co-teaching、JoCoR、DivideMix 以及多网络扩展思路  
> 研究背景：高密度肌电（HD-sEMG）手势识别中的噪声标签学习（Learning with Noisy Labels, LNL）

---

## 0. 今天的核心成长结论

今天最重要的收获不是“记住了几个论文名”，而是逐渐分清了下面几件事：

1. **CE baseline 不是模型架构，而是普通监督训练方式。**
2. **Co-teaching、JoCoR、DivideMix 也不是具体 backbone，而是噪声标签学习算法框架。**
3. **模型架构和训练算法是两件事：**
   - 模型架构：ResNet1D、CNN、Transformer、TCN、3D-CNN 等。
   - 训练算法：CE、Co-teaching、JoCoR、DivideMix 等。
4. **Co-teaching 的本质不是修正标签，而是筛选 small-loss 样本。**
5. **DivideMix 比 Co-teaching 更进一步：它不直接相信可疑样本的人工标签，但也不完全丢掉，而是把它当作 unlabeled data 做半监督学习。**
6. **对 HD-sEMG 来说，不能只依赖 loss 判断样本是否干净，因为 high-loss 样本也可能是困难但真实的干净样本。**
7. **你提出的“多网络委员会”想法是合理的，它可以作为后续创新方向，但需要注意网络之间要有差异性，否则多个网络可能一起犯同样的错。**

---

## 1. 最开始的问题：我要读哪些 LNL 论文？

一开始的目标是：

> 找到 Learning with Noisy Labels（LNL）领域，尤其是时序信号、生理信号、EMG/sEMG/HD-sEMG 相关论文，并筛选出有算法架构、值得精读的文章。

最初我们把文献分成几类：

### 1.1 通用 LNL 经典方法

这些方法虽然很多来自图像分类，但它们是 LNL 的基础框架，老师问“经典 LNL 方法”时基本绕不开。

| 方法 | 代表论文 | 核心思想 |
|---|---|---|
| CE baseline | Cross Entropy baseline | 普通交叉熵训练，作为基础对照组 |
| Co-teaching | Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels | 两个网络互相选择 small-loss 样本 |
| JoCoR | Combating Noisy Labels by Agreement: A Joint Training Method with Co-Regularization | 两个网络共同训练，并要求预测一致 |
| DivideMix | DivideMix: Learning with Noisy Labels as Semi-supervised Learning | 把样本分成 clean labeled 和 noisy unlabeled，再做半监督学习 |
| GCE | Generalized Cross Entropy | 在 CE 和 MAE 之间折中，鲁棒损失 |
| SCE | Symmetric Cross Entropy | CE + Reverse CE，增强噪声鲁棒性 |
| MAE | Mean Absolute Error | 理论上抗噪，但深度网络中容易欠拟合 |
| APL | Active Passive Loss | active loss + passive loss 互补 |
| FINE | FINE Samples for Learning with Noisy Labels | 基于特征空间挑选 clean samples |
| DISC | Learning from Noisy Labels via Dynamic Instance-Specific Selection and Correction | 动态样本选择与标签修正 |

### 1.2 时序 LNL 方法

这些更贴近 HD-sEMG，因为肌电本质是时序信号。

| 方法 | 论文 | 对 HD-sEMG 的启发 |
|---|---|---|
| Scale-teaching | Robust Multi-scale Training for Time Series Classification with Noisy Labels | 多尺度窗口、多尺度肌电模式 |
| CTW | Confident Time-Warping for Time-Series Label-Noise Learning | 只对可靠样本做 time-warping |
| Temporal Label Noise | Learning under Temporal Label Noise | 标签噪声随动作阶段变化 |
| RTS | Learning Robustly from Time Series Data with Noisy Label | 同时考虑 noisy label 和类别不平衡 |
| TS-CoT | A Co-training Approach for Noisy Time Series Learning | 多视角时序表征学习 |

### 1.3 生理信号 LNL 方法

这类虽然不一定直接是 HD-sEMG，但和 EEG、ECG、PPG 等生理信号相关，适合做领域动机。

| 方法 | 论文 | 信号类型 | 启发 |
|---|---|---|---|
| ALR-CNN | sEMG-Based Gesture Recognition Using Deep Learning From Noisy Labels | sEMG | 直接做 sEMG noisy labels，是最贴近你的前人工作 |
| BUNDL | Bayesian Uncertainty-aware Deep Learning with Noisy Labels | EEG | 标签歧义和不确定性建模 |
| Negative-ResNet | Noisy Ambulatory Electrocardiogram Signal Classification Scheme | ECG | Positive learning + negative learning |
| PPG label noise analysis | Impact of Label Noise on Physiological Signal Classification | PPG | 生理信号 label noise 对性能影响的分析 |

---

## 2. 老师给的表格给出了“明路”

后来你上传了老师给的结果表格。这个表格比较了不同方法在不同噪声设置下的表现，例如：

- None：无噪声标签
- 20% symmetric noise
- 40% symmetric noise
- 20% asymmetric noise
- 40% asymmetric noise

表格中的方法包括：

- CE
- GCE
- SCE
- MAE
- Mixup
- DISC
- Co-teaching
- Co-teaching+
- JoCoR
- FINE
- DivideMix
- ANNE

当时我们得出的判断是：

> 不应该再漫无目的地读很多论文，而应该围绕老师表格中的方法，逐个理解算法框架、可靠样本判断方式、错标处理方式，以及是否适合改造成 HD-sEMG 方法。

因此我们把方法分成三梯队：

### 2.1 第一梯队：必须精读

| 方法 | 原因 |
|---|---|
| JoCoR | 表格里综合表现强，必须理解 joint training 和 co-regularization |
| Co-teaching | 经典 LNL 方法，表格里也较强 |
| DISC | clean 和 noisy 设置下都强，值得挖机制 |
| DivideMix | 强基线，需要理解 clean/noisy 划分与半监督训练 |
| FINE | feature-based clean sample selection，和 HD-sEMG 表征问题相关 |

### 2.2 第二梯队：baseline 方法

| 方法 | 作用 |
|---|---|
| GCE | 鲁棒 loss 经典 |
| SCE | 鲁棒 loss 经典 |
| MAE | 理论抗噪，但容易 underfit |
| APL | active + passive loss |
| Mixup | 数据增强 baseline |

### 2.3 第三梯队：补充方法

| 方法 | 作用 |
|---|---|
| Co-teaching+ | 解释 disagreement strategy 为什么在 HD-sEMG 可能不稳定 |
| ANNE | 需要确认代码库里的具体实现 |
| CE | 基础对照组，不是 LNL 方法 |

---

## 3. 入门概念：CE、CE baseline、模型架构分别是什么？

### 3.1 CE 是什么？

CE = Cross Entropy，交叉熵损失。

在分类任务中，模型输出每个类别的概率：

\[
p_c = \frac{e^{z_c}}{\sum_{k=1}^{C} e^{z_k}}
\]

人工标签通常是 one-hot 形式：

\[
y = [0, 1, 0, 0]
\]

多分类交叉熵为：

\[
L_{CE} = -\sum_{c=1}^{C} y_c \log(p_c)
\]

因为 one-hot 标签只有真实类别为 1，所以可以简化为：

\[
L_{CE} = -\log(p_{\text{true}})
\]

直观理解：

| 模型给人工标签类别的概率 | CE loss | 意义 |
|---:|---:|---|
| 0.99 | 很小 | 模型非常相信标签 |
| 0.80 | 小 | 模型基本相信标签 |
| 0.50 | 中等 | 模型不确定 |
| 0.10 | 大 | 模型不相信标签 |
| 0.01 | 很大 | 模型强烈不相信标签 |

### 3.2 CE baseline 是什么？

你曾经问：

> CE baseline 就是最普通的训练方法是什么意思，这是一个模型架构吗？

答案是：

> **CE baseline 不是模型架构，而是普通交叉熵训练方式。**

例如：

| 项目 | 含义 |
|---|---|
| ResNet1D | 模型架构 |
| CNN | 模型架构 |
| Transformer | 模型架构 |
| CE baseline | 用普通 CE loss 训练模型 |
| Co-teaching | 一种噪声标签训练策略 |
| JoCoR | 一种噪声标签训练策略 |
| DivideMix | 一种噪声标签训练策略 |

所以：

\[
\text{ResNet1D + CE baseline}
\]

意思是：

> 使用 ResNet1D 作为 backbone，不做样本筛选、不做标签修正，只用普通 CE loss 训练。

### 3.3 CE 在 noisy label 中的问题

如果人工标签是对的，CE 很合理。

但如果人工标签错了，例如：

\[
\text{真实动作：握拳}, \quad \widetilde{y}=\text{伸腕}
\]

CE 仍然会强迫模型把这个样本学成“伸腕”。

所以 CE 的问题是：

> 它完全相信人工标签，不判断标签是否可信。

这就是为什么 LNL 方法要在 CE baseline 之上做改进。

---

## 4. 第一个核心算法：Co-teaching

### 4.1 论文参考

**论文：**

> Han et al., *Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels*, NeurIPS 2018.

**方法：**

> Co-teaching，双网络互教式噪声标签学习。

### 4.2 最开始的困惑

你当时的疑问很关键：

> 最开始的权重都是随机的，loss 也是随机的，凭什么根据 loss 判断 clean/noisy？

这个问题抓住了 Co-teaching 的核心矛盾。

后来我们澄清：

> Co-teaching 不是在第 0 步就能判断 clean/noisy，而是利用训练早期逐渐出现的 loss 分化。

训练初期，模型几乎随机，loss 排名没有意义。  
随着训练进行：

- 模式稳定、标签一致的 easy clean samples 会先被拟合；
- 标签和信号冲突的 noisy samples loss 会下降得慢；
- 因此 small-loss 样本中 clean sample 的比例更高。

这就是 **memorization effect**。

### 4.3 Co-teaching 的核心直觉

普通 CE：

\[
\text{所有样本都参与训练}
\]

Co-teaching：

\[
\text{只选择当前 batch 中 small-loss 样本参与训练}
\]

但它不是一个网络自己选自己学，而是两个网络互相选：

\[
\text{网络 A 选 small-loss 样本给网络 B 学}
\]

\[
\text{网络 B 选 small-loss 样本给网络 A 学}
\]

### 4.4 Co-teaching 的算法流程

```text
输入 noisy training set
        ↓
初始化两个网络 f1, f2
        ↓
每个 mini-batch 中：
        ↓
f1 计算每个样本 loss
f2 计算每个样本 loss
        ↓
f1 选择 small-loss 样本 B1_small
f2 选择 small-loss 样本 B2_small
        ↓
f1 用 f2 选出的样本更新
f2 用 f1 选出的样本更新
```

### 4.5 严格数学框架

带噪声标签数据集：

\[
\widetilde{\mathcal{D}} = \{(x_i,\widetilde{y}_i)\}_{i=1}^{N}
\]

其中：

- \(x_i\)：HD-sEMG 样本
- \(y_i\)：真实标签，但训练时未知
- \(\widetilde{y}_i\)：人工观测标签，可能有错

两个网络：

\[
f_1(x;\theta_1), \quad f_2(x;\theta_2)
\]

对 mini-batch \(\mathcal{B}\) 中每个样本计算 CE loss：

\[
\ell_i^{(1)} = -\log p_{i,\widetilde{y}_i}^{(1)}
\]

\[
\ell_i^{(2)} = -\log p_{i,\widetilde{y}_i}^{(2)}
\]

定义 remember rate：

\[
R(t)=1-\min\left(\frac{t}{T_k}\tau,\tau\right)
\]

其中：

- \(t\)：当前 epoch
- \(\tau\)：预估噪声率
- \(T_k\)：筛选强度逐渐增加的 epoch 数

每个 batch 选择样本数：

\[
K(t)=\lfloor R(t)B \rfloor
\]

网络 1 选择的 small-loss 样本：

\[
\mathcal{B}_1^{small}
=
\operatorname{arg\,min}_{\mathcal{S}\subset \mathcal{B}, |\mathcal{S}|=K(t)}
\sum_{i\in \mathcal{S}}\ell_i^{(1)}
\]

网络 2 选择：

\[
\mathcal{B}_2^{small}
=
\operatorname{arg\,min}_{\mathcal{S}\subset \mathcal{B}, |\mathcal{S}|=K(t)}
\sum_{i\in \mathcal{S}}\ell_i^{(2)}
\]

交叉更新：

\[
\mathcal{L}_1(\theta_1)
=
\frac{1}{|\mathcal{B}_2^{small}|}
\sum_{i\in \mathcal{B}_2^{small}}
-\log p_{i,\widetilde{y}_i}^{(1)}
\]

\[
\mathcal{L}_2(\theta_2)
=
\frac{1}{|\mathcal{B}_1^{small}|}
\sum_{i\in \mathcal{B}_1^{small}}
-\log p_{i,\widetilde{y}_i}^{(2)}
\]

更新：

\[
\theta_1 \leftarrow \theta_1 - \eta \nabla_{\theta_1}\mathcal{L}_1
\]

\[
\theta_2 \leftarrow \theta_2 - \eta \nabla_{\theta_2}\mathcal{L}_2
\]

### 4.6 Co-teaching 的本质

普通 CE baseline：

\[
\min_{\theta}\sum_{i\in \mathcal{B}}\ell_i(\theta)
\]

Co-teaching：

\[
\min_{\theta_1}\sum_{i\in \mathcal{B}_2^{small}}\ell_i^{(1)}(\theta_1)
\]

\[
\min_{\theta_2}\sum_{i\in \mathcal{B}_1^{small}}\ell_i^{(2)}(\theta_2)
\]

也就是说：

> Co-teaching 把“所有样本训练”变成了“被另一个网络筛选出的 small-loss 样本训练”。

### 4.7 你最终理解到的关键点

你问：

> 所以这个只可以舍弃错误标签，而不能修正错误标签然后加以训练，对不对？

答案是：

> 对，原始 Co-teaching 是 sample selection，不是 label correction。

它对 high-loss 样本的处理是：

\[
\text{暂时不用}
\]

而不是：

\[
\text{把标签改对后再用}
\]

### 4.8 Co-teaching 对 HD-sEMG 的局限

HD-sEMG 中 high-loss 不一定是错标，也可能是：

- 动作起始段
- 动作结束段
- 过渡段
- 低力度动作
- 相似手势
- 被试差异
- 电极接触不好
- 通道空间模式异常

所以 Co-teaching 适合作为 baseline，但不能作为最终方法。

---

## 5. 第二个核心算法：JoCoR

### 5.1 论文参考

**论文：**

> Wei et al., *Combating Noisy Labels by Agreement: A Joint Training Method with Co-Regularization*, CVPR 2020.

### 5.2 和 Co-teaching 的关系

Co-teaching 是：

> 两个网络互相选择 small-loss 样本。

JoCoR 是：

> 两个网络共同训练，并且要求预测一致。

也就是说，JoCoR 不是只看“和人工标签的 CE 小不小”，还看：

> 两个网络对同一个样本的预测是否一致。

### 5.3 JoCoR 的核心 loss

两个网络输出：

\[
p_i^{(1)} = f_1(x_i;\theta_1)
\]

\[
p_i^{(2)} = f_2(x_i;\theta_2)
\]

监督损失：

\[
\ell_{sup}
=
CE(p_i^{(1)},\widetilde{y}_i)
+
CE(p_i^{(2)},\widetilde{y}_i)
\]

一致性损失：

\[
\ell_{con}
=
D_{KL}(p_i^{(1)}\|p_i^{(2)})
+
D_{KL}(p_i^{(2)}\|p_i^{(1)})
\]

joint loss：

\[
\ell_i
=
(1-\lambda)\ell_{sup}
+
\lambda\ell_{con}
\]

其中 \(\lambda\) 控制监督损失和一致性损失的权重。

### 5.4 JoCoR 如何判断样本可靠

Co-teaching 看：

\[
CE \text{ 是否小}
\]

JoCoR 看：

\[
\text{CE 是否小} + \text{两个网络是否一致}
\]

如果一个样本：

- 和人工标签一致，CE 小；
- 两个网络预测也一致；

那么它更可能是可靠样本。

如果一个样本：

- CE 大；
- 两个网络预测差异也大；

那么它可能是错标、边界样本或困难样本。

### 5.5 JoCoR 和 Co-teaching 的区别

| 对比项 | Co-teaching | JoCoR |
|---|---|---|
| 网络数量 | 两个 | 两个 |
| 样本选择依据 | 每个网络自己的 CE loss | 两个网络的 joint loss |
| 是否要求预测一致 | 不要求 | 要求 |
| 更新方式 | 交叉更新 | 共同更新 |
| 是否修正标签 | 不修正 | 不直接修正 |
| 本质 | sample selection | sample selection + agreement regularization |

### 5.6 对 HD-sEMG 的理解

JoCoR 对 HD-sEMG 有意义，因为：

> 如果一个 HD-sEMG 样本空间激活稳定、标签正确，两个网络更可能给出一致预测；如果样本处于动作边界、标签错了、信号弱或相似手势混淆，两个网络更可能出现分歧。

但 JoCoR 仍然不直接修正标签。它主要还是：

\[
\text{选择更可信的样本训练}
\]

---

## 6. 第三个核心算法：DivideMix

### 6.1 论文参考

**论文：**

> Li et al., *DivideMix: Learning with Noisy Labels as Semi-supervised Learning*, ICLR 2020.

### 6.2 为什么 Co-teaching 后要学 DivideMix？

Co-teaching 的处理逻辑是：

\[
\text{可疑样本先不学}
\]

DivideMix 的逻辑是：

\[
\text{可疑样本不信它的人工标签，但仍然作为无标签数据使用}
\]

这是一个很重要的升级。

### 6.3 DivideMix 的一句话理解

> DivideMix 把 noisy label learning 转化为 semi-supervised learning。

也就是：

1. 先把训练样本分成 clean 和 noisy；
2. clean 样本当作 labeled data；
3. noisy 样本当作 unlabeled data；
4. 用半监督学习方法继续训练。

### 6.4 DivideMix 的主要流程

```text
noisy dataset
      ↓
warm-up 训练两个网络
      ↓
计算每个样本的 loss
      ↓
用 GMM 拟合 loss 分布
      ↓
把样本划分成 clean labeled set 和 noisy unlabeled set
      ↓
clean 样本：用原标签或 refined label
noisy 样本：当 unlabeled，用 pseudo label
      ↓
MixMatch / MixUp 半监督训练
```

### 6.5 为什么要 GMM？

DivideMix 认为 loss 分布大致由两部分混合而成：

- clean samples：loss 小
- noisy samples：loss 大

因此用 Gaussian Mixture Model（GMM）拟合每个样本的 loss：

\[
p(\ell_i)=\pi_1\mathcal{N}(\ell_i;\mu_1,\sigma_1^2)
+
\pi_2\mathcal{N}(\ell_i;\mu_2,\sigma_2^2)
\]

其中：

- 低均值高斯：clean component
- 高均值高斯：noisy component

每个样本属于 clean component 的概率可以记为：

\[
w_i = P(\text{clean}|\ell_i)
\]

如果：

\[
w_i > \tau
\]

则样本被认为是 clean labeled sample。

否则被认为是 noisy unlabeled sample。

### 6.6 DivideMix 如何处理 clean 样本？

对于 clean 样本，DivideMix 不完全死信原标签，而是做 label refinement。

假设原标签 one-hot 为 \(\widetilde{y}_i\)，模型预测为 \(p_i\)，clean 概率为 \(w_i\)，可以构造 refined label：

\[
\hat{y}_i = w_i \widetilde{y}_i + (1-w_i)p_i
\]

意思是：

- 如果样本很干净，更多相信原标签；
- 如果样本不太确定，更多参考模型预测。

### 6.7 DivideMix 如何处理 noisy 样本？

对于 noisy 样本，不再相信其人工标签，而是把它当 unlabeled data。

用模型预测生成 pseudo label：

\[
q_i = \frac{1}{2}(p_i^{(1)} + p_i^{(2)})
\]

然后用半监督一致性训练。

### 6.8 DivideMix 的本质

Co-teaching：

\[
\text{high-loss 样本不学}
\]

DivideMix：

\[
\text{high-loss 样本不信标签，但作为 unlabeled 继续学}
\]

所以 DivideMix 是：

> sample selection + label refinement + semi-supervised learning

它比 Co-teaching 更充分利用数据。

### 6.9 对 HD-sEMG 的价值

HD-sEMG 数据获取成本高，不应该轻易丢掉 high-loss 样本。

很多 high-loss 样本可能是：

- 动作边界
- 过渡段
- 低强度动作
- 相似手势
- 被试差异
- 电极接触问题

因此 DivideMix 的思想很适合：

> 可疑样本不直接强监督，但可以作为 unlabeled data 做一致性学习。

---

## 7. 你提出的新思考：为什么不用多个网络？

在学了 Co-teaching、JoCoR、DivideMix 后，你提出一个很重要的问题：

> 既然两个网络互相作用有效，为什么不直接使用多个网络？多个网络一起判断，命中 clean 样本和区分 noisy 样本的概率是不是会更高？还可以挑选最好的几个网络进入下一步优化或半监督标签修正。

这个想法是合理的，它对应 LNL 里的：

> multi-network / ensemble / committee-based noisy label learning

### 7.1 为什么这个想法有道理？

假设有 \(M\) 个网络：

\[
f_1, f_2, \dots, f_M
\]

每个网络对样本 \(x_i\) 输出概率：

\[
p_i^{(m)}
\]

可以计算对人工标签的平均置信度：

\[
C_i = \frac{1}{M}\sum_{m=1}^{M}p_{i,\widetilde{y}_i}^{(m)}
\]

也可以计算网络之间的分歧：

\[
V_i =
\frac{1}{M}
\sum_{m=1}^{M}
\left(
p_{i,\widetilde{y}_i}^{(m)} - C_i
\right)^2
\]

定义可靠性：

\[
R_i = C_i - \alpha V_i
\]

其中：

- \(C_i\) 高：多个网络都相信原标签；
- \(V_i\) 低：多个网络意见一致；
- \(R_i\) 高：样本更可靠。

### 7.2 多网络如何处理样本？

可以分为三类：

| 类型 | 条件 | 处理 |
|---|---|---|
| reliable clean | 多网络都相信原标签 | 用原标签 CE 训练 |
| likely mislabeled | 多网络都不信原标签，但一致预测另一个类 | 用 soft pseudo label 修正 |
| uncertain | 网络分歧大 | 当作 unlabeled，用一致性或半监督训练 |

### 7.3 为什么不是网络越多越好？

主要问题包括：

1. 多个网络可能共同犯错；
2. 如果网络结构、数据增强、loss 都一样，它们不一定真正独立；
3. 多网络可能更偏向 easy samples；
4. 计算成本增加；
5. 如果没有 clean validation set，很难判断哪个网络“最好”。

### 7.4 对 HD-sEMG 更合理的多网络方式

不建议简单复制：

\[
\text{ResNet1D} \times 5
\]

更合理的是多视角网络：

| 网络 | 输入视角 | 关注点 |
|---|---|---|
| Temporal Net | channel × time | 时间波形 |
| Spatial Net | HD-sEMG grid | 电极空间激活 |
| Spatio-temporal Net | H × W × T | 空间激活随时间演化 |
| Spectral Net | STFT/CWT/频带能量 | 频域肌电特征 |

这可以形成：

> 多网络委员会式 HD-sEMG 噪声标签修正框架。

### 7.5 这个想法的论文级表述

可以命名为：

> Committee-guided Spatio-temporal Label Correction for HD-sEMG Gesture Recognition under Noisy Labels

整体框架：

```text
HD-sEMG noisy dataset
        ↓
多视角网络委员会
        ↓
Temporal / Spatial / Spectral / Spatio-temporal predictions
        ↓
计算可靠性：
平均置信度 + 网络分歧 + 类别平衡 + 时间一致性 + 空间拓扑一致性
        ↓
样本分组：
reliable clean / likely mislabeled / uncertain
        ↓
训练：
clean → 原标签 CE
mislabeled → soft pseudo label correction
uncertain → unlabeled consistency learning
```

---

## 8. 今天理解上的关键转变

### 8.1 从“论文名很多”转向“算法机制”

一开始关注的是：

> 有哪些论文？

后来变成：

> 每个算法到底怎么判断样本可靠？怎么处理错标？能不能迁移到 HD-sEMG？

这是非常重要的转变。

### 8.2 从“模型架构”转向“训练算法”

你逐渐分清了：

- ResNet1D / CNN / Transformer 是 backbone；
- CE / Co-teaching / JoCoR / DivideMix 是 training strategy；
- LNL 论文的核心通常不是换 backbone，而是换训练方式、样本选择方式和标签处理方式。

### 8.3 从“loss 大就是错标”转向“high-loss 也可能是 hard clean”

你已经意识到 Co-teaching 的局限：

> high-loss 可能是 noisy，也可能是 HD-sEMG 中真实但困难的样本。

这对后续创新非常关键。

### 8.4 从“两个网络互教”拓展到“多网络委员会”

你提出：

> 多个网络是不是可以提高识别 clean/noisy 的概率？

这个想法可以转化为：

> 多视角 HD-sEMG committee-based label correction。

这是比单纯复现经典 LNL 更贴近你领域的方向。

---

## 9. 三个已学算法总对比

| 方法 | 网络数量 | 判断样本可靠方式 | 处理疑似错标 | 是否修正标签 | 是否利用可疑样本 |
|---|---:|---|---|---|---|
| CE baseline | 1 | 不判断 | 全部照学 | 否 | 是，但可能学错 |
| Co-teaching | 2 | small-loss | 暂时不用 | 否 | 否或少量 |
| JoCoR | 2 | joint loss = CE + 一致性 | 暂时不用 | 否 | 否或少量 |
| DivideMix | 2 | loss + GMM clean probability | 当 unlabeled | 半修正/伪标签 | 是 |
| 你的多网络设想 | M | 多网络置信度 + 分歧 + 时空结构 | 分为 clean/mislabeled/uncertain | 可以 soft correction | 是 |

---

## 10. 给老师的阶段性汇报版本

你可以这样说：

> 我今天重点学习了三个经典 LNL 框架：Co-teaching、JoCoR 和 DivideMix。Co-teaching 的核心是利用 deep network 的 memorization effect，认为训练早期 small-loss 样本更可能是 clean sample，因此用两个网络互相选择 small-loss 样本训练。它本质是 sample selection，不修正标签。JoCoR 在此基础上加入 co-regularization，让两个网络不仅要和标签拟合，还要预测一致，因此用 joint loss 判断样本可靠性。DivideMix 则进一步把 LNL 转化为半监督学习：先通过 loss 分布和 GMM 把样本划分成 clean labeled 和 noisy unlabeled，再用 MixMatch/MixUp 继续利用可疑样本。  
>
> 对 HD-sEMG 来说，这些方法可以作为基础 baseline，但它们主要依赖 loss 或预测置信度，没有充分利用高密度肌电的电极空间拓扑、相邻时间窗口一致性和动作边界信息。下一步我认为可以考虑多视角网络委员会方法，让时间网络、空间网络、时频网络和时空网络共同判断样本可靠性，并将样本分为 reliable clean、likely mislabeled 和 uncertain 三类，分别采用原标签训练、软标签修正和无标签一致性训练。

---

## 11. 下一步学习路线

建议后续按下面顺序继续学。

### 11.1 第四个算法：DISC

原因：

- 它是 dynamic instance-specific selection and correction；
- 相比 Co-teaching / JoCoR，它不只是筛样本，还涉及标签修正；
- 和你后续“自修正”目标更贴近。

学习重点：

| 问题 | 需要理解 |
|---|---|
| 它怎么判断样本可靠？ | dynamic instance-specific selection |
| 它怎么修正标签？ | correction mechanism |
| 它和 DivideMix 有什么不同？ | 是否显式半监督、是否动态修正 |
| 能否改成 HD-sEMG？ | 加入时序一致性和空间拓扑 |

### 11.2 第五个算法：sEMG ALR-CNN

原因：

- 它是直接 sEMG noisy label 工作；
- 和你的领域最直接相关；
- 适合从“通用 LNL”过渡到“HD-sEMG-specific LNL”。

学习重点：

| 问题 | 需要理解 |
|---|---|
| sEMG-Map 是什么？ | 如何构造 sEMG 表示 |
| ALR 是什么？ | 如何自动修正标签 |
| 它修正的是哪类噪声？ | rest-movement-rest 边界标签 |
| HD-sEMG 如何改造？ | 用二维电极拓扑替代人工 sEMG-Map |

### 11.3 第六个算法：Scale-teaching

原因：

- 时序 LNL 顶会方法；
- 适合 HD-sEMG 多尺度窗口；
- 可以补上 Co-teaching / DivideMix 不考虑时序尺度的不足。

---

## 12. 今天可以保存下来的核心理解

最重要的三句话：

1. **Co-teaching：我不知道哪个标签该改成什么，所以先只学 small-loss 可能干净的样本。**
2. **JoCoR：我不仅看样本和标签是否匹配，还看两个网络对这个样本是否预测一致。**
3. **DivideMix：我不完全相信 high-loss 样本的人工标签，但我也不浪费它，而是把它当作无标签数据继续利用。**

对 HD-sEMG 的核心判断：

> 高密度肌电中的噪声标签不只是随机类别翻转，更多可能来自动作边界不准、过渡段、相似手势、低力度动作、电极接触差和被试差异。因此，未来方法不能只看 loss，还要结合时间一致性、空间电极拓扑和多视角模型一致性。

---

## 13. 参考论文

1. Han, B., Yao, Q., Yu, X., Niu, G., Xu, M., Hu, W., Tsang, I., & Sugiyama, M.  
   **Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels.**  
   NeurIPS 2018.  
   https://arxiv.org/abs/1804.06872

2. Wei, H., Feng, L., Chen, X., & An, B.  
   **Combating Noisy Labels by Agreement: A Joint Training Method with Co-Regularization.**  
   CVPR 2020.  
   https://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_Combating_Noisy_Labels_by_Agreement_A_Joint_Training_Method_with_CVPR_2020_paper.pdf

3. Li, J., Socher, R., & Hoi, S. C. H.  
   **DivideMix: Learning with Noisy Labels as Semi-supervised Learning.**  
   ICLR 2020.  
   https://arxiv.org/abs/2002.07394

4. Fatayer, T., et al.  
   **sEMG-Based Gesture Recognition Using Deep Learning From Noisy Labels.**  
   IEEE Journal of Biomedical and Health Informatics, 2022.  
   https://ieeexplore.ieee.org/document/9786842

5. Li, Y., et al.  
   **DISC: Learning from Noisy Labels via Dynamic Instance-Specific Selection and Correction.**  
   CVPR 2023.  
   https://openaccess.thecvf.com/content/CVPR2023/papers/Li_DISC_Learning_From_Noisy_Labels_via_Dynamic_Instance-Specific_Selection_and_CVPR_2023_paper.pdf

6. Scale-teaching:  
   **Robust Multi-scale Training for Time Series Classification with Noisy Labels.**  
   NeurIPS 2023.  
   https://proceedings.neurips.cc/paper_files/paper/2023/file/6a6ecedac816a24f92ad1f444b1edcb0-Paper-Conference.pdf

---

## 14. 后续记录建议

后续每学一个算法，都建议用同一套模板记录：

```text
1. 论文标题：
2. 方法名称：
3. 解决什么问题：
4. 输入是什么：
5. 网络结构是什么：
6. 如何判断样本可靠：
7. 如何处理疑似错标：
8. 是否修正标签：
9. 是否利用可疑样本：
10. 数学公式：
11. 和 Co-teaching / JoCoR / DivideMix 的区别：
12. 如何迁移到 HD-sEMG：
13. 一段给老师复述的话：
```

这样积累 8–10 个算法后，你就能形成完整的 LNL 算法谱系。
