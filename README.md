# GIRF-Core：Gauge-Invariant Relation Field

## 1. 核心思想

GIRF（Gauge-Invariant Relation Field，规范不变关系场）从一个基本问题出发：

> 文本序号与视觉时空坐标属于不同的坐标系，二者不应在跨模态 attention 中被直接相减。

原始多模态 RoPE 对 Query $i$ 和 Key $j$ 使用：

$$
A_{ij}=q_i^\top R(p_i-p_j)k_j.
$$

对于文本—视觉 token pair，这隐含使用：

$$
p_i^T-p_j^V.
$$

但 $p_i^T$ 是文本序列位置，$p_j^V$ 是视觉 $t/h/w$ 位置，它们没有天然共享的原点、尺度和单位。插入无关文本后，即使图像、问题和答案都没有变化，文本绝对位置仍会移动，从而产生没有语义依据的跨模态相位变化。

GIRF 不为两个模态强行构造统一绝对坐标，而是使用两个模态各自内部的关系量，直接定义跨模态关系。

## 2. Modality-pair positional attention

GIRF 先判断 Query 和 Key 的模态，再选择对应的位置规则：

$$
A_{ij}=
\begin{cases}
q_i^\top R_T(p_i^T-p_j^T)k_j, & TT,\\[2mm]
q_i^\top R_V(p_i^V-p_j^V)k_j, & VV,\\[2mm]
q_i^\top R_C(\theta_{ij}^{TV})k_j, & TV,\\[2mm]
q_i^\top R_C(\theta_{ij}^{VT})k_j, & VT.
\end{cases}
$$

其中：

- 文本—文本保留文本相对距离；
- 视觉—视觉保留视觉内部 $t/h/w$ 相对结构；
- 文本—视觉不再计算文本序号减视觉坐标；
- 视觉—文本使用相反方向的同一跨模态关系。

## 3. 模态内部规范坐标

### 3.1 文本内部角色

文本 token $i$ 在当前问题或答案段中的规范位置为：

$$
\tau_i=
2\frac{p_i^T-a_T}{\max(L_T-1,1)}-1,
$$

其中 $a_T$ 是当前文本段起点，$L_T$ 是该文本段长度。

因此，在当前文本段之前插入长度为 $\Delta_T$ 的 distractor 时：

$$
p_i^T\rightarrow p_i^T+\Delta_T,
\qquad
a_T\rightarrow a_T+\Delta_T,
$$

但：

$$
\tau_i\rightarrow\tau_i.
$$

### 3.2 视觉内部位置

视觉 token $j$ 在当前图像或视频中的规范坐标为：

$$
x_j=2\frac{w_j-w_{\min}}{\max(w_{\max}-w_{\min},1)}-1,
$$

$$
y_j=2\frac{h_j-h_{\min}}{\max(h_{\max}-h_{\min},1)}-1,
$$

$$
z_j=2\frac{t_j-t_{\min}}{\max(t_{\max}-t_{\min},1)}-1.
$$

这些坐标均位于 $[-1,1]$，只描述视觉 token 在当前视觉实体内部的位置。视觉坐标整体平移不会改变 $x_j,y_j,z_j$。

### 3.3 视觉实体关系

多图输入中，图像身份与图像内部坐标分开表示。定义：

$$
\epsilon_{ij}
=\eta(e_i^{\mathrm{ref}})-\eta(e_j),
\qquad
\eta(e)=\frac{2}{\pi}\arctan(e),
$$

其中 $e_j$ 是视觉 token 所属实体，$e_i^{\mathrm{ref}}$ 是当前文本段通过输入结构所引用的视觉实体，$\eta$ 是固定、有界且保持实体次序差异的势函数。

当：

$$
\epsilon_{ij}=0,
$$

视觉 token 属于当前文本段直接引用的图像；非零值表示它来自其他视觉实体。

该关系由图像占位符和输入结构确定，不需要模型预测。

## 4. Gauge-Invariant Relation Field

GIRF 的跨模态关系向量为：

$$
\boxed{
r_{ij}^{TV}
=
\begin{bmatrix}
\tau_i,&
\epsilon_{ij},&
z_j,&
y_j,&
x_j
\end{bmatrix}
}.
$$

它分别表示：

1. 文本 token 在当前文本段中的角色；
2. 文本所引用实体与视觉 token 所属实体的关系；
3. 视觉 token 的内部时间位置；
4. 视觉 token 的内部纵向位置；
5. 视觉 token 的内部横向位置。

该关系不包含：

- 全局文本 token index；
- 图像在统一序列中的绝对 offset；
- 未中心化的视觉坐标；
- 模型 hidden state；
- MLP、Adapter 或逐层逐头 gate。

## 5. Gauge invariance

如果文本和视觉坐标系分别进行独立原点平移：

$$
p^T\rightarrow p^T+\Delta_T,
\qquad
p^V\rightarrow p^V+\Delta_V,
$$

模态内部规范坐标保持不变，因此：

$$
\boxed{
r_{ij}\left(p^T+\Delta_T,p^V+\Delta_V\right)
=r_{ij}\left(p^T,p^V\right)
}.
$$

对于相同图像和问题，在二者之间插入不同长度的无关文本时：

$$
\boxed{
r_{ij}^{0K}=r_{ij}^{1K}=r_{ij}^{4K}=r_{ij}^{8K}
}.
$$

这保证跨模态位置关系不会因为上下文长度或任意坐标原点而漂移。

## 6. Structural discriminability

Gauge invariance 只要求关系不受任意坐标变换影响，但 GIRF 还要求真实的模态内部结构保持可区分。

对于同一文本 Query，如果两个视觉 token 位于不同的内部位置：

$$
(z_{j_1},y_{j_1},x_{j_1})
\neq
(z_{j_2},y_{j_2},x_{j_2}),
$$

则它们的跨模态关系也不同：

$$
\boxed{
r_{ij_1}^{TV}\neq r_{ij_2}^{TV}
}.
$$

对于不同视觉实体中的 token，只要其实体关系不同：

$$
\epsilon_{ij_1}\neq\epsilon_{ij_2},
$$

也有：

$$
r_{ij_1}^{TV}\neq r_{ij_2}^{TV}.
$$

因此 GIRF 的核心不是简单地消除跨模态位置，而是同时满足：

$$
\boxed{
\text{Gauge invariance}
+
\text{structural discriminability}
}.
$$

## 7. 确定性旋转通道分配

GIRF 不使用含义不明确的可学习投影矩阵。它将 rotary channel 确定性地划分为：

$$
\mathcal K_T,quad
\mathcal K_E,quad
\mathcal K_Z,quad
\mathcal K_Y,quad
\mathcal K_X,
$$

分别承载文本角色、视觉实体、视觉时间、视觉高度和视觉宽度。

定义轴分配函数：

$$
a(k)\in\{T,E,Z,Y,X\}.
$$

第 $k$ 个旋转通道的跨模态相位为：

$$
\boxed{
\theta_{ij,k}^{TV}
=\omega_k r_{ij,a(k)}^{TV}
},
$$

其中 $\omega_k$ 是目标模型原有的 RoPE 频率。通道按照固定比例分配，不依赖模型层数、attention head 数量或任务训练。

### 当前实现说明

当前 LLaVA 数据 batch 没有显式传递问题段和答案段边界。为避免从全局 token index 推导 $\tau_i$、进而破坏 distractor 平移不变性，首版代码将文本角色势安全地设为中性值：

$$
\tau_i=0.
$$

实体、时间、高度和宽度关系已经启用。只有当数据管线能够提供可靠、模型无关的语义段边界时，才会启用非零 $\tau_i$；这不会改变 GIRF attention 接口或通道分配。

反方向关系满足：

$$
\boxed{
r_{ji}^{VT}=-r_{ij}^{TV}
},
$$

因此：

$$
\theta_{ji,k}^{VT}=-\theta_{ij,k}^{TV}.
$$

## 8. 最终跨模态 attention

文本—视觉 attention 为：

$$
\boxed{
A_{ij}^{TV}
=q_i^\top R_C(\theta_{ij}^{TV})k_j
}.
$$

视觉—文本 attention 为：

$$
\boxed{
A_{ji}^{VT}
=q_j^\top R_C(-\theta_{ij}^{TV})k_i
}.
$$

GIRF 只改变跨模态位置关系。Q/K/V 内容、attention mask、Key 数量和 softmax 归一化均保持不变。

## 9. GIRF 与 DIPE 的本质差别

### 9.1 DIPE：稳定的跨模态感知邻近

DIPE 的主要目标是消除随着文本长度增长的跨模态距离惩罚。其跨模态关系可以抽象为一个锚定的稳定邻近关系：

$$
r_{ij}^{TV,\mathrm{DIPE}}\approx\rho_0.
$$

它主要回答：

> 如何避免图像因为中间文本变长而在位置上远离问题？

### 9.2 GIRF：类型化的跨模态关系几何

GIRF 不使用单一跨模态邻近关系，而是定义：

$$
r_{ij}^{TV,\mathrm{GIRF}}
=
[\tau_i,\epsilon_{ij},z_j,y_j,x_j].
$$

它回答：

> 当文本与视觉没有共享坐标系时，如何通过各自内部的不变量定义可区分的跨模态关系？

二者的主要区别为：

| 方面 | DIPE | GIRF |
|---|---|---|
| 主要问题 | 长文本导致跨模态距离惩罚 | 异构模态坐标不可直接比较 |
| 跨模态关系 | 锚定的稳定感知邻近 | 类型化的内部关系向量 |
| 文本长度稳定性 | 保持稳定 | 保持稳定 |
| 视觉内部位置区分 | 取决于固定位置设计 | 显式保留 $z/y/x$ |
| 多图实体关系 | 不是主要建模对象 | 显式保留 $\epsilon_{ij}$ |
| 坐标原点不变性 | 主要针对模态间距离 | 对文本和视觉独立平移显式成立 |
| 模型特定训练模块 | 不需要 | 不需要 |
| 方法目标 | 距离不变 | 不变性与结构可区分性同时成立 |

最核心的区别是：

$$
\boxed{
\mathrm{DIPE}:
\quad
\text{跨模态关系}=\text{稳定邻近}
}
$$

$$
\boxed{
\mathrm{GIRF}:
\quad
\text{跨模态关系}
=\text{文本内部角色}
+\text{视觉实体关系}
+\text{视觉内部结构}
}
$$

## 10. 潜在优势与边界

GIRF 在理论上可能相对于 DIPE 提供以下优势：

- 在保持长上下文稳定性的同时，保留视觉区域差异；
- 显式区分多张图像及其内部位置；
- 自然扩展到视频时间关系；
- 不依赖 layer/head 结构，便于跨模型迁移；
- 不需要额外训练模块。



## 11. 核心总结

GIRF-Core 只包含四个必要组成：

1. 不同 modality pair 使用不同位置规则；
2. 跨模态关系由模态内部规范坐标构造；
3. 关系对文本和视觉坐标的独立平移保持不变；
4. 在保持不变性的同时保留视觉位置和实体差异。

核心公式为：

$$
\boxed{
r_{ij}^{TV}
=
[\tau_i,\epsilon_{ij},z_j,y_j,x_j]
}
$$

$$
\boxed{
\theta_{ij,k}^{TV}
=\omega_k r_{ij,a(k)}^{TV}
}
$$

$$
\boxed{
A_{ij}^{TV}
=q_i^\top R_C(\theta_{ij}^{TV})k_j
}
$$

以及：

$$
\boxed{
r_{ij}(p^T+\Delta_T,p^V+\Delta_V)
=r_{ij}(p^T,p^V)
}.
$$

一句话概括：

> GIRF取消文本与视觉共享坐标的假设，用规范不变且结构可区分的模态内部关系重新定义跨模态位置。
