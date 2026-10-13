# GIRF 预实验设计：基于 Visual Haystacks 的受控跨模态实体定位

> **实验名称：** Controlled Cross-Modal Entity Grounding  
> **研究对象：** GIRF（Gauge-Invariant Relation Fields for Cross-Modal Position Encoding）  
> **数据来源：** Visual Haystacks，Single-Needle 任务  
> **状态：** 待实施的实验方案；下述结果均为研究假设，不是已获得的实验结果。

## 1. 实验目的与研究假设

### 1.1 GIRF 需要验证什么

GIRF 的核心问题是：文本顺序、视觉实体身份和视觉实体内部空间具有不同的位置语义。如果跨模态注意力直接混合这些异质坐标，可能产生不合理的跨模态对应关系。GIRF 以 typed relation field 显式分离：

- **文本局部进度（T）**：文本 token 相对于其连续文本段起点的位置；
- **视觉实体关系（E）**：文本查询相对于因果参考视觉实体的关系；
- **实体内部几何（Z, Y, X）**：视觉 token 在图片或视频实体内部的局部时空坐标。

本预实验主要考察 **实体身份（E）** 的作用，以及文本描述能否准确绑定到对应视觉实体；不把它称为 Gauge Invariance 的数学证明，也不将其等同于完整的图内空间（Y/X）验证。

### 1.2 核心假设

- **H1：实体定位。** 在相同图像内容及文本问题下，仅改变目标图片的序列位置，GIRF 能更准确地识别对应视觉实体。
- **H2：跨图属性绑定。** 当查询的目标属性出现在非相关图片中时，GIRF 能减少跨图属性错误关联。
- **H3：机制对应。** 若显式实体关系有助于上述能力，则完整 GIRF 应优于其 **w/o Entity Relation** 消融（以实际结果为准）。

**重要边界：** 改变图片排列本身会改变真实的视觉实体顺序；GIRF 不要求不同排列下所有位置相位完全一致。该实验测量的是实体检索和属性绑定的*行为稳定性及准确性*，不是 Proposition 1 的直接证明。

## 2. 数据集与准备

### 2.1 官方资源

- 数据集：[Visual Haystacks（Hugging Face）](https://huggingface.co/datasets/tsunghanwu/visual_haystacks)
- 评测仓库：[visual-haystacks/vhs_benchmark](https://github.com/visual-haystacks/vhs_benchmark)
- 原论文：[ICLR 2025 论文页面](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a264726ebd222124514a32bf0143b83d-Abstract-Conference.html)
- 图片来源：COCO 2017（按标注所需准备对应 `train2017`、`val2017` 图像）。

Visual Haystacks 的 **Single-Needle** 问题要求先定位具有 *anchor/needle object* 的相关图片，再判断该图片是否具有 *target object*，原任务答案为 Yes/No。

推荐使用：

| 数据划分 | 规模 | 作用 |
|---|---:|---|
| `single_needle/VHs_small` | 100 个基础问题 | 调试和了解位置偏差现象 |
| `single_needle/VHs_large` | 1,000 个基础问题 | 构建固定的正式诊断子集 |

**标注核对：** 下载后先核实当前仓库版本的文件结构和 JSON 字段，不能只依据示例假设所有文件字段完全一致。预计使用 `id`、`needle`、`target`、`pos_image`、`neg_image`、`conversations` 等字段。

### 2.2 获取数据（参考命令）

```bash
hf download tsunghanwu/visual_haystacks \
  single_needle/VHs_large/visual_haystack_5.json \
  --repo-type dataset \
  --local-dir data/VHs_qa
```

以上文件作为**候选入口**；实际操作前确认远端路径存在，并根据下载的 JSON 结构调整读取脚本。它来自官方五图设置，我们会进一步构造四图的派生诊断条件。单独下载 JSON 不包含完整 COCO 图片。

## 3. 构建固定子集

### 3.1 推荐规模

| 阶段 | 基础问题数 | 作用 |
|---|---:|---|
| Pilot | 40 | 检查输入、标注、图片、推理接口、输出解析及任务难度 |
| 正式实验 | 200 | 预先固定样本集合进行配对统计 |
| 可选扩展 | 200–300 | 若存在统计不确定性或需验证稳健性，再扩大规模 |

**不按方法预测结果选择样本。** 在模型测试前固定随机种子（例如 `seed=42`）、样本 ID 和剔除规则。

### 3.2 样本筛选规则

1. **Single-Needle：** 每题恰有 1 张标注相关图片（`pos_image`），避免多个相关实体。
2. **可构建四图：** 至少有 3 张有效、互不重复的干扰图片（`neg_image`）。
3. **答案平衡：** 正式集尽量包括 100 个 Yes 和 100 个 No。
4. **语义唯一性：** 确认干扰图中不存在另一个同样符合 anchor 描述的实体；对于对象标注遗漏或歧义样本，按预定标准排除。
5. **文件质量：** 所有图像文件存在、可读取，对象可辨认；记录原始 COCO 路径和尺寸。
6. **类别分布：** 尽量防止少数 anchor/target 对象类别占据大量样本；可按类别分层抽样。
7. **开发/正式集去重：** 对 `VHs_small` 和 `VHs_large` 的重叠情况进行 ID 和图像组合核查。

### 3.3 Hard Negative 分组（重点）

从答案为 **No** 的样本中标记：

- **Hard Negative：** 相关图片不含 target object，但至少一张干扰图片包含 target object。
- **Easy Negative：** 相关图片不含 target object，所选干扰图片也不包含该对象。

例如：相关图片有猫、没有笔记本电脑；某张干扰图片有笔记本电脑、没有猫；问题是“在有猫的图片里是否存在笔记本电脑？”正确答案为 No。

Hard Negative 是研究**跨图属性误关联**的关键诊断条件。可以用 COCO 类别标注辅助筛选，但要核实对象标注是否遗漏，必要时人工复核。若严格条件下样本不足，报告实际数量，不强行凑数或根据模型输出来选择。

### 3.4 保存固定清单

建议将每个基础问题写入 `subset_manifest.json`（以下为示意结构，字段名按实际数据调整）：

```json
{
  "base_id": "example_001",
  "source_id": "original_vhs_id",
  "anchor": "cat",
  "target": "laptop",
  "qa_answer": "no",
  "positive_image": "path/to/P.jpg",
  "distractor_images": ["path/to/D1.jpg", "path/to/D2.jpg", "path/to/D3.jpg"],
  "negative_type": "hard",
  "seed": 42
}
```

还需保存数据版本/commit、图像文件校验信息、筛选脚本与排除日志，以便重复构建相同实验。

## 4. 核心受控实验：目标图片位置重排

### 4.1 固定四张图片，仅改变目标位置

对每个基础问题固定相关图片 `P` 与三张干扰图片 `D1,D2,D3`。为同一问题生成以下四个排列：

| 目标位置 | 输入图片顺序 | Entity Identification 的正确编号 | 原始 QA 答案 |
|---|---|---:|---|
| Position 1 | P, D1, D2, D3 | 1 | 不变 |
| Position 2 | D1, P, D2, D3 | 2 | 不变 |
| Position 3 | D1, D2, P, D3 | 3 | 不变 |
| Position 4 | D1, D2, D3, P | 4 | 不变 |

**严格控制：** 同一问题的四个版本中，图片像素、干扰图片集合及彼此相对顺序、问题文本、Yes/No 标注均保持不变，仅目标图片在序列中的插入位置发生变化。

这些是**从官方标注构造的派生实验**，不是 Visual Haystacks 的标准四图测试集。

### 4.2 Task A：Entity Identification（新增定位任务）

**目的：** 测量文本查询能否定位相关视觉实体。

统一英文 Prompt 示例：

```text
You are given four images in order: Image 1, Image 2, Image 3, and Image 4.
Which image contains a cat?
Answer with the image number only (1-4).
```

从 `needle` 生成 anchor 描述，从输入排列得到正确图片编号。不允许给模型额外提供正例索引、图片 ID 或数据标注。

该任务不是官方 Visual Haystacks 原始 Yes/No 指标，论文中需标为 **derived entity-identification diagnostic**。

### 4.3 Task B：Cross-Image Attribute Binding（原始问题形式）

**目的：** 判断模型是否正确地将 target object 归属到包含 anchor object 的那张图片。

统一 Prompt 示例：

```text
You are given four images in order: Image 1, Image 2, Image 3, and Image 4.
For the image containing a cat, is there a laptop?
Answer yes or no only.
```

真实实验优先保留官方 `conversations` 中的实际问题与标签，只做必要的统一输出格式处理。Task A 与 Task B **独立推理**，避免先输出图片编号帮助后续 Yes/No 任务而改变原始设定。

## 5. Oracle 单图对照及可选扩展

### 5.1 单图 Oracle

每道基础问题只输入相关图片 `P`，使用其原始属性判断问题（必要时调整成明确的单图表述）。评估该图本身是否可被正确理解。

对 Task B 定义：

\[
\Delta_{\mathrm{multi}}=A_{\mathrm{oracle}}-A_{\mathrm{4img}}.
\]

如果某方法在单图上就答错，它的四图错误可能来自视觉理解、对象识别或对象幻觉，而不一定源自实体混淆。正式主结果仍需报告**全部预注册问题**，可以额外报告“所有方法单图均答对”的共同子集分析。

### 5.2 可选的图像数量扩展

在四图实验有效后，再预先固定 2/4/8 图协议：

\[
N_{\mathrm{images}}\in\{2,4,8\}.
\]

但随图片数量增加，干扰内容和视觉 token 总量也在增加。因此，多图数量曲线不能直接当作位置编码机制的单因素因果证据。应记录视觉 token 数、截断、分辨率，并设计可比的视觉预算对照。

**Resolution Gate 说明：** 四图结果主要验证实体关系，并不自动证明分辨率门控有效。需要检查具体阈值下的门控是否激活，再决定是否追加 `w/o Resolution Gate` 和更大实体数量实验。

## 6. 比较方法与统一推理设置

### 6.1 主要方法

| 方法 | 比较目的 |
|---|---|
| MRoPE | Backbone 基线 |
| DIPE | Anchor-based 跨模态位置解耦 |
| Circle-RoPE | 另一种跨模态几何表示 |
| GIRF (Full) | 完整 Typed Relation Field |
| GIRF w/o Entity Relation | 验证 E 分量的贡献（建议的关键消融） |

如已有 `GIRF w/o Resolution Gate` checkpoint，可用于后续实体数量实验，但不是第一轮必须展示的主方法。

### 6.2 公平性与输入控制

- 使用各自**已经完成对应位置编码训练**的 checkpoint，不在同一个 checkpoint 上临时替换 RoPE 来当作公平比较。
- 使用相同的图片、问题、处理流程、评价脚本及生成设置；建议 greedy decoding、temperature = 0。
- 使用相同的实体标签方式（Image 1/2/3/4）及答案解析规则。
- 统一图像处理与分辨率预算；记录 NaFlex 实际视觉 token 数。
- 优先确保样本实际 token 长度在目前训练协议的 **8,192 tokens** 范围内；不得悄然截断图像或问题。
- 输出格式错误、拒答、越界编号要记录并按照预定义规则计分，不得静默丢弃。
- 在独立开发集上检查提示词，不根据正式测试集成绩调整不同方法的提示词。

## 7. 指标设计

### 7.1 核心指标

**Entity Identification Accuracy（EIA）**：

\[
A_{\mathrm{entity}}(p)=\frac{1}{N}\sum_{i=1}^{N}\mathbf 1[\hat e_{i,p}=p],
\]

其中 \(p\in\{1,2,3,4\}\) 是目标图片位置。

**Attribute Binding Accuracy（ABA）**：对 Task B 按原始 Yes/No 标签计算准确率，可报告整体结果及各位置结果。

### 7.2 诊断指标

**Hard-negative False Positive Rate（FPR）**：

\[
\mathrm{FPR}_{\mathrm{hard}} =
\frac{\#\{\text{hard-negative samples predicted Yes}\}}
     {\#\{\text{hard-negative samples}\}}.
\]

此指标越低越好，但 **错误回答 Yes 不能自动等同于已证明跨模态相位错误**，也可能来自视觉幻觉、对象识别或语言偏差。

**Position Sensitivity**：按四个目标位置展示 EIA/ABA，附总体平均、最差位置成绩及不同位置之间的波动。不能只靠曲线更平坦就声称模型更好——曲线可能稳定地低准确率。

**Oracle-to-Multi Gap**：同一问题在单图与四图 Task B 的准确率差，帮助理解多图干扰代价。

**Confusion Matrix**：Task A 的真实目标实体位置 × 预测实体编号（1–4），展示是否倾向误报到某个固定序列位置。

### 7.3 统计方式

- 四种排列属于同一个基础问题的**配对观测**，不能视为四个独立样本。
- 建议以基础问题 ID 为 resampling 单位进行 paired / clustered bootstrap，报告 95% 置信区间。
- 对 Hard Negative 分组报告实际基础问题数，必要时分别给出 Yes/No 子集结果。
- 样本子集和评价指标应预先固定，并保存所有样本级预测供复核。

## 8. 建议的执行顺序与评测预算

### Phase 1：Pilot（先排查，不用于最终优势筛选）

1. 下载 JSON，解析标注并解析所需 COCO 图片路径。
2. 抽取 40 个基础问题，构造四种排列及两种任务。
3. 先测试 MRoPE 与 GIRF，检查输出能否解析、单图 Oracle 能力是否合理、题目是否存在歧义或严重地板/天花板效应。
4. 根据**通用数据和接口问题**修复方案，随后冻结正式协议；不能根据哪种方法胜出来筛选样本。

Pilot 的四图生成调用预计为：

\[
40\times4\times2\times2=640\text{ 次（不含 Oracle）}.
\]

### Phase 2：正式测试

1. 固定 200 个基础问题（按筛选规则）。
2. 每题四种位置，共 `200 × 4 = 800` 个四图输入/任务。
3. Task A 与 Task B 分别测试，每种方法 `200 × 4 × 2 = 1600` 次生成。
4. 若测试 5 个方法，总计约 **8,000 次四图生成**；单图 Oracle Task B 还需每种方法约 200 次额外生成，即全部约 **9,000 次**。
5. 统一汇总样本级预测、类别错误、位置曲线和配对区间。
6. 如结果足够明确，再追加不同图片数量、MMNeedle 2×2 子集或门控专项测试。

这些调用次数是实验预算估计，不代表已经运行。

### 输出文件建议

```text
experiments/visual_haystacks/
  README.md
  subset_manifest.json
  exclusion_log.json
  derived_4img_inputs.jsonl
  prompts_entity.txt
  prompts_binding.txt
  predictions/
    mrope_entity.jsonl
    mrope_binding.jsonl
    dipe_entity.jsonl
    dipe_binding.jsonl
    circle_rope_entity.jsonl
    circle_rope_binding.jsonl
    girf_entity.jsonl
    girf_binding.jsonl
    girf_wo_entity_entity.jsonl
    girf_wo_entity_binding.jsonl
  metrics_by_question.csv
  metrics_summary.csv
  figures/
    entity_accuracy_by_position.pdf
    hard_negative_fpr.pdf
    entity_confusion_matrix.pdf
```

## 9. 预期结果与判断规则（非实测）

预期支持 GIRF 的一组**可检验假设**是：

1. 相同图片集合经位置重排后，GIRF 具有更高的实体识别准确率，且不会只依赖某个有利的目标位置。
2. 对 Hard Negative，GIRF 的 False Positive Rate 更低，体现更少的跨图属性错误关联。
3. 完整 GIRF 相对 `w/o Entity Relation` 表现更好，说明 E 分量对实体定位具有实际帮助。

但需做好如下解释：

| 实际观测 | 合理结论 | 是否适合放入 Introduction |
|---|---|---|
| EIA、属性判断与 Hard Negative 均有一致改善，消融支持 | 有行为证据表明显式实体关系改善跨图绑定 | 适合 |
| 仅 Yes/No 总体准确率改善，EIA 无改善 | 尚不能把收益归因于更准确的实体选择 | 宜放在实验分析，避免强机制叙事 |
| 全部模型单图 Oracle 就很弱 | 视觉感知可能是主要瓶颈 | 应重新评估诊断有效性 |
| 全部方法接近满分 | 任务区分度不足 | 采用预定义的更难设置扩大验证 |

**不可提前承诺的结论：** 不应声称 VHs 直接证明了 Gauge Invariance、所有跨模态相位偏差被消除，或 Resolution Gate 的改善已经得到验证。

## 10. Introduction 放置位置、图表与文案

### 10.1 建议放置位置

在现有 Introduction 中，先指出 RoPE 跨模态位置语义不匹配与视觉实体关系模糊问题，再插入 **Motivating Diagnostic**，最后引出 GIRF 的 typed relation field。

叙事顺序：

1. 文本顺序、视觉实体身份及图内局部空间对应不同类型的位置语义。
2. 混合位置索引可能导致跨图实体错误关联。
3. 使用受控 VHs 实验说明在目标内容不变时，实体位置与干扰图片可影响文本—视觉绑定。
4. 引出 GIRF 的结构化跨模态关系，以及显式实体身份的作用。

这组实验应称为 **Controlled Cross-Modal Entity Grounding / Motivating Diagnostic**，不是 Gauge Invariance Evaluation。

### 10.2 Introduction 主图设计

**Panel (a): Entity Identification vs. Target Position**

- 横轴：目标图片位置 1、2、3、4。
- 纵轴：Entity Identification Accuracy (%)。
- 曲线：MRoPE、DIPE、Circle-RoPE、GIRF（可为关键结果增加 w/o Entity Relation）。
- 显示整体均值、置信区间，避免只展示最有利的样本。

**Panel (b): Cross-Image False Association**

- 横轴：方法，或 Easy vs. Hard Negative 条件。
- 纵轴：False Positive Rate (%)，越低越好。
- 重点展示 Hard Negative 中干扰图片含 target 时的错误情况。
- 使用与 Panel (a) 同一批基础问题来源及清楚的分组说明。

**与 DIPE 的区分：** 主图不以增加文本—图像间隔来展示 visual fading，而是展示在固定图像内容下的视觉实体识别和跨图属性误关联。

### 10.3 可先使用的英文 Introduction 段落（不提前宣称效果）

> To examine how cross-modal positional representations affect visual-entity grounding, we construct a controlled diagnostic using the single-needle setting of Visual Haystacks. For each question, we retain the same four images and textual query while systematically varying the position of the relevant visual entity. We evaluate both the identification of the target entity and the association of visual attributes with that entity, including challenging cases where the queried attribute appears in irrelevant images. By keeping the visual content and query fixed across permutations, this evaluation isolates the effect of visual-entity placement on cross-modal grounding. It enables us to investigate whether conventional positional representations lead to incorrect cross-image associations and whether explicitly modeling entity relations, as in GIRF, can mitigate such errors.

跑完实验后，应根据**实际结果与置信区间**增加一句具体观察；不要在此阶段填写虚构的增益或宣称已验证机制。

### 10.4 图注草案

> **Figure X: Controlled cross-modal entity grounding on Visual Haystacks.** We keep the visual content and question fixed while varying the position of the relevant image. (a) Entity identification accuracy across target positions. (b) False positive rate on hard negative examples, where the queried target object appears in distractor images but not in the relevant image. This diagnostic examines cross-image association errors under controlled visual-entity placement.

## 11. 实验的证据边界与注意事项

1. **输入重排不是坐标原点变换。** 它改变了视觉实体的真实顺序，因此不能直接验证 Gauge Invariance Proposition。
2. **回答错误不一定源自位置相位。** 视觉识别、对象幻觉、序列位置偏差和因果注意力等都可能起作用。Oracle、消融、统一输入能够减少混杂，但不能让行为实验变成严格的相位因果证明。
3. **Hard Negative 需要事先定义。** 不能根据模型输出或事后挑选失败案例来建立正式子集。
4. **VHs 衍生任务与官方评测要区分。** 图片编号识别以及四图重排是我们自行构建的诊断，不可宣称为官方原始分数。
5. **门控结论需独立检验。** 实体数量曲线并不能自动证明 Resolution Gate；要检查实际激活并与 w/o Resolution Gate 对比。
6. **训练差异与公平性。** 该实验沿用 GIRF 文章中各方法在统一训练协议下得到的 checkpoint，不拿不同论文原始绝对分数直接对比。

## 12. 最终建议

**第一轮先完成**：`VHs_large` 固定 200 个基础问题 → 每题四图、四种目标位置 → Entity Identification 与原始 Yes/No 两类任务 → 单图 Oracle → MRoPE / DIPE / Circle-RoPE / GIRF / w/o Entity Relation 比较 → 位置准确率曲线与 Hard Negative FPR。

若实验结果支持 H1–H3，再将双面板诊断图放入 Introduction。若没有形成清楚的“实体识别改善 + 属性误关联减少 + 消融对应”证据，则应如实将结果放入实验分析，而不是强行作为开篇动机。后续可用 MMNeedle 2×2 子集单独检验实体身份与局部位置的联合定位。
