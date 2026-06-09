# BIOME-Bench

**A Benchmark for Biomolecular Interaction Inference and Multi-Omics Pathway Mechanism Elucidation**

<div align="center">
  <img src="imgs/logo.png" width="48% biological-atom" /> <br/>
    <span>*Logo由Nano Banana生成</span>
</div>
<p align="center">
    <b>🌐 Language:</b> 中文 | <a href="README.md">English</a>
</p>



## 🌟 简介

**BIOME-Bench** 是一个基于科学文献的评估框架，旨在评估大语言模型在**生物分子互作推理**（Biomolecular Interaction Inference, BII）和**端到端多组学通路作用机制阐释**（Multi-Omics Pathway Mechanism Elucidation, MPME）方面的能力。

在多组学研究中，研究者通常采用通路富集分析（Pathway Enrichment Analysis）来解释复杂的分子变化。如下图所示：

<div align="center">
  <img src="imgs/pathway_analysis_pipeline.svg" width="48% biological-atom" />
</div>



然而，这种传统的富集分析法面临着几个关键的瓶颈：

1. **维护滞后性 (Curation Lag)**：通路知识库的更新往往滞后于最新的学术发现。
2. **功能冗余 (Functional Redundancy)**：富集结果往往包含大量重叠的基因集，产生冗余的通路列表，难以确定优先级。
3. **上下文不敏感 (Context-insensitivity)**：富集分值忽略了分子的具体状态（如磷酸化）、干预的方向性以及连接扰动实体与表型所需的逻辑因果结构。

当前研究方法尝试利用大语言模型解决上述问题，但目前**缺乏一个能够评估模型“端到端”机制阐释能力的标准化基准**——即模型能否直接从“扰动观测”推导出连贯的“因果机制链条”。

**BIOME-Bench** 填补了这一空白。它要求模型在给定**扰动实体**和**通路背景**的情况下，直接生成连贯的、状态感知的机制假说。

下图展示了我们从文献中构建这一基准测试的核心工作流：

<div align="center">
  <img src="imgs/workflow.svg" width="80% biological-atom" />
</div>



## 🏗️ 数据构建方法

BIOME-Bench 的数据构建工作流将通路信息与文献证据转化为结构化、经验证的知识表示，包含以下四个关键阶段：

### 阶段 I：文献检索与相关性过滤 (Literature Retrieval and Relevance Filtering)

为了确保生物学有效性，构建过程始于严谨的文献获取。令 $\mathcal{P} = \lbrace p_1, p_2, \dots, p_n\rbrace$ 表示预定义的 KEGG 通路集。每个通路 $p_i$ 由其名称 $N_{p_i}$ 和关联物种 $S_{p_i}$ 定义。

- **MeSH 引导的文献检索**： 为每个通路 $p_i$，我们在 PubMed 数据库上使用医学主题词（MeSH）进行结构化检索，以提高召回精度和语义一致性。最终的 PubMed 查询构造为通路相关 MeSH 术语与物种限制的交集：

$$
Q(p_i) = \mathrm{MeSH}(N_{p_i}) \wedge \mathrm{MeSH}(S_{p_i}).
$$

  执行 $Q(p_i)$ 得到初始候选文档集 $D_{\text{candidate}}(p_i) = \lbrace d_1, d_2, \dots, d_m \rbrace$。

- **基于 LLM 的语义与机制相关性评分**： MeSH 引导的检索虽然保证了高召回率，但 MeSH 注释本身不能确保文章包含通路特定的机制证据。我们使用参数为 $\theta$ 的 LLM 评估器为文档-通路对 $(d, p_i)$ 分配相关性分数 $s \in [0, 10]$：

$$
f_{\theta}(d, p_i)=g_{\theta}(\mathbf{S}),
\qquad
\mathbf{S}=
\begin{bmatrix}
S_{\text{subj}}\\
S_{\text{spec}}\\
S_{\text{mol}}\\
S_{\text{ctx}}
\end{bmatrix}.
$$

  其中 $\mathbf{S}$ 包含四个维度的评分：

  - **通路主体聚焦度 (**$S_{\text{subj}}$**)**：评估文章是否将通路的生物过程作为主要研究对象。
  - **物种一致性 (**$S_{\text{spec}}$**)**：评估研究物种是否匹配，并考虑模式生物的关联性。
  - **分子匹配度 (**$S_{\text{mol}}$**)**：文章是否提及通路定义的关键分子实体（基因、酶、代谢物等）。
  - **上下文调控描述 (**$S_{\text{ctx}}$**)**：文章是否描述了通路调控（如激活、抑制）而非仅提及通路存在。

  仅保留 $s \geq 8$ 的文档进入相关文档集 $D_{\text{relevant}}(p_i)$。

### 阶段 II：信息提取与实体标准化 (Information Extraction and Entity Standardization)

- **基于 LLM 的机制提取**： 对于 $D_{\text{relevant}}(p_i)$ 中的每个文档 $d$，利用 LLM 提取原始实体（包括化学品、基因/蛋白质和表型）集 $E_{\text{raw}}$ 以及连贯的自然语言机制描述 $M_{\text{text}}$。

- **实体归一化与本体映射**： 为了确保与外部资源的互操作性，利用解析函数 $\phi(e)$ 将 $E_{\text{raw}}$ 映射至规范标识符（化学品映射至 PubChem CID，基因/蛋白质映射至 NCBI Gene ID/UniProt ID）。仅保留所有实体均能成功标准化的文档：

$$
E_{\text{std}}=\left\lbrace \phi(e)\mid e\in E_{\text{raw}} \wedge \forall e'\in E_{\text{raw}},\ \phi(e')\neq\emptyset \right\rbrace
$$

### 阶段 III：知识结构化与校验 (Knowledge Structuring and Validation)

此阶段将提取的机制信息转换为细粒度的知识图谱表示。

- **核心四元组提取**：从 $M_{\text{text}}$ 提取核心结构 $T_{\text{core}} = (e_s, r, e_t, c)$，其中 $r$ 为受控词表中的关系类型， $c$ 为实验条件。

- **生物状态标注**：引入源/目标实体的生物状态 $\sigma_s, \sigma_t$（如突变、过表达、磷酸化），构建**状态感知的六元组 (State-aware Hexaplet)**：

$$
T_{\text{final}} = (e_s, \sigma_s, r, e_t, \sigma_t, c).
$$

这种表述使基准能够区分细微但关键的机制差异（如蛋白丰度变化与翻译后修饰的区别）。

- **专家验证**：由分子生物学专家对知识图谱条目进行抽样核验，确保其准确性与证据一致性。

### 阶段 IV：任务公式化 (Task Formulation)

基于上述知识表示，BIOME-Bench 定义了两个任务：

- **Task A (BII - 生物分子互作推理)**：预测给定通路背景 $p_i$、源实体及其状态、目标实体及其状态以及实验条件下的精准互作关系：

$$
\hat{r} = \arg\max_{r \in \mathcal{R}} P\bigl(r \mid p_i, e_s, \sigma_s, e_t, \sigma_t, c\bigr).
$$

- **Task B (MPME - 多组学通路作用机制阐释)**：模拟真实的组学分析场景，给定通路背景 $p_i$ 和差异观测集 $E_{\text{diff}} \subseteq E_{\text{std}}$，要求模型生成能够解释分子间如何互相作用并导致表型后果的机制描述 $\hat{Y}$。

## 📈 Benchmark统计

| **Species** | **Pathways** | **Entities** | **Processes & Phenotypes** | **Task A: Biomolecular Interaction Inference** | **Task B: Multi-Omics Pathway Mechanism Elucidation** |
| ----------- | ------------ | ------------ | -------------------------- | ---------------------------------------------- | ----------------------------------------------------- |
| `hsa`       | 80           | 1,349        | 1,781                      | 4,032                                          | 490                                                   |
| `mmu`       | 80           | 1,356        | 1,860                      | 4,162                                          | 496                                                   |
| `rno`       | 80           | 1,141        | 1,265                      | 3,384                                          | 361                                                   |
| **Total**   | **240**      | **3,846**    | **4,906**                  | **11,578**                                     | **1,347**                                             |

**BIOME-Bench** 是一个涵盖了三种常用生物——**人 (`hsa`)**、**小鼠 (`mmu`)** 和 **大鼠 (`rno`)** 的多物种评测基准。上表汇总了其核心统计数据，包括经过人工策展的通路数量、标准化实体、过程与表型术语、机制分析实例以及知识图谱关系。总体而言，该基准包含 **1,347** 个用于多组学通路作用机制阐释的实例，以及 **11,578** 个用于生物分子相互作用推断的实例，两者均在一致的通路背景下进行评估。

## 🧩 评估协议与指标

### Task A: 生物分子互作推理

采用关系标签的 **Accuracy** 和 **Macro-F1** 进行评估。

### Task B: 多组学通路机制阐释

针对生成的解释，我们采用多维度评估策略：

1. **LLM-as-a-Judge**: 使用 **Qwen3-32B** 作为裁判模型，根据 Ground Truth $M_{\text{text}}$ 对生成的解释 $\hat{Y}$ 在四个维度上进行评分（1-5 分）：**表型覆盖度 (Phenotype Coverage)**、**因果推理 (Causal Reasoning)**、**事实性 (Factuality)** 以及 **幻觉控制 (Hallucination Control)**。

2. **结构化知识评估 (Structured Knowledge Evaluation)**: 基于文献派生的知识图谱，采用闭集评估协议。使用 **Qwen3-32B** 作为提取模型，仅允许从标准化知识图中选择元组来支撑解释 $\hat{Y}$ 。事实完整性通过 **Coverage** 衡量：

$$
\text{Coverage} = \frac{|\mathcal{T}_{\text{pred}}|}{|\mathcal{T}_{\text{GT}}|}, \mathcal{T}_{\text{pred}}\subseteq\mathcal{T}_{\text{GT}}
$$

3. **语义嵌入相似度 (Semantic Embedding Similarity)**: 计算生成解释 $\hat{Y}$ 与标准机制文本 $M_{\text{text}}$ 向量表示之间的余弦相似度。

## 📊 实验结果

下表展示了不同模型在 BIOME-Bench 上的性能表现：

<div align="center">
  <table style="border-collapse: collapse; width: 100%; font-family: sans-serif; font-size: 13px; text-align: center;">
    <thead>
      <tr style="border-top: 2px solid black;">
        <th rowspan="4" style="border-bottom: 1px solid black; padding: 8px;"><strong>Model</strong></th>
        <th colspan="2" style="padding: 8px;"><strong>Biomolecular Interaction</strong></th>
        <th colspan="6" rowspan="2" style="padding: 8px;"><strong>Multi-Omics Pathway Mechanism Elucidation</strong></th>
        <th rowspan="4" style="border-bottom: 1px solid black; padding: 8px;"><strong>Avg.</strong></th>
      </tr>
      <tr>
        <th colspan="2" style="border-bottom: 1px solid black; padding: 8px;"><strong>Inference</strong></th>
      </tr>
      <tr>
        <th rowspan="2" style="border-bottom: 1px solid black; padding: 8px;">Acc</th>
        <th rowspan="2" style="border-bottom: 1px solid black; padding: 8px;">Macro-F1</th>
        <th colspan="4" style="border-bottom: 1px solid black; padding: 8px;">LLM-as-a-Judge</th>
        <th rowspan="2" style="border-bottom: 1px solid black; padding: 8px;">Similarity</th>
        <th rowspan="2" style="border-bottom: 1px solid black; padding: 8px;">Coverage</th>
      </tr>
      <tr>
        <th style="border-bottom: 1px solid black; padding: 8px;">Phenotype Coverage</th>
        <th style="border-bottom: 1px solid black; padding: 8px;">Causal Reasoning</th>
        <th style="border-bottom: 1px solid black; padding: 8px;">Factuality</th>
        <th style="border-bottom: 1px solid black; padding: 8px;">Hallucination</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="text-align: left; padding: 6px;">Qwen3-14B</td>
        <td>47.43</td><td>43.72</td><td>3.12</td><td>3.31</td><td>3.97</td><td>4.64</td><td>78.73</td><td>42.38</td><td>64.13</td>
      </tr>
      <tr>
        <td style="text-align: left; padding: 6px;">Qwen3-32B</td>
        <td>41.84</td><td>40.51</td><td>3.00</td><td>3.26</td><td>3.89</td><td>4.79</td><td>78.98</td><td>45.43</td><td>63.20</td>
      </tr>
      <tr>
        <td style="text-align: left; padding: 6px;">Qwen3-235B</td>
        <td>51.41</td><td>46.21</td><td>3.66</td><td>4.32</td><td>4.54</td><td>4.40</td><td>77.34</td><td>42.22</td><td>69.45</td>
      </tr>
      <tr>
        <td style="text-align: left; padding: 6px;">GLM-4.6</td>
        <td>53.60</td><td>50.08</td><td>3.50</td><td>4.14</td><td>4.32</td><td>4.18</td><td>76.89</td><td>39.95</td><td>67.92</td>
      </tr>
      <tr>
        <td style="text-align: left; padding: 6px;">DeepSeek-V3.2-R1</td>
        <td>53.10</td><td>47.52</td><td>3.28</td><td>4.31</td><td>4.20</td><td>4.10</td><td>75.12</td><td>40.76</td><td>66.79</td>
      </tr>
      <tr>
        <td style="text-align: left; padding: 6px;">Gemini3-Pro</td>
        <td>52.34</td><td>46.54</td><td>3.60</td><td>4.57</td><td>4.59</td><td>4.54</td><td>77.21</td><td>41.13</td><td>69.74</td>
      </tr>
      <tr>
        <td style="text-align: left; padding: 6px;">GPT-5.2</td>
        <td>54.66</td><td>50.70</td><td>3.68</td><td>4.58</td><td>4.69</td><td>4.62</td><td>71.38</td><td>37.49</td><td>70.70</td>
      </tr>
      <tr>
        <td style="text-align: left; padding: 6px;">Doubao-Seed-1.8</td>
        <td>55.42</td><td>50.40</td><td>3.81</td><td>4.69</td><td>4.69</td><td>4.57</td><td>74.92</td><td>39.72</td><td>71.96</td>
      </tr>
      <tr style="border-bottom: 2px solid black;">
        <td style="text-align: left; padding: 6px;">S1-Base-671B</td>
        <td>54.68</td><td>50.41</td><td><strong>4.02</strong></td><td>4.48</td><td><strong>4.76</strong></td><td><strong>4.83</strong></td><td>77.36</td><td>44.45</td><td><strong>73.59</strong></td>
      </tr>
    </tbody>
  </table>
</div>

Qwen3-32B 评测模型（judge）对语义扰动的敏感性如下图所示。图中记录了改写（rewrite）与扰动（perturb）后的得分。Drop% 表示从改写到扰动后的得分相对下降百分比：

<div align="center">
  <img src="imgs/llm_as_a_judge.svg" width="48% biological-atom" />
</div>


图中结果验证了LLM-as-a-Judge的有效性。

下图展示了生物分子互作推断的错误混淆矩阵。行代表标注真值（gold）关系类型，列代表模型预测类型。颜色深浅表示真值被误分类为预测类型的具体数量。



<div align="center">
  <img src="imgs/kg_relation_confusion_matrix.svg" width="45% biological-atom" />
</div>


图中结果表明：模型倾向于将精细的生物机制误判为粗粒度的因果或调节关系（如 `leads_to`），且难以准确区分直接调节与通路层面的因果联系。这种对模糊关系的过度解读（如将 `regulates` 极化）以及细粒度辨析能力的缺失，反映了当前模型在处理复杂生物逻辑时的局限性。

## 🚀 快速上手

### 1. 环境准备

确保已安装 Python 3.10+。

```bash
# 克隆仓库
git clone url
cd BIOME-Bench

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 (`config/config.json`)

在运行之前，您**必须**配置模型终端。以下是完整的配置样例，包含待评估模型、裁判模型和 Embedding 模型的设置：

```json
{
  "EvalModel": {
    "api_config": {
      "model": "EvalModel",
      "base_url": "http://localhost:8000/v1",
      "api_key": "EMPTY",
      "timeout": 60,
      "max_retries": 10
    },
    "generation_config": {
      "temperature": 0.0,
      "max_tokens": 10240,
      "no_think": false,
      "thinking_rules_file": "config/thinking_rules.json"
    },
    "save_every": 10
  },
  "JudgeModel": {
    "api_config": {
      "model": "Qwen3-32B",
      "base_url": "http://localhost:8001/v1",
      "api_key": "sk-..."
    },
    "generation_config": {
      "max_tokens": 10240,
      "temperature": 0.0
    }
  },
  "EmbedModel": {
    "api_config": {
      "model": "Qwen3-8B-Embedding",
      "base_url": "http://localhost:8002/v1",
      "api_key": "sk-..."
    }
  }
}
```

### 3. 一键运行 Demo

使用 `data/` 中提供的示例数据快速验证流程：

```bash
# 运行 Demo（使用只有一条数据的测试集进行快速验证）
python run_demo.py --threads 1
```

### 4. 一键运行完整测评

```bash
# 运行完整测评（使用完整数据集和指定配置）
python run_pipeline.py --config config/my_config.json --threads 1
```

## 🛠️ 手动使用流程

### 阶段 1: 模型推理

使用 `evaluation` 模块对数据集进行批处理推理。

```bash
# Task-A
python -m evaluation run \
  --data data/TASK-A.jsonl \
  --task-type relation_prediction \
  --config config/config.json \
  --threads 1

# Task-B
python -m evaluation run \
  --data data/TASK-B.jsonl \
  --task-type mechanism_analysis \
  --config config/config.json \
  --threads 1
```

如果你的接口支持高并发推理，可以将`threads`调高以加快评测。

### 阶段 2: 评分与指标计算

#### 1. Task A: 关系预测准确率

```bash
python metrics/biomolecular_interaction_inference_acc.py \
  --input outputs/your_run/results.jsonl
```

#### 2. Task B: 机制分析 (LLM-as-a-Judge)

**注意**：必须提供与物种对应的知识库文件 (`--db`)。

```bash
python metrics/LLM-as-a-Judge.py \
  --results outputs/your_run/results.jsonl \
  --db data/hsa.jsonl data/mmu.jsonl data/rno.jsonl \
  --output outputs/your_run/judge_results.jsonl \
  --threads 1
```

## 📂 项目结构

```
BIOME-Bench/
├── config/              # 配置文件与推理规则
├── data/                # 经过清洗的知识库与测试集
├── evaluation/          # 核心推理引擎
├── metrics/             # 评分模块与裁判 Prompt
└── outputs/             # 实验结果与评估报告
```

## 📄 数据说明

以下是`data\TASK-B.jsonl`中的一条多组学通路作用机制阐释数据：

```json
{
    "messages": [
        {
            "role": "user",
            "content": "**Pathway Context**: Bladder cancer - Homo sapiens (human)\n\nDetermine the specific biological relationship between the Source and Target entities under the given Condition.\nYou must select the relationship strictly from the following vocabulary:\n['activates', 'inhibits', 'upregulates_expression', 'downregulates_expression', 'regulates', 'binds', 'dissociates_from', 'phosphorylates', 'dephosphorylates', 'ubiquitinates', 'glycosylates', 'methylates', 'produces', 'consumes', 'converts_to', 'leads_to', 'increases_level', 'decreases_level']\n\n**Source**: SIGLEC12 (elevated expression)\n**Target**: oncogenic signaling (upregulation)\n**Condition**: in bladder cancer\n\nRespond with the exact relation name from the list above."
        },
        {
            "role": "assistant",
            "content": "upregulates_expression"
        }
    ],
    "pathway_id": "hsa05219",
    "pubmed_id": "41303731",
    "species": "hsa"
}
```

其对应的详细通路信息、文献信息、文献相关性评估结果、标准化的实体和知识图谱可以在对应物种的`data\hsa.jsonl`中根据`pathway_id`和`pubmed_id`找到：

```json
{
    "id": "hsa05219",
    "name": "Bladder cancer - Homo sapiens (human)",
    "description": "The urothelium covers the luminal surface of almost the entire urinary tract, extending from the renal pelvis, through the ureter and bladder, to the proximal urethra. The majority of urothelial carcinoma are bladder carcinomas, and urothelial carcinomas of the renal pelvis and ureter account for only approximately 7% of the total. Urothelial tumours arise and evolve through divergent phenotypic pathways. Some tumours progress from urothelial hyperplasia to low-grade non-invasive superficial papillary tumours. More aggressive variants arise either from flat, high-grade carcinoma in situ (CIS) and progress to invasive tumours, or they arise de novo as invasive tumours. Low-grade papillary tumors frequently show a constitutive activation of the receptor tyrosine kinase-Ras pathway, exhibiting activating mutations in the HRAS and fibroblast growth factor receptor 3 (FGFR3) genes. In contrast, CIS and invasive tumors frequently show alterations in the TP53 and RB genes and pathways. Invasion and metastases are promoted by several factors that alter the tumour microenvironment, including the aberrant expression of E-cadherins (E-cad), matrix metalloproteinases (MMPs), angiogenic factors such as vascular endothelial growth factor (VEGF).",
    "genes": [
        "1019"
    ],
    "pubmed": [
        {
            "pmid": "41303731",
            "title": "Decoding SIGLEC12 in Bladder Cancer: In Silico Profiling of Expression, Tumor-Immune Interactions, and Prognostic Impact.",
            "abstract": "Background and Objectives: Siglec-XII, encoded by SIGLEC12, is a unique sialic acid-binding immunoglobulin-like lectin. It lacks a highly conserved R122 residue for sialic acid recognition in humans. Although it is upregulated in bladder cancer (BCa), its role in tumorigenesis remains largely unexplored. This study aims to investigate the expression patterns of SIGLEC12 in BCa and its correlation with disease features. Materials and Methods: An integrated analysis of transcriptomic data and clinical profiles was conducted using various databases and tools, including UALCAN, GEPIA, TIMER, CAMOIP, and CPADs. The analyses encompassed SIGLEC12 expression, survival rates, immune infiltration levels, promoter methylation, and correlation with drug response. Results: SIGLEC12 expression was higher in both low-grade papillary and high-grade invasive non-papillary BCa. Higher SIGLEC12 expression resulting from low promoter hypomethylation was detected at the stage II-IV of BCa, and was unrelated to disease stages and metastatic stages. Elevated SIGLEC12 expression correlated with increased immune cell infiltration, higher expression of oncogenic and immune checkpoint blockade-related genes, and drug resistance signatures. Mutation analysis confirmed the absence of the canonical R122 missense mutation, indicating that the structural integrity and potential functionality of Siglec-XII are preserved in BCa. Conclusions: SIGLEC12 may have sialic acid recognition functions and serve as a potential early biomarker of BCa.",
            "authors": "Rathore V; Lin WW",
            "fulltext_url": null,
            "keywords": "Bladder cancer[MeSH Terms] AND human[MeSH Terms]",
            "llm_relevance_assessment": {
                "relevance_score": 8,
                "relevance_level": "High",
                "species_check": "The pathway is human (Homo sapiens), and the article uses human-derived bladder cancer transcriptomic data and clinical profiles. Species match is valid.",    
                "evidence_summary": [
                    "SIGLEC12 expression was higher in both low-grade papillary and high-grade invasive non-papillary BCa.",
                    "Elevated SIGLEC12 expression correlated with increased immune cell infiltration, higher expression of oncogenic and immune checkpoint blockade-related genes, and drug resistance signatures."
                ],
                "reasoning": "The article directly investigates SIGLEC12 in the context of human bladder cancer, aligning with the KEGG pathway 'Bladder cancer - Homo sapiens'. It provides evidence of its expression patterns, association with immune infiltration, and potential role in oncogenic and immune checkpoint-related pathways. While it does not explicitly discuss all the genetic alterations (e.g., FGFR3, TP53, RB) mentioned in the pathway, it offers meaningful biological context and pathway-level insights into tumor-immune interactions and disease progression, placing it in the high relevance category."
            },
            "standardized_entities": {
                "chemicals": [],
                "genes_proteins": [
                    {
                        "original": "SIGLEC12",
                        "standard_name": "SIGLEC12",
                        "status": "success",
                        "source_db": "NCBI_Gene",
                        "entrez_id": "89858",
                        "official_symbol": "SIGLEC12",
                        "full_name": "sialic acid binding Ig like lectin 12",
                        "summary": "Sialic acid-binding immunoglobulin-like lectins (SIGLECs) are a family of cell surface proteins belonging to the immunoglobulin superfamily. They mediate protein-carbohydrate interactions by selectively binding to different sialic acid moieties present on glycolipids and glycoproteins. This gene encodes a member of the SIGLEC3-like subfamily of SIGLECs. Members of this subfamily are characterized by an extracellular V-set immunoglobulin-like domain followed by two C2-set immunoglobulin-like domains, and the cytoplasmic tyrosine-based motifs ITIM and SLAM-like. The encoded protein, upon tyrosine phosphorylation, has been shown to recruit the Src homology 2 domain-containing protein-tyrosine phosphatases SHP1 and SHP2. It has been suggested that the protein is involved in the negative regulation of macrophage signaling by functioning as an inhibitory receptor. This gene is located in a cluster with other SIGLEC3-like genes on 19q13.4. Alternative splicing results in multiple transcript variants. [provided by RefSeq, Aug 2013].",
                        "go_process": [
                            "cell adhesion",
                            "cell adhesion"
                        ],
                        "uniprot_id": "Q96PQ1"
                    },
                    {
                        "original": "Siglec-XII",
                        "standard_name": "SIGLEC12",
                        "status": "success",
                        "source_db": "NCBI_Gene",
                        "entrez_id": "89858",
                        "official_symbol": "SIGLEC12",
                        "full_name": "sialic acid binding Ig like lectin 12",
                        "summary": "Sialic acid-binding immunoglobulin-like lectins (SIGLECs) are a family of cell surface proteins belonging to the immunoglobulin superfamily. They mediate protein-carbohydrate interactions by selectively binding to different sialic acid moieties present on glycolipids and glycoproteins. This gene encodes a member of the SIGLEC3-like subfamily of SIGLECs. Members of this subfamily are characterized by an extracellular V-set immunoglobulin-like domain followed by two C2-set immunoglobulin-like domains, and the cytoplasmic tyrosine-based motifs ITIM and SLAM-like. The encoded protein, upon tyrosine phosphorylation, has been shown to recruit the Src homology 2 domain-containing protein-tyrosine phosphatases SHP1 and SHP2. It has been suggested that the protein is involved in the negative regulation of macrophage signaling by functioning as an inhibitory receptor. This gene is located in a cluster with other SIGLEC3-like genes on 19q13.4. Alternative splicing results in multiple transcript variants. [provided by RefSeq, Aug 2013].",
                        "go_process": [
                            "cell adhesion",
                            "cell adhesion"
                        ],
                        "uniprot_id": "Q96PQ1"
                    },
                    {
                        "original": "R122",
                        "standard_name": "arginine at position 122 of SIGLEC12",
                        "status": "success",
                        "source_db": "NCBI_Gene",
                        "entrez_id": "89858",
                        "official_symbol": "SIGLEC12",
                        "full_name": "sialic acid binding Ig like lectin 12",
                        "summary": "Sialic acid-binding immunoglobulin-like lectins (SIGLECs) are a family of cell surface proteins belonging to the immunoglobulin superfamily. They mediate protein-carbohydrate interactions by selectively binding to different sialic acid moieties present on glycolipids and glycoproteins. This gene encodes a member of the SIGLEC3-like subfamily of SIGLECs. Members of this subfamily are characterized by an extracellular V-set immunoglobulin-like domain followed by two C2-set immunoglobulin-like domains, and the cytoplasmic tyrosine-based motifs ITIM and SLAM-like. The encoded protein, upon tyrosine phosphorylation, has been shown to recruit the Src homology 2 domain-containing protein-tyrosine phosphatases SHP1 and SHP2. It has been suggested that the protein is involved in the negative regulation of macrophage signaling by functioning as an inhibitory receptor. This gene is located in a cluster with other SIGLEC3-like genes on 19q13.4. Alternative splicing results in multiple transcript variants. [provided by RefSeq, Aug 2013].",
                        "go_process": [
                            "cell adhesion",
                            "cell adhesion"
                        ],
                        "uniprot_id": "Q96PQ1"
                    },
                    {
                        "original": "E-cad",
                        "standard_name": "E-cadherin",
                        "status": "success",
                        "source_db": "NCBI_Gene",
                        "entrez_id": "999",
                        "official_symbol": "CDH1",
                        "full_name": "cadherin 1",
                        "summary": "This gene encodes a classical cadherin of the cadherin superfamily. Alternative splicing results in multiple transcript variants, at least one of which encodes a preproprotein that is proteolytically processed to generate the mature glycoprotein. This calcium-dependent cell-cell adhesion protein is comprised of five extracellular cadherin repeats, a transmembrane region and a highly conserved cytoplasmic tail. Mutations in this gene are correlated with gastric, breast, colorectal, thyroid and ovarian cancer. Loss of function of this gene is thought to contribute to cancer progression by increasing proliferation, invasion, and/or metastasis. The ectodomain of this protein mediates bacterial adhesion to mammalian cells and the cytoplasmic domain is required for internalization. This gene is present in a gene cluster with other members of the cadherin family on chromosome 16. [provided by RefSeq, Nov 2015].",
                        "go_process": [
                            "cell morphogenesis",
                            "desmosome assembly",
                            "cell-cell junction assembly"
                        ],
                        "uniprot_id": "P12830"
                    },
                    {
                        "original": "MMPs",
                        "standard_name": "matrix metalloproteinases",
                        "status": "success",
                        "source_db": "NCBI_Gene",
                        "entrez_id": "4319",
                        "official_symbol": "MMP10",
                        "full_name": "matrix metallopeptidase 10",
                        "summary": "This gene encodes a member of the peptidase M10 family of matrix metalloproteinases (MMPs). Proteins in this family are involved in the breakdown of extracellular matrix in normal physiological processes, such as embryonic development, reproduction, and tissue remodeling, as well as in disease processes, such as arthritis and metastasis. The encoded preproprotein is proteolytically processed to generate the mature protease. This secreted protease breaks down fibronectin, laminin, elastin, proteoglycan core protein, gelatins, and several types of collagen. The gene is part of a cluster of MMP genes on chromosome 11. [provided by RefSeq, Jan 2016].",
                        "go_process": [
                            "proteolysis",
                            "proteolysis",
                            "extracellular matrix disassembly"
                        ],
                        "uniprot_id": "P09238"
                    },
                    {
                        "original": "VEGF",
                        "standard_name": "vascular endothelial growth factor",
                        "status": "success",
                        "source_db": "NCBI_Gene",
                        "entrez_id": "2277",
                        "official_symbol": "VEGFD",
                        "full_name": "vascular endothelial growth factor D",
                        "summary": "The protein encoded by this gene is a member of the platelet-derived growth factor/vascular endothelial growth factor (PDGF/VEGF) family and is active in angiogenesis, lymphangiogenesis, and endothelial cell growth. This secreted protein undergoes a complex proteolytic maturation, generating multiple processed forms which bind and activate VEGFR-2 and VEGFR-3 receptors. This protein is structurally and functionally similar to vascular endothelial growth factor C. Read-through transcription has been observed between this locus and the upstream PIR (GeneID 8544) locus. [provided by RefSeq, Feb 2011].",
                        "go_process": [
                            "angiogenesis",
                            "response to hypoxia",
                            "sprouting angiogenesis"
                        ],
                        "uniprot_id": "O43915"
                    },
                    {
                        "original": "HRAS",
                        "standard_name": "HRAS",
                        "status": "success",
                        "source_db": "NCBI_Gene",
                        "entrez_id": "3265",
                        "official_symbol": "HRAS",
                        "full_name": "HRas proto-oncogene, GTPase",
                        "summary": "This gene belongs to the Ras oncogene family, whose members are related to the transforming genes of mammalian sarcoma retroviruses. The products encoded by these genes function in signal transduction pathways. These proteins can bind GTP and GDP, and they have intrinsic GTPase activity. This protein undergoes a continuous cycle of de- and re-palmitoylation, which regulates its rapid exchange between the plasma membrane and the Golgi apparatus. Mutations in this gene cause Costello syndrome, a disease characterized by increased growth at the prenatal stage, growth deficiency at the postnatal stage, predisposition to tumor formation, cognitive disability, skin and musculoskeletal abnormalities, distinctive facial appearance and cardiovascular abnormalities. Defects in this gene are implicated in a variety of cancers, including bladder cancer, follicular thyroid cancer, and oral squamous cell carcinoma. Multiple transcript variants, which encode different isoforms, have been identified for this gene. [provided by RefSeq, Jul 2008].",
                        "go_process": [
                            "MAPK cascade",
                            "MAPK cascade",
                            "regulation of transcription by RNA polymerase II"
                        ],
                        "uniprot_id": "P01112"
                    },
                    {
                        "original": "FGFR3",
                        "standard_name": "fibroblast growth factor receptor 3",
                        "status": "success",
                        "source_db": "NCBI_Gene",
                        "entrez_id": "2261",
                        "official_symbol": "FGFR3",
                        "full_name": "fibroblast growth factor receptor 3",
                        "summary": "This gene encodes a member of the fibroblast growth factor receptor (FGFR) family, with its amino acid sequence being highly conserved between members and among divergent species. FGFR family members differ from one another in their ligand affinities and tissue distribution. A full-length representative protein would consist of an extracellular region, composed of three immunoglobulin-like domains, a single hydrophobic membrane-spanning segment and a cytoplasmic tyrosine kinase domain. The extracellular portion of the protein interacts with fibroblast growth factors, setting in motion a cascade of downstream signals, ultimately influencing mitogenesis and differentiation. This particular family member binds acidic and basic fibroblast growth hormone and plays a role in bone development and maintenance. Mutations in this gene lead to craniosynostosis and multiple types of skeletal dysplasia. [provided by RefSeq, Aug 2017].",
                        "go_process": [
                            "MAPK cascade",
                            "skeletal system development",
                            "ossification"
                        ],
                        "uniprot_id": "P22607"
                    },
                    {
                        "original": "TP53",
                        "standard_name": "TP53",
                        "status": "success",
                        "source_db": "NCBI_Gene",
                        "entrez_id": "7157",
                        "official_symbol": "TP53",
                        "full_name": "tumor protein p53",
                        "summary": "This gene encodes a tumor suppressor protein containing transcriptional activation, DNA binding, and oligomerization domains. The encoded protein responds to diverse cellular stresses to regulate expression of target genes, thereby inducing cell cycle arrest, apoptosis, senescence, DNA repair, or changes in metabolism. Mutations in this gene are associated with a variety of human cancers, including hereditary cancers such as Li-Fraumeni syndrome. Alternative splicing of this gene and the use of alternate promoters result in multiple transcript variants and isoforms. Additional isoforms have also been shown to result from the use of alternate translation initiation codons from identical transcript variants (PMIDs: 12032546, 20937277). [provided by RefSeq, Dec 2016].",
                        "go_process": [
                            "negative regulation of transcription by RNA polymerase II",
                            "negative regulation of transcription by RNA polymerase II",
                            "negative regulation of transcription by RNA polymerase II"
                        ],
                        "uniprot_id": "P04637"
                    },
                    {
                        "original": "RB",
                        "standard_name": "RB1",
                        "status": "success",
                        "source_db": "NCBI_Gene",
                        "entrez_id": "5925",
                        "official_symbol": "RB1",
                        "full_name": "RB transcriptional corepressor 1",
                        "summary": "The protein encoded by this gene is a negative regulator of the cell cycle and was the first tumor suppressor gene found. The encoded protein also stabilizes constitutive heterochromatin to maintain the overall chromatin structure. The active, hypophosphorylated form of the protein binds transcription factor E2F1. Defects in this gene are a cause of childhood cancer retinoblastoma (RB), bladder cancer, and osteogenic sarcoma. [provided by RefSeq, Jul 2008].",
                        "go_process": [
                            "G1/S transition of mitotic cell cycle",
                            "negative regulation of transcription by RNA polymerase II",
                            "negative regulation of transcription by RNA polymerase II"
                        ],
                        "uniprot_id": "P06400"
                    },
                    {
                        "original": "E-cadherins",
                        "standard_name": "E-cadherin",
                        "status": "success",
                        "source_db": "NCBI_Gene",
                        "entrez_id": "999",
                        "official_symbol": "CDH1",
                        "full_name": "cadherin 1",
                        "summary": "This gene encodes a classical cadherin of the cadherin superfamily. Alternative splicing results in multiple transcript variants, at least one of which encodes a preproprotein that is proteolytically processed to generate the mature glycoprotein. This calcium-dependent cell-cell adhesion protein is comprised of five extracellular cadherin repeats, a transmembrane region and a highly conserved cytoplasmic tail. Mutations in this gene are correlated with gastric, breast, colorectal, thyroid and ovarian cancer. Loss of function of this gene is thought to contribute to cancer progression by increasing proliferation, invasion, and/or metastasis. The ectodomain of this protein mediates bacterial adhesion to mammalian cells and the cytoplasmic domain is required for internalization. This gene is present in a gene cluster with other members of the cadherin family on chromosome 16. [provided by RefSeq, Nov 2015].",
                        "go_process": [
                            "cell morphogenesis",
                            "desmosome assembly",
                            "cell-cell junction assembly"
                        ],
                        "uniprot_id": "P12830"
                    }
                ],
                "processes_phenotypes": [
                    {
                        "original": "Bladder cancer",
                        "type": "phenotype"
                    },
                    {
                        "original": "Immune cell infiltration",
                        "type": "phenotype"
                    },
                    {
                        "original": "Oncogenic signaling",
                        "type": "phenotype"
                    },
                    {
                        "original": "Immune checkpoint blockade",
                        "type": "phenotype"
                    },
                    {
                        "original": "Drug resistance",
                        "type": "phenotype"
                    },
                    {
                        "original": "Tumor progression",
                        "type": "phenotype"
                    },
                    {
                        "original": "Metastasis",
                        "type": "phenotype"
                    }
                ]
            },
            "knowledge_graph": [
                {
                    "source": "SIGLEC12",
                    "source_state": "elevated expression",
                    "relation": "upregulates_expression",
                    "target": "oncogenic signaling",
                    "target_state": "upregulation",
                    "condition": "in bladder cancer"
                },
                {
                    "source": "SIGLEC12",
                    "source_state": "elevated expression",
                    "relation": "upregulates_expression",
                    "target": "immune checkpoint blockade",
                    "target_state": "upregulation",
                    "condition": "in bladder cancer"
                },
                {
                    "source": "SIGLEC12",
                    "source_state": "elevated expression",
                    "relation": "increases_level",
                    "target": "immune cell infiltration",
                    "target_state": "increased",
                    "condition": "in bladder cancer"
                },
                {
                    "source": "SIGLEC12",
                    "source_state": "elevated expression",
                    "relation": "leads_to",
                    "target": "drug resistance",
                    "target_state": "drug resistance signatures",
                    "condition": "in bladder cancer"
                },
                {
                    "source": "alterations in TP53",
                    "source_state": "alterations",
                    "relation": "leads_to",
                    "target": "tumor progression",
                    "target_state": "invasive tumor progression",
                    "condition": "in bladder cancer"
                },
                {
                    "source": "alterations in RB1",
                    "source_state": "alterations",
                    "relation": "leads_to",
                    "target": "tumor progression",
                    "target_state": "invasive tumor progression",
                    "condition": "in bladder cancer"
                },
                {
                    "source": "HRAS mutations",
                    "source_state": "mutations",
                    "relation": "leads_to",
                    "target": "low-grade papillary tumors",
                    "target_state": "Present",
                    "condition": "in bladder cancer"
                },
                {
                    "source": "FGFR3 mutations",
                    "source_state": "mutations",
                    "relation": "leads_to",
                    "target": "low-grade papillary tumors",
                    "target_state": "Present",
                    "condition": "in bladder cancer"
                },
                {
                    "source": "E-cadherin loss",
                    "source_state": "loss",
                    "relation": "facilitates",
                    "target": "tumor invasion",
                    "target_state": "Present",
                    "condition": "General"
                },
                {
                    "source": "matrix metalloproteinases",
                    "source_state": "Present",
                    "relation": "facilitates",
                    "target": "tumor invasion",
                    "target_state": "Present",
                    "condition": "General"
                },
                {
                    "source": "vascular endothelial growth factor",
                    "source_state": "Present",
                    "relation": "promotes",
                    "target": "angiogenesis",
                    "target_state": "Present",
                    "condition": "in bladder cancer"
                },
                {
                    "source": "angiogenesis",
                    "source_state": "Present",
                    "relation": "supports",
                    "target": "tumor growth",
                    "target_state": "Present",
                    "condition": "in bladder cancer"
                },
                {
                    "source": "angiogenesis",
                    "source_state": "Present",
                    "relation": "supports",
                    "target": "metastasis",
                    "target_state": "Present",
                    "condition": "in bladder cancer"
                }
            ]
        }
    ]
}
```

## 🙏 致谢

感谢以下项目对本工作的支持：

- [KEGG](https://www.kegg.jp/)
- [PubMed](https://pubmed.ncbi.nlm.nih.gov/)
- [PubChem](https://pubchem.ncbi.nlm.nih.gov/)
- [PubChemPy](https://github.com/mcs07/PubChemPy)
- [Genes - NCBI](https://uud.ncbi.nlm.nih.gov/home/genes/)
- [UniProt](https://www.uniprot.org/)
- [MyGene.info](https://github.com/biothings/mygene.info)
- [Qwen3](https://github.com/QwenLM/Qwen3)
- [GLM-4.6](https://huggingface.co/zai-org/GLM-4.6)
- [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)
- [Gemini 3](https://blog.google/products/gemini/gemini-3/)
- [GPT-5.2](https://openai.com/zh-Hans-CN/index/introducing-gpt-5-2/)
- [Doubao-Seed-1.8](https://console.volcengine.com/ark/region:ark+cn-beijing/model/detail?Id=doubao-seed-1-8)
- [Intern-S1](https://github.com/InternLM/Intern-S1)
- [Qwen3-Embedding](https://github.com/QwenLM/Qwen3-Embedding)
- [vllm](https://github.com/vllm-project/vllm)

## 📝 引用

```
@misc{biomebench2025,
  title = {BIOME-Bench: A Benchmark for Biomolecular Interaction Inference and Multi-Omics Pathway Mechanism Elucidation},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository}
}
```