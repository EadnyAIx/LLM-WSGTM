# LLM-WSGTM

LLM-WSGTM 是一个面向主题建模实验的研究代码项目。模型以文档语义表示为基础，通过文档—主题与主题—词之间的最优传输关系学习主题分布，并在训练过程中加入锚点条件先验、主题边界约束和覆盖优化等机制。

当前仓库包含模型训练、主题词导出、指标计算、结果保存和可视化代码，同时保留了独立的 Agent 扩展目录，用于 MCP、Skills 和 RAG 相关实验。Agent 部分与主题模型主训练流程相互独立，不影响模型的基本运行。

## 项目结构

```text
LLM-WSGTM/
├── llm_wsgtm/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── mcp_client.py
│   │   ├── rag.py
│   │   └── skills.py
│   ├── __init__.py
│   ├── anchor_prior.py
│   ├── core.py
│   ├── coverage_optimizer.py
│   ├── data.py
│   ├── experiment.py
│   ├── llm_client.py
│   ├── math_utils.py
│   ├── metrics.py
│   ├── model.py
│   ├── semantic_labeler.py
│   ├── transport.py
│   └── visualization.py
├── README.md
├── pyproject.toml
├── requirements.txt
├── run_experiment.py
└── train.py
```

其中：

- `model.py`：LLM-WSGTM 模型接口与训练流程。
- `core.py`：主题模型核心计算，包括主题表示、主题词分布及相关损失项。
- `transport.py`：基于 Sinkhorn 迭代的最优传输计算。
- `anchor_prior.py`：根据主题词构建锚点条件先验。
- `coverage_optimizer.py`：训练后的主题词覆盖优化。
- `semantic_labeler.py`：结合主题词和代表文档生成主题名称与描述。
- `metrics.py`：主题连贯性、多样性、覆盖度、Purity 和 NMI 等指标。
- `experiment.py`：数据读取、模型训练、评测及实验结果保存。
- `agents/`：MCP、Skills、RAG 和 Agent 路由相关扩展代码。

## 运行环境

建议使用 Python 3.10 及以上版本。

主要依赖包括：

```text
torch
numpy
pandas
scipy
sentence-transformers
scikit-learn
gensim
topmost
plotly
tqdm
requests
mcp
```

如本机已正确安装支持 CUDA 的 PyTorch，默认配置会优先使用 GPU；否则使用 CPU。

## 安装

克隆仓库：

```bash
git clone https://github.com/EadnyAIx/LLM-WSGTM.git
cd LLM-WSGTM
```

创建虚拟环境：

```bash
python -m venv .venv
```

Windows：

```bash
.venv\Scripts\activate
```

Linux 或 macOS：

```bash
source .venv/bin/activate
```

安装依赖：

```bash
pip install -r requirements.txt
```

也可以按项目方式安装：

```bash
pip install -e .
```

## 数据集

默认从项目根目录下的 `datasets` 文件夹读取数据。目录形式如下：

```text
LLM-WSGTM/
├── datasets/
│   └── NeurIPS/
│       ├── train_texts.txt
│       └── train_labels.txt
├── models/
└── runs/
```

`train_texts.txt` 为必需文件，每行对应一篇文档。

`train_labels.txt` 为可选文件，每行对应一篇文档的类别标签。存在标签时，评测阶段会额外计算 Purity 和 NMI。

如果指定的数据集目录不存在，程序会尝试通过 TopMost 下载对应数据集。

## 文档语义模型

默认配置使用本地 SentenceTransformer 模型：

```text
models/all-MiniLM-L6-v2
```

对应配置项为：

```python
"document_embedding_model": "models/all-MiniLM-L6-v2"
```

如果使用其他 SentenceTransformer 模型，只需要修改该路径。

## Ollama 配置

锚点条件先验和主题语义命名模块需要本地 Ollama 服务。

默认地址：

```text
http://127.0.0.1:11434
```

默认模型：

```text
llama3:8b
```

可以通过环境变量修改 Ollama 地址。

Linux 或 macOS：

```bash
export OLLAMA_HOST=http://127.0.0.1:11434
```

Windows PowerShell：

```powershell
$env:OLLAMA_HOST="http://127.0.0.1:11434"
```

如果当前实验不使用 Ollama，可以关闭相关模块：

```json
{
  "anchor_prior_enable": false,
  "semantic_labeling_enable": false
}
```

## 直接运行

使用默认配置训练：

```bash
python train.py
```

也可以运行：

```bash
python run_experiment.py
```

默认配置位于 `run_experiment.py` 的 `BASE_CONFIG` 中，包括数据集、主题数、训练轮数、学习率以及各模块参数。

## 使用配置文件

可以新建一个 JSON 文件，例如 `config.json`：

```json
{
  "dataset_name": "NeurIPS",
  "num_topics": 60,
  "epochs": 300,
  "learning_rate": 0.0006,
  "anchor_prior_enable": true,
  "coverage_optimizer_enable": true,
  "semantic_labeling_enable": true,
  "ollama_model": "llama3:8b"
}
```

运行：

```bash
python run_experiment.py --config config.json
```

也可以直接覆盖少量参数：

```bash
python run_experiment.py --overrides '{"num_topics": 100, "epochs": 200}'
```

## 主要参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `dataset_name` | `NeurIPS` | 数据集名称 |
| `num_topics` | `60` | 主题数量 |
| `num_top_words` | `20` | 每个主题导出的关键词数量 |
| `top_n_for_metrics` | `10` | 计算指标时使用的主题词数量 |
| `epochs` | `300` | 训练轮数 |
| `learning_rate` | `6e-4` | Adam 学习率 |
| `document_topic_alpha` | `0.8` | 文档—主题传输参数 |
| `topic_word_alpha` | `0.6` | 主题—词传输参数 |
| `theta_temperature` | `0.4` | 文档主题分布温度参数 |
| `anchor_prior_enable` | `true` | 是否启用锚点条件先验 |
| `anchor_prior_after_epochs` | `5` | 开始构建锚点先验的训练轮次 |
| `anchor_prior_weight` | `0.05` | 锚点先验损失权重 |
| `coverage_optimizer_enable` | `true` | 是否进行覆盖优化 |
| `semantic_labeling_enable` | `true` | 是否生成主题语义标签 |
| `ollama_model` | `llama3:8b` | Ollama 模型名称 |
| `seed` | `2024` | 随机种子 |

其他参数可以直接查看 `run_experiment.py` 中的 `BASE_CONFIG`。

## Python 调用

模型主类为 `LLMWSGTM`：

```python
from llm_wsgtm import LLMWSGTM

model = LLMWSGTM(
    num_topics=50,
    document_embedding_model="models/all-MiniLM-L6-v2",
    anchor_prior_enable=False,
)

top_words, theta = model.fit_transform(
    documents,
    epochs=200,
    learning_rate=6e-4,
)

beta = model.get_beta()
```

对新文档进行主题分布推断：

```python
theta_new = model.transform(documents=new_documents)
```

## 模型组成

### 语义表示

文档通过 SentenceTransformer 编码得到语义向量。主题表示、词表示和文档表示在统一的向量空间中参与后续主题学习。

### 文档—主题与主题—词传输

模型分别计算文档到主题、主题到词的最优传输关系，并由此得到文档主题分布和主题词分布。相关计算位于 `core.py` 和 `transport.py`。

### 语义边界约束

训练过程中通过主题表示约束、主题词语义一致性和文档主题分布约束减少主题之间的重叠，并保持主题内部语义的一致性。

### 锚点条件先验

启用锚点先验后，模型会根据当前主题词生成锚点词与反锚点词，并将其映射到训练词表，形成主题词先验分布。该先验通过附加损失参与后续训练。

### 覆盖优化

模型训练完成后，可以对主题关键词进行覆盖优化。该过程在候选词中搜索能够提升文档覆盖的替换词，只调整最终导出的关键词，不修改已经学习到的 `beta` 和 `theta`。

### 主题语义命名

主题命名模块根据主题关键词和代表文档生成主题名称、简要说明和补充关键词，结果分别保存为 JSON 和文本文件。

## 评测指标

当前实验流程使用以下指标：

- Topic Diversity
- Coherence，优先计算 `c_v`，失败时使用文档级 NPMI
- Document Coverage
- Purity
- NMI
- 主题关键词平均 Jaccard 重叠度

其中 Purity 和 NMI 需要数据集提供标签。

## 实验输出

每次运行会在 `runs/` 下建立一个带时间戳的目录，例如：

```text
runs/llm-wsgtm_20260819-220000/
```

根据启用的模块，目录中可能包含：

```text
config.json
beta.pt
theta.pt
llm_wsgtm.pt
evaluation.json
coverage_report.json
topic_top_words.txt
topic_labels.json
topic_labels.txt
```

其中：

- `config.json`：本次实验参数。
- `beta.pt`：主题词分布。
- `theta.pt`：训练语料的文档主题分布。
- `llm_wsgtm.pt`：训练后的模型。
- `evaluation.json`：评测结果。
- `coverage_report.json`：覆盖优化前后的结果。
- `topic_top_words.txt`：各主题关键词。
- `topic_labels.json`、`topic_labels.txt`：主题语义命名结果。

## Agent 扩展

`llm_wsgtm/agents/` 为独立扩展目录，目前包含 Skills、RAG、MCP 和 Agent 路由代码。该部分不参与默认主题模型训练，需要时可单独调用。

### Skills

`SkillRegistry` 用于注册和调用本地 Python 函数：

```python
from llm_wsgtm.agents import SkillRegistry

skills = SkillRegistry()
skills.register("topic_count", lambda topics: len(topics), description="Count topics")
result = skills.invoke("topic_count", topics=[1, 2, 3])
```

### RAG

`DenseRAGIndex` 提供简单的内存向量检索：

```python
from llm_wsgtm.agents import DenseRAGIndex

rag = DenseRAGIndex(encoder)
rag.add(["document one", "document two"])
results = rag.search("query", top_k=2)
```

### MCP

`MCPClient` 用于连接通过标准输入输出启动的 MCP Server：

```python
from llm_wsgtm.agents import MCPClient

client = MCPClient(["python", "server.py"])
tools = client.list_tools()
```

### Agent 路由

`TopicModelAgent` 用于组合本地 Skills、RAG 检索结果和 MCP 工具，并根据输入选择对应的调用方式。

## 复现实验

默认随机种子为 `2024`，同时设置 NumPy 和 PyTorch。不同硬件、CUDA/cuDNN 版本、依赖版本、文档语义模型以及 Ollama 模型版本都可能造成实验结果差异。
