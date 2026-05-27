<div align="center">

<img src="assets/logo.png" width="160" alt="LLMRouterBench">

# LLMRouterBench

### 大规模 LLM 路由评测基准与统一框架

[English](README.md) | [中文](README_zh.md)

[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](http://arxiv.org/abs/2601.07206)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow.svg)](https://huggingface.co/datasets/NPULH/LLMRouterBench)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)]()

<p align="center">
  <a href="#新闻">新闻</a> ·
  <a href="#概述">概述</a> ·
  <a href="#实验结果">实验结果</a> ·
  <a href="#安装">安装</a> ·
  <a href="#快速开始">快速开始</a> ·
  <a href="#数据集">数据集</a> ·
  <a href="#模型池">模型池</a> ·
  <a href="#引用">引用</a>
</p>

</div>

---

## 新闻

- **2026 年 4 月**：🎉🎉🎉 本文已被 **ACL 2026 Findings** 接收！

---

## 概述

<div align="center">

**33 个模型 | 21+ 数据集 | 10 种路由算法 | 40万+ 实例 | 约18亿 tokens**

</div>

LLMRouterBench 是一个大规模 LLM 路由评测基准与统一框架。我们整合了 21 个数据集、33 个模型的标准化输出，同时支持纯性能和性能-成本两种路由范式，并为 10 种代表性路由方法提供了开箱即用的适配器。

<div align="center">
<img src="assets/PPT_fig1_ZH.png" width="95%" alt="LLMRouterBench 概览">
</div>

### 核心亮点

- **双路由范式**: **纯性能** | **性能-成本**权衡
- **前沿模型池**: 20 个 7B 级**轻量**模型（Qwen3-8B、DS-Qwen3、NVIDIA-Nemo 等）+ 来自 8 家厂商的 13 个**旗舰**模型（GPT-5、Gemini-2.5-Pro、Claude-4、DeepSeek-V3.1 等）
- **多样化高难度数据集**: **数学**（AIME、LiveMathBench）、**代码**（LiveCodeBench、SWE-Bench）、**逻辑**（BBH、KORBench）、**知识**（HLE、SimpleQA）、**情感**（EmoryNLP、MELD）、**指令遵循**（ArenaHard）、**工具使用**（τ²-Bench）
- **代表性路由方法**: RouterDC (NeurIPS'24)、EmbedLLM (ICLR'25)、MODEL-SAT (AAAI'25)、Avengers (AAAI'26)、HybridLLM (ICLR'24)、FrugalGPT (TMLR'24)、RouteLLM (ICLR'25)、GraphRouter (ICLR'25)、Avengers-Pro (DAI'25 最佳论文)、OpenRouter
- **数据收集成本**: 约 1000 GPU 小时 + 3000 美元 API 开销
- **标准化数据字段**（逐实例）: `origin_query`, `prompt`, `prediction(raw output)`, `ground_truth`, `score`, `prompt_tokens`, `completion_tokens`, `cost`
- **模块化架构**: **Collector**（统一 LLM API）→ **Evaluator**（数据集评分）→ **Adaptor**（算法格式适配）
---

## 关键发现

### 纯性能设置

**没有哪个模型能在所有领域称霸，模型间呈现出明显的互补优势。** 如下图所示，数学任务上 Intern-S1-mini、Qwen3-8B 表现更优，代码任务上 Qwen-Coder、Fin-R1 更胜一筹，这正是 LLM 路由的核心前提。

<div align="center">
<img src="assets/Figure8-perf-main-1-row.png" width="95%" alt="各领域模型性能">
</div>

**主流路由方法性能相近，但距离 Oracle 仍有明显差距。** 我们将路由方法与三个基线对比：**Random**（随机选模型）、**Best Single**（平均准确率最高的单一模型）、**Oracle**（为每条查询选择最优模型，即理论上界）。核心指标：
- **AvgAcc**: 所有数据集的平均准确率
- **Gain@R / Gain@B**: 相对 Random / Best Single 的提升幅度
- **Gap@O**: 与 Oracle 的差距（越小越好）

尽管方法各异，主流路由器（EmbedLLM、GraphRouter、MODEL-SAT、Avengers）的表现趋于一致。值得一提的是，Avengers 无需神经网络训练即可达到同等水平。各方法与 **Dataset Oracle**（为每个数据集选最优模型，图中斜线柱）相近，说明当前路由收益主要来自粗粒度的领域区分。而与实例级 Oracle 之间仍存在显著 Gap@O，主因是 **模型召回失败**——当仅少数模型能答对时，路由器往往选不中。

<div align="center">
<img src="assets/Figure7-perf-metrics-03.png" width="95%" alt="性能指标">
</div>

### 性能-成本设置

**有效路由能带来显著收益，但并非所有路由器都能做到。** 性能-成本权衡的核心指标：
- **PerfGain**: 相对 Best Single 的最大性能提升（取最高准确率配置）
- **CostSave**: 在不低于 Best Single 准确率前提下的最大成本节省
- **Pareto 前沿**: 最优配置集合——不存在同时更便宜且更准的配置
- **ParetoDist**: 到 Pareto 前沿的平均距离（越小越好）

顶级方法可达 4% PerfGain 和 31.7% CostSave。但部分路由器（包括商业化的 OpenRouter）未能超越 Best Single。Avengers-Pro 以接近零的 ParetoDist 占据 Pareto 前沿。

<table>
<tr>
<td width="50%">
<div align="center">
<img src="assets/Figure10-2-PanelC.png" width="100%" alt="性能提升与成本节省">
<br>
<sub>相对于 GPT-5 的 PerfGain 和 CostSave。</sub>
</div>
</td>
<td width="50%">
<div align="center">
<img src="assets/Figure11-1-ParetoDist.png" width="100%" alt="Pareto 前沿">
<br>
<sub>准确率 vs. 成本及 Pareto 前沿。</sub>
</div>
</td>
</tr>
</table>

> 更多发现（Embedding 消融实验、模型池规模影响、延迟分析等）详见论文。

<details>
<summary><b>原始数据表</b></summary>

<div align="center">
<img src="assets/Table6.png" width="95%" alt="表6: 纯性能设置">
</div>

<div align="center">
<img src="assets/Table9.png" width="95%" alt="表9: 性能-成本设置">
</div>

<div align="center">
<img src="assets/Table10.png" width="95%" alt="表10: 推理成本">
</div>

</details>

---

## 安装

```bash
git clone https://github.com/ynulihao/LLMRouterBench.git
cd LLMRouterBench
pip install -r requirements.txt
```

## 快速开始

<table>
<tr>
<td>

**1. 收集数据**
```bash
python -m data_collector.cli run \
    config/data_collector_small_model_config.yaml
```

</td>
<td>

**2. 分析**
```python
from baselines import BaselineDataLoader, BaselineAggregator

loader = BaselineDataLoader("config/baseline_config.yaml")
records = loader.load_all_records()
agg = BaselineAggregator(records, data_loader=loader)
agg.print_summary_tables(
    score_as_percent=True,
    test_mode=False
)
```

</td>
<td>

**3. 训练路由器**
```bash
python -m baselines.adaptors.avengerspro_adaptor \
    --config config/baseline_config.yaml \
    --seed 42 \
    --split-ratio 0.7 \
    --output-dir baselines/AvengersPro/data/small_models_seed_42

python -m baselines.AvengersPro.simple_cluster_router \
    --config baselines/AvengersPro/config/simple_config_small_models_42.json \
    --output baselines/AvengersPro/logs/simple_config_small_models_42.json
```

</td>
</tr>
</table>

---



## 配置说明

LLMRouterBench 支持两种路由范式，分别对应不同配置文件：

| 设置 | 说明 | Collector 配置 | Adaptor 配置 |
|:---|:---|:---|:---|
| **纯性能** | 20 个轻量模型（约 7B） | `config/data_collector_small_model_config.yaml` | `config/baseline_config.yaml` |
| **性能-成本** | 13 个旗舰模型（含成本信息） | `config/data_collector_proprietary_model_config.yaml` | `config/baseline_config_performance_cost.yaml` |

## 核心组件

LLMRouterBench 采用模块化的三组件架构：

<div align="center">
<img src="assets/figure2-framework.png" width="95%" alt="框架架构">
</div>

<table>
<tr>
<th width="33%">Collector（数据收集）</th>
<th width="33%">Evaluator（结果评估）</th>
<th width="33%">Adaptor（格式适配）</th>
</tr>
<tr>
<td valign="top">

统一的 LLM API 调用接口：
- 自动缓存与失败重试
- 成本追踪
- Token 统计

[文档](data_collector/README.md)

</td>
<td valign="top">

支持 21+ 数据集的定制化评估：
- 数据集专属评估模块
- 多样化评分（正则匹配、LLM-as-Judge）
- **新增**: 高难度数据集（FrontierScience、SGIBench、SFE）及多模态支持

[文档](evaluation/README.md)

</td>
<td valign="top">

为 10 种路由算法适配数据格式：
- 算法专属输入格式
- 统一的训练/测试划分
- 支持域内/域外（ID/OOD）设置

[文档](baselines/README.md)

</td>
</tr>
</table>

### 自定义

| 扩展类型 | 操作步骤 |
|:---|:---|
| **添加新模型** | 1. 在 `config/data_collector_*.yaml` 添加模型配置 2. 运行 Collector 生成结果 |
| **添加新数据集** | 1. 在 `evaluation/` 下创建评估模块 2. 在配置文件中注册 |
| **添加新路由方法** | 1. 将算法代码放入 `baselines/` 2. 在 `baselines/adaptors/` 实现适配器 |
---

## 数据集

LLMRouterBench 涵盖 21 个数据集，分布于两种设置：

<details>
<summary><b>纯性能设置（15 个数据集）</b></summary>

| 类别 | 数据集 | 缩写 | 样本数 | 指标 |
|:---|:---|:---:|---:|:---:|
| **数学** | AIME | AIME | 60 | 准确率, 0-shot |
| | MATH500 | M500. | 500 | 准确率, 0-shot |
| | MATHBench | MBen. | 150 | 准确率, 0-shot |
| **代码** | HumanEval | HE. | 164 | Pass@1, 0-shot |
| | MBPP | MBPP | 974 | Pass@1, 0-shot |
| | LiveCodeBench | LCB. | 1055 | Pass@1, 0-shot |
| **逻辑** | BBH | BBH | 1080 | 准确率, 3-shot |
| | KORBench | KOR. | 1250 | 准确率, 3-shot |
| | Knights & Knaves | K&K. | 700 | 准确率, 0-shot |
| **知识** | MMLU-Pro | MP. | 1000 | 准确率, 0-shot |
| | GPQA | GPQA | 198 | 准确率, 0-shot |
| | FinQA | FQA. | 1147 | 准确率, 0-shot |
| | MedQA | MQA. | 1273 | 准确率, 0-shot |
| **情感** | EmoryNLP | Emory. | 697 | 准确率, 0-shot |
| | MELD | MELD | 1232 | 准确率, 0-shot |

</details>

<details>
<summary><b>性能-成本设置（10 个数据集）</b></summary>

| 类别 | 数据集 | 缩写 | 样本数 | 指标 |
|:---|:---|:---:|---:|:---:|
| **数学** | AIME | AIME | 60 | 准确率, 0-shot |
| | LiveMathBench | LMB. | 121 | 准确率, 0-shot |
| **代码** | LiveCodeBench | LCB. | 1055 | Pass@1, 0-shot |
| | SWE-Bench | SWE. | 500 | Pass@1, 0-shot |
| **知识** | GPQA | GPQA | 198 | 准确率, 0-shot |
| | HLE | HLE | 2158 | LLM 判断, 0-shot |
| | MMLU-Pro | MP. | 3000 | 准确率, 0-shot |
| | SimpleQA | SQA. | 4326 | LLM 判断, 0-shot |
| **指令遵循** | ArenaHard | AHARD. | 750 | LLM 判断, 0-shot |
| **工具使用** | τ²-Bench | TAU2. | 278 | 成功率, 0-shot |

</details>

## 模型池

<details>
<summary><b>纯性能设置（20 个模型）</b></summary>

| 模型 | 缩写 | 参数量 |
|:---|:---:|:---:|
| DeepHermes-3-Llama-3-8B-Preview | DH-Llama3-it | 8B |
| DeepSeek-R1-0528-Qwen3-8B | DS-Qwen3 | 8B |
| DeepSeek-R1-Distill-Qwen-7B | DS-Qwen | 7B |
| Fin-R1 | Fin-R1 | 7B |
| GLM-Z1-9B-0414 | GLM-Z1 | 9B |
| Intern-S1-mini | Intern-S1-mini | 8B |
| Llama-3.1-8B-Instruct | Llama-3.1-it | 8B |
| Llama-3.1-8B-UltraMedical | UltraMedical | 8B |
| Llama-3.1-Nemotron-Nano-8B-v1 | Llama-Nemo | 8B |
| MiMo-7B-RL-0530 | MiMo-RL | 7B |
| MiniCPM4.1-8B | MiniCPM | 8B |
| NVIDIA-Nemotron-Nano-9B-v2 | NVIDIA-Nemo | 9B |
| OpenThinker3-7B | OpenThinker | 7B |
| Qwen2.5-Coder-7B-Instruct | Qwen-Coder | 7B |
| Qwen3-8B | Qwen3-8B | 8B |
| Cogito-v1-preview-llama-8B | Cogito-v1 | 8B |
| Gemma-2-9b-it | Gemma-2-it | 9B |
| Glm-4-9b-chat | Glm-4-chat | 9B |
| Granite-3.3-8b-instruct | Granite-3.3-it | 8B |
| Internlm3-8b-instruct | Internlm3-it | 8B |

</details>

<details>
<summary><b>性能-成本设置（13 个模型）</b></summary>

| 模型 | 缩写 | 输入价格 | 输出价格 |
|:---|:---:|---:|---:|
| Claude-sonnet-4 | Claude-v4 | $3.00/1M | $15.00/1M |
| Gemini-2.5-flash | Gemini-Flash | $0.30/1M | $2.50/1M |
| Gemini-2.5-pro | Gemini-Pro | $1.25/1M | $10.00/1M |
| GPT-5-chat | GPT-5-Chat | $1.25/1M | $10.00/1M |
| GPT-5-medium | GPT-5 | $1.25/1M | $10.00/1M |
| Qwen3-235b-a22b-2507 | Qwen3-235B | $0.09/1M | $0.60/1M |
| Qwen3-235b-a22b-thinking-2507 | Qwen3-Thinking | $0.30/1M | $2.90/1M |
| Deepseek-v3-0324 | DeepSeek-V3 | $0.25/1M | $0.88/1M |
| Deepseek-v3.1-terminus | DS-V3.1-Tms | $0.27/1M | $1.00/1M |
| Deepseek-r1-0528 | DeepSeek-R1 | $0.50/1M | $2.15/1M |
| GLM-4.6 | GLM-4.6 | $0.60/1M | $2.20/1M |
| Kimi-k2-0905 | Kimi-K2 | $0.50/1M | $2.00/1M |
| Intern-s1 | Intern-S1 | $0.18/1M | $0.54/1M |

</details>

---

## 数据下载

LLMRouterBench 将标准化 JSON 记录存储于 `results/bench/`，并提供预收集结果的下载包：

<div align="center">
<table>
<tr>
<td align="center">
<strong>百度网盘</strong><br>
<a href="https://pan.baidu.com/s/1bfa_eX3bhuo7wgNlD_dbpA?pwd=mmbf">bench-release.tar.gz</a><br>
<sub>提取码: mmbf</sub>
</td>
<td align="center">
<strong>Google Drive</strong><br>
<a href="https://drive.google.com/file/d/12pupoZDjqziZ2JPspH60MCC8fdXWgnX1/view?usp=drive_link">bench-release.tar.gz</a>
</td>
<td align="center">
<strong>Hugging Face</strong><br>
<a href="https://huggingface.co/datasets/NPULH/LLMRouterBench">bench-release.tar.gz</a>
</td>
</tr>
</table>
</div>

```bash
# 解压到 results 目录
tar xzf bench-release.tar.gz
```

解压后的目录结构：
```
results/
└── bench/
    ├── aime/
    ├── bbh/
    ├── humaneval/
    ├── mmlu_pro/
    └── ...
```

详见 [results/download.md](results/download.md)

<details>
<summary><b>结果文件结构</b></summary>

结果以 JSON 格式存储在 `results/bench/<dataset>/<split>/<model>/<timestamp>.json`：

```json
{
  "performance": 0.85,
  "time_taken": 120.5,
  "prompt_tokens": 50000,
  "completion_tokens": 20000,
  "cost": 0.15,
  "counts": 100,
  "records": [
    {
      "index": 1,
      "origin_query": "What is the sum of 2+2?",
      "prompt": "Question: What is the sum of 2+2?\nAnswer:",
      "prediction": "4",
      "ground_truth": "4",
      "score": 1.0,
      "prompt_tokens": 15,
      "completion_tokens": 5,
      "cost": 0.0001,
      "raw_output": "The sum of 2+2 is 4."
    }
  ]
}
```

</details>

<details>
<summary><b>数据查看器示例</b></summary>

```python
from baselines import BaselineDataLoader, BaselineAggregator

loader = BaselineDataLoader('config/baseline_config.yaml')
records = loader.load_all_records()

test_mode = False
score_as_percent = True
train_ratio = 0.7
random_seed = 3407

agg = BaselineAggregator(records, data_loader=loader)
agg.print_summary_tables(
    score_as_percent=score_as_percent,
    test_mode=test_mode,
    random_seed=random_seed,
    train_ratio=train_ratio,
)

agg.save_summary_tables_to_excel(
    output_file='small_models_total.xlsx',
    score_as_percent=score_as_percent,
    test_mode=test_mode,
    train_ratio=train_ratio,
    random_seed=random_seed
)
```

</details>

---

## 项目结构

```
LLMRouterBench/
├── data_collector/     # Collector 模块
├── evaluation/         # Evaluator（21 个数据集）
├── baselines/          # Adaptor 与路由算法
├── generators/         # 模型 API 接口
├── common/cache/       # 缓存系统
├── external_bench/     # 第三方集成
├── config/             # 配置文件
└── results/            # 基准测试结果
```

## 🗓️ 路线图

**近期更新**
- ✅ 集成三个高难度基准（FrontierScience、SGIBench、SFE）
- ✅ 扩展多模态路由评测支持

**长期目标**
- 更广泛的模型覆盖
- 扩展基准测试集
- 更多基线方法
- 扩展路由范式

## 相关工作

### 与现有路由基准的对比

<div align="center">
<img src="assets/Table1_2.png" width="95%" alt="与现有路由基准的对比">
<br>
</div>

现有路由基准存在以下局限：

- **RouterBench**: 仅限早期模型和 8 个相对简单的数据集。
- **EmbedLLM & RouterEval**: 聚焦开源模型，缺乏推理成本信息。
- **FusionFactory**: 使用估计成本评测开源模型。
- **RouterArena**: 各路由器使用的模型池不一致，有碍公平对比，且缺乏逐 prompt、逐模型的数据。

## 引用

如果 LLMRouterBench 对您的研究有帮助，请考虑引用我们的论文：
```bibtex
@article{li2026llmrouterbench,
  title={LLMRouterBench: A Massive Benchmark and Unified Framework for LLM Routing},
  author={Li, Hao and Zhang, Yiqun and Guo, Zhaoyan and Wang, Chenxu and Tang, Shengji and Zhang, Qiaosheng and Chen, Yang and Qi, Biqing and Ye, Peng and Bai, Lei and others},
  journal={arXiv preprint arXiv:2601.07206},
  year={2026}
}
```
本工作是我们 LLM 路由系列研究的一部分，如您感兴趣，请同时参考并引用：
```bibtex
@inproceedings{zhang2025avengers,
  title        = {The Avengers: A Simple Recipe for Uniting Smaller Language Models to Challenge Proprietary Giants},
  author       = {Zhang, Yiqun and Li, Hao and Wang, Chenxu and Chen, Linyao and Zhang, Qiaosheng and Ye, Peng and Feng, Shi and Wang, Daling and Wang, Zhen and Wang, Xinrun and others},
  booktitle    = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year         = {2025},
  note         = {Oral presentation},
  url          = {https://arxiv.org/abs/2505.19797}
}
@inproceedings{zhang2025beyond,
  title        = {Beyond gpt-5: Making llMs cheaper and better via performance-efficiency optimized routing},
  author       = {Zhang, Yiqun and Li, Hao and Chen, Jianhao and Zhang, Hangfan and Ye, Peng and Bai, Lei and Hu, Shuyue},
  booktitle    = {Distributed AI (DAI) conference},
  year         = {2025},
  note         = {Best Paper Award},
  url          = {https://arxiv.org/abs/2508.12631}
}
@inproceedings{wang2025icl,
  title        = {ICL-Router: In-Context Learned Model Representations for LLM Routing},
  author       = {Wang, Chenxu and Li, Hao and Zhang, Yiqun and Chen, Linyao and Chen, Jianhao and Jian, Ping and Ye, Peng and Zhang, Qiaosheng and Hu, Shuyue},
  booktitle    = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year         = {2025},
  note         = {Poster},
  url          = {https://arxiv.org/abs/2510.09719}
}
@article{chen2025learning,
  title        = {Learning Compact Representations of LLM Abilities via Item Response Theory},
  author       = {Chen, Jianhao and Wang, Chenxu and Zhang, Gengrui and Ye, Peng and Bai, Lei and Hu, Wei and Qu, Yuzhong and Hu, Shuyue},
  journal      = {arXiv preprint arXiv:2510.00844},
  year         = {2025},
  url          = {https://arxiv.org/abs/2510.00844v1}
}
```

---

<div align="center">

**LLMRouterBench** — 推动 LLM 路由研究

[提交 Issue](https://github.com/ynulihao/LLMRouterBench/issues) · [功能建议](https://github.com/ynulihao/LLMRouterBench/issues)

</div>
