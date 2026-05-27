# LLMRouterBench（llmbenchmark）内置资源

本目录包含 DuoRoute 运行所需的 **LLMRouterBench 配置与 benchmark 数据入口**，无需再引用仓库外路径。

## 目录说明

```text
llmbenchmark/
├── config/                              # baseline 过滤、模型列表、collector pricing
│   ├── baseline_config.yaml             # 15 数据集 × 20 个 8B 模型（DuoRoute 默认）
│   ├── baseline_config_performance_cost.yaml
│   └── data_collector_small_model_config.yaml
├── baselines/GraphRouter/configs/
│   └── adaptor_config.yaml              # API 模型的 llm_descriptions（补充 model card）
├── results/
│   ├── download.md                      # 官方数据下载说明
│   └── bench -> ...                     # benchmark JSON（当前为 symlink）
├── README.md
└── README_zh.md
```

## 关于 `results/bench`

当前 `results/bench` **符号链接**到本机已有的 LLMRouterBench 数据：

```text
llmbenchmark/results/bench -> /home/phx/LLMRouterBench/results/bench
```

换机器或需要独立副本时，可任选其一：

```bash
# 方式 A：复制（约 6.6GB）
rsync -a /path/to/LLMRouterBench/results/bench/ llmbenchmark/results/bench/

# 方式 B：按 download.md 下载 bench-release.tar.gz 解压到 llmbenchmark/results/
```

## DuoRoute 默认用法

在仓库根目录：

```bash
python3 scripts/build_rewards.py \
  --config llmbenchmark/config/baseline_config.yaml \
  --bench-dir llmbenchmark/results/bench \
  --output-dir data/seed42 \
  --model-cards configs/model_cards.yaml \
  --llmbench-collector-config llmbenchmark/config/data_collector_small_model_config.yaml \
  --llmbench-graphrouter-config llmbenchmark/baselines/GraphRouter/configs/adaptor_config.yaml
```

或使用 `configs/default.yaml` 里已写好的相对路径。

## 未包含内容

为控制体积，**未复制** LLMRouterBench 的 `baselines/` 训练代码（约 25GB，SPO/EmbedLLM 等）。DuoRoute 本身不依赖那些代码，只使用本目录下的 **config + bench JSON + model descriptions**。
