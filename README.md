# DuoRoute

双通道 **dense reward** LLM 路由框架：训练时 Channel A（query + model-card）与 Channel B（query + response + model-card）共同拟合 Oracle reward；部署时默认只用 Channel A，必要时用 B 对已生成 response 做 fallback 重选。

本仓库代码与 **LLMRouterBench 对齐评估脚本** 在 GitHub；实验数据与 checkpoint 通过 [Release 数据包](docs/DATA_DOWNLOAD.md) 单独下载。

## 快速开始

**本仓库仅含代码**；实验数据（~750MB）需先从 [Releases](https://github.com/panhongxing-sds/duoroute/releases) 下载：

```bash
cd DuoRoute
pip install -e .
bash scripts/download_data.sh          # 下载并解压 data + checkpoint
python3 tests/test_duoroute.py
```

或见 [docs/DATA_DOWNLOAD.md](docs/DATA_DOWNLOAD.md)。

数据就绪后：

```bash
# 小模型池 query-only 评估
python3 scripts/eval_duoroute.py \
  --data-dir data/seed42_small \
  --checkpoint outputs/checkpoints/seed42_small/best.pt

# 统一基线对比
python3 scripts/eval_unified_baselines.py \
  --data-dir data/seed42_small \
  --checkpoint outputs/checkpoints/seed42_small/best.pt

# LLMRouterBench 对齐三表（8 卡并行）
python3 scripts/run_llmbench_aligned_eval.py --gpus 0,1,2,3,4,5,6,7
```

## 实验数据

| Pool | 路径 | 模型数 | 数据集 | train/val/test |
|------|------|--------|--------|----------------|
| **小模型池** | `data/seed42_small/` | 20 | 10 | 5814 / 824 / 1669 |
| **旗舰池** | `data/seed42_flagship/` | 13 | 9 | 2966 / 416 / 851 |

每个 pool 目录包含：

- `{train,val,test}/grouped.npz` — reward / performance / cost / mask 矩阵
- `{train,val,test}/meta.json` — prompt 与 response 文本（体积较大，**必需**）
- `question_embeddings.f16.npz` — query embedding（2048 维，`text-embedding-3-large`）
- `model_embeddings.f16.npz` — model-card embedding
- `train/response_embeddings.f16.npz` — Channel B 训练用 response embedding
- `model_cards.json`, `config.json`, `embedding_meta.json`

已训练 checkpoint：

- `outputs/checkpoints/seed42_small/best.pt`
- `outputs/checkpoints/seed42_flagship/best.pt`

默认训练超参（两池相同）：30 epoch，`α=0.5`, `β=0.1`, `T=0.5`, `λ=0.2`, `hidden_dim=64`, `distill_warmup=3`。

## 项目结构

```text
DuoRoute/
├── configs/              # 训练/评估/embedding 配置
│   ├── small_subset.yaml # seed42_small
│   ├── flagship.yaml     # seed42_flagship
│   ├── default.yaml      # 通用默认超参
│   └── embedding.yaml    # 重建 embedding 时的 backend 配置
├── src/duoroute/         # 核心包（data, model, trainer, embedding_io, …）
├── scripts/              # CLI：train / eval / compress / baselines
├── data/
│   ├── seed42_small/     # ★ 主实验 pool
│   └── seed42_flagship/  # ★ 旗舰 pool
├── outputs/
│   ├── checkpoints/      # ★ 已训练权重
│   └── eval/             # 评估结果（可重新生成）
├── llmbenchmark/         # 内置 LLMRouterBench 配置（bench JSON 为外部 symlink）
├── tests/
└── docs/                 # 详细结构与上传清单
```

更完整的目录说明见 [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)。

## 核心设计

| 模块 | 说明 |
|------|------|
| **Model embedding** | model-card 描述 embedding（非 ID table），支持未见模型 zero-shot |
| **Oracle reward** | `R = performance − λ × norm_cost(per_query)`，统一越大越好 |
| **Channel A** | `query_emb + model_card_emb → reward` |
| **Channel B** | `query_emb + response_emb + model_card_emb → reward` |
| **Distill warmup** | 前 N epoch 不开 `L_distill`，避免 A 过早模仿 B |
| **Fallback** | A 的 top1−top2 margin < δ 时，用 B 在已有 responses 上重选 |
| **LOMO** | leave-one-model-out 训练时 mask 掉 held-out 模型列 |

Loss：`L = L_reg + α·L_rank + β·L_distill`

## Embedding 压缩

预构建 embedding 已压缩为 `.f16.npz`（float16 + gzip）。加载器会**自动优先**读取 `.f16.npz`，否则回退 `.pth`；运行时转为 float32，与原始精度误差 ≤ 6e-5。

```bash
# 压缩（上传前可选，去掉原始 .pth 可再省 ~1.1GB）
python3 scripts/compress_embeddings.py --data-dirs data/seed42_small data/seed42_flagship

# 解压回 float32 .pth
python3 scripts/decompress_embeddings.py
```

## 安装

```bash
pip install -e .
pip install -e ".[embeddings]"   # 仅重新构建 embedding 时需要 sentence-transformers
```

## 训练

```bash
# 小模型池
python3 scripts/train_duoroute.py \
  --config configs/small_subset.yaml \
  --data-dir data/seed42_small \
  --output-dir outputs/checkpoints/seed42_small

# 旗舰池
python3 scripts/train_duoroute.py \
  --config configs/flagship.yaml \
  --data-dir data/seed42_flagship \
  --output-dir outputs/checkpoints/seed42_flagship
```

## 评估

### Query-only 部署

```bash
python3 scripts/eval_duoroute.py \
  --data-dir data/seed42_flagship \
  --checkpoint outputs/checkpoints/seed42_flagship/best.pt
```

### 统一 split 基线对比

`eval_unified_baselines.py` 在同一 train/test split 上对比 SingleBest、DuoRoute、AvengersPro（复用本地 `question_embeddings`，不调 API）：

```bash
python3 scripts/eval_unified_baselines.py \
  --data-dir data/seed42_flagship \
  --checkpoint outputs/checkpoints/seed42_flagship/best.pt
```

### LLMRouterBench 对齐三表

`run_llmbench_aligned_eval.py` 输出三套评价：

1. **Performance-oriented**（两池）：AvgAcc, Gain@B, Gap@O
2. **Fixed utility λ=0.2**（旗舰）：AvgAcc, AvgReward, Regret
3. **Pareto frontier**（旗舰）：DuoRoute λ sweep vs AvengersPro balance sweep

```bash
python3 scripts/run_llmbench_aligned_eval.py --pool all --gpus 0,1,2,3,4,5,6,7
```

结果写入 `outputs/eval/llmbench_aligned_eval.json` 与 `.md`。

### Fallback 阈值校准

需要 `train/response_embeddings.*`（Channel B）：

```bash
python3 scripts/calibrate_fallback.py \
  --data-dir data/seed42_small \
  --checkpoint outputs/checkpoints/seed42_small/best.pt
```

### Leave-one-model-out

```bash
python3 scripts/run_leave_one_model_out.py \
  --data-dir data/seed42_small \
  --held-out-model Qwen3-8B \
  --output-dir outputs/lomo/qwen3
```

## 从头重建数据（可选）

若接收方**没有**预构建 `data/seed42_*`，需要：

1. 提供 `llmbenchmark/results/bench/` 原始 bench JSON（~6.6GB，默认 symlink 到外部路径）
2. 配置 embedding API（`configs/embedding.yaml` + 本地密钥文件，**勿上传**）

```bash
# 构建 reward 矩阵
python3 scripts/build_rewards.py \
  --config configs/small_subset.yaml \
  --output-dir data/seed42_small

# 预计算 embedding（需 API 或本地 embedding 模型）
python3 scripts/build_embeddings.py --data-dir data/seed42_small

# 压缩 embedding
python3 scripts/compress_embeddings.py --data-dirs data/seed42_small
```

使用预构建数据时**不需要** bench JSON 与 embedding API。

## 上传与打包

最小可复现包（约 **3.5 GB**）应包含：

| 类别 | 路径 |
|------|------|
| 代码 | `src/`, `scripts/`, `configs/`（不含 `*.local.yaml`）, `tests/`, `docs/` |
| 数据 | `data/seed42_small/`, `data/seed42_flagship/`（`.f16.npz` + `meta.json` + `grouped.npz`） |
| 权重 | `outputs/checkpoints/seed42_*/best.pt` |
| 配置引用 | `llmbenchmark/config/`, `llmbenchmark/baselines/` |

可省略：`*.pth` 原始 embedding（有 `.f16.npz` 时）、`data/seed42_exp/`、`outputs/eval/`、`llmbenchmark/results/bench/`（不重建数据时）。

详见 [docs/DATA_MANIFEST.md](docs/DATA_MANIFEST.md) 与 [docs/UPLOAD_NOTES.md](docs/UPLOAD_NOTES.md)。

## 与 LLMRouterBench 的关系

- 配置与 bench 入口内置在 `llmbenchmark/`（见 `llmbenchmark/README.md`）
- DuoRoute **不 import** LLMRouterBench 的 SPO / EmbedLLM 等 baseline 训练代码
- 评估指标与 `run_llmbench_aligned_eval.py` 对齐；AvengersPro 路由通过 `run_avengerspro_cached.py` 使用本地 embedding 复现

## 文档

| 文档 | 内容 |
|------|------|
| [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) | 完整目录树、脚本-数据依赖 |
| [docs/DATA_MANIFEST.md](docs/DATA_MANIFEST.md) | 上传文件清单与体积 |
| [docs/UPLOAD_NOTES.md](docs/UPLOAD_NOTES.md) | 打包注意事项 |
