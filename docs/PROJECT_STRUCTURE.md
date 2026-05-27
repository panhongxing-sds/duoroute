# DuoRoute 项目结构

本文档说明 `/home/phx/DuoRoute` 的目录布局、各目录用途，以及如何训练/评估与加载压缩 embedding。

## 目录树

```text
DuoRoute/
├── configs/                         # 训练/评估/embedding 配置
│   ├── default.yaml                 # 默认训练超参
│   ├── small_subset.yaml            # seed42_small 实验配置
│   ├── flagship.yaml                # seed42_flagship 实验配置
│   ├── embedding.yaml               # Qwen3-Embedding 构建配置
│   ├── model_cards.yaml             # 模型描述（zero-shot 泛化）
│   └── embedding_api.local.yaml     # 本地 API 密钥（不上传）
│
├── src/duoroute/                    # 核心 Python 包
│   ├── bench_loader.py              # 读取 LLMRouterBench JSON
│   ├── build_data.py                # 从 bench 构建 grouped reward 数据
│   ├── build_embeddings.py          # 预计算 query/response/model embedding
│   ├── data.py                      # DuoRouteGroupedData（grouped.npz + meta.json）
│   ├── embedding_io.py              # 压缩/解压 embedding（.f16.npz）
│   ├── encoders.py                  # embedding 加载与编码
│   ├── reward_builder.py            # Oracle reward 矩阵
│   ├── model.py / losses.py         # 双通道 router
│   ├── trainer.py / train.py        # 训练循环
│   ├── inference.py / evaluator.py  # 推理与评估
│   ├── prompt_ids.py                # 全局 query → embedding 行号映射
│   └── zero_shot.py / calibration.py
│
├── scripts/                         # CLI 入口
│   ├── build_rewards.py             # 构建 data/seed42_* 原始 split
│   ├── build_embeddings.py          # 构建 embedding（调用 duoroute.build_embeddings）
│   ├── compress_embeddings.py       # float32 .pth → float16 .f16.npz
│   ├── decompress_embeddings.py     # .f16.npz → float32 .pth
│   ├── train_duoroute.py            # 训练
│   ├── eval_duoroute.py             # query-only 评估
│   ├── calibrate_fallback.py        # fallback 阈值校准（需 response embedding）
│   ├── eval_unified_baselines.py    # SingleBest / DuoRoute / AvengersPro 统一基线
│   ├── run_llmbench_aligned_eval.py   # LLMRouterBench 对齐三表评估
│   ├── run_avengerspro_cached.py    # 使用本地 question embedding 的 AvengersPro
│   ├── build_comparison_tables.py   # 性能/均衡对比表
│   └── run_leave_one_model_out.py   # LOMO 实验
│
├── data/                            # 预构建实验数据（见 DATA_MANIFEST.md）
│   ├── seed42_small/                # ★ 必需：8B 小模型池，10 数据集
│   ├── seed42_flagship/             # ★ 必需：performance-cost 旗舰池
│   ├── seed42_exp/                  # 可选：早期实验副本，脚本未引用
│   └── seed42_synth/                # 可选：合成 smoke 数据
│
├── outputs/                         # 训练/评估产物（.gitignore，上传时按需包含）
│   ├── checkpoints/seed42_small/    # ★ 必需：已训练 checkpoint
│   ├── checkpoints/seed42_flagship/ # ★ 必需：已训练 checkpoint
│   ├── eval/                        # 可选：评估 JSON/MD（可重新生成）
│   └── avengerspro/                 # 可选：AvengersPro 缓存 eval 导出
│
├── llmbenchmark/                    # 内置 LLMRouterBench 配置
│   ├── config/                      # baseline_config_*.yaml
│   ├── baselines/GraphRouter/       # 模型描述 adaptor 配置
│   └── results/bench/               # → 外部 symlink（~6.6GB 原始 bench JSON）
│
├── tests/test_duoroute.py
├── docs/                            # 结构与数据清单
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 数据目录结构（以 `data/seed42_small/` 为例）

```text
seed42_small/
├── config.json                      # 构建参数（bench 配置路径、split 比例等）
├── model_cards.json                 # 该 pool 的模型描述
├── embedding_meta.json              # embedding 构建元信息（backend、维度、response 统计）
├── question_embeddings.pth          # 原始 float32（上传可省略，见压缩版）
├── question_embeddings.f16.npz      # ★ 压缩版 query embedding（优先加载）
├── model_embeddings.pth / .f16.npz  # model-card embedding
├── train/
│   ├── grouped.npz                  # [N,K] reward/performance/cost/mask 数组
│   ├── meta.json                    # prompt/response 文本与模型列表（体积大，必需）
│   └── response_embeddings.pth / .f16.npz  # Channel B 训练用
├── val/   {grouped.npz, meta.json}
└── test/  {grouped.npz, meta.json}
```

## 必需 vs 可选数据

| 类别 | 路径 | 说明 |
|------|------|------|
| **必需** | `data/seed42_small/`, `data/seed42_flagship/` | 训练与全部 eval 脚本使用的两个 pool |
| **必需** | 各 split 的 `grouped.npz` + `meta.json` | reward 矩阵与 prompt/response 文本 |
| **必需** | `*.f16.npz` 或 `*.pth` embedding | 至少保留一种；推荐仅上传 `.f16.npz` |
| **必需** | `outputs/checkpoints/seed42_*/best.pt` | 已发布实验的 router 权重 |
| **可选** | `data/seed42_exp/` (~2.3GB) | 无脚本引用，可不上传 |
| **可选** | `data/seed42_synth/` (~430MB) | 单元/smoke 测试 |
| **可选** | `*.pth` 原始 embedding | 有 `.f16.npz` 时可不上传（节省 ~1.1GB） |
| **可选** | `data/*/build_*.log` | 构建日志 |
| **可选** | `outputs/eval/`, `outputs/avengerspro/` | 可重新运行脚本生成 |
| **外部** | `llmbenchmark/results/bench/` | 仅 **重新构建** `data/seed42_*` 时需要（~6.6GB） |

## 压缩 embedding

### 格式

- 后缀：`.f16.npz`（NumPy `savez_compressed`，内部 float16 + gzip）
- 加载器（`embedding_io.py` / `encoders.py`）**自动优先**加载 `.f16.npz`，否则回退 `.pth`
- 加载后统一转为 float32，训练/评估行为与原始一致（float16 最大误差 ~6e-5）

### 压缩

```bash
python3 scripts/compress_embeddings.py
# 或指定目录
python3 scripts/compress_embeddings.py --data-dirs data/seed42_small data/seed42_flagship
```

### 解压（如需恢复原始 .pth）

```bash
python3 scripts/decompress_embeddings.py
python3 scripts/decompress_embeddings.py --overwrite
```

## 运行流程

### 安装

```bash
cd DuoRoute
pip install -e .
pip install -e ".[embeddings]"   # 仅重新构建 embedding 时需要
```

### 训练

```bash
python3 scripts/train_duoroute.py \
  --config configs/small_subset.yaml \
  --data-dir data/seed42_small \
  --output-dir outputs/checkpoints/seed42_small
```

### 评估

```bash
# Query-only
python3 scripts/eval_duoroute.py \
  --data-dir data/seed42_small \
  --checkpoint outputs/checkpoints/seed42_small/best.pt

# 统一基线（SingleBest / DuoRoute / AvengersPro cached）
python3 scripts/eval_unified_baselines.py \
  --data-dir data/seed42_small \
  --checkpoint outputs/checkpoints/seed42_small/best.pt

# LLMRouterBench 对齐三表
python3 scripts/run_llmbench_aligned_eval.py --pools all
```

### Fallback 校准（需要 train response embedding）

```bash
python3 scripts/calibrate_fallback.py \
  --data-dir data/seed42_small \
  --checkpoint outputs/checkpoints/seed42_small/best.pt
```

## 脚本引用的数据文件

| 组件 | 读取的文件 |
|------|-----------|
| `trainer.py` | `{data_dir}/question_embeddings.*`, `model_embeddings.*`, `train/response_embeddings.*`, `{split}/grouped.npz`, `meta.json` |
| `eval_unified_baselines.py` | 同上 + `best.pt` |
| `run_llmbench_aligned_eval.py` | seed42_small + seed42_flagship 全套 |
| `run_avengerspro_cached.py` | `question_embeddings.*`, `prompt_ids` 映射 |
| `calibrate_fallback.py` | query/model/response embedding + val/test grouped 数据 |

## 上传建议

1. **包含** `src/`, `scripts/`, `configs/`, `tests/`, `llmbenchmark/config`, `llmbenchmark/baselines`
2. **包含** `data/seed42_small`, `data/seed42_flagship`（embedding 用 `.f16.npz`，可省略 `.pth`）
3. **包含** `outputs/checkpoints/seed42_small`, `outputs/checkpoints/seed42_flagship`
4. **不包含** `configs/embedding_api.local.yaml`、API 密钥、`__pycache__`
5. **bench 原始 JSON**（6.6GB）仅在接收方需要从头构建数据时才需提供；使用预构建 `data/seed42_*` 则不需要

详见 [DATA_MANIFEST.md](./DATA_MANIFEST.md)。
