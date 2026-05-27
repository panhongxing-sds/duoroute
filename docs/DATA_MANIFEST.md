# DuoRoute 上传数据清单

生成日期：2026-05-27  
路径根目录：`/home/phx/DuoRoute`

本文档列出**复现已发布实验**应包含的文件。大小为 `du -sh` 快照（含已生成的压缩 embedding）。

---

## 总体积估算

| 打包方案 | 约计大小 | 说明 |
|----------|----------|------|
| **最小可复现包** | **~3.5 GB** | 代码 + seed42_small/flagship（含 meta.json）+ `.f16.npz` + checkpoint，**不含**原始 `.pth` embedding |
| 含原始 `.pth` embedding | ~4.6 GB | 额外 +1.1 GB float32 embedding |
| 含 `data/seed42_exp` | +2.3 GB | 非必需 |
| 含 `llmbenchmark/results/bench` | +6.6 GB | 仅重建数据时需要 |

---

## 1. 代码与配置（必需，~1 MB）

| 路径 | 大小 | 必需 |
|------|------|------|
| `src/duoroute/` | 300K | ✓ |
| `scripts/` | 136K | ✓ |
| `configs/`（除 `*.local.yaml`） | 32K | ✓ |
| `tests/` | 8K | ✓ |
| `llmbenchmark/config/` | 76K | ✓ |
| `llmbenchmark/baselines/` | 24K | ✓ |
| `llmbenchmark/README*.md` | 24K | ✓ |
| `pyproject.toml`, `requirements.txt`, `README.md` | 16K | ✓ |
| `docs/PROJECT_STRUCTURE.md`, `docs/DATA_MANIFEST.md` | — | ✓ |

---

## 2. `data/seed42_small`（必需，上传压缩版 ~1.8 GB）

主实验 pool：8B 小模型，10 数据集。被 `train_duoroute.py`、`eval_unified_baselines.py`、`run_llmbench_aligned_eval.py` 等引用。

| 路径 | 大小 | 上传 | 用途 |
|------|------|------|------|
| `train/meta.json` | 1.2G | ✓ | prompt/response 文本、模型列表 |
| `test/meta.json` | 327M | ✓ | 测试 split 文本 |
| `val/meta.json` | 161M | ✓ | 验证 split 文本 |
| `train/response_embeddings.f16.npz` | 88M | ✓ | Channel B 训练 |
| `question_embeddings.f16.npz` | 31M | ✓ | query embedding |
| `model_embeddings.f16.npz` | 76K | ✓ | model-card embedding |
| `train/grouped.npz` | 788K | ✓ | reward/performance 矩阵 |
| `test/grouped.npz` | 232K | ✓ | |
| `val/grouped.npz` | 116K | ✓ | |
| `config.json`, `model_cards.json`, `embedding_meta.json` | 24K | ✓ | 元数据 |
| `train/response_embeddings.pth` | 909M | 可选 | 有 `.f16.npz` 可省略 |
| `question_embeddings.pth` | 65M | 可选 | 有 `.f16.npz` 可省略 |
| `model_embeddings.pth` | 164K | 可选 | 有 `.f16.npz` 可省略 |
| `build_*.log` | ~150K | 可选 | 构建日志 |

---

## 3. `data/seed42_flagship`（必需，上传压缩版 ~650 MB）

旗舰 performance-cost pool（13 模型）。评估脚本与 seed42_small 并列使用。

| 路径 | 大小 | 上传 | 用途 |
|------|------|------|------|
| `train/meta.json` | 220M | ✓ | |
| `test/meta.json` | 65M | ✓ | |
| `val/meta.json` | 32M | ✓ | |
| `train/response_embeddings.f16.npz` | 87M | ✓ | |
| `question_embeddings.f16.npz` | 16M | ✓ | |
| `model_embeddings.f16.npz` | 52K | ✓ | |
| `train/test/val/grouped.npz` | ~560K | ✓ | |
| `config.json`, `model_cards.json`, `embedding_meta.json`, `duoroute_config.yaml` | 28K | ✓ | |
| `*.pth` embedding | ~336M | 可选 | 有 `.f16.npz` 可省略 |
| `build_*.log` | ~200K | 可选 | |

---

## 4. Checkpoints（必需，~200 MB）

| 路径 | 大小 | 必需 | 用途 |
|------|------|------|------|
| `outputs/checkpoints/seed42_small/best.pt` | 115M | ✓ | 已训练 router |
| `outputs/checkpoints/seed42_small/results.json` | 20K | 推荐 | 训练指标 |
| `outputs/checkpoints/seed42_flagship/best.pt` | 83M | ✓ | |
| `outputs/checkpoints/seed42_flagship/results.json` | 20K | 推荐 | |
| `test_pred_*.npy`, `train.log` | ~180K | 可选 | 预测缓存/日志 |

---

## 5. 评估产物（可选，可重新生成）

| 路径 | 大小 | 说明 |
|------|------|------|
| `outputs/eval/*.json`, `*.md` | 64K | `eval_unified_baselines.py`, `run_llmbench_aligned_eval.py` 输出 |
| `outputs/avengerspro/seed42_*/duoroute_unified/` | ~170M | AvengersPro cached eval 的 JSONL 导出 |
| `outputs/avengerspro/**/run.log`, `nohup.log` | 可变 | 可省略 |

---

## 6. 建议排除（不影响复现）

| 路径 | 大小 | 原因 |
|------|------|------|
| `data/seed42_exp/` | 2.3G | **无任何脚本引用**；早期实验副本 |
| `data/seed42_synth/` | 430M | 合成 smoke 数据；`tests/` 可独立运行 |
| `data/*/question_embeddings.pth` 等 | 1.1G | 已有 `.f16.npz`；可用 `decompress_embeddings.py` 恢复 |
| `configs/embedding_api.local.yaml` | — | 含 API 密钥 |
| `src/**/__pycache__`, `*.egg-info` | — | 运行时产物 |
| `llmbenchmark/results/bench/` | 6.6G | 外部 symlink；使用预构建 `data/seed42_*` 时不需要 |

⚠️ **排除 `meta.json` 或 `grouped.npz` 将导致无法训练/评估。**  
⚠️ **排除 `.f16.npz` 且无 `.pth` 时将回退到 hash embedding（仅 smoke 测试质量）。**

---

## 7. Embedding 压缩对照表

| 文件 | 原始 (.pth) | 压缩 (.f16.npz) | 压缩比 |
|------|-------------|-----------------|--------|
| seed42_small/question_embeddings | 65M | 31M | 2.16× |
| seed42_small/model_embeddings | 164K | 76K | 2.16× |
| seed42_small/train/response_embeddings | 909M | 88M | **10.4×** |
| seed42_flagship/question_embeddings | 34M | 16M | 2.16× |
| seed42_flagship/model_embeddings | 108K | 52K | 2.18× |
| seed42_flagship/train/response_embeddings | 302M | 87M | 3.5× |
| **合计** | **~1.31 GB** | **~230 MB** | **~5.7×** |

格式：`float16` 数组 + NPZ gzip；加载后转 `float32`。  
验证：`python3 scripts/compress_embeddings.py` 后运行 encoders 加载，max abs diff ≤ 6e-5。

---

## 8. 外部依赖数据

| 资源 | 位置 | 何时需要 |
|------|------|----------|
| LLMRouterBench bench JSON | `llmbenchmark/results/bench/` → `/home/phx/LLMRouterBench/results/bench` | 运行 `scripts/build_rewards.py` **从头构建** data 时 |
| Embedding API / GPU | `configs/embedding.yaml` | 运行 `scripts/build_embeddings.py` 重新编码时 |

使用本仓库预构建的 `data/seed42_small` 与 `data/seed42_flagship` **不需要** bench JSON。

---

## 9. 快速校验

```bash
# 确认压缩 embedding 可加载
python3 -c "
from pathlib import Path; import sys; sys.path.insert(0,'src')
from duoroute.embedding_io import load_embedding_tensor, resolve_embedding_path
for d in ['data/seed42_small','data/seed42_flagship']:
    p = resolve_embedding_path(f'{d}/question_embeddings.pth')
    t = load_embedding_tensor(p)
    print(d, p.name, tuple(t.shape))
"
```

预期输出包含 `question_embeddings.f16.npz` 与 shape `(8307, 2048)` / `(4233, 2048)`。
