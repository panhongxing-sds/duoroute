# 上传说明

## 大文件处理

| 类型 | 模式 | 建议 |
|------|------|------|
| Embedding | `*.f16.npz` | 上传压缩版；省略对应 `*.pth` |
| Reward 文本 | `*/meta.json` | **必须上传**（占大部分体积） |
| Checkpoint | `outputs/checkpoints/*/best.pt` | 必须上传 |
| Bench 原始数据 | `llmbenchmark/results/bench/` | 默认不上传（6.6GB 外部 symlink） |

## Git LFS（可选）

若使用 Git 管理大文件，可在 `.gitattributes` 中跟踪：

```
*.f16.npz filter=lfs diff=lfs merge=lfs -text
data/**/meta.json filter=lfs diff=lfs merge=lfs -text
outputs/checkpoints/**/best.pt filter=lfs diff=lfs merge=lfs -text
```

当前 `.gitignore` 已忽略 `*.pth`、`*.npz`、`outputs/`；上传 tarball 时不受 gitignore 限制。

## 打包示例

```bash
cd /home/phx
tar -czvf duoroute-upload.tar.gz \
  --exclude='DuoRoute/data/seed42_exp' \
  --exclude='DuoRoute/data/seed42_synth' \
  --exclude='DuoRoute/data/*/*.pth' \
  --exclude='DuoRoute/data/*/*/*.pth' \
  --exclude='DuoRoute/**/__pycache__' \
  --exclude='DuoRoute/configs/*.local.yaml' \
  --exclude='DuoRoute/outputs/avengerspro/**/run.log' \
  --exclude='DuoRoute/outputs/avengerspro/**/nohup.log' \
  DuoRoute/
```

上传前请确认 `data/seed42_small/**/*.f16.npz` 与 `data/seed42_flagship/**/*.f16.npz` 已存在（运行 `python3 scripts/compress_embeddings.py`）。
