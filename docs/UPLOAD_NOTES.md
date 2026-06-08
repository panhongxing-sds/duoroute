# 上传说明（维护者）

## RegretRouter 数据发布流程

1. 打包旗舰池数据：

```bash
REGRETROUTER_DATA_ROOT=/path/to/data bash scripts/package_data.sh
# 输出: data_pack/regretrouter_data_flagship.tar.zst (~200 MB)
```

2. 在 GitHub 创建 Release，标签 **`v1.0.0-data`**，上传 `regretrouter_data_flagship.tar.zst`。

3. 验证下载脚本：

```bash
rm -rf data/seed42_flagship
bash scripts/download_data.sh
python3 tests/test_duoroute.py
```

## 数据包文件清单（15 个）

- `data/seed42_flagship/config.json`, `model_cards.json`, `embedding_meta.json`
- `data/seed42_flagship/question_embeddings.f16.npz`, `model_embeddings.f16.npz`
- `data/seed42_flagship/{train,val,test}/grouped.npz`
- `data/seed42_flagship/{train,val,test}/meta.json.gz`
- `data/seed42_flagship/{train,val,test}/response_embeddings.f16.npz`

**不包含** 原始 `*.pth` 与未压缩 `meta.json`（单文件可超 100MB，不适合 git）。

## 为何不用 Git LFS

GitHub Release tarball 方案更简单：克隆代码快、数据按需下载、无 LFS 配额成本。
