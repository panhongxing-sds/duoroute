# 数据下载说明

代码仓库不含大体积数据（embedding、meta）。请从 **GitHub Release** 下载旗舰池数据包。

## 一键安装（推荐）

```bash
cd RegretRouter
bash scripts/setup.sh
```

## 仅下载数据

```bash
bash scripts/download_data.sh
python3 tests/test_duoroute.py
```

默认从 Release `v1.0.0-data` 下载 `regretrouter_data_flagship.tar.zst` 并解压到项目根目录。

环境变量（可选）：

| 变量 | 默认 | 说明 |
|------|------|------|
| `REGRETROUTER_DATA_TAG` | `v1.0.0-data` | Release 标签 |
| `REGRETROUTER_DATA_REPO` | `panhongxing-sds/duoroute` | GitHub 仓库 |

## 手动下载

1. 打开 [Releases](https://github.com/panhongxing-sds/duoroute/releases)
2. 下载 `regretrouter_data_flagship.tar.zst`（约 200–250 MB）
3. 解压：

```bash
tar -I zstd -xf regretrouter_data_flagship.tar.zst -C /path/to/RegretRouter
```

## 数据包内容（seed42_flagship）

| 路径 | 说明 |
|------|------|
| `data/seed42_flagship/{train,val,test}/grouped.npz` | 性能/成本/utility 矩阵 |
| `data/seed42_flagship/{train,val,test}/meta.json.gz` | 元数据（自动解压读取） |
| `data/seed42_flagship/*/response_embeddings.f16.npz` | 压缩 response embedding |
| `data/seed42_flagship/question_embeddings.f16.npz` | query embedding |
| `data/seed42_flagship/model_embeddings.f16.npz` | model card embedding |
| `data/seed42_flagship/model_cards.json` | 模型卡片 |

解压后 `DuoRouteGroupedData.load` 会自动读取 `meta.json.gz`；`embedding_io` 优先加载 `.f16.npz`。

## 自行打包（维护者）

```bash
# 若数据在 HDD 上
REGRETROUTER_DATA_ROOT=/HDDDATA/phx/DuoRoute/data bash scripts/package_data.sh
# 输出: data_pack/regretrouter_data_flagship.tar.zst
```

上传至 GitHub Release，标签建议 `v1.0.0-data`。
