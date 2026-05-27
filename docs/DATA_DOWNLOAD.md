# 数据下载说明

代码仓库不含大体积数据（embedding、meta、checkpoint）。请从 **GitHub Release** 下载数据包。

## 一键下载

```bash
cd DuoRoute
bash scripts/download_data.sh
python3 tests/test_duoroute.py
```

默认从 Release `v0.1.0-data` 下载 `duoroute-data.tar.zst` 并解压到项目根目录。

## 手动下载

1. 打开 [Releases](https://github.com/panhongxing-sds/duoroute/releases)
2. 下载 `duoroute-data.tar.zst`（约 700–800 MB）
3. 解压：

```bash
tar -I zstd -xf duoroute-data.tar.zst -C /path/to/DuoRoute
```

## 数据包内容

| 路径 | 说明 |
|------|------|
| `data/seed42_small/` | 小模型池 split + `.f16.npz` embedding + `meta.json.gz` |
| `data/seed42_flagship/` | 旗舰池 |
| `outputs/checkpoints/seed42_*/best.pt` | 已训练 router |

解压后 `DuoRouteGroupedData.load` 会自动读取 `meta.json.gz`。

## 自行打包（维护者）

```bash
bash scripts/package_data.sh
# 输出默认 ../duoroute-data.tar.zst
```
