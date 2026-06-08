# RegretRouter

**RegretRouter** 是面向 LLM 路由的 K=3 递归决策聚焦路由器；**RegretRouter + Cascade** 在其上叠加 Perfect Verifier、Query×Model Selector 与 Top-5 Rerank，用于旗舰池 N=851 主表实验。

本仓库为 **独立复现项目**（与历史 DuoRoute 双通道训练代码分离）。Python 包目录仍为 `src/duoroute/`（内部模块名），对外品牌与 PyPI 包名为 `regretrouter`。

## 快速开始

```bash
git clone https://github.com/panhongxing-sds/duoroute.git RegretRouter
cd RegretRouter
bash scripts/setup.sh
```

`setup.sh` 会：`pip install -e .` → 下载/解压 `seed42_flagship` 数据 → 运行冒烟测试。

数据包（~200–250 MB 压缩）从 [GitHub Releases](https://github.com/panhongxing-sds/duoroute/releases) 获取；详见 [docs/DATA_DOWNLOAD.md](docs/DATA_DOWNLOAD.md)。

## 复现主表（N=851，seeds 41/42/43）

完整协议见 **[docs/REGRETROUTER_REPRODUCTION.md](docs/REGRETROUTER_REPRODUCTION.md)**。

```bash
# 快速冒烟（推荐首次验证）
python3 scripts/run_multiseed_main_tables.py --quick

# 完整主实验：RegretRouter + Cascade，3-seed × 5-λ
python3 scripts/run_multiseed_main_tables.py --seeds 41 42 43 --epochs 28

# SOTA 基线（需本地 LLMRouterBench，可选）
python3 scripts/run_llmrouterbench_flagship.py

# 导出与汇总论文表
python3 scripts/export_main_tables.py
python3 scripts/build_final_paper_table.py

# RegretRouter benchmark / 机制消融
python3 scripts/benchmark_r_dfl_ap.py --full-test
python3 scripts/run_mechanism_ablations.py --seeds 41 42 43 --epochs 28
```

**主表指标**：AvgAcc、Gain@R、Gain@B、Gap@O、Regret@O、AvgCost（raw perf utility，LLMRouterBench §3.2 对齐）。

## 实验数据

| Pool | 路径 | 模型数 | test |
|------|------|--------|------|
| **旗舰池（主表）** | `data/seed42_flagship/` | 13 | **851** |

每个 pool 含 `{train,val,test}/grouped.npz`、`meta.json.gz`、`.f16.npz` 压缩 embedding 等（无 >100MB 单文件，适配 GitHub Release）。

## 项目结构

```text
RegretRouter/
├── src/duoroute/
│   ├── regretrouter.py              # RegretRouter 训练/推理（canonical）
│   ├── pool_data.py                 # N=851 数据加载
│   ├── main_table.py                # 主表导出共享逻辑
│   ├── reward_builder.py            # Oracle / cascade utility
│   ├── bench_metrics.py             # LLMRouterBench 对齐指标
│   ├── rdf_router.py                # RecursiveDFLRouter 核心
│   ├── rdf_query_model_selector.py
│   ├── rdf_vg_cascade.py
│   └── rdf_cascade_decomp.py
├── scripts/
│   ├── setup.sh                     # ★ 一键安装 + 数据
│   ├── download_data.sh             # 从 Release 拉取数据
│   ├── package_data.sh              # 维护者打包数据
│   ├── run_multiseed_main_tables.py # ★ 主表入口
│   └── ...
├── data_pack/                       # 本地数据包（不提交 git）
├── docs/REGRETROUTER_REPRODUCTION.md
└── tests/test_duoroute.py
```

## 输出

| 路径 | 内容 |
|------|------|
| `outputs/cascade/TABLES_DATA.json` | 3-seed 主表数据 |
| `outputs/cascade/TABLES_LATEX.tex` | LaTeX 主表 |
| `outputs/regretrouter/` | RegretRouter benchmark 报告 |

## 维护者：发布数据包

```bash
# 从本地 data/ 或 HDD 路径打包
REGRETROUTER_DATA_ROOT=/path/to/data bash scripts/package_data.sh
# 上传 data_pack/regretrouter_data_flagship.tar.zst 到 GitHub Release v1.0.0-data
```

详见 [docs/UPLOAD_NOTES.md](docs/UPLOAD_NOTES.md)。
