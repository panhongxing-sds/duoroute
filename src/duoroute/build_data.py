#!/usr/bin/env python3
from duoroute.data import build_splits_from_bench
from duoroute.utils import load_yaml, project_root, resolve_project_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build DuoRoute grouped data")
    parser.add_argument("--config", default=None, help="bench yaml or DuoRoute runtime yaml (e.g. configs/flagship.yaml)")
    parser.add_argument("--bench-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-cost", type=float, default=0.2)
    parser.add_argument("--cost-mode", choices=["global", "per_query"], default="per_query")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-records-per-dataset", type=int, default=None)
    parser.add_argument("--max-prompts-per-dataset", type=int, default=None)
    parser.add_argument("--repair-zero-cost", action="store_true")
    parser.add_argument("--pricing-config", default=None)
    parser.add_argument("--model-cards", default=None)
    parser.add_argument("--llmbench-graphrouter-config", default=None)
    parser.add_argument("--llmbench-collector-config", default=None)
    args = parser.parse_args()

    default_cfg = load_yaml(project_root() / "configs/default.yaml").get("duoroute", {})
    config_path = args.config or default_cfg.get("bench_config", "llmbenchmark/config/baseline_config.yaml")
    config_path = str(resolve_project_path(config_path))
    raw_cfg = load_yaml(config_path)
    if "duoroute" in raw_cfg:
        duoroute_cfg = {**default_cfg, **raw_cfg.get("duoroute", {})}
        bench_config_path = str(resolve_project_path(duoroute_cfg.get("bench_config", config_path)))
    elif "baseline" in raw_cfg:
        duoroute_cfg = default_cfg
        bench_config_path = config_path
    else:
        duoroute_cfg = default_cfg
        bench_config_path = config_path

    max_records_per_dataset = args.max_records_per_dataset
    if max_records_per_dataset is None:
        max_records_per_dataset = duoroute_cfg.get("max_records_per_dataset")
    max_prompts_per_dataset = args.max_prompts_per_dataset
    if max_prompts_per_dataset is None:
        max_prompts_per_dataset = duoroute_cfg.get("max_prompts_per_dataset")

    build_splits_from_bench(
        config_path=bench_config_path,
        bench_dir=str(resolve_project_path(args.bench_dir or duoroute_cfg.get("bench_dir"))),
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_seed=args.seed,
        lambda_cost=args.lambda_cost,
        cost_mode=args.cost_mode,
        max_records=args.max_records,
        max_records_per_dataset=max_records_per_dataset,
        max_prompts_per_dataset=max_prompts_per_dataset,
        repair_zero_cost=args.repair_zero_cost,
        pricing_config_path=args.pricing_config or duoroute_cfg.get("llmbench_collector_config"),
        model_cards_path=str(resolve_project_path(args.model_cards or duoroute_cfg.get("model_cards"))),
        llmbench_graphrouter_config=str(resolve_project_path(
            args.llmbench_graphrouter_config or duoroute_cfg.get("llmbench_graphrouter_config")
        )),
        llmbench_collector_config=str(resolve_project_path(
            args.llmbench_collector_config or duoroute_cfg.get("llmbench_collector_config")
        )),
    )


if __name__ == "__main__":
    main()
