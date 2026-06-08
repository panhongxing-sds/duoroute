#!/usr/bin/env python3
from duoroute.trainer import TrainConfig, train_duoroute
from duoroute.utils import load_yaml, project_root


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train DuoRoute")
    parser.add_argument("--config", default=str(project_root() / "configs/default.yaml"))
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    cfg_yaml = load_yaml(args.config).get("duoroute", {})
    train_cfg = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=int(cfg_yaml.get("epochs", 30)),
        batch_size=int(cfg_yaml.get("batch_size", 64)),
        lr=float(cfg_yaml.get("lr", 1e-3)),
        hidden_dim=int(cfg_yaml.get("hidden_dim", 64)),
        embed_dim=int(cfg_yaml.get("embed_dim", 256)),
        alpha=float(cfg_yaml.get("alpha", 0.5)),
        beta=float(cfg_yaml.get("beta", 0.1)),
        temperature=float(cfg_yaml.get("temperature", 0.5)),
        reward_target=str(cfg_yaml.get("reward_target", "oracle_reward")),
        distill_warmup_epochs=int(cfg_yaml.get("distill_warmup_epochs", 3)),
        seed=int(cfg_yaml.get("seed", 42)),
        max_samples=args.max_samples if args.max_samples is not None else cfg_yaml.get("max_samples"),
        use_id_fallback=bool(cfg_yaml.get("use_id_fallback", False)),
        model_cards_path=cfg_yaml.get("model_cards"),
        llmbench_graphrouter_config=cfg_yaml.get("llmbench_graphrouter_config"),
        llmbench_collector_config=cfg_yaml.get("llmbench_collector_config"),
    )
    train_duoroute(train_cfg, config_path=args.config)


if __name__ == "__main__":
    main()
