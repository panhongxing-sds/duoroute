"""Grouped DuoRoute tensors, splits, and reward matrix export."""

from __future__ import annotations

import gzip
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from loguru import logger

from duoroute.model_cards import cards_for_models, load_model_cards
from duoroute.reward_builder import rebuild_grouped_rewards
from duoroute.schema import BenchRecord


def _prompt_key(record: BenchRecord) -> str:
    text = record.prompt or record.origin_query
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
    return f"{record.dataset_id}::{record.split}::{digest}"


def _serialize_raw_output(raw) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    return str(raw)


def split_by_dataset_then_prompt(
    records: List[BenchRecord],
    train_ratio: float,
    random_seed: int = 42,
    ood_datasets: Optional[List[str]] = None,
) -> Tuple[List[BenchRecord], List[BenchRecord]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")

    random.seed(random_seed)
    ood = set(ood_datasets or [])
    by_dataset: Dict[str, List[BenchRecord]] = defaultdict(list)
    for record in records:
        by_dataset[record.dataset_id].append(record)

    train_records: List[BenchRecord] = []
    test_records: List[BenchRecord] = []
    for dataset_id, dataset_records in by_dataset.items():
        if dataset_id in ood:
            test_records.extend(dataset_records)
            continue
        prompt_to_records: Dict[str, List[BenchRecord]] = defaultdict(list)
        for record in dataset_records:
            prompt_to_records[record.prompt].append(record)
        unique_prompts = list(prompt_to_records.keys())
        unique_prompts.sort(key=lambda p: min(r.record_index for r in prompt_to_records[p]))
        n_train = int(len(unique_prompts) * train_ratio)
        indices = list(range(len(unique_prompts)))
        random.shuffle(indices)
        train_idx = set(indices[:n_train])
        for idx, prompt in enumerate(unique_prompts):
            bucket = train_records if idx in train_idx else test_records
            bucket.extend(prompt_to_records[prompt])

    logger.info(f"Split: {len(train_records)} train / {len(test_records)} test records")
    return train_records, test_records


def _load_model_pricing(config_path: Optional[str]) -> Dict[str, Tuple[float, float]]:
    if not config_path:
        return {}
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    pricing: Dict[str, Tuple[float, float]] = {}
    for model_cfg in cfg.get("models", []):
        name = model_cfg.get("name")
        prices = model_cfg.get("pricing") or {}
        if not name or not prices:
            continue
        pricing[name] = (
            float(prices.get("prompt_price_per_million", 0.0) or 0.0),
            float(prices.get("completion_price_per_million", 0.0) or 0.0),
        )
    return pricing


def repair_zero_cost_records(
    records: Sequence[BenchRecord],
    pricing: Dict[str, Tuple[float, float]],
) -> Dict[str, int]:
    stats = {"zero_cost_before": 0, "zero_cost_after": 0, "repaired": 0}
    for record in records:
        if float(record.cost or 0.0) != 0.0:
            continue
        stats["zero_cost_before"] += 1
        prompt_tokens = int(record.prompt_tokens or 0)
        completion_tokens = int(record.completion_tokens or 0)
        if prompt_tokens <= 0 and completion_tokens <= 0:
            continue
        if record.model_name not in pricing:
            continue
        prompt_price, completion_price = pricing[record.model_name]
        estimate = (prompt_tokens * prompt_price + completion_tokens * completion_price) / 1_000_000.0
        if estimate > 0:
            record.cost = float(estimate)
            stats["repaired"] += 1
    stats["zero_cost_after"] = sum(1 for r in records if float(r.cost or 0.0) == 0.0)
    return stats


@dataclass
class DuoRouteGroupedData:
    model_names: List[str]
    prompt_keys: List[str]
    dataset_ids: List[str]
    performance: np.ndarray
    cost: np.ndarray
    utility: np.ndarray
    oracle_reward: np.ndarray
    mask: np.ndarray
    prompt_texts: List[str]
    response_texts: List[List[str]]
    model_cards: Optional[List[dict]] = None
    oracle_reward_exp: Optional[np.ndarray] = None

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path / "grouped.npz",
            performance=self.performance,
            cost=self.cost,
            utility=self.utility,
            oracle_reward=self.oracle_reward,
            mask=self.mask,
        )
        meta = {
            "model_names": self.model_names,
            "prompt_keys": self.prompt_keys,
            "dataset_ids": self.dataset_ids,
            "prompt_texts": self.prompt_texts,
            "response_texts": self.response_texts,
            "model_cards": self.model_cards or [],
        }
        with open(path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "DuoRouteGroupedData":
        path = Path(path)
        arrays = np.load(path / "grouped.npz")
        gz_meta = path / "meta.json.gz"
        if gz_meta.exists():
            with gzip.open(gz_meta, "rt", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            with open(path / "meta.json", encoding="utf-8") as f:
                meta = json.load(f)
        oracle = arrays["oracle_reward"] if "oracle_reward" in arrays else arrays["utility"]
        oracle_exp = arrays["oracle_reward_exp"] if "oracle_reward_exp" in arrays else None
        response_texts = meta.get("response_texts")
        if response_texts is None:
            n, k = arrays["performance"].shape
            response_texts = [[""] * k for _ in range(n)]
        return cls(
            model_names=meta["model_names"],
            prompt_keys=meta["prompt_keys"],
            dataset_ids=meta["dataset_ids"],
            performance=arrays["performance"],
            cost=arrays["cost"],
            utility=arrays["utility"],
            oracle_reward=oracle,
            mask=arrays["mask"],
            prompt_texts=meta["prompt_texts"],
            response_texts=response_texts,
            model_cards=meta.get("model_cards"),
            oracle_reward_exp=oracle_exp,
        )


def records_to_grouped(
    records: Sequence[BenchRecord],
    model_names: Optional[Sequence[str]] = None,
    *,
    lambda_cost: float = 0.2,
    cost_mode: str = "per_query",
    cost_penalty: str = "linear",
    model_cards: Optional[Dict[str, dict]] = None,
) -> DuoRouteGroupedData:
    models = sorted({r.model_name for r in records}) if model_names is None else list(model_names)
    model_to_idx = {name: idx for idx, name in enumerate(models)}
    k = len(models)

    groups: Dict[str, Dict] = defaultdict(
        lambda: {"dataset_id": "", "prompt_text": "", "perf": {}, "cost": {}, "responses": {}}
    )
    for record in records:
        if record.model_name not in model_to_idx:
            continue
        key = _prompt_key(record)
        group = groups[key]
        group["dataset_id"] = record.dataset_id
        group["prompt_text"] = record.prompt or record.origin_query
        group["perf"][record.model_name] = float(record.score if record.score is not None else 0.0)
        group["cost"][record.model_name] = float(record.cost if record.cost is not None else 0.0)
        group["responses"][record.model_name] = _serialize_raw_output(record.raw_output)

    prompt_keys = sorted(groups.keys())
    n = len(prompt_keys)
    performance = np.full((n, k), 0.0, dtype=np.float32)
    cost = np.zeros((n, k), dtype=np.float32)
    mask = np.zeros((n, k), dtype=bool)
    dataset_ids: List[str] = []
    prompt_texts: List[str] = []
    response_texts: List[List[str]] = []

    for i, key in enumerate(prompt_keys):
        group = groups[key]
        dataset_ids.append(group["dataset_id"])
        prompt_texts.append(group["prompt_text"])
        row_responses = [""] * k
        for model_name, score in group["perf"].items():
            j = model_to_idx[model_name]
            performance[i, j] = score
            cost[i, j] = group["cost"][model_name]
            mask[i, j] = True
            row_responses[j] = group["responses"].get(model_name, "")
        response_texts.append(row_responses)

    oracle_reward = rebuild_grouped_rewards(
        performance,
        cost,
        lambda_cost=lambda_cost,
        cost_mode=cost_mode,  # type: ignore[arg-type]
        cost_penalty=cost_penalty,  # type: ignore[arg-type]
    )
    card_dicts = []
    if model_cards:
        card_dicts = [
            {
                "name": c.name,
                "feature": c.feature,
                "input_price": c.input_price,
                "output_price": c.output_price,
            }
            for c in cards_for_models(models, model_cards)
        ]

    return DuoRouteGroupedData(
        model_names=models,
        prompt_keys=prompt_keys,
        dataset_ids=dataset_ids,
        performance=performance,
        cost=cost,
        utility=oracle_reward,
        oracle_reward=oracle_reward,
        mask=mask,
        prompt_texts=prompt_texts,
        response_texts=response_texts,
        model_cards=card_dicts or None,
    )


def cap_prompts_per_dataset(
    records: Sequence[BenchRecord],
    *,
    max_prompts_per_dataset: Optional[int] = None,
    max_records_per_dataset: Optional[int] = None,
    random_seed: int = 42,
) -> List[BenchRecord]:
    """Subsample by whole prompts; only keep questions with a full K-arm response set."""
    if not records:
        return []
    if max_prompts_per_dataset is None and max_records_per_dataset is None:
        return list(records)

    by_dataset: Dict[str, Dict[str, List[BenchRecord]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_dataset[record.dataset_id][record.prompt].append(record)

    rng = random.Random(random_seed)
    kept: List[BenchRecord] = []
    stats: Dict[str, dict[str, int]] = {}
    for dataset_id, prompts_map in sorted(by_dataset.items()):
        group_sizes = [len(group) for group in prompts_map.values()]
        if not group_sizes:
            continue
        expected_k = max(group_sizes)
        complete_prompts = [prompt for prompt, group in prompts_map.items() if len(group) == expected_k]
        incomplete_prompts = len(prompts_map) - len(complete_prompts)

        prompt_limit = max_prompts_per_dataset
        if max_records_per_dataset is not None:
            derived = max(1, max_records_per_dataset // max(expected_k, 1))
            prompt_limit = derived if prompt_limit is None else min(prompt_limit, derived)

        if prompt_limit is None or len(complete_prompts) <= prompt_limit:
            selected_prompts = complete_prompts
        else:
            rng.shuffle(complete_prompts)
            selected_prompts = complete_prompts[:prompt_limit]

        selected_records = [record for prompt in selected_prompts for record in prompts_map[prompt]]
        kept.extend(selected_records)
        stats[dataset_id] = {
            "prompts_total": len(prompts_map),
            "prompts_complete": len(complete_prompts),
            "prompts_incomplete_skipped": incomplete_prompts,
            "prompts_kept": len(selected_prompts),
            "records_kept": len(selected_records),
            "models_per_prompt": expected_k,
        }

    logger.info(
        f"Capped by prompt groups "
        f"(max_prompts={max_prompts_per_dataset}, max_records={max_records_per_dataset}): {stats}"
    )
    return kept


def build_splits_from_bench(
    *,
    config_path: str,
    output_dir: str,
    bench_dir: Optional[str] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    random_seed: int = 42,
    ood_datasets: Optional[List[str]] = None,
    lambda_cost: float = 0.2,
    cost_mode: str = "per_query",
    max_records: Optional[int] = None,
    max_prompts_per_dataset: Optional[int] = None,
    max_records_per_dataset: Optional[int] = None,
    repair_zero_cost: bool = False,
    pricing_config_path: Optional[str] = None,
    model_cards_path: Optional[str] = None,
    llmbench_graphrouter_config: Optional[str] = None,
    llmbench_collector_config: Optional[str] = None,
) -> Dict[str, DuoRouteGroupedData]:
    from duoroute.bench_loader import BenchLoader

    loader = BenchLoader(config_path=config_path, results_dir=bench_dir)
    if max_records is not None:
        records: List[BenchRecord] = []
        for i, record in enumerate(loader.iter_records()):
            if i >= max_records:
                break
            records.append(record)
    else:
        records = loader.load_all_records()

    if max_prompts_per_dataset is not None or max_records_per_dataset is not None:
        records = cap_prompts_per_dataset(
            records,
            max_prompts_per_dataset=max_prompts_per_dataset,
            max_records_per_dataset=max_records_per_dataset,
            random_seed=random_seed,
        )

    if repair_zero_cost:
        stats = repair_zero_cost_records(records, _load_model_pricing(pricing_config_path))
        logger.info(f"Zero-cost repair stats: {stats}")

    models = sorted({r.model_name for r in records})
    cards = load_model_cards(
        cards_path=model_cards_path,
        llmbench_graphrouter_config=llmbench_graphrouter_config,
        llmbench_collector_config=llmbench_collector_config,
        model_names=models,
    )

    train_records, test_records = split_by_dataset_then_prompt(
        records,
        train_ratio=train_ratio + val_ratio,
        random_seed=random_seed,
        ood_datasets=ood_datasets,
    )

    rng = np.random.default_rng(random_seed)
    by_dataset_prompt: Dict[str, Dict[str, List[BenchRecord]]] = defaultdict(lambda: defaultdict(list))
    for record in train_records:
        by_dataset_prompt[record.dataset_id][record.prompt].append(record)

    val_records: List[BenchRecord] = []
    new_train: List[BenchRecord] = []
    val_frac = val_ratio / max(train_ratio + val_ratio, 1e-8)
    for _, prompts in by_dataset_prompt.items():
        prompt_keys = list(prompts.keys())
        rng.shuffle(prompt_keys)
        n_val = int(len(prompt_keys) * val_frac)
        val_prompts = set(prompt_keys[:n_val])
        for prompt, recs in prompts.items():
            if prompt in val_prompts:
                val_records.extend(recs)
            else:
                new_train.extend(recs)

    grouped_kwargs = dict(
        lambda_cost=lambda_cost,
        cost_mode=cost_mode,
        model_cards=cards,
    )
    splits = {
        "train": records_to_grouped(new_train, models, **grouped_kwargs),
        "val": records_to_grouped(val_records, models, **grouped_kwargs),
        "test": records_to_grouped(test_records, models, **grouped_kwargs),
    }

    out = Path(output_dir)
    for name, grouped in splits.items():
        grouped.save(out / name)
        logger.info(f"{name}: N={grouped.performance.shape[0]} queries, K={len(grouped.model_names)}")

    cards_out = {name: card.__dict__ for name, card in cards.items()}
    with open(out / "model_cards.json", "w", encoding="utf-8") as f:
        json.dump(cards_out, f, indent=2, ensure_ascii=False)

    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "bench_config": config_path,
                "bench_dir": bench_dir,
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "random_seed": random_seed,
                "lambda_cost": lambda_cost,
                "cost_mode": cost_mode,
                "max_prompts_per_dataset": max_prompts_per_dataset,
                "max_records_per_dataset": max_records_per_dataset,
                "format": "duoroute_v2",
            },
            f,
            indent=2,
        )
    return splits
