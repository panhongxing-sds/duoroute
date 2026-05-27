#!/usr/bin/env python3
"""Evaluate singlebest / DuoRoute / AvengersPro on the same DuoRoute split."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from duoroute.data import DuoRouteGroupedData
from duoroute.encoders import build_model_embeddings, load_embedding_dim, load_or_build_query_embeddings
from duoroute.evaluator import evaluate_predictions
from duoroute.inference import predict_channel_a
from duoroute.model import DuoRouteModel
from duoroute.model_cards import cards_for_models, load_model_cards
from duoroute.prompt_ids import assign_prompt_ids, load_global_prompt_map
from duoroute.utils import load_yaml, project_root


def grouped_to_jsonl(grouped: DuoRouteGroupedData) -> list[dict]:
    items: list[dict] = []
    for i in range(len(grouped.prompt_keys)):
        records: dict[str, float] = {}
        usages: dict[str, dict] = {}
        for k, name in enumerate(grouped.model_names):
            score = float(grouped.performance[i, k]) if grouped.mask[i, k] else 0.0
            records[name] = score
            usages[name] = {
                "completion_tokens": 0,
                "cost": float(grouped.cost[i, k]) if grouped.mask[i, k] else 0.0,
                "prompt_tokens": 0,
            }
        items.append(
            {
                "query": grouped.prompt_texts[i],
                "dataset": grouped.dataset_ids[i],
                "index": i,
                "prompt_key": grouped.prompt_keys[i],
                "records": records,
                "usages": usages,
            }
        )
    return items


def write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_baseline_scores(path: Path, test_items: list[dict]) -> None:
    model_dataset_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for item in test_items:
        dataset = item["dataset"]
        for model, score in item["records"].items():
            model_dataset_scores[model][dataset].append(float(score))
    baseline_scores = {
        model: {
            dataset: round(float(np.mean(scores)) * 100.0, 2)
            for dataset, scores in datasets.items()
        }
        for model, datasets in model_dataset_scores.items()
    }
    path.write_text(json.dumps(baseline_scores, indent=2, ensure_ascii=False), encoding="utf-8")


def export_unified_avengerspro_data(data_dir: Path, out_dir: Path) -> dict[str, str]:
    train = DuoRouteGroupedData.load(data_dir / "train")
    test = DuoRouteGroupedData.load(data_dir / "test")
    train_items = grouped_to_jsonl(train)
    test_items = grouped_to_jsonl(test)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    test_path = out_dir / "test.jsonl"
    baseline_path = out_dir / "baseline_scores.json"
    write_jsonl(train_path, train_items)
    write_jsonl(test_path, test_items)
    write_baseline_scores(baseline_path, test_items)
    return {
        "train": str(train_path),
        "test": str(test_path),
        "baseline_scores": str(baseline_path),
        "train_n": len(train_items),
        "test_n": len(test_items),
    }


def eval_singlebest(train: DuoRouteGroupedData, test: DuoRouteGroupedData) -> dict:
    model_means = []
    for k in range(len(train.model_names)):
        mask_k = train.mask[:, k]
        model_means.append(float(train.oracle_reward[mask_k, k].mean()) if mask_k.any() else -1e9)
    best_idx = int(np.argmax(model_means))
    pred = np.zeros_like(test.oracle_reward, dtype=np.float32)
    pred[:, best_idx] = 1.0
    metrics = evaluate_predictions(
        test.oracle_reward,
        pred,
        test.mask,
        performance=test.performance,
        cost=test.cost,
    )
    return {
        "best_model": train.model_names[best_idx],
        "best_model_idx": best_idx,
        **metrics.to_dict(),
    }


def eval_duoroute(test: DuoRouteGroupedData, data_dir: Path, checkpoint: Path, cfg: dict) -> dict:
    text_to_pid = load_global_prompt_map(data_dir)
    prompt_ids = assign_prompt_ids(test.prompt_texts, text_to_pid)
    cards = load_model_cards(cards_path=str(data_dir / "model_cards.json"), model_names=test.model_names)
    query_emb = load_or_build_query_embeddings(
        sorted(text_to_pid.keys()),
        embed_path=str(data_dir / "question_embeddings.pth"),
    )
    model_emb = build_model_embeddings(
        cards_for_models(test.model_names, cards),
        embed_path=str(data_dir / "model_embeddings.pth"),
    )
    response_dim = load_embedding_dim(data_dir, fallback=int(cfg.get("embed_dim", 2048)))
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = DuoRouteModel(query_emb, model_emb, hidden_dim=int(cfg.get("hidden_dim", 64)), response_dim=response_dim)
    model.load_state_dict(ckpt["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    pred_a = predict_channel_a(model, torch.from_numpy(prompt_ids).to(device))
    metrics = evaluate_predictions(
        test.oracle_reward,
        pred_a,
        test.mask,
        performance=test.performance,
        cost=test.cost,
    )
    return metrics.to_dict()


def eval_avengerspro_unified(results_path: Path, expected_test_n: int) -> dict:
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    if "results" in payload:
        ap = payload["results"]
        n = int(ap["total_queries"])
        sample_acc = float(ap["correct_routes"] / n) if n else 0.0
        dataset_avg = float(ap.get("accuracy", 0.0))
        embedding_source = None
    else:
        ap = payload
        n = int(ap["total_queries"])
        sample_acc = float(ap.get("avg_acc", 0.0))
        dataset_avg = float(ap.get("dataset_avg_acc", 0.0)) * 100.0
        embedding_source = ap.get("embedding_source")
    if n != expected_test_n:
        raise ValueError(f"AvengersPro test n={n}, expected DuoRoute test n={expected_test_n}")
    return {
        "avg_acc": sample_acc,
        "avg_reward": sample_acc,
        "n_queries": n,
        "correct_routes": float(ap.get("correct_routes", sample_acc * n)),
        "dataset_avg_pct": dataset_avg,
        "embedding_source": embedding_source,
        "notes": "Same DuoRoute test JSONL; cached question_embeddings.pth; avg_acc = sample-level",
    }


def build_avengerspro_config(export_info: dict, config_path: Path, api_key_env: str = "EMBEDDING_API_KEY") -> None:
    config = {
        "train_data_path": export_info["train"],
        "test_data_path": export_info["test"],
        "baseline_scores_path": export_info["baseline_scores"],
        "n_clusters": 16,
        "seed": 42,
        "max_router": 1,
        "top_k": 1,
        "beta": 9.0,
        "max_workers": 8,
        "cluster_batch_size": 1000,
        "max_tokens": 8000,
        "embedding_model": "text-embedding-3-large",
        "embedding_base_url": "https://api.shubiaobiao.cn/v1",
        "embedding_api_key": api_key_env,
        "excluded_models": [],
        "ood_datasets": [],
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def run_avengerspro_cached(
    data_dir: Path,
    train_jsonl: Path,
    test_jsonl: Path,
    output_path: Path,
    *,
    n_clusters: int = 16,
    seed: int = 42,
) -> None:
    """Run AvengersPro cluster routing with DuoRoute question_embeddings.pth (no API)."""
    from run_avengerspro_cached import run_cached_avengerspro

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = run_cached_avengerspro(
        data_dir=data_dir,
        train_jsonl=train_jsonl,
        test_jsonl=test_jsonl,
        n_clusters=n_clusters,
        seed=seed,
    )
    output_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(project_root() / "configs/small_subset.yaml"))
    parser.add_argument("--data-dir", default="data/seed42_small")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/seed42_small/best.pt")
    parser.add_argument("--skip-avengerspro-run", action="store_true")
    parser.add_argument("--n-clusters", type=int, default=16)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = project_root() / data_dir
    cfg = load_yaml(args.config).get("duoroute", {})

    export_dir = project_root() / "outputs/avengerspro/seed42_small/duoroute_unified"
    export_info = export_unified_avengerspro_data(data_dir, export_dir)
    config_path = export_dir / "simple_cluster_config.json"
    results_path = export_dir / "results.json"
    if not args.skip_avengerspro_run:
        run_avengerspro_cached(
            data_dir,
            Path(export_info["train"]),
            Path(export_info["test"]),
            results_path,
            n_clusters=args.n_clusters,
            seed=int(cfg.get("seed", 42)),
        )

    train = DuoRouteGroupedData.load(data_dir / "train")
    test = DuoRouteGroupedData.load(data_dir / "test")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = project_root() / checkpoint

    out = {
        "pool": data_dir.name,
        "split_info": {
            "train_n": export_info["train_n"],
            "test_n": export_info["test_n"],
            "split_note": "Unified DuoRoute train/val/test; AvengersPro train=test split uses DuoRoute train only, eval on DuoRoute test only.",
        },
        "methods": {
            "singlebest": eval_singlebest(train, test),
            "duoroute": eval_duoroute(test, data_dir, checkpoint, cfg),
        },
    }
    if results_path.exists():
        out["methods"]["avengerspro"] = eval_avengerspro_unified(results_path, export_info["test_n"])
    else:
        out["methods"]["avengerspro"] = None

    eval_dir = project_root() / "outputs/eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = eval_dir / f"{data_dir.name}_baselines_unified.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
