#!/usr/bin/env python3
"""AvengersPro-style cluster router using precomputed DuoRoute query embeddings (no API)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from duoroute.embedding_io import load_embedding_tensor, resolve_embedding_path
from duoroute.prompt_ids import load_global_prompt_map
from duoroute.utils import project_root


def _load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _lookup_embeddings(texts: list[str], text_to_pid: dict[str, int], emb: torch.Tensor) -> np.ndarray:
    missing = [t for t in texts if t not in text_to_pid]
    if missing:
        raise KeyError(f"{len(missing)} queries missing from cached embeddings (first: {missing[0][:80]!r})")
    ids = [text_to_pid[t] for t in texts]
    return emb[torch.as_tensor(ids, dtype=torch.long)].numpy()


def _compute_balance_cluster_rankings(
    cluster_labels: np.ndarray,
    train_items: list[dict],
    available_models: list[str],
    *,
    performance_weight: float,
    cost_sensitivity: float,
    min_accuracy_threshold: float = 0.0,
) -> dict[int, dict]:
    cluster_data: dict[int, list[dict]] = defaultdict(list)
    for i, cluster_id in enumerate(cluster_labels):
        cluster_data[int(cluster_id)].append(train_items[i])

    rankings: dict[int, dict] = {}
    for cluster_id, records_list in cluster_data.items():
        model_data: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"successes": [], "costs": []})
        for item in records_list:
            records = item.get("records", {})
            usages = item.get("usages") or {}
            for model_name in available_models:
                if model_name not in records:
                    continue
                score = float(records[model_name])
                usage = usages.get(model_name, {}) if isinstance(usages, dict) else {}
                cost = float(usage.get("cost", 0.0) or 0.0)
                model_data[model_name]["successes"].append(score)
                model_data[model_name]["costs"].append(cost)

        metrics: dict[str, dict[str, float]] = {}
        all_costs: list[float] = []
        for model_name in available_models:
            succ = model_data[model_name]["successes"]
            costs = model_data[model_name]["costs"]
            if succ:
                all_costs.extend(costs)
                metrics[model_name] = {
                    "accuracy": float(np.mean(succ)),
                    "avg_cost": float(np.mean(costs)),
                    "total_queries": float(len(succ)),
                }
            else:
                metrics[model_name] = {"accuracy": 0.0, "avg_cost": float("inf"), "total_queries": 0.0}

        valid = [m for m in metrics.values() if m["total_queries"] > 0 and m["avg_cost"] != float("inf")]
        if valid:
            max_cost = max(m["avg_cost"] for m in valid)
            max_acc = max(m["accuracy"] for m in valid)
            min_acc = min(m["accuracy"] for m in valid)
        else:
            max_cost = max_acc = min_acc = 1.0

        balance_scores: dict[str, float] = {}
        for model_name, m in metrics.items():
            if m["total_queries"] <= 0:
                balance_scores[model_name] = 0.0
                continue
            acc_range = max_acc - min_acc
            norm_acc = (m["accuracy"] - min_acc) / acc_range if acc_range > 0 else 1.0
            norm_cost = m["avg_cost"] / max_cost if max_cost > 0 else 0.0
            cost_score = 1.0 - norm_cost
            score = performance_weight * norm_acc + cost_sensitivity * cost_score
            balance_scores[model_name] = score if m["accuracy"] >= min_accuracy_threshold else 0.0

        sorted_models = sorted(balance_scores.items(), key=lambda x: x[1], reverse=True)
        rankings[cluster_id] = {
            "total": len(records_list),
            "balance_scores": dict(sorted_models),
            "ranking": [name for name, _ in sorted_models],
        }
    return rankings


def _compute_cluster_rankings(
    cluster_labels: np.ndarray,
    train_records: list[dict[str, float]],
    available_models: list[str],
) -> dict[int, dict]:
    cluster_data: dict[int, list[dict[str, float]]] = defaultdict(list)
    for i, cluster_id in enumerate(cluster_labels):
        cluster_data[int(cluster_id)].append(train_records[i])

    rankings: dict[int, dict] = {}
    for cluster_id, records_list in cluster_data.items():
        model_performance: dict[str, list[float]] = defaultdict(list)
        for rec in records_list:
            for model_name, score in rec.items():
                if model_name in available_models:
                    model_performance[model_name].append(float(score))
        model_scores = {
            model: (float(np.mean(scores)) if scores else 0.0)
            for model, scores in model_performance.items()
        }
        for model in available_models:
            model_scores.setdefault(model, 0.0)
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        rankings[cluster_id] = {
            "total": len(records_list),
            "scores": dict(sorted_models),
            "ranking": [name for name, _ in sorted_models],
        }
    return rankings


def _route_batch(
    query_embeddings: np.ndarray,
    *,
    normalizer: Normalizer,
    cluster_centers: np.ndarray,
    cluster_rankings: dict[int, dict],
    available_models: list[str],
    top_k: int,
    beta: float,
    max_router: int,
) -> list[list[str]]:
    x = normalizer.transform(query_embeddings)
    distances = 1.0 - x @ cluster_centers.T
    results: list[list[str]] = []
    for i in range(len(query_embeddings)):
        query_distances = distances[i]
        closest = np.argsort(query_distances)[:top_k]
        closest_distances = query_distances[closest]
        logits = -beta * closest_distances
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()

        expert_scores: dict[str, float] = defaultdict(float)
        for cluster_idx, prob in zip(closest, probs):
            info = cluster_rankings.get(int(cluster_idx))
            if not info:
                continue
            ranking = info["ranking"]
            for model_name in available_models:
                if model_name in ranking:
                    rank = ranking.index(model_name)
                    expert_scores[model_name] += prob * (1.0 / (rank + 1))
        for model_name in available_models:
            expert_scores.setdefault(model_name, 0.0)
        selected = [
            name
            for name, _ in sorted(expert_scores.items(), key=lambda x: x[1], reverse=True)[:max_router]
        ]
        results.append(selected)
    return results


def build_cluster_state(
    *,
    data_dir: Path,
    train_jsonl: Path,
    test_jsonl: Path,
    n_clusters: int = 16,
    seed: int = 42,
) -> dict[str, Any]:
    train_items = _load_jsonl(train_jsonl)
    test_items = _load_jsonl(test_jsonl)
    text_to_pid = load_global_prompt_map(data_dir)
    emb = load_embedding_tensor(data_dir / "question_embeddings.pth")
    test_emb = _lookup_embeddings([x["query"] for x in test_items], text_to_pid, emb)
    train_records = [item["records"] for item in train_items]
    available_models = sorted({m for rec in train_records for m in rec.keys()})
    normalizer = Normalizer(norm="l2")
    train_norm = normalizer.fit_transform(train_emb)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels = kmeans.fit_predict(train_norm)
    return {
        "train_items": train_items,
        "test_items": test_items,
        "train_records": train_records,
        "available_models": available_models,
        "normalizer": normalizer,
        "cluster_centers": kmeans.cluster_centers_,
        "cluster_labels": cluster_labels,
        "test_emb": test_emb,
    }


def route_cluster_state(
    state: dict[str, Any],
    *,
    mode: str = "simple",
    performance_weight: float = 0.7,
    cost_sensitivity: float = 0.3,
    top_k: int = 1,
    beta: float = 9.0,
    max_router: int = 1,
) -> list[list[str]]:
    if mode == "balance":
        rankings = _compute_balance_cluster_rankings(
            state["cluster_labels"],
            state["train_items"],
            state["available_models"],
            performance_weight=performance_weight,
            cost_sensitivity=cost_sensitivity,
        )
    elif mode == "simple":
        rankings = _compute_cluster_rankings(
            state["cluster_labels"], state["train_records"], state["available_models"]
        )
    else:
        raise ValueError(mode)
    return _route_batch(
        state["test_emb"],
        normalizer=state["normalizer"],
        cluster_centers=state["cluster_centers"],
        cluster_rankings=rankings,
        available_models=state["available_models"],
        top_k=top_k,
        beta=beta,
        max_router=max_router,
    )


def run_cached_avengerspro(
    *,
    data_dir: Path,
    train_jsonl: Path,
    test_jsonl: Path,
    n_clusters: int = 16,
    seed: int = 42,
    top_k: int = 1,
    beta: float = 9.0,
    max_router: int = 1,
    mode: str = "simple",
    performance_weight: float = 0.7,
    cost_sensitivity: float = 0.3,
) -> dict[str, Any]:
    train_items = _load_jsonl(train_jsonl)
    test_items = _load_jsonl(test_jsonl)
    text_to_pid = load_global_prompt_map(data_dir)
    emb = load_embedding_tensor(data_dir / "question_embeddings.pth")

    train_queries = [item["query"] for item in train_items]
    test_queries = [item["query"] for item in test_items]
    train_emb = _lookup_embeddings(train_queries, text_to_pid, emb)
    test_emb = _lookup_embeddings(test_queries, text_to_pid, emb)
    train_records = [item["records"] for item in train_items]

    model_set = set()
    for rec in train_records:
        model_set.update(rec.keys())
    available_models = sorted(model_set)

    normalizer = Normalizer(norm="l2")
    train_norm = normalizer.fit_transform(train_emb)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels = kmeans.fit_predict(train_norm)
    cluster_centers = kmeans.cluster_centers_
    if mode == "balance":
        cluster_rankings = _compute_balance_cluster_rankings(
            cluster_labels,
            train_items,
            available_models,
            performance_weight=performance_weight,
            cost_sensitivity=cost_sensitivity,
        )
    elif mode == "simple":
        cluster_rankings = _compute_cluster_rankings(cluster_labels, train_records, available_models)
    else:
        raise ValueError(f"Unknown mode {mode!r}; expected 'simple' or 'balance'")

    selected_batches = _route_batch(
        test_emb,
        normalizer=normalizer,
        cluster_centers=cluster_centers,
        cluster_rankings=cluster_rankings,
        available_models=available_models,
        top_k=top_k,
        beta=beta,
        max_router=max_router,
    )

    correct = 0.0
    dataset_perf: dict[str, dict[str, float]] = defaultdict(lambda: {"correct": 0.0, "total": 0.0})
    model_stats: Counter[str] = Counter()
    per_query: list[dict] = []

    for item, models in zip(test_items, selected_batches):
        score = 0.0
        if models:
            score = max(float(item["records"].get(m, 0.0)) for m in models)
        correct += score
        ds = item["dataset"]
        dataset_perf[ds]["correct"] += score
        dataset_perf[ds]["total"] += 1.0
        for m in models:
            model_stats[m] += 1
        per_query.append({"dataset": ds, "selected_models": models, "score": score})

    n = len(test_items)
    sample_acc = correct / n if n else 0.0
    dataset_accs = [
        v["correct"] / v["total"] for v in dataset_perf.values() if v["total"] > 0
    ]
    dataset_avg = float(np.mean(dataset_accs)) if dataset_accs else 0.0

    return {
        "mode": mode,
        "avg_acc": sample_acc,
        "correct_routes": correct,
        "total_queries": n,
        "dataset_avg_acc": dataset_avg,
        "selected_routes": selected_batches,
        "dataset_performance": {
            ds: {
                "correct": vals["correct"],
                "total": vals["total"],
                "accuracy": (vals["correct"] / vals["total"] if vals["total"] else 0.0),
            }
            for ds, vals in sorted(dataset_perf.items())
        },
        "model_selection_stats": dict(model_stats),
        "embedding_source": str(resolve_embedding_path(data_dir / "question_embeddings.pth")),
        "n_clusters": n_clusters,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--test-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-clusters", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = project_root() / data_dir
    out = run_cached_avengerspro(
        data_dir=data_dir,
        train_jsonl=Path(args.train_jsonl),
        test_jsonl=Path(args.test_jsonl),
        n_clusters=args.n_clusters,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
