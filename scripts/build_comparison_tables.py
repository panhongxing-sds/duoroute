#!/usr/bin/env python3
"""Build performance-only and balance comparison tables (local embeddings, no API)."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
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

from eval_unified_baselines import export_unified_avengerspro_data
from run_avengerspro_cached import run_cached_avengerspro


PERFORMANCE_WEIGHT = 0.7
COST_SENSITIVITY = 0.3


@dataclass
class PoolSpec:
    name: str
    data_dir: Path
    config_path: Path
    checkpoint: Path


def _masked_argmax(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    neg_inf = -1e9
    masked = np.where(mask, values, neg_inf)
    return masked.argmax(axis=1)


def eval_singlebest(
    train: DuoRouteGroupedData,
    test: DuoRouteGroupedData,
    *,
    select_by: str,
) -> dict:
    matrix = train.performance if select_by == "performance" else train.oracle_reward
    model_means = []
    for k in range(len(train.model_names)):
        mask_k = train.mask[:, k]
        model_means.append(float(matrix[mask_k, k].mean()) if mask_k.any() else -1e9)
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
        "select_by": select_by,
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


def _score_avengerspro_from_routes(test_items: list[dict], routes: list[list[str]], test: DuoRouteGroupedData) -> dict:
    name_to_idx = {n: i for i, n in enumerate(test.model_names)}
    pred = np.zeros_like(test.oracle_reward, dtype=np.float32)
    for row, models in enumerate(routes):
        if not models:
            continue
        model = models[0]
        if model in name_to_idx:
            pred[row, name_to_idx[model]] = 1.0
    metrics = evaluate_predictions(
        test.oracle_reward,
        pred,
        test.mask,
        performance=test.performance,
        cost=test.cost,
    )
    return metrics.to_dict()


def evaluate_pool(pool: PoolSpec) -> dict:
    cfg = load_yaml(pool.config_path).get("duoroute", {})
    train = DuoRouteGroupedData.load(pool.data_dir / "train")
    test = DuoRouteGroupedData.load(pool.data_dir / "test")

    export_dir = project_root() / "outputs" / "avengerspro" / pool.name / "duoroute_unified"
    export_info = export_unified_avengerspro_data(pool.data_dir, export_dir)
    train_jsonl = Path(export_info["train"])
    test_jsonl = Path(export_info["test"])

    perf_ap = run_cached_avengerspro(
        data_dir=pool.data_dir,
        train_jsonl=train_jsonl,
        test_jsonl=test_jsonl,
        mode="simple",
    )
    bal_ap = run_cached_avengerspro(
        data_dir=pool.data_dir,
        train_jsonl=train_jsonl,
        test_jsonl=test_jsonl,
        mode="balance",
        performance_weight=PERFORMANCE_WEIGHT,
        cost_sensitivity=COST_SENSITIVITY,
    )

    test_items = []
    with test_jsonl.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_items.append(json.loads(line))

    perf_ap_metrics = _score_avengerspro_from_routes(test_items, perf_ap["selected_routes"], test)
    bal_ap_metrics = _score_avengerspro_from_routes(test_items, bal_ap["selected_routes"], test)

    return {
        "pool": pool.name,
        "test_n": int(test.oracle_reward.shape[0]),
        "performance": {
            "singlebest": eval_singlebest(train, test, select_by="performance"),
            "duoroute": eval_duoroute(test, pool.data_dir, pool.checkpoint, cfg),
            "avengerspro": {
                **perf_ap_metrics,
                "router": "simple",
            },
        },
        "balance": {
            "singlebest": eval_singlebest(train, test, select_by="oracle_reward"),
            "duoroute": eval_duoroute(test, pool.data_dir, pool.checkpoint, cfg),
            "avengerspro": {
                **bal_ap_metrics,
                "router": "balance",
                "performance_weight": PERFORMANCE_WEIGHT,
                "cost_sensitivity": COST_SENSITIVITY,
            },
        },
        "notes": {
            "split": "DuoRoute train/test unified; test_n matches all methods",
            "performance_singlebest": "train 上 mean(performance) 最高的单模型",
            "balance_singlebest": "train 上 mean(oracle_reward=perf-0.2*norm_cost) 最高的单模型",
            "balance_duoroute": "同一 checkpoint；主看 avg_reward",
            "balance_avengerspro": "AvengersPro balance_cluster 逻辑；pw=0.7, cs=0.3",
            "embedding": "本地 question_embeddings.pth，无 API",
        },
    }


def _row(method: str, m: dict, *, primary: str) -> dict:
    return {
        "method": method,
        "avg_acc": round(float(m["avg_acc"]), 4),
        "avg_reward": round(float(m["avg_reward"]), 4),
        "avg_cost": round(float(m["avg_cost"]), 6),
        "routing_regret": round(float(m["routing_regret"]), 4),
        "primary": round(float(m[primary]), 4),
        "best_model": m.get("best_model"),
    }


def _markdown_table(title: str, rows: list[dict], primary_key: str, primary_label: str) -> str:
    lines = [
        f"## {title}",
        "",
        f"| 池 | 方法 | avg_acc | {primary_label} | avg_cost | routing_regret | 备注 |",
        "|----|------|---------|--------------|----------|----------------|------|",
    ]
    for r in rows:
        note = r.get("best_model") or r.get("router") or ""
        primary_val = r["avg_acc"] if primary_key == "avg_acc" else r["primary"]
        lines.append(
            f"| {r['pool']} | {r['method']} | {r['avg_acc']:.4f} | {primary_val:.4f} | "
            f"{r['avg_cost']:.6f} | {r['routing_regret']:.4f} | {note} |"
        )
    return "\n".join(lines)


def main() -> None:
    pools = [
        PoolSpec(
            name="seed42_small",
            data_dir=project_root() / "data/seed42_small",
            config_path=project_root() / "configs/small_subset.yaml",
            checkpoint=project_root() / "outputs/checkpoints/seed42_small/best.pt",
        ),
        PoolSpec(
            name="seed42_flagship",
            data_dir=project_root() / "data/seed42_flagship",
            config_path=project_root() / "configs/flagship.yaml",
            checkpoint=project_root() / "outputs/checkpoints/seed42_flagship/best.pt",
        ),
    ]

    all_results = {}
    perf_rows: list[dict] = []
    bal_rows: list[dict] = []

    for pool in pools:
        result = evaluate_pool(pool)
        all_results[pool.name] = result
        for method, metrics in result["performance"].items():
            row = _row(method, metrics, primary="avg_acc")
            row["pool"] = pool.name
            if method == "singlebest":
                row["best_model"] = metrics.get("best_model")
            elif method == "avengerspro":
                row["router"] = "simple"
            perf_rows.append(row)
        for method, metrics in result["balance"].items():
            row = _row(method, metrics, primary="avg_reward")
            row["pool"] = pool.name
            if method == "singlebest":
                row["best_model"] = metrics.get("best_model")
            elif method == "avengerspro":
                row["router"] = "balance"
            bal_rows.append(row)

    out_dir = project_root() / "outputs/eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "comparison_tables.json"
    md_path = out_dir / "comparison_tables.md"
    json_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")

    md = "\n\n".join(
        [
            "# DuoRoute 对比表（统一 split，本地 embedding，无 API）",
            "",
            _markdown_table(
                "表 1：纯 Performance（Simple / 按 accuracy 选 Singlebest）",
                perf_rows,
                "avg_acc",
                "avg_acc",
            ),
            "",
            _markdown_table(
                "表 2：Balance（Balance router / 按 oracle_reward 选 Singlebest；DuoRoute 看 avg_reward）",
                bal_rows,
                "avg_reward",
                "avg_reward",
            ),
        ]
    )
    md_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
