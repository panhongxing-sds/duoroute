#!/usr/bin/env python3
"""LLMRouterBench-aligned evaluation: 3 tables, local embeddings, multi-GPU inference."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from duoroute.bench_metrics import (
    BenchRoutingMetrics,
    best_single_idx_by_train,
    compute_bench_metrics,
    duoroute_utility_sweep,
    oracle_route_performance_tiebreak_cost,
    pareto_frontier_points,
    pred_matrix_from_choices,
    summarize_method_frontier,
)
from duoroute.data import DuoRouteGroupedData
from duoroute.encoders import build_model_embeddings, load_embedding_dim, load_or_build_query_embeddings
from duoroute.inference import predict_channel_a
from duoroute.model import DuoRouteModel
from duoroute.model_cards import cards_for_models, load_model_cards
from duoroute.prompt_ids import assign_prompt_ids, load_global_prompt_map
from duoroute.reward_builder import build_oracle_reward
from duoroute.utils import load_yaml, project_root

from eval_unified_baselines import export_unified_avengerspro_data
from run_avengerspro_cached import build_cluster_state, route_cluster_state

LAMBDA_SWEEP = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
AP_WEIGHT_SWEEP = [
    (1.0, 0.0),
    (0.9, 0.1),
    (0.8, 0.2),
    (0.7, 0.3),
    (0.5, 0.5),
    (0.3, 0.7),
    (0.0, 1.0),
]
TRAIN_LAMBDA = 0.2


def _build_duoroute_model(
    data_dir: Path,
    checkpoint: Path,
    cfg: dict,
    split: DuoRouteGroupedData,
) -> Tuple[DuoRouteModel, np.ndarray]:
    text_to_pid = load_global_prompt_map(data_dir)
    prompt_ids = assign_prompt_ids(split.prompt_texts, text_to_pid)
    cards = load_model_cards(cards_path=str(data_dir / "model_cards.json"), model_names=split.model_names)
    query_emb = load_or_build_query_embeddings(
        sorted(text_to_pid.keys()),
        embed_path=str(data_dir / "question_embeddings.pth"),
    )
    model_emb = build_model_embeddings(
        cards_for_models(split.model_names, cards),
        embed_path=str(data_dir / "model_embeddings.pth"),
    )
    response_dim = load_embedding_dim(data_dir, fallback=int(cfg.get("embed_dim", 2048)))
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = DuoRouteModel(
        query_emb,
        model_emb,
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        response_dim=response_dim,
    )
    model.load_state_dict(ckpt["model"])
    return model, prompt_ids


@torch.no_grad()
def _predict_shard(
    model: DuoRouteModel,
    prompt_ids: np.ndarray,
    local_device: int,
) -> np.ndarray:
    device = torch.device(f"cuda:{local_device}")
    model.to(device)
    model.eval()
    return model.forward_a(torch.from_numpy(prompt_ids).to(device)).cpu().numpy()


def load_duoroute_pred_a(
    data_dir: Path,
    checkpoint: Path,
    cfg: dict,
    split: DuoRouteGroupedData,
    *,
    local_device_ids: Sequence[int] | None = None,
) -> np.ndarray:
    model, prompt_ids = _build_duoroute_model(data_dir, checkpoint, cfg, split)
    if not torch.cuda.is_available():
        return predict_channel_a(model, torch.from_numpy(prompt_ids))

    devices = list(local_device_ids or [0])
    if len(devices) == 1:
        return _predict_shard(model, prompt_ids, devices[0])

    n_dev = len(devices)
    chunks: List[Tuple[int, int, np.ndarray]] = []
    for i, dev in enumerate(devices):
        start = i * len(prompt_ids) // n_dev
        end = (i + 1) * len(prompt_ids) // n_dev
        if start < end:
            chunks.append((dev, start, prompt_ids[start:end]))

    parts: Dict[int, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=n_dev) as ex:
        futures = {
            ex.submit(_predict_shard, copy.deepcopy(model), chunk, dev): start
            for dev, start, chunk in chunks
        }
        for fut in as_completed(futures):
            start = futures[fut]
            parts[start] = fut.result()

    out = np.empty((len(prompt_ids), split.performance.shape[1]), dtype=np.float32)
    for dev, start, chunk in chunks:
        out[start : start + len(chunk)] = parts[start]
    return out


def _build_ap_state(pool_name: str, data_dir: Path) -> dict:
    export_dir = project_root() / "outputs/avengerspro" / pool_name / "duoroute_unified"
    export_info = export_unified_avengerspro_data(data_dir, export_dir)
    return build_cluster_state(
        data_dir=data_dir,
        train_jsonl=Path(export_info["train"]),
        test_jsonl=Path(export_info["test"]),
    )


def routes_to_metrics(
    chosen: np.ndarray,
    test: DuoRouteGroupedData,
    train: DuoRouteGroupedData,
    *,
    true_u: np.ndarray,
    best_single_by: str,
) -> BenchRoutingMetrics:
    bs_idx = best_single_idx_by_train(
        train.performance,
        train.mask,
        by=best_single_by,
        train_cost=train.cost,
        lambda_cost=TRAIN_LAMBDA,
    )
    pred_u = pred_matrix_from_choices(test.performance.shape[0], test.performance.shape[1], chosen)
    return compute_bench_metrics(
        performance=test.performance,
        cost=test.cost,
        mask=test.mask,
        pred_u=pred_u,
        true_u=true_u,
        best_single_idx=bs_idx,
    )


def routes_from_ap(state: dict, routes: List[List[str]], test: DuoRouteGroupedData) -> np.ndarray:
    name_to_idx = {n: i for i, n in enumerate(test.model_names)}
    chosen = np.zeros(test.performance.shape[0], dtype=np.int64)
    for i, models in enumerate(routes):
        if models and models[0] in name_to_idx:
            chosen[i] = name_to_idx[models[0]]
    return chosen


def metrics_from_ap_routes(
    routes: List[List[str]],
    test: DuoRouteGroupedData,
    train: DuoRouteGroupedData,
    true_u: np.ndarray,
    best_single_by: str,
) -> BenchRoutingMetrics:
    chosen = routes_from_ap({}, routes, test)
    return routes_to_metrics(chosen, test, train, true_u=true_u, best_single_by=best_single_by)


def eval_table1_performance(
    pool_name: str,
    data_dir: Path,
    checkpoint: Path,
    cfg: dict,
    *,
    pred_a: np.ndarray | None = None,
    ap_state: dict | None = None,
    local_device_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    train = DuoRouteGroupedData.load(data_dir / "train")
    test = DuoRouteGroupedData.load(data_dir / "test")
    true_u = test.performance.copy()
    bs_idx = best_single_idx_by_train(train.performance, train.mask, by="performance")

    if pred_a is None:
        pred_a = load_duoroute_pred_a(
            data_dir, checkpoint, cfg, test, local_device_ids=local_device_ids
        )
    if ap_state is None:
        ap_state = _build_ap_state(pool_name, data_dir)

    rows: dict[str, dict] = {}

    # Random
    rng = np.random.default_rng(42)
    rand = np.array([int(rng.choice(np.where(test.mask[i])[0])) for i in range(len(test.mask))])
    rows["random"] = routes_to_metrics(rand, test, train, true_u=true_u, best_single_by="performance").__dict__

    # Best Single (performance)
    bs = np.full(test.performance.shape[0], bs_idx, dtype=np.int64)
    m = routes_to_metrics(bs, test, train, true_u=true_u, best_single_by="performance")
    rows["singlebest"] = {**m.__dict__, "best_model": train.model_names[bs_idx]}

    # Oracle
    oracle = oracle_route_performance_tiebreak_cost(test.performance, test.cost, test.mask)
    rows["oracle"] = routes_to_metrics(oracle, test, train, true_u=true_u, best_single_by="performance").__dict__

    # DuoRoute (trained utility router; report AvgAcc on performance objective)
    du = np.argmax(np.where(test.mask, pred_a, -1e9), axis=1)
    rows["duoroute"] = routes_to_metrics(du, test, train, true_u=true_u, best_single_by="performance").__dict__

    # AvengersPro simple
    ap_routes = route_cluster_state(ap_state, mode="simple")
    rows["avengerspro"] = metrics_from_ap_routes(
        ap_routes, test, train, true_u, "performance"
    ).__dict__

    return {
        "pool": pool_name,
        "setting": "performance_oriented",
        "test_n": int(test.performance.shape[0]),
        "metrics": rows,
    }


def eval_table2_fixed_utility(
    pool_name: str,
    data_dir: Path,
    checkpoint: Path,
    cfg: dict,
    *,
    pred_a: np.ndarray | None = None,
    ap_state: dict | None = None,
    local_device_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    train = DuoRouteGroupedData.load(data_dir / "train")
    test = DuoRouteGroupedData.load(data_dir / "test")
    true_u = build_oracle_reward(test.performance, test.cost, lambda_cost=TRAIN_LAMBDA)
    if pred_a is None:
        pred_a = load_duoroute_pred_a(
            data_dir, checkpoint, cfg, test, local_device_ids=local_device_ids
        )
    if ap_state is None:
        ap_state = _build_ap_state(pool_name, data_dir)

    rows: dict[str, dict] = {}
    bs_idx = best_single_idx_by_train(
        train.performance, train.mask, by="oracle_reward", train_cost=train.cost, lambda_cost=TRAIN_LAMBDA
    )
    bs = np.full(test.performance.shape[0], bs_idx, dtype=np.int64)
    m = routes_to_metrics(bs, test, train, true_u=true_u, best_single_by="oracle_reward")
    rows["singlebest"] = {**m.__dict__, "best_model": train.model_names[bs_idx]}

    du = np.argmax(np.where(test.mask, pred_a, -1e9), axis=1)
    rows["duoroute"] = routes_to_metrics(du, test, train, true_u=true_u, best_single_by="oracle_reward").__dict__

    ap_routes = route_cluster_state(ap_state, mode="balance", performance_weight=0.7, cost_sensitivity=0.3)
    rows["avengerspro_balance"] = metrics_from_ap_routes(
        ap_routes, test, train, true_u, "oracle_reward"
    ).__dict__

    return {
        "pool": pool_name,
        "setting": "fixed_utility_lambda_0.2",
        "lambda_cost": TRAIN_LAMBDA,
        "test_n": int(test.performance.shape[0]),
        "metrics": rows,
    }


def _eval_ap_balance_point(
    ap_state: dict,
    pw: float,
    cs: float,
    test: DuoRouteGroupedData,
    train: DuoRouteGroupedData,
) -> Tuple[BenchRoutingMetrics, float, float]:
    routes = route_cluster_state(
        ap_state, mode="balance", performance_weight=pw, cost_sensitivity=cs
    )
    m = metrics_from_ap_routes(routes, test, train, test.performance, "performance")
    return m, pw, cs


def eval_table3_frontier(
    pool_name: str,
    data_dir: Path,
    checkpoint: Path,
    cfg: dict,
    *,
    pred_a: np.ndarray | None = None,
    ap_state: dict | None = None,
    local_device_ids: Sequence[int] | None = None,
    ap_workers: int = 1,
) -> dict[str, Any]:
    train = DuoRouteGroupedData.load(data_dir / "train")
    test = DuoRouteGroupedData.load(data_dir / "test")
    if pred_a is None:
        pred_a = load_duoroute_pred_a(
            data_dir, checkpoint, cfg, test, local_device_ids=local_device_ids
        )
    if ap_state is None:
        ap_state = _build_ap_state(pool_name, data_dir)

    bs_idx = best_single_idx_by_train(train.performance, train.mask, by="performance")
    bs_acc = float(test.performance[:, bs_idx].mean())
    bs_cost = float(test.cost[test.mask[:, bs_idx], bs_idx].mean())

    all_points: List[Tuple[str, float, float, float, BenchRoutingMetrics]] = []
    method_configs: Dict[str, List[Tuple[float, float, float]]] = {
        "singlebest": [],
        "duoroute": [],
        "avengerspro": [],
    }

    # Singlebest: one operating point
    bs = np.full(test.performance.shape[0], bs_idx, dtype=np.int64)
    m = routes_to_metrics(bs, test, train, true_u=test.performance, best_single_by="performance")
    method_configs["singlebest"].append((m.sample_avg_acc, m.avg_cost, 0.0))
    all_points.append(("singlebest", 0.0, m.sample_avg_acc, m.avg_cost, m))

    # DuoRoute lambda sweep (inference-time utility sweep; model trained at λ=0.2)
    sweep = duoroute_utility_sweep(pred_a, test.cost, test.mask, LAMBDA_SWEEP, train_lambda=TRAIN_LAMBDA)
    du_rows = []
    for lam, pred_u in sweep.items():
        chosen = np.argmax(pred_u, axis=1)
        true_u = build_oracle_reward(test.performance, test.cost, lambda_cost=lam)
        m = routes_to_metrics(chosen, test, train, true_u=true_u, best_single_by="performance")
        method_configs["duoroute"].append((m.sample_avg_acc, m.avg_cost, lam))
        all_points.append(("duoroute", lam, m.sample_avg_acc, m.avg_cost, m))
        du_rows.append({"lambda": lam, **m.__dict__})

    # AvengersPro: simple + balance weight sweep
    ap_rows = []
    simple_routes = route_cluster_state(ap_state, mode="simple")
    m = metrics_from_ap_routes(simple_routes, test, train, test.performance, "performance")
    method_configs["avengerspro"].append((m.sample_avg_acc, m.avg_cost, 0.0))
    all_points.append(("avengerspro", 0.0, m.sample_avg_acc, m.avg_cost, m))
    ap_rows.append({"mode": "simple", "pw": 1.0, "cs": 0.0, **m.__dict__})

    balance_sweep = [(pw, cs) for pw, cs in AP_WEIGHT_SWEEP if not (pw == 1.0 and cs == 0.0)]
    if ap_workers > 1 and balance_sweep:
        with ThreadPoolExecutor(max_workers=ap_workers) as ex:
            futures = [
                ex.submit(_eval_ap_balance_point, ap_state, pw, cs, test, train)
                for pw, cs in balance_sweep
            ]
            balance_results = [fut.result() for fut in as_completed(futures)]
        balance_results.sort(key=lambda x: (x[1], x[2]))
    else:
        balance_results = [
            _eval_ap_balance_point(ap_state, pw, cs, test, train) for pw, cs in balance_sweep
        ]

    for m, pw, cs in balance_results:
        method_configs["avengerspro"].append((m.sample_avg_acc, m.avg_cost, cs))
        all_points.append(("avengerspro", cs, m.sample_avg_acc, m.avg_cost, m))
        ap_rows.append({"mode": "balance", "pw": pw, "cs": cs, **m.__dict__})

    global_frontier = pareto_frontier_points([(p[3], p[2]) for p in all_points])

    summaries = {}
    for method, configs in method_configs.items():
        cfg_tuples = [(a, c, w) for a, c, w in configs]
        summaries[method] = summarize_method_frontier(
            cfg_tuples, bs_acc=bs_acc, bs_cost=bs_cost, global_frontier=global_frontier
        ).__dict__

    return {
        "pool": pool_name,
        "setting": "performance_cost_frontier",
        "best_single": {"model": train.model_names[bs_idx], "avg_acc": bs_acc, "avg_cost": bs_cost},
        "lambda_sweep": LAMBDA_SWEEP,
        "ap_weight_sweep": AP_WEIGHT_SWEEP,
        "summaries": summaries,
        "duoroute_points": du_rows,
        "avengerspro_points": ap_rows,
        "global_pareto_frontier": global_frontier,
    }


def _md_table1(row: dict) -> str:
    lines = [
        "| 方法 | AvgAcc | Gain@B | Gap@O | AvgCost |",
        "|------|-------:|-------:|------:|--------:|",
    ]
    for method, m in row["metrics"].items():
        lines.append(
            f"| {method} | {m['sample_avg_acc']:.4f} | {m['gain_at_best_single']:+.4f} | "
            f"{m['gap_at_oracle']:.4f} | {m['avg_cost']:.6f} |"
        )
    return "\n".join(lines)


def _md_table2(row: dict) -> str:
    lines = [
        "| 方法 | AvgAcc | AvgReward(λ=0.2) | AvgCost | RoutingRegret |",
        "|------|-------:|-----------------:|--------:|--------------:|",
    ]
    for method, m in row["metrics"].items():
        lines.append(
            f"| {method} | {m['sample_avg_acc']:.4f} | {m['avg_acc']:.4f} | "
            f"{m['avg_cost']:.6f} | {m['routing_regret']:.4f} |"
        )
    return "\n".join(lines)


def _md_table3(row: dict) -> str:
    lines = [
        "| 方法 | Best AvgAcc | PerfGain | MinCost@≥BS | CostSave | ParetoDist | #configs |",
        "|------|------------:|---------:|------------:|---------:|-----------:|---------:|",
    ]
    for method, s in row["summaries"].items():
        lines.append(
            f"| {method} | {s['best_avg_acc']:.4f} | {s['perf_gain']:+.4f} | "
            f"{s['lowest_cost_at_least_bs_acc']:.6f} | {s['cost_save']:+.4f} | "
            f"{s['pareto_dist']:.4f} | {s['n_configs']} |"
        )
    return "\n".join(lines)


def evaluate_pool(
    name: str,
    data_dir: Path,
    cfg_path: Path,
    ckpt: Path,
    gpu_ids: List[int],
    ap_workers: int,
) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    local_ids = list(range(len(gpu_ids))) if gpu_ids else None
    cfg = load_yaml(cfg_path).get("duoroute", {})

    test = DuoRouteGroupedData.load(data_dir / "test")
    print(f"[{name}] DuoRoute inference on GPU(s) {gpu_ids} ({test.performance.shape[0]} queries)...", flush=True)
    pred_a = load_duoroute_pred_a(data_dir, ckpt, cfg, test, local_device_ids=local_ids)
    print(f"[{name}] Building AvengersPro cluster state...", flush=True)
    ap_state = _build_ap_state(name, data_dir)

    out: dict[str, Any] = {
        "pool": name,
        "gpus": gpu_ids,
        "local_device_ids": local_ids or [],
    }
    t1 = eval_table1_performance(
        name, data_dir, ckpt, cfg, pred_a=pred_a, ap_state=ap_state, local_device_ids=local_ids
    )
    out["table1_performance"] = t1

    if name == "seed42_flagship":
        t2 = eval_table2_fixed_utility(
            name, data_dir, ckpt, cfg, pred_a=pred_a, ap_state=ap_state, local_device_ids=local_ids
        )
        t3 = eval_table3_frontier(
            name,
            data_dir,
            ckpt,
            cfg,
            pred_a=pred_a,
            ap_state=ap_state,
            local_device_ids=local_ids,
            ap_workers=ap_workers,
        )
        out["table2_fixed_utility"] = t2
        out["table3_frontier"] = t3
    print(f"[{name}] Done.", flush=True)
    return out


def _pool_worker(args: Tuple[str, Path, Path, Path, List[int], int]) -> dict[str, Any]:
    return evaluate_pool(*args)


def _split_gpus(all_gpus: List[int], n_pools: int) -> List[List[int]]:
    if n_pools <= 1:
        return [all_gpus]
    groups: List[List[int]] = [[] for _ in range(n_pools)]
    for i, g in enumerate(all_gpus):
        groups[i % n_pools].append(g)
    return [g for g in groups if g]


def _parse_gpus(gpu_str: str) -> List[int]:
    return [int(x.strip()) for x in gpu_str.split(",") if x.strip()]


POOL_SPECS = [
    (
        "seed42_small",
        project_root() / "data/seed42_small",
        project_root() / "configs/small_subset.yaml",
        project_root() / "outputs/checkpoints/seed42_small/best.pt",
    ),
    (
        "seed42_flagship",
        project_root() / "data/seed42_flagship",
        project_root() / "configs/flagship.yaml",
        project_root() / "outputs/checkpoints/seed42_flagship/best.pt",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="LLMRouterBench-aligned DuoRoute/AvengersPro eval")
    parser.add_argument(
        "--gpus",
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated physical GPU ids (default: all 8)",
    )
    parser.add_argument(
        "--pool",
        choices=["seed42_small", "seed42_flagship", "all"],
        default="all",
        help="Which model pool to evaluate",
    )
    parser.add_argument(
        "--ap-workers",
        type=int,
        default=0,
        help="Thread workers for AvengersPro balance sweep (0=auto)",
    )
    args = parser.parse_args()

    all_gpus = _parse_gpus(args.gpus)
    specs = [s for s in POOL_SPECS if args.pool == "all" or s[0] == args.pool]
    gpu_groups = _split_gpus(all_gpus, len(specs))
    ap_workers = args.ap_workers or max(1, len(all_gpus) // max(1, len(specs)))

    print(
        f"Using GPUs {all_gpus} -> {dict(zip([s[0] for s in specs], gpu_groups))}, "
        f"ap_workers={ap_workers}",
        flush=True,
    )

    worker_args = [
        (name, data_dir, cfg_path, ckpt, gpu_groups[i], ap_workers)
        for i, (name, data_dir, cfg_path, ckpt) in enumerate(specs)
    ]

    if len(worker_args) == 1:
        pool_results = [evaluate_pool(*worker_args[0])]
    else:
        with ProcessPoolExecutor(max_workers=len(worker_args)) as ex:
            pool_results = list(ex.map(_pool_worker, worker_args))

    out: dict[str, Any] = {
        "gpus": all_gpus,
        "gpu_assignment": {r["pool"]: r["gpus"] for r in pool_results},
        "ap_workers": ap_workers,
    }
    md_parts = ["# LLMRouterBench 对齐评估（本地 embedding，无 API）", ""]
    md_parts.append(f"GPU 分配: {out['gpu_assignment']}")
    md_parts.append("")

    for r in pool_results:
        name = r["pool"]
        t1 = r["table1_performance"]
        out[f"{name}_table1_performance"] = t1
        md_parts += [f"## 表1 Performance-oriented — {name}", _md_table1(t1), ""]
        if name == "seed42_flagship":
            t2 = r["table2_fixed_utility"]
            t3 = r["table3_frontier"]
            out[f"{name}_table2_fixed_utility"] = t2
            out[f"{name}_table3_frontier"] = t3
            md_parts += [
                f"## 表2 Fixed utility λ=0.2 — {name}",
                _md_table2(t2),
                "",
                f"## 表3 Performance-cost frontier — {name}",
                _md_table3(t3),
                "",
            ]

    out_dir = project_root() / "outputs/eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "llmbench_aligned_eval.json"
    md_path = out_dir / "llmbench_aligned_eval.md"
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text("\n".join(md_parts), encoding="utf-8")
    print("\n".join(md_parts))
    print(f"\nWrote {json_path}\nWrote {md_path}")


if __name__ == "__main__":
    main()
