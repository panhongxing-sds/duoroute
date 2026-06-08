#!/usr/bin/env python3
"""Export paper tables: Main Results (multi-λ), Ablation (λ=0.2), Per-dataset (851 pool)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import torch

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from duoroute.bench_metrics import best_single_idx_by_train, oracle_route_performance_tiebreak_cost
from duoroute.main_table import (
    DEFAULT_LAMBDAS,
    METRIC_COLS,
    aggregate_rows,
    best_single_idx,
    eval_at_lambda,
    fmt_mean_std,
    load_pool_851,
    _train_rdfl,
)
from duoroute.pool_data import (
    DEFAULT_LAMBDA,
    enrich_pool_data,
    eval_row,
    load_flagship,
    per_dataset_breakdown,
    rebuild_utilities,
)
from duoroute.regretrouter import OURS_METHOD_NAME, RDFLAPTrainConfig, infer_choices, make_router, train_regretrouter
from duoroute.reward_builder import rebuild_grouped_rewards
from duoroute.utils import project_root, set_seed

try:
    from run_llmrouterbench_flagship import (  # noqa: E402
        SOTA_METHODS,
        load_sota_routes,
        run_method as run_router_sota_method,
    )
except ImportError:
    SOTA_METHODS = ROUTER_SOTA_METHODS
    load_sota_routes = None  # type: ignore
    run_router_sota_method = None  # type: ignore

# Main table: 9 rows (Oracle, BestSingle, AP-balance, 5 SOTA, RegretRouter).
MAIN_METHOD_ORDER = [
    "Oracle-utility",
    "BestSingle",
    "Official-AvengersPro-balance",
    "GraphRouter-lite",
    "EmbedLLM",
    "RouterDC-lite",
    "RouteLLM",
    "HybridLLM-lite",
    OURS_METHOD_NAME,
]

MAIN_BASELINE_METHODS = [
    "Oracle-utility",
    "BestSingle",
    "Official-AvengersPro-balance",
]

MAIN_LEARNED_METHODS = [OURS_METHOD_NAME]

AP_ANCHOR_METHOD = "Official-AvengersPro-balance"

ROUTER_SOTA_METHODS = [
    "GraphRouter-lite",
    "EmbedLLM",
    "RouterDC-lite",
    "RouteLLM",
    "HybridLLM-lite",
]
DEFAULT_ROUTER_SOTA_CACHE = "outputs/llmrouterbench_flagship"

PER_DATASET_METHODS = [
    "Oracle-utility",
    "BestSingle",
    "Official-AvengersPro-balance",
    OURS_METHOD_NAME,
]


def _masked_argmax(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    neg_inf = -1e9
    masked = np.where(mask, values, neg_inf)
    return masked.argmax(axis=1)


def oracle_utility_route(data: dict, lambda_cost: float) -> np.ndarray:
    u = rebuild_grouped_rewards(data["test_perf"], data["test_cost"], lambda_cost=lambda_cost)
    return _masked_argmax(u, data["test_mask"])


def constant_route(n: int, idx: int) -> np.ndarray:
    return np.full(n, int(idx), dtype=np.int64)


def random_route(mask: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, _ = mask.shape
    chosen = np.zeros(n, dtype=np.int64)
    for i in range(n):
        avail = np.where(mask[i])[0]
        chosen[i] = int(rng.choice(avail)) if len(avail) else 0
    return chosen


def always_cheapest_route(cost: np.ndarray, mask: np.ndarray) -> np.ndarray:
    inf = np.inf
    masked = np.where(mask, cost, inf)
    return masked.argmin(axis=1)


def always_expensive_route(cost: np.ndarray, mask: np.ndarray) -> np.ndarray:
    neg_inf = -np.inf
    masked = np.where(mask, cost, neg_inf)
    return masked.argmax(axis=1)


def dataset_oracle_route(data: dict, lambda_cost: float) -> np.ndarray:
    train_ds = data.get("train_dataset_ids")
    test_ds = data.get("test_dataset_ids")
    if train_ds is None or test_ds is None:
        raise KeyError("dataset_oracle requires train_dataset_ids and test_dataset_ids")
    train_ds = np.asarray(train_ds)
    test_ds = np.asarray(test_ds)
    u = rebuild_grouped_rewards(data["train_perf"], data["train_cost"], lambda_cost=lambda_cost)
    best_by_ds: dict = {}
    for d in sorted(set(train_ds.tolist())):
        m = train_ds == d
        if not m.any():
            continue
        sub_u = u[m]
        sub_mask = data["train_mask"][m]
        means = []
        for j in range(sub_u.shape[1]):
            mk = sub_mask[:, j]
            means.append(float(sub_u[mk, j].mean()) if mk.any() else -1e9)
        best_by_ds[d] = int(np.argmax(means))
    n = len(test_ds)
    chosen = np.zeros(n, dtype=np.int64)
    for i in range(n):
        chosen[i] = best_by_ds.get(test_ds[i], 0)
    return chosen


def _train_sdf(data: dict, seed: int, epochs: int, ap_init_eps: float = 0.05) -> np.ndarray:
    d = data["train_h"].shape[1]
    k = len(data["model_names"])
    cost = torch.tensor(data["cost"], dtype=torch.float32)
    cfg = RDFLAPTrainConfig(seed=seed, epochs=epochs, ap_init_eps=ap_init_eps, init_mode="apinit")
    model = make_router(d, k, kind="sdf", cfg=cfg)
    train_regretrouter(
        model,
        torch.from_numpy(data["train_h"]).float(),
        torch.from_numpy(data["train_u"]).float(),
        torch.from_numpy(data["train_mask"]).bool(),
        torch.from_numpy(data["ap_balance_train_idx"]).long(),
        cost,
        torch.from_numpy(data["val_h"]).float(),
        torch.from_numpy(data["val_u"]).float(),
        torch.from_numpy(data["val_mask"]).bool(),
        torch.from_numpy(data["ap_balance_val_idx"]).long(),
        cfg,
    )
    return infer_choices(
        model,
        torch.from_numpy(data["test_h"]).float(),
        torch.from_numpy(data["test_mask"]).bool(),
        cost,
        torch.from_numpy(data["ap_balance_test_idx"]).long(),
        cfg=cfg,
    )


def baseline_routes(data: dict, seed: int, *, lambda_cost: float) -> dict[str, np.ndarray]:
    n = data["test_perf"].shape[0]
    bs = best_single_idx(data, lambda_cost)
    return {
        "Oracle-utility": oracle_utility_route(data, lambda_cost),
        "BestSingle": constant_route(n, bs),
        "Dataset-Oracle": dataset_oracle_route(data, lambda_cost),
        "Random": random_route(data["test_mask"], seed + int(lambda_cost * 1000)),
        "Always-Cheapest": always_cheapest_route(data["test_cost"], data["test_mask"]),
        "Always-Most-Expensive": always_expensive_route(data["test_cost"], data["test_mask"]),
        "Official-AvengersPro-simple": data["ap_test_idx"],
        "Official-AvengersPro-balance": data["ap_balance_test_idx"],
        "Official-AvengersPro-costfirst": data["ap_costfirst_test_idx"],
    }


def learned_routes(
    data: dict,
    seed: int,
    epochs: int,
    *,
    lambda_train: float,
    include_no_feedback: bool,
) -> dict[str, np.ndarray]:
    d = rebuild_utilities(data, lambda_cost=lambda_train)
    d = enrich_pool_data(d, seed, lambda_cost=lambda_train)
    out = {
        "S-DFL-MLP": _train_sdf(d, seed, epochs),
        OURS_METHOD_NAME: _train_rdfl(
            d,
            seed,
            RDFLAPTrainConfig(
                seed=seed,
                epochs=epochs,
                K=3,
                router_version="v1",
                predictor_mode="concat",
                init_mode="uniform",
                feedback_mode="full",
                loss_mode="regret",
                step_loss=False,
                weight_decay=1e-5,
                temperature=1.0,
            ),
        ),
    }
    if include_no_feedback:
        out["RegretRouter-no-feedback"] = _train_rdfl(
            d,
            seed,
            RDFLAPTrainConfig(
                seed=seed,
                epochs=epochs,
                K=3,
                ap_init_eps=0.05,
                init_mode="apinit",
                feedback_mode="none",
            ),
        )
    return out


def write_main_results(out_dir: Path, payload: dict) -> None:
    agg = payload["aggregated"]
    lambdas = payload["lambdas"]
    lines = [
        f"# Main Results (flagship 851, vs {AP_ANCHOR_METHOD})",
        "",
        f"Seeds: {payload['seeds']}; λ sweep: {lambdas}.",
        f"ΔAcc / ΔGap@O anchor: **{AP_ANCHOR_METHOD}** (pw=0.7, cs=0.3).",
        "Primary: **Gap@O↓**; secondary: Acc, AvgUtility, Gain@B, AvgCost.",
        "",
    ]
    allowed = set(MAIN_METHOD_ORDER)
    for lam in lambdas:
        sub = [a for a in agg if a.get("lambda") == lam and a.get("method") in allowed]
        lines.append(f"## λ = {lam}")
        lines.append("")
        lines.append(
            "| Method | Acc | AvgCost | AvgUtility | Gap@O | Gain@B | ΔAcc(pp) | ΔGap@O |"
        )
        lines.append("|--------|----:|--------:|-----------:|------:|-------:|---------:|--------:|")
        order = {m: i for i, m in enumerate(MAIN_METHOD_ORDER)}
        sub.sort(key=lambda a: order.get(a["method"], 99))
        for a in sub:
            lines.append(
                f"| {a['method']} | "
                f"{fmt_mean_std(a['avg_acc'], a.get('avg_acc_std', 0))} | "
                f"{fmt_mean_std(a['avg_cost'], a.get('avg_cost_std', 0), nd=6)} | "
                f"{fmt_mean_std(a['avg_utility'], a.get('avg_utility_std', 0))} | "
                f"{fmt_mean_std(a['gap_at_oracle'], a.get('gap_at_oracle_std', 0))} | "
                f"{fmt_mean_std(a['gain_at_best_single'], a.get('gain_at_best_single_std', 0))} | "
                f"{fmt_mean_std(a.get('delta_acc_pp', 0), a.get('delta_acc_pp_std', 0), nd=2)} | "
                f"{fmt_mean_std(a.get('delta_gap_at_oracle', 0), a.get('delta_gap_at_oracle_std', 0))} |"
            )
        lines.append("")
    (out_dir / "main_results.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "main_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


ABLATION_ROWS = [
    ("perf-only", "perf_only"),
    ("K=1", "k1"),
    ("K=2", "k2"),
    ("K=3", "k3"),
    ("K=5", "k5"),
    ("init-uniform", "init_uniform"),
    ("init-AP-eps0", "init_ap0"),
    ("init-AP-eps0.05", "init_ap005"),
    ("init-AP-eps0.1", "init_ap01"),
    ("feedback-none", "fb_none"),
    ("feedback-shuffle", "fb_shuffle"),
    ("feedback-detach", "fb_detach"),
    ("AP-simple", "ap_ref"),
    ("S-DFL-MLP", "sdf_ref"),
    ("R-DFL-K3-APinit", "rdfl_ref"),
]


def run_ablation_row(data: dict, seed: int, epochs: int, tag: str) -> tuple[str, np.ndarray]:
    base = enrich_pool_data(rebuild_utilities(data, lambda_cost=DEFAULT_LAMBDA), seed, lambda_cost=DEFAULT_LAMBDA)
    if tag == "perf_only":
        d = dict(data)
        for split in ("train", "val", "test"):
            d[f"{split}_u"] = data[f"{split}_perf"].astype(np.float32).copy()
        d = enrich_pool_data(d, seed, lambda_cost=0.0)
        return "perf-only", _train_rdfl(
            d, seed, RDFLAPTrainConfig(seed=seed, epochs=epochs, K=3, ap_init_eps=0.05, init_mode="apinit")
        )
    if tag.startswith("k"):
        k = int(tag[1:])
        return f"K={k}", _train_rdfl(
            base,
            seed,
            RDFLAPTrainConfig(seed=seed, epochs=epochs, K=k, ap_init_eps=0.05, init_mode="apinit", feedback_mode="full"),
        )
    if tag == "init_uniform":
        return "init-uniform", _train_rdfl(
            base,
            seed,
            RDFLAPTrainConfig(seed=seed, epochs=epochs, K=3, ap_init_eps=0.05, init_mode="uniform", feedback_mode="full"),
        )
    if tag == "init_ap0":
        return "init-AP-eps0", _train_rdfl(
            base,
            seed,
            RDFLAPTrainConfig(seed=seed, epochs=epochs, K=3, ap_init_eps=0.0, init_mode="apinit", feedback_mode="full"),
        )
    if tag == "init_ap005":
        return "init-AP-eps0.05", _train_rdfl(
            base,
            seed,
            RDFLAPTrainConfig(seed=seed, epochs=epochs, K=3, ap_init_eps=0.05, init_mode="apinit", feedback_mode="full"),
        )
    if tag == "init_ap01":
        return "init-AP-eps0.1", _train_rdfl(
            base,
            seed,
            RDFLAPTrainConfig(seed=seed, epochs=epochs, K=3, ap_init_eps=0.1, init_mode="apinit", feedback_mode="full"),
        )
    if tag == "fb_none":
        return "feedback-none", _train_rdfl(
            base,
            seed,
            RDFLAPTrainConfig(seed=seed, epochs=epochs, K=3, ap_init_eps=0.05, init_mode="apinit", feedback_mode="none"),
        )
    if tag == "fb_shuffle":
        return "feedback-shuffle", _train_rdfl(
            base,
            seed,
            RDFLAPTrainConfig(seed=seed, epochs=epochs, K=3, ap_init_eps=0.05, init_mode="apinit", feedback_mode="shuffle"),
        )
    if tag == "fb_detach":
        return "feedback-detach", _train_rdfl(
            base,
            seed,
            RDFLAPTrainConfig(seed=seed, epochs=epochs, K=3, ap_init_eps=0.05, init_mode="apinit", feedback_mode="detach"),
        )
    if tag == "ap_ref":
        return "AP-simple", base["ap_test_idx"]
    if tag == "sdf_ref":
        return "S-DFL-MLP", _train_sdf(base, seed, epochs)
    if tag == "rdfl_ref":
        return "R-DFL-K3-APinit", _train_rdfl(
            base,
            seed,
            RDFLAPTrainConfig(seed=seed, epochs=epochs, K=3, ap_init_eps=0.05, init_mode="apinit", feedback_mode="full"),
        )
    raise ValueError(tag)


def export_ablation(out_dir: Path, seeds: list[int], epochs: int, *, quick: bool) -> dict:
    rows: list[dict] = []
    tags = [t for _, t in ABLATION_ROWS]
    if quick:
        tags = ["ap_ref", "k3", "fb_none", "rdfl_ref", "perf_only"]
    for seed in seeds:
        set_seed(seed)
        data = load_pool_851(seed)
        for label, tag in ABLATION_ROWS:
            if tag not in tags:
                continue
            print(f"  ablation seed={seed} {label}...", flush=True)
            method, chosen = run_ablation_row(data, seed, epochs, tag)
            row = eval_at_lambda(data, method, chosen, lambda_eval=DEFAULT_LAMBDA, seed=seed)
            row["ablation"] = label
            rows.append(row)
    payload = {
        "lambda": DEFAULT_LAMBDA,
        "pool": "851",
        "seeds": seeds,
        "rows": rows,
        "aggregated": aggregate_rows(rows, ["ablation", "method"]),
    }
    agg = payload["aggregated"]
    lines = [
        f"# Ablation (851, λ={DEFAULT_LAMBDA}, Δ vs {AP_ANCHOR_METHOD})",
        "",
        f"Seeds: {seeds}.",
        "",
        "| Ablation | Method | Acc | Gain@B | Gap@O | AvgCost | ΔAcc(pp) | ΔGap@O |",
        "|----------|--------|----:|-------:|------:|--------:|---------:|--------:|",
    ]
    order = {a[0]: i for i, a in enumerate(ABLATION_ROWS)}
    agg.sort(key=lambda a: order.get(a.get("ablation", ""), 99))
    for a in agg:
        lines.append(
            f"| {a.get('ablation', a['method'])} | {a['method']} | "
            f"{fmt_mean_std(a['avg_acc'], a.get('avg_acc_std', 0))} | "
            f"{fmt_mean_std(a['gain_at_best_single'], a.get('gain_at_best_single_std', 0))} | "
            f"{fmt_mean_std(a['gap_at_oracle'], a.get('gap_at_oracle_std', 0))} | "
            f"{fmt_mean_std(a['avg_cost'], a.get('avg_cost_std', 0), nd=6)} | "
            f"{fmt_mean_std(a.get('delta_acc_pp', 0), a.get('delta_acc_pp_std', 0), nd=2)} | "
            f"{fmt_mean_std(a.get('delta_gap_at_oracle', 0), a.get('delta_gap_at_oracle_std', 0))} |"
        )
    (out_dir / "ablation_table.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "ablation_table.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def router_sota_routes(
    seed: int,
    *,
    cache_dir: Path,
    methods: list[str],
    train: bool,
    sota_epochs: int,
    gpu: str | None,
    quick_sota: bool,
) -> dict[str, np.ndarray]:
    if load_sota_routes is None or run_router_sota_method is None:
        raise RuntimeError("run_llmrouterbench_flagship.py is required for --router-sota")
    use_methods = [m for m in methods if m in ROUTER_SOTA_METHODS]
    if not train:
        cached = load_sota_routes(cache_dir, seed, use_methods)
        if cached is None:
            raise FileNotFoundError(
                f"Missing router SOTA cache under {cache_dir}/seed{seed}. "
                "Run scripts/run_llmrouterbench_flagship.py first."
            )
        return cached
    from run_llmrouterbench_flagship import DEFAULT_LB_ROOT, flagship_data_dir  # noqa: E402

    data_dir = flagship_data_dir(seed)
    ep = 3 if quick_sota else sota_epochs
    for method in use_methods:
        run_router_sota_method(
            method,
            data_dir=data_dir,
            out_root=cache_dir,
            seed=seed,
            epochs=ep,
            lb_root=DEFAULT_LB_ROOT,
            gpu=gpu,
            train=True,
            export_only=False,
            lambda_train=DEFAULT_LAMBDA,
        )
    cached = load_sota_routes(cache_dir, seed, use_methods)
    if cached is None:
        raise RuntimeError(f"Router SOTA training finished but cache missing for seed={seed}")
    return cached


def export_main(
    out_dir: Path,
    seeds: list[int],
    lambdas: list[float],
    epochs: int,
    *,
    lambda_retrain: bool,
    include_no_feedback: bool,
    quick: bool,
    router_sota: bool = False,
    router_sota_cache: Path | None = None,
    router_sota_train: bool = False,
    router_sota_methods: list[str] | None = None,
    router_sota_epochs: int = 30,
    router_sota_gpu: str | None = "0",
) -> dict:
    rows: list[dict] = []
    learned_cache: dict[tuple[int, float], dict[str, np.ndarray]] = {}

    for seed in seeds:
        set_seed(seed)
        base = load_pool_851(seed)
        print(f"\n=== main seed={seed} ===", flush=True)

        for lam in lambdas:
            routes = baseline_routes(base, seed, lambda_cost=lam)
            for method in MAIN_BASELINE_METHODS:
                rows.append(eval_at_lambda(base, method, routes[method], lambda_eval=lam, seed=seed))

        if lambda_retrain:
            for lam in lambdas:
                print(f"  learned retrain λ={lam}...", flush=True)
                learned_cache[(seed, lam)] = learned_routes(
                    base, seed, epochs, lambda_train=lam, include_no_feedback=include_no_feedback
                )
        else:
            print(f"  learned train once λ={DEFAULT_LAMBDA}...", flush=True)
            once = learned_routes(
                base, seed, epochs, lambda_train=DEFAULT_LAMBDA, include_no_feedback=include_no_feedback
            )
            for lam in lambdas:
                learned_cache[(seed, lam)] = once

        for lam in lambdas:
            for method, chosen in learned_cache[(seed, lam)].items():
                if method not in MAIN_LEARNED_METHODS:
                    continue
                rows.append(eval_at_lambda(base, method, chosen, lambda_eval=lam, seed=seed))

        if router_sota:
            cache = router_sota_cache or (project_root() / DEFAULT_ROUTER_SOTA_CACHE)
            methods = router_sota_methods or ROUTER_SOTA_METHODS
            print(f"  router SOTA (cache={cache})...", flush=True)
            sota = router_sota_routes(
                seed,
                cache_dir=cache,
                methods=methods,
                train=router_sota_train,
                sota_epochs=router_sota_epochs,
                gpu=router_sota_gpu,
                quick_sota=quick,
            )
            for lam in lambdas:
                for method, chosen in sota.items():
                    rows.append(eval_at_lambda(base, method, chosen, lambda_eval=lam, seed=seed))

    payload = {
        "pool": "851",
        "data_dir": "data/seed42_flagship",
        "seeds": seeds,
        "lambdas": lambdas,
        "lambda_retrain": lambda_retrain,
        "epochs": epochs,
        "router_sota": router_sota,
        "router_sota_cache": str(router_sota_cache or DEFAULT_ROUTER_SOTA_CACHE),
        "rows": rows,
        "aggregated": aggregate_rows(rows, ["method", "lambda"]),
    }
    write_main_results(out_dir, payload)
    return payload


def export_per_dataset(
    out_dir: Path,
    seeds: list[int],
    epochs: int,
    *,
    metric: str = "gap_at_oracle",
) -> dict:
    all_cells: list[dict] = []
    datasets: set = set()

    for seed in seeds:
        set_seed(seed)
        data = load_pool_851(seed)
        ap_anchor = data["ap_balance_test_idx"]
        routes = {
            "Oracle-utility": oracle_utility_route(data, DEFAULT_LAMBDA),
            "BestSingle": constant_route(len(ap_anchor), best_single_idx(data, DEFAULT_LAMBDA)),
            "Official-AvengersPro-balance": ap_anchor,
            OURS_METHOD_NAME: learned_routes(
                data, seed, epochs, lambda_train=DEFAULT_LAMBDA, include_no_feedback=False
            )[OURS_METHOD_NAME],
        }
        data["best_single_idx"] = best_single_idx(data, DEFAULT_LAMBDA)
        for method in PER_DATASET_METHODS:
            chosen = routes[method]
            for drow in per_dataset_breakdown(data, chosen, ap_anchor):
                ds = drow["dataset_id"]
                datasets.add(ds)
                all_cells.append(
                    {
                        "seed": seed,
                        "method": method,
                        "dataset_id": ds,
                        "n": drow["n"],
                        metric: drow.get(metric, drow.get("gap_at_oracle")),
                        "avg_acc": drow.get("avg_acc"),
                    }
                )

    ds_order = sorted(datasets)
    methods = PER_DATASET_METHODS
    agg: dict[tuple, list[float]] = {}
    for c in all_cells:
        key = (c["method"], c["dataset_id"])
        agg.setdefault(key, []).append(float(c[metric]))

    table = []
    for method in methods:
        row = {"method": method}
        for ds in ds_order:
            vals = agg.get((method, ds), [])
            row[ds] = mean(vals) if vals else None
            row[f"{ds}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
        table.append(row)

    payload = {
        "pool": "851",
        "lambda": DEFAULT_LAMBDA,
        "metric": metric,
        "seeds": seeds,
        "datasets": ds_order,
        "cells": all_cells,
        "table": table,
    }

    header = ["Method"] + ds_order
    lines = [
        f"# Per-dataset ({metric}, λ={DEFAULT_LAMBDA}, N=851)",
        "",
        f"Seeds: {seeds}.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in table:
        cells = [row["method"]]
        for ds in ds_order:
            v = row.get(ds)
            if v is None:
                cells.append("—")
            else:
                cells.append(fmt_mean_std(v, row.get(f"{ds}_std", 0)))
        lines.append("| " + " | ".join(cells) + " |")

    (out_dir / "per_dataset_table.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "per_dataset_table.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    p = argparse.ArgumentParser(
        description="Export paper tables (Main multi-λ, Ablation, Per-dataset) on flagship 851 pool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoke (seed 42, few epochs, λ=0.2 only):
  python scripts/export_main_tables.py --quick

  # Full paper tables (3 seeds, multi-λ, retrain learned per λ):
  python scripts/export_main_tables.py --seeds 41 42 43 --epochs 28

  # Main + ablation only (skip per-dataset):
  python scripts/export_main_tables.py --tables main ablation --seeds 42
        """,
    )
    p.add_argument("--output-dir", default="outputs/r_dfl_ap")
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--epochs", type=int, default=28)
    p.add_argument("--lambdas", nargs="+", type=float, default=None)
    p.add_argument(
        "--lambda-retrain",
        action="store_true",
        help="Retrain S-DFL / R-DFL for each λ (default: train once at λ=0.2, eval all λ)",
    )
    p.add_argument("--no-lambda-retrain", action="store_true", help="Alias: single train at λ=0.2")
    p.add_argument(
        "--tables",
        nargs="+",
        default=["main", "ablation", "per_dataset"],
        choices=["main", "ablation", "per_dataset"],
    )
    p.add_argument("--quick", action="store_true", help="seed 42, epochs=3, λ=[0.2], subset ablations")
    p.add_argument("--skip-no-feedback", action="store_true")
    p.add_argument(
        "--router-sota",
        action="store_true",
        help="Include LLMRouterBench SOTA rows from cache or --router-sota-train",
    )
    p.add_argument(
        "--router-sota-cache",
        type=Path,
        default=None,
        help=f"Cache dir for test_choices.npy (default: {DEFAULT_ROUTER_SOTA_CACHE})",
    )
    p.add_argument(
        "--router-sota-train",
        action="store_true",
        help="Train SOTA routers before export (slow; prefer run_llmrouterbench_flagship.py)",
    )
    p.add_argument(
        "--router-sota-methods",
        nargs="+",
        default=None,
        choices=ROUTER_SOTA_METHODS,
    )
    p.add_argument("--router-sota-epochs", type=int, default=30)
    p.add_argument("--router-sota-gpu", default="0")
    args = p.parse_args()

    if args.quick:
        args.seeds = [42]
        args.epochs = min(args.epochs, 3)
        lambdas = [DEFAULT_LAMBDA]
    else:
        lambdas = list(args.lambdas) if args.lambdas else list(DEFAULT_LAMBDAS)

    lambda_retrain = args.lambda_retrain and not args.no_lambda_retrain
    out = project_root() / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    summary = {"elapsed_sec": 0, "tables": args.tables, "seeds": args.seeds, "lambdas": lambdas}
    if "main" in args.tables:
        print("Export Main Results...", flush=True)
        main_payload = export_main(
            out,
            args.seeds,
            lambdas,
            args.epochs,
            lambda_retrain=lambda_retrain,
            include_no_feedback=not args.skip_no_feedback,
            quick=args.quick,
            router_sota=args.router_sota,
            router_sota_cache=args.router_sota_cache,
            router_sota_train=args.router_sota_train,
            router_sota_methods=args.router_sota_methods,
            router_sota_epochs=args.router_sota_epochs,
            router_sota_gpu=args.router_sota_gpu,
        )
        summary["main"] = {
            "n_rows": len(main_payload["rows"]),
            "n_agg": len(main_payload["aggregated"]),
            "files": ["main_results.md", "main_results.json"],
        }
    if "ablation" in args.tables:
        print("Export Ablation...", flush=True)
        ab_payload = export_ablation(out, args.seeds, args.epochs, quick=args.quick)
        summary["ablation"] = {"n_rows": len(ab_payload["rows"]), "files": ["ablation_table.md", "ablation_table.json"]}
    if "per_dataset" in args.tables:
        print("Export Per-dataset...", flush=True)
        pd_payload = export_per_dataset(out, args.seeds, args.epochs)
        summary["per_dataset"] = {
            "datasets": pd_payload["datasets"],
            "files": ["per_dataset_table.md", "per_dataset_table.json"],
        }

    summary["elapsed_sec"] = time.time() - t0
    (out / "export_tables_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nDone in {summary['elapsed_sec']:.0f}s -> {out}")


if __name__ == "__main__":
    main()
