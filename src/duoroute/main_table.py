"""Shared helpers for RegretRouter main-table export and multi-seed runs."""

from __future__ import annotations

from statistics import mean, pstdev

import numpy as np
import torch

from duoroute.bench_metrics import best_single_idx_by_train
from duoroute.pool_data import (
    DEFAULT_LAMBDA,
    enrich_pool_data,
    eval_row,
    load_flagship,
    rebuild_utilities,
)
from duoroute.regretrouter import (
    OURS_METHOD_NAME,
    RDFLAPTrainConfig,
    infer_choices,
    make_router,
    train_regretrouter,
)

DEFAULT_LAMBDAS = [0.0, 0.1, 0.2, 0.5, 0.8]
METRIC_COLS = [
    "avg_acc",
    "avg_cost",
    "avg_utility",
    "gap_at_oracle",
    "gain_at_best_single",
    "delta_acc_pp",
    "delta_gap_at_oracle",
]


def best_single_idx(data: dict, lambda_cost: float) -> int:
    return best_single_idx_by_train(
        data["train_perf"],
        data["train_mask"],
        by="oracle_reward",
        train_cost=data["train_cost"],
        lambda_cost=lambda_cost,
    )


def eval_at_lambda(
    data: dict,
    method: str,
    chosen: np.ndarray,
    *,
    lambda_eval: float,
    seed: int,
) -> dict:
    d = rebuild_utilities(data, lambda_cost=lambda_eval)
    bs = best_single_idx(d, lambda_eval)
    row = eval_row(
        d["test_perf"],
        d["test_u"],
        d["test_mask"],
        d["test_cost"],
        d["ap_balance_test_idx"],
        chosen,
        best_single_idx=bs,
        random_seed=seed,
        lambda_cost=lambda_eval,
        train_perf=d["train_perf"],
        train_mask=d["train_mask"],
        train_cost=d["train_cost"],
    )
    row["method"] = method
    row["lambda"] = lambda_eval
    row["pool"] = data.get("name", "851")
    row["seed"] = seed
    return row


def train_regretrouter_route(data: dict, seed: int, cfg: RDFLAPTrainConfig) -> np.ndarray:
    d = data["train_h"].shape[1]
    k = len(data["model_names"])
    cost = torch.tensor(data["cost"], dtype=torch.float32)
    model = make_router(d, k, kind="rdfl", cfg=cfg, cost=cost)
    train_perf = (
        torch.from_numpy(data["train_perf"]).float()
        if cfg.rare_weight_mode != "none"
        else None
    )
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
        train_perf=train_perf,
    )
    return infer_choices(
        model,
        torch.from_numpy(data["test_h"]).float(),
        torch.from_numpy(data["test_mask"]).bool(),
        cost,
        torch.from_numpy(data["ap_balance_test_idx"]).long(),
        cfg=cfg,
    )


# Backward-compatible alias used by export_main_tables / run_multiseed_main_tables.
_train_rdfl = train_regretrouter_route


def load_pool_851(seed: int, lambda_cost: float = DEFAULT_LAMBDA) -> dict:
    raw = load_flagship(seed, filter_four=False)
    data = enrich_pool_data(raw, seed, lambda_cost=lambda_cost)
    data["name"] = "851"
    return data


def aggregate_rows(rows: list[dict], key_fields: list[str]) -> list[dict]:
    from collections import defaultdict

    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        k = tuple(r.get(f) for f in key_fields)
        buckets[k].append(r)
    out = []
    for k, rs in sorted(buckets.items()):
        agg = {key_fields[i]: k[i] for i in range(len(key_fields))}
        agg["n_seeds"] = len(rs)
        for field in METRIC_COLS + ["rescued", "harmed", "net"]:
            vals = [float(x[field]) for x in rs if field in x]
            if vals:
                agg[field] = mean(vals)
                agg[f"{field}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
        out.append(agg)
    return out


def fmt_mean_std(val: float, std: float, *, nd: int = 4) -> str:
    if std and std > 1e-12:
        return f"{val:.{nd}f}±{std:.{nd}f}"
    return f"{val:.{nd}f}"


__all__ = [
    "DEFAULT_LAMBDAS",
    "METRIC_COLS",
    "OURS_METHOD_NAME",
    "aggregate_rows",
    "best_single_idx",
    "eval_at_lambda",
    "fmt_mean_std",
    "load_pool_851",
    "train_regretrouter_route",
    "_train_rdfl",
]
