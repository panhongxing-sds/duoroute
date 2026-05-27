"""LLMRouterBench-aligned routing metrics and Pareto frontier helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from duoroute.reward_builder import build_oracle_reward, normalize_cost


@dataclass
class BenchRoutingMetrics:
    sample_avg_acc: float
    avg_acc: float
    gain_at_best_single: float
    gain_at_random: float
    gap_at_oracle: float
    routing_regret: float
    avg_cost: float
    cost_save_vs_best_single: float
    n_queries: int
    best_single_model_idx: int


def _masked_argmax(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    neg_inf = -1e9
    masked = np.where(mask, values, neg_inf)
    return masked.argmax(axis=1)


def oracle_route_performance_tiebreak_cost(
    performance: np.ndarray,
    cost: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Oracle: max performance; tie-break lowest cost."""
    n, k = performance.shape
    chosen = np.zeros(n, dtype=np.int64)
    neg_inf = -1e9
    perf_m = np.where(mask, performance, neg_inf)
    cost_m = np.where(mask, cost, np.inf)
    for i in range(n):
        best_perf = perf_m[i].max()
        candidates = np.where((perf_m[i] >= best_perf - 1e-9) & mask[i])[0]
        if len(candidates) == 0:
            chosen[i] = int(np.argmax(perf_m[i]))
            continue
        chosen[i] = int(candidates[np.argmin(cost_m[i, candidates])])
    return chosen


def pred_matrix_from_choices(n: int, k: int, chosen: np.ndarray) -> np.ndarray:
    pred = np.full((n, k), -1e9, dtype=np.float32)
    pred[np.arange(n), chosen] = 1.0
    return pred


def compute_bench_metrics(
    *,
    performance: np.ndarray,
    cost: np.ndarray,
    mask: np.ndarray,
    pred_u: np.ndarray,
    true_u: np.ndarray,
    best_single_idx: int,
    random_seed: int = 42,
) -> BenchRoutingMetrics:
    mask = mask.astype(bool)
    neg_inf = -1e9
    true_m = np.where(mask, true_u, neg_inf)
    perf_m = np.where(mask, performance, neg_inf)
    pred_m = np.where(mask, pred_u, neg_inf)

    n = performance.shape[0]
    batch = np.arange(n)
    chosen = _masked_argmax(pred_m, mask)
    routed_u = true_m[batch, chosen]
    routed_perf = perf_m[batch, chosen]

    oracle_u = true_m.max(axis=1)
    regret = oracle_u - routed_u

    rng = np.random.default_rng(random_seed)
    rand_chosen = []
    for i in range(n):
        avail = np.where(mask[i])[0]
        rand_chosen.append(int(rng.choice(avail)) if len(avail) else 0)
    rand_chosen = np.asarray(rand_chosen)
    random_u = true_m[batch, rand_chosen]

    bs_u = float(true_m[:, best_single_idx].mean()) if mask[:, best_single_idx].any() else 0.0
    avg_u = float(routed_u.mean())
    sample_acc = float(routed_perf.mean())

    avg_cost = float(cost[batch, chosen].mean())
    bs_cost = float(cost[mask[:, best_single_idx], best_single_idx].mean()) if mask[:, best_single_idx].any() else 0.0
    cost_save = float((bs_cost - avg_cost) / max(bs_cost, 1e-12))

    return BenchRoutingMetrics(
        sample_avg_acc=sample_acc,
        avg_acc=avg_u,
        gain_at_best_single=avg_u - bs_u,
        gain_at_random=avg_u - float(random_u.mean()),
        gap_at_oracle=float(regret.mean()),
        routing_regret=float(regret.mean()),
        avg_cost=avg_cost,
        cost_save_vs_best_single=cost_save,
        n_queries=n,
        best_single_model_idx=best_single_idx,
    )


def best_single_idx_by_train(
    train_perf: np.ndarray,
    train_mask: np.ndarray,
    *,
    by: str = "performance",
    train_cost: Optional[np.ndarray] = None,
    lambda_cost: float = 0.2,
) -> int:
    if by == "performance":
        matrix = train_perf
    elif by == "oracle_reward":
        if train_cost is None:
            raise ValueError("train_cost required for oracle_reward selection")
        matrix = build_oracle_reward(train_perf, train_cost, lambda_cost=lambda_cost)
    else:
        raise ValueError(by)
    means = []
    for k in range(matrix.shape[1]):
        mk = train_mask[:, k]
        means.append(float(matrix[mk, k].mean()) if mk.any() else -1e9)
    return int(np.argmax(means))


def duoroute_utility_sweep(
    pred_a: np.ndarray,
    cost: np.ndarray,
    mask: np.ndarray,
    lambdas: Sequence[float],
    *,
    train_lambda: float = 0.2,
) -> Dict[float, np.ndarray]:
    c_norm = normalize_cost(cost, mode="per_query")
    out: Dict[float, np.ndarray] = {}
    for lam in lambdas:
        utility = pred_a + (train_lambda - float(lam)) * c_norm
        out[float(lam)] = np.where(mask, utility, -1e9)
    return out


def pareto_frontier_points(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Non-dominated (cost, acc) points; minimize cost, maximize acc."""
    pareto: List[Tuple[float, float]] = []
    for i, (c1, a1) in enumerate(points):
        dominated = False
        for j, (c2, a2) in enumerate(points):
            if i == j:
                continue
            if c2 <= c1 and a2 >= a1 and (c2 < c1 or a2 > a1):
                dominated = True
                break
        if not dominated:
            pareto.append((c1, a1))
    return sorted(pareto)


def pareto_distance(cost: float, acc: float, frontier: List[Tuple[float, float]]) -> float:
    if not frontier:
        return 0.0
    # Normalize by global scale for distance
    costs = [p[0] for p in frontier] + [cost]
    accs = [p[1] for p in frontier] + [acc]
    c_scale = max(max(costs) - min(costs), 1e-12)
    a_scale = max(max(accs) - min(accs), 1e-12)
    best = min(
        np.sqrt(((cost - fc) / c_scale) ** 2 + ((acc - fa) / a_scale) ** 2)
        for fc, fa in frontier
    )
    return float(best)


@dataclass
class FrontierSummary:
    best_avg_acc: float
    perf_gain: float
    lowest_cost_at_least_bs_acc: float
    cost_save: float
    pareto_dist: float
    n_configs: int


def summarize_method_frontier(
    configs: List[Tuple[float, float, float]],
    *,
    bs_acc: float,
    bs_cost: float,
    global_frontier: List[Tuple[float, float]],
) -> FrontierSummary:
    """configs: list of (avg_acc, avg_cost, lambda_or_weight)."""
    if not configs:
        return FrontierSummary(0, 0, 0, 0, 0, 0)
    accs = [c[0] for c in configs]
    costs = [c[1] for c in configs]
    best_acc = max(accs)
    perf_gain = best_acc / max(bs_acc, 1e-12) - 1.0

    eligible = [(a, co) for a, co in zip(accs, costs) if a >= bs_acc - 1e-9]
    if eligible:
        min_cost = min(co for _, co in eligible)
    else:
        min_cost = min(costs)
    cost_save = 1.0 - min_cost / max(bs_cost, 1e-12)

    dists = [pareto_distance(co, acc, global_frontier) for acc, co in zip(accs, costs)]
    return FrontierSummary(
        best_avg_acc=best_acc,
        perf_gain=perf_gain,
        lowest_cost_at_least_bs_acc=min_cost,
        cost_save=cost_save,
        pareto_dist=float(np.mean(dists)),
        n_configs=len(configs),
    )
