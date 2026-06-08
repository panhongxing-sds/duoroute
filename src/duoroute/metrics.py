"""Routing evaluation metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence

import numpy as np


@dataclass
class RoutingMetrics:
    avg_reward: float
    avg_acc: float
    routing_regret: float
    top1_oracle_match: float
    near_tie_rate: float
    avg_cost: float
    oracle_reward: float
    random_reward: float
    n_queries: int

    def to_dict(self) -> Dict[str, float]:
        return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in asdict(self).items()}


def _masked_argmax(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    neg_inf = -1e9
    masked = np.where(mask, values, neg_inf)
    return masked.argmax(axis=1)


def compute_routing_metrics_from_choices(
    oracle_reward: np.ndarray,
    mask: np.ndarray,
    chosen: np.ndarray,
    *,
    performance: Optional[np.ndarray] = None,
    cost: Optional[np.ndarray] = None,
    random_seed: int = 42,
    near_tie_eps: float = 0.01,
) -> RoutingMetrics:
    mask = mask.astype(bool)
    neg_inf = -1e9
    true_m = np.where(mask, oracle_reward, neg_inf)
    perf_m = np.where(mask, performance if performance is not None else oracle_reward, neg_inf)
    oracle_best = true_m.max(axis=1)
    idx = np.arange(oracle_reward.shape[0])
    routed_reward = true_m[idx, chosen]
    routed_perf = perf_m[idx, chosen]
    regret = oracle_best - routed_reward
    oracle_idx = _masked_argmax(true_m, mask)
    top1_match = float((chosen == oracle_idx).mean())
    near_tie = float((regret <= near_tie_eps).mean())
    rng = np.random.default_rng(random_seed)
    random_choice = []
    for i in range(oracle_reward.shape[0]):
        avail = np.where(mask[i])[0]
        random_choice.append(int(rng.choice(avail)) if len(avail) else 0)
    random_choice = np.asarray(random_choice)
    random_reward = float(true_m[idx, random_choice].mean())
    avg_cost = 0.0
    if cost is not None:
        avg_cost = float(cost[idx, chosen].mean())
    return RoutingMetrics(
        avg_reward=float(routed_reward.mean()),
        avg_acc=float(routed_perf.mean()),
        routing_regret=float(regret.mean()),
        top1_oracle_match=top1_match,
        near_tie_rate=near_tie,
        avg_cost=avg_cost,
        oracle_reward=float(oracle_best.mean()),
        random_reward=random_reward,
        n_queries=int(oracle_reward.shape[0]),
    )


def compute_routing_metrics(
    oracle_reward: np.ndarray,
    pred_reward: np.ndarray,
    mask: np.ndarray,
    *,
    performance: Optional[np.ndarray] = None,
    cost: Optional[np.ndarray] = None,
    random_seed: int = 42,
    near_tie_eps: float = 0.01,
) -> RoutingMetrics:
    mask = mask.astype(bool)
    neg_inf = -1e9
    true_m = np.where(mask, oracle_reward, neg_inf)
    pred_m = np.where(mask, pred_reward, neg_inf)
    perf_m = np.where(mask, performance if performance is not None else oracle_reward, neg_inf)

    oracle_best = true_m.max(axis=1)
    chosen = _masked_argmax(pred_m, mask)
    idx = np.arange(oracle_reward.shape[0])
    routed_reward = true_m[idx, chosen]
    routed_perf = perf_m[idx, chosen]
    regret = oracle_best - routed_reward

    oracle_idx = _masked_argmax(true_m, mask)
    top1_match = float((chosen == oracle_idx).mean())
    near_tie = float((regret <= near_tie_eps).mean())

    rng = np.random.default_rng(random_seed)
    random_choice = []
    for i in range(oracle_reward.shape[0]):
        avail = np.where(mask[i])[0]
        random_choice.append(int(rng.choice(avail)) if len(avail) else 0)
    random_choice = np.asarray(random_choice)
    random_reward = float(true_m[idx, random_choice].mean())

    avg_cost = 0.0
    if cost is not None:
        avg_cost = float(cost[idx, chosen].mean())

    return RoutingMetrics(
        avg_reward=float(routed_reward.mean()),
        avg_acc=float(routed_perf.mean()),
        routing_regret=float(regret.mean()),
        top1_oracle_match=top1_match,
        near_tie_rate=near_tie,
        avg_cost=avg_cost,
        oracle_reward=float(oracle_best.mean()),
        random_reward=random_reward,
        n_queries=int(oracle_reward.shape[0]),
    )
