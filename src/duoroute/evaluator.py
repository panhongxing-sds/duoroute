"""Evaluation helpers."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from duoroute.metrics import (
    RoutingMetrics,
    compute_routing_metrics,
    compute_routing_metrics_from_choices,
)


def evaluate_predictions(
    oracle_reward: np.ndarray,
    pred_reward: np.ndarray,
    mask: np.ndarray,
    *,
    performance: Optional[np.ndarray] = None,
    cost: Optional[np.ndarray] = None,
    random_seed: int = 42,
) -> RoutingMetrics:
    return compute_routing_metrics(
        oracle_reward,
        pred_reward,
        mask,
        performance=performance,
        cost=cost,
        random_seed=random_seed,
    )


def evaluate_choices(
    oracle_reward: np.ndarray,
    mask: np.ndarray,
    chosen: np.ndarray,
    *,
    performance: Optional[np.ndarray] = None,
    cost: Optional[np.ndarray] = None,
    random_seed: int = 42,
) -> RoutingMetrics:
    return compute_routing_metrics_from_choices(
        oracle_reward,
        mask,
        chosen,
        performance=performance,
        cost=cost,
        random_seed=random_seed,
    )


def pareto_points(
    metrics_by_method: Dict[str, RoutingMetrics],
) -> Dict[str, dict]:
    return {
        name: {"avg_reward": m.avg_reward, "avg_acc": m.avg_acc, "avg_cost": m.avg_cost}
        for name, m in metrics_by_method.items()
    }
