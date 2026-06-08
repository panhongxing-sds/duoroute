"""Fallback threshold calibration on dev set."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from duoroute.evaluator import evaluate_predictions
from duoroute.inference import route_with_fallback


def calibrate_fallback_threshold(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    oracle_reward: np.ndarray,
    mask: np.ndarray,
    *,
    performance: np.ndarray,
    cost: np.ndarray,
    thresholds: List[float] | None = None,
) -> List[Dict[str, float]]:
    if thresholds is None:
        thresholds = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    rows: List[Dict[str, float]] = []
    for delta in thresholds:
        decision = route_with_fallback(pred_a, pred_b, mask, delta=delta)
        idx = np.arange(len(decision.chosen))
        routed_reward = oracle_reward[idx, decision.chosen]
        routed_perf = performance[idx, decision.chosen]
        routed_cost = cost[idx, decision.chosen]
        regret = oracle_reward.max(axis=1) - routed_reward
        rows.append(
            {
                "threshold": float(delta),
                "avg_reward": float(routed_reward.mean()),
                "avg_acc": float(routed_perf.mean()),
                "avg_cost": float(routed_cost.mean()),
                "fallback_rate": float(decision.fallback_used.mean()),
                "routing_regret": float(regret.mean()),
            }
        )
    return rows
