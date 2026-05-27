"""Oracle reward construction."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np


def normalize_cost(cost: np.ndarray, *, mode: Literal["global", "per_query"] = "per_query", eps: float = 1e-8) -> np.ndarray:
    values = cost.astype(np.float32)
    if mode == "global":
        c_min = float(values.min())
        c_max = float(values.max())
        return (values - c_min) / (c_max - c_min + eps)
    c_min = values.min(axis=1, keepdims=True)
    c_max = values.max(axis=1, keepdims=True)
    return (values - c_min) / (c_max - c_min + eps)


def build_oracle_reward(
    quality: np.ndarray,
    cost: np.ndarray,
    *,
    lambda_cost: float = 0.2,
    cost_mode: Literal["global", "per_query"] = "per_query",
    quality_key: str = "performance",
) -> np.ndarray:
    """
    R_{q,k} = quality_{q,k} - lambda * normalized_cost_{q,k}

    quality can be performance (0/1), judge score, etc. Larger is better.
    """
    q = quality.astype(np.float32)
    if lambda_cost <= 0:
        return q
    c_norm = normalize_cost(cost, mode=cost_mode)
    reward = q - float(lambda_cost) * c_norm
    return reward.astype(np.float32)


def rebuild_grouped_rewards(
    performance: np.ndarray,
    cost: np.ndarray,
    *,
    lambda_cost: float = 0.2,
    cost_mode: Literal["global", "per_query"] = "per_query",
) -> np.ndarray:
    return build_oracle_reward(performance, cost, lambda_cost=lambda_cost, cost_mode=cost_mode)
