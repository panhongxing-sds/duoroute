"""Oracle reward construction."""

from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np

CostPenalty = Literal["linear", "exp", "binary_clip_exp", "binary_multiplicative"]


def normalize_cost(
    cost: np.ndarray,
    *,
    mode: Literal["global", "per_query"] = "per_query",
    eps: float = 1e-8,
    ignore_zero_cost: bool | None = None,
) -> np.ndarray:
    """
    Min-max normalize cost to [0, 1].

    For ``per_query`` mode, default ``ignore_zero_cost=True``: missing/zero costs
  (e.g. unrouted models) are not used as the per-query minimum, so they do not
  automatically get the lowest penalty.
    """
    if ignore_zero_cost is None:
        ignore_zero_cost = mode == "per_query"
    values = cost.astype(np.float32)
    if mode == "global":
        c_min = float(values.min())
        c_max = float(values.max())
        return (values - c_min) / (c_max - c_min + eps)
    if not ignore_zero_cost:
        c_min = values.min(axis=1, keepdims=True)
        c_max = values.max(axis=1, keepdims=True)
        return (values - c_min) / (c_max - c_min + eps)
    pos = values > 0
    c_max = values.max(axis=1, keepdims=True)
    masked_min = np.where(pos, values, np.inf)
    c_min_pos = masked_min.min(axis=1, keepdims=True)
    c_min_pos = np.where(np.isfinite(c_min_pos), c_min_pos, 0.0)
    denom = c_max - c_min_pos + eps
    return np.where(pos, (values - c_min_pos) / denom, 0.0).astype(np.float32)


def normalize_quality(
    quality: np.ndarray,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """Per-query min-max normalize quality/performance to [0, 1]."""
    q = quality.astype(np.float32)
    q_min = q.min(axis=1, keepdims=True)
    q_max = q.max(axis=1, keepdims=True)
    return (q - q_min) / (q_max - q_min + eps)


def raw_perf_utility_matrix(
    perf: np.ndarray,
    cost: np.ndarray,
    *,
    lambda_cost: float,
    cost_mode: Literal["global", "per_query"] = "per_query",
) -> np.ndarray:
    """
    Raw 0/1 performance utility (no perf_tilde).

    U(q,m;λ) = (1-λ)·perf_{q,m} + λ·(1 - c_norm_{q,m})
    with per-query min-max cost normalization (ignore_zero_cost=True).
    """
    lam = float(lambda_cost)
    c_norm = normalize_cost(
        cost,
        mode=cost_mode,
        ignore_zero_cost=True if cost_mode == "per_query" else False,
    )
    p = perf.astype(np.float32)
    return ((1.0 - lam) * p + lam * (1.0 - c_norm)).astype(np.float32)


def raw_perf_routed_utility(
    perf_matrix: np.ndarray,
    chosen: np.ndarray,
    routed_cost: np.ndarray,
    per_query_cost_matrix: np.ndarray,
    *,
    lambda_cost: float,
) -> np.ndarray:
    """Routed utility with raw perf and per-query cost bounds (cumulative cost allowed >1)."""
    idx = np.arange(len(chosen))
    lam = float(lambda_cost)
    q = perf_matrix.astype(np.float32)[idx, chosen.astype(np.int64)]
    c_norm = normalize_routed_cost(routed_cost, per_query_cost_matrix)
    return ((1.0 - lam) * q + lam * (1.0 - c_norm)).astype(np.float32)


def raw_perf_cascade_oracle_utility(
    perf_matrix: np.ndarray,
    cost_matrix: np.ndarray,
    mask: np.ndarray,
    m1: np.ndarray,
    *,
    lambda_cost: float,
) -> np.ndarray:
    """Best two-call cascade utility per query (raw perf); oracle for Regret@O_cas."""
    perf = perf_matrix.astype(np.float32)
    cost = cost_matrix.astype(np.float32)
    mask = mask.astype(bool)
    n, _ = perf.shape
    idx = np.arange(n)
    lam = float(lambda_cost)

    c1 = cost[idx, m1.astype(np.int64)]
    u_stop = raw_perf_routed_utility(perf, m1, c1, cost, lambda_cost=lambda_cost)

    c1_col = c1[:, None]
    cum_cost = c1_col + cost
    c_min_pos, c_max = per_query_cost_bounds(cost, ignore_zero_cost=True)
    denom = c_max - c_min_pos + 1e-8
    c_norm_cum = np.where(
        cum_cost > 0,
        (cum_cost - c_min_pos[:, None]) / denom[:, None],
        0.0,
    ).astype(np.float32)
    u_reroute = (1.0 - lam) * perf + lam * (1.0 - c_norm_cum)
    u_reroute = np.where(mask, u_reroute, -1e9)
    u_best_reroute = u_reroute.max(axis=1)
    return np.maximum(u_stop, u_best_reroute).astype(np.float32)


def llmrouterbench_linear_utility(
    quality: np.ndarray,
    cost: np.ndarray,
    *,
    lambda_cost: float,
    cost_mode: Literal["global", "per_query"] = "per_query",
) -> np.ndarray:
    """LLMRouterBench: U = alpha*perf_tilde + (1-alpha)*(1 - c_norm), alpha = 1 - lambda_cost."""
    alpha = 1.0 - float(lambda_cost)
    q_tilde = normalize_quality(quality)
    c_norm = normalize_cost(
        cost,
        mode=cost_mode,
        ignore_zero_cost=True if cost_mode == "per_query" else False,
    )
    return (alpha * q_tilde + (1.0 - alpha) * (1.0 - c_norm)).astype(np.float32)


def compute_cost_penalty(
    cost: np.ndarray,
    *,
    lambda_cost: float = 0.2,
    cost_mode: Literal["global", "per_query"] = "per_query",
    cost_penalty: CostPenalty = "linear",
    exp_gamma: float | None = None,
) -> np.ndarray:
    """
    Cost penalty term (non-negative, larger = more expensive).

    linear: ``lambda * c_norm``
    exp:    ``exp(gamma * c_norm) - 1``  with ``gamma = exp_gamma or lambda_cost``

    ``c_norm`` uses per-query min-max with zero-cost ignored (see ``normalize_cost``).
    """
    c_norm = normalize_cost(
        cost,
        mode=cost_mode,
        ignore_zero_cost=True if cost_mode == "per_query" else False,
    )
    if cost_penalty == "linear":
        return (float(lambda_cost) * c_norm).astype(np.float32)
    if cost_penalty == "exp":
        gamma = float(lambda_cost if exp_gamma is None else exp_gamma)
        x = np.clip(gamma * c_norm, 0.0, 50.0)
        return np.expm1(x).astype(np.float32)
    raise ValueError(f"Unknown cost_penalty: {cost_penalty}")


def build_oracle_reward(
    quality: np.ndarray,
    cost: np.ndarray,
    *,
    lambda_cost: float = 0.2,
    cost_mode: Literal["global", "per_query"] = "per_query",
    cost_penalty: CostPenalty = "linear",
    exp_gamma: float | None = None,
    quality_key: str = "performance",
) -> np.ndarray:
    """
    Oracle utility (larger is better).

    linear (LLMRouterBench): ``U = alpha * perf_tilde + (1-alpha) * (1 - c_norm)``,
    ``alpha = 1 - lambda_cost``; per-query min-max on perf and cost (ignore_zero_cost).

    exp (legacy): ``R = quality - (exp(gamma * c_norm) - 1)``  (``gamma`` defaults to ``lambda_cost``)

    quality can be performance (0/1), judge score, etc.
    """
    q = quality.astype(np.float32)
    c_norm = normalize_cost(
        cost,
        mode=cost_mode,
        ignore_zero_cost=True if cost_mode == "per_query" else False,
    )
    if cost_penalty == "binary_clip_exp":
        # Binary split:
        # - correct (q>=0.5): always positive, clipped by ``pos_floor``
        # - incorrect (q<0.5): non-positive exponential penalty
        q_bin = (q >= 0.5).astype(np.float32)
        pos_floor = 0.05
        pos = np.clip(1.0 - float(lambda_cost) * c_norm, pos_floor, 1.0)
        neg_gamma = float(4.0 if exp_gamma is None else exp_gamma)
        neg = -np.expm1(np.clip(neg_gamma * c_norm, 0.0, 50.0))
        return np.where(q_bin > 0.0, pos, neg).astype(np.float32)
    if cost_penalty == "binary_multiplicative":
        # Symmetric multiplicative shaping for binary quality.
        # correct:  exp(-gamma*c_norm) in (0,1]
        # incorrect: -(1-exp(-gamma*c_norm)) in [-1,0)
        q_bin = (q >= 0.5).astype(np.float32)
        gamma = float(1.0 if exp_gamma is None else exp_gamma)
        decay = np.exp(-np.clip(gamma * c_norm, 0.0, 50.0)).astype(np.float32)
        return (q_bin * decay - (1.0 - q_bin) * (1.0 - decay)).astype(np.float32)
    if cost_penalty == "linear":
        return llmrouterbench_linear_utility(
            q, cost, lambda_cost=lambda_cost, cost_mode=cost_mode,
        )
    pen = compute_cost_penalty(
        cost,
        lambda_cost=lambda_cost,
        cost_mode=cost_mode,
        cost_penalty=cost_penalty,
        exp_gamma=exp_gamma,
    )
    return (q - pen).astype(np.float32)


def per_query_cost_bounds(
    cost: np.ndarray,
    *,
    ignore_zero_cost: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-query (c_min_pos, c_max) matching ``normalize_cost(..., mode='per_query')``."""
    values = cost.astype(np.float32)
    if ignore_zero_cost:
        pos = values > 0
        c_max = values.max(axis=1)
        masked_min = np.where(pos, values, np.inf)
        c_min_pos = masked_min.min(axis=1)
        c_min_pos = np.where(np.isfinite(c_min_pos), c_min_pos, 0.0)
    else:
        c_min_pos = values.min(axis=1)
        c_max = values.max(axis=1)
    return c_min_pos.astype(np.float32), c_max.astype(np.float32)


def normalize_routed_cost(
    routed_cost: np.ndarray,
    per_query_cost_matrix: np.ndarray,
    *,
    eps: float = 1e-8,
    ignore_zero_cost: bool = True,
) -> np.ndarray:
    """
    Normalize per-query routed costs with the same per-query bounds as oracle ``c_norm``.

    ``routed_cost`` may be a single-model lookup or cascade_true cumulative sum; cumulative
    values can exceed per-query ``c_max`` (normalized cost may be >1).
    """
    c_min_pos, c_max = per_query_cost_bounds(
        per_query_cost_matrix, ignore_zero_cost=ignore_zero_cost,
    )
    denom = c_max - c_min_pos + eps
    routed = routed_cost.astype(np.float32)
    return np.where(routed > 0, (routed - c_min_pos) / denom, 0.0).astype(np.float32)


def routed_llmrouterbench_utility(
    perf_matrix: np.ndarray,
    chosen: np.ndarray,
    routed_cost: np.ndarray,
    per_query_cost_matrix: np.ndarray,
    *,
    lambda_cost: float = 0.2,
) -> np.ndarray:
    """
    LLMRouterBench routed utility aligned with ``build_oracle_reward`` / ``test_u``.

    ``U = alpha * perf_tilde[chosen] + (1-alpha) * (1 - c_norm_routed)``;
    ``perf_tilde`` from per-query min-max over ``perf_matrix``; cost bounds from
    ``per_query_cost_matrix`` (cumulative routed cost may exceed per-query ``c_max``).
    """
    idx = np.arange(len(chosen))
    q_tilde = normalize_quality(perf_matrix.astype(np.float32))
    q = q_tilde[idx, chosen.astype(np.int64)]
    c_norm = normalize_routed_cost(routed_cost, per_query_cost_matrix)
    alpha = 1.0 - float(lambda_cost)
    return (alpha * q + (1.0 - alpha) * (1.0 - c_norm)).astype(np.float32)


def cascade_oracle_utility(
    perf_matrix: np.ndarray,
    cost_matrix: np.ndarray,
    mask: np.ndarray,
    m1: np.ndarray,
    *,
    lambda_cost: float = 0.2,
) -> np.ndarray:
    """
    Best cascade_true utility per query: keep m1 or reroute to any m2 (pay c(m1)+c(m2)).

    Same LLMRouterBench formula as ``build_oracle_reward`` / ``routed_llmrouterbench_utility``.
    Use as the oracle reference for Gap@O when evaluating two-call cascades.
    """
    perf = perf_matrix.astype(np.float32)
    cost = cost_matrix.astype(np.float32)
    mask = mask.astype(bool)
    n, k = perf.shape
    idx = np.arange(n)
    q_tilde = normalize_quality(perf)
    alpha = 1.0 - float(lambda_cost)

    c1 = cost[idx, m1.astype(np.int64)]
    u_stop = routed_llmrouterbench_utility(
        perf, m1, c1, cost, lambda_cost=lambda_cost,
    )

    c1_col = c1[:, None]
    cum_cost = c1_col + cost
    c_min_pos, c_max = per_query_cost_bounds(cost, ignore_zero_cost=True)
    denom = c_max - c_min_pos + 1e-8
    c_norm_cum = np.where(
        cum_cost > 0,
        (cum_cost - c_min_pos[:, None]) / denom[:, None],
        0.0,
    ).astype(np.float32)
    u_reroute = alpha * q_tilde + (1.0 - alpha) * (1.0 - c_norm_cum)
    u_reroute = np.where(mask, u_reroute, -1e9)
    u_best_reroute = u_reroute.max(axis=1)
    return np.maximum(u_stop, u_best_reroute).astype(np.float32)


def build_routed_oracle_utility(
    routed_cost: np.ndarray,
    per_query_cost_matrix: np.ndarray,
    *,
    lambda_cost: float = 0.2,
    perf_matrix: np.ndarray | None = None,
    chosen: np.ndarray | None = None,
    routed_quality: np.ndarray | None = None,
) -> np.ndarray:
    """
    Routed utility (LLMRouterBench), aligned with training ``test_u`` / ``train_u``.

    **Pool routing (preferred):** ``perf_matrix`` [N,K] + ``chosen`` [N] — uses perf_tilde.

    **External cascade (e.g. FrugalGPT):** ``routed_quality`` [N] scalar outcomes; cost bounds
    still from ``per_query_cost_matrix``; quality is used directly (no cross-model tilde).
    """
    if perf_matrix is not None and chosen is not None:
        return routed_llmrouterbench_utility(
            perf_matrix, chosen, routed_cost, per_query_cost_matrix, lambda_cost=lambda_cost,
        )
    if routed_quality is not None:
        c_norm = normalize_routed_cost(routed_cost, per_query_cost_matrix)
        alpha = 1.0 - float(lambda_cost)
        q = routed_quality.astype(np.float32)
        return (alpha * q + (1.0 - alpha) * (1.0 - c_norm)).astype(np.float32)
    raise ValueError("build_routed_oracle_utility requires perf_matrix+chosen or routed_quality")


def rebuild_grouped_rewards(
    performance: np.ndarray,
    cost: np.ndarray,
    *,
    lambda_cost: float = 0.2,
    cost_mode: Literal["global", "per_query"] = "per_query",
    cost_penalty: CostPenalty = "linear",
    exp_gamma: float | None = None,
) -> np.ndarray:
    return build_oracle_reward(
        performance,
        cost,
        lambda_cost=lambda_cost,
        cost_mode=cost_mode,
        cost_penalty=cost_penalty,
        exp_gamma=exp_gamma,
    )
