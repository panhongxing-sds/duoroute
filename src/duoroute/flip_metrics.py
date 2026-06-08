"""Flip statistics between two routing policies."""

from __future__ import annotations

import numpy as np


def route_choices(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    neg = -1e9
    return np.where(mask, scores, neg).argmax(axis=1)


def flip_stats(
    oracle: np.ndarray,
    mask: np.ndarray,
    chosen_a: np.ndarray,
    chosen_b: np.ndarray,
    *,
    query_filter: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compare policy A (chosen_a) vs B (chosen_b) by oracle reward on routed arm.
    helpful: reward(B) > reward(A); harmful: reward(B) < reward(A).
    """
    n = oracle.shape[0]
    if query_filter is None:
        query_filter = np.ones(n, dtype=bool)
    idx = np.arange(n)[query_filter]
    if len(idx) == 0:
        return {
            "n_queries": 0.0,
            "flip_rate": 0.0,
            "helpful_flip_rate": 0.0,
            "harmful_flip_rate": 0.0,
            "neutral_flip_rate": 0.0,
            "net_helpful_flip_rate": 0.0,
            "acc_a": 0.0,
            "acc_b": 0.0,
        }

    ra = oracle[idx, chosen_a[idx]]
    rb = oracle[idx, chosen_b[idx]]
    flipped = chosen_a[idx] != chosen_b[idx]
    helpful = flipped & (rb > ra + 1e-8)
    harmful = flipped & (rb < ra - 1e-8)
    neutral = flipped & ~(helpful | harmful)
    n_f = float(len(idx))
    flip_n = float(flipped.sum())
    return {
        "n_queries": n_f,
        "flip_rate": float(flip_n / n_f),
        "helpful_flip_rate": float(helpful.sum() / n_f),
        "harmful_flip_rate": float(harmful.sum() / n_f),
        "neutral_flip_rate": float(neutral.sum() / n_f),
        "net_helpful_flip_rate": float((helpful.sum() - harmful.sum()) / n_f),
        "acc_a": float(ra.mean()),
        "acc_b": float(rb.mean()),
    }
