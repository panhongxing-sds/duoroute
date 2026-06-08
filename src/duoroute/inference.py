"""Inference modes for DuoRoute."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from duoroute.model import DuoRouteModel


@dataclass
class RoutingDecision:
    chosen: np.ndarray
    fallback_used: np.ndarray
    pred_a: np.ndarray
    pred_b: np.ndarray | None = None


def top1_margin(pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
    neg_inf = -1e9
    masked = np.where(mask, pred, neg_inf)
    top2 = np.partition(masked, -2, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    return margin


@torch.no_grad()
def predict_channel_a(model: DuoRouteModel, prompt_ids: torch.Tensor) -> np.ndarray:
    model.eval()
    return model.forward_a(prompt_ids).cpu().numpy()


@torch.no_grad()
def predict_channel_b(
    model: DuoRouteModel,
    prompt_ids: torch.Tensor,
    response_emb: torch.Tensor,
) -> np.ndarray:
    model.eval()
    return model.forward_b(prompt_ids, response_emb).cpu().numpy()


def route_query_only(pred_a: np.ndarray, mask: np.ndarray) -> RoutingDecision:
    neg_inf = -1e9
    masked = np.where(mask, pred_a, neg_inf)
    chosen = masked.argmax(axis=1)
    return RoutingDecision(
        chosen=chosen,
        fallback_used=np.zeros(len(chosen), dtype=bool),
        pred_a=pred_a,
    )


def route_with_fallback(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    mask: np.ndarray,
    *,
    delta: float,
) -> RoutingDecision:
    """
    Pre-generation router A; if A margin < delta, verify/reroute with B on available responses.
    B is only used post-generation, never as a free pre-generation oracle.
    """
    margin = top1_margin(pred_a, mask)
    fallback = margin < delta
    chosen_a = route_query_only(pred_a, mask).chosen
    neg_inf = -1e9
    chosen_b = np.where(mask, pred_b, neg_inf).argmax(axis=1)
    chosen = np.where(fallback, chosen_b, chosen_a)
    return RoutingDecision(
        chosen=chosen,
        fallback_used=fallback,
        pred_a=pred_a,
        pred_b=pred_b,
    )
