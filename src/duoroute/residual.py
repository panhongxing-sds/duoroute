"""Masked residual routing: A prior + response-aware correction."""

from __future__ import annotations

import numpy as np
import torch

from duoroute.model import DuoRouteModel

DEFAULT_RESPONSE_NORM_EPS = 1e-6


def response_present_mask(
    response_emb: torch.Tensor,
    *,
    norm_eps: float = DEFAULT_RESPONSE_NORM_EPS,
) -> torch.Tensor:
    """(batch, k) bool — non-zero response embedding."""
    return torch.linalg.norm(response_emb, dim=-1) > norm_eps


@torch.no_grad()
def predict_delta_b(
    model: DuoRouteModel,
    prompt_ids: torch.Tensor,
    response_emb: torch.Tensor,
) -> np.ndarray:
    model.eval()
    return model.forward_b(prompt_ids, response_emb).cpu().numpy()


def compose_residual_scores(
    pred_a: torch.Tensor,
    delta_b: torch.Tensor,
    response_emb: torch.Tensor,
    *,
    gamma: float,
    norm_eps: float = DEFAULT_RESPONSE_NORM_EPS,
) -> torch.Tensor:
    """
    score_final = pred_a + gamma * delta_b * has_response
    pred_a is typically detached when training delta only.
    """
    resp_mask = response_present_mask(response_emb, norm_eps=norm_eps).float()
    return pred_a + float(gamma) * delta_b * resp_mask


def route_argmax(
    scores: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    neg_inf = -1e9
    masked = np.where(mask, scores, neg_inf)
    return masked.argmax(axis=1)


def top1_margin_np(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    neg_inf = -1e9
    masked = np.where(mask, scores, neg_inf)
    if masked.shape[1] < 2:
        return np.zeros(masked.shape[0], dtype=np.float32)
    top2 = np.partition(masked, -2, axis=1)[:, -2:]
    return (top2[:, 1] - top2[:, 0]).astype(np.float32)


def route_teacher_confidence_gate(
    pred_a: np.ndarray,
    delta_b: np.ndarray,
    mask: np.ndarray,
    has_response: np.ndarray,
    *,
    gamma: float,
    top_k: int = 2,
    tau: float = 0.03,
) -> np.ndarray:
    """
    If Channel A margin >= tau, keep A; else top-k residual rerank on score_final.
    """
    n = pred_a.shape[0]
    final = pred_a + float(gamma) * delta_b * has_response.astype(np.float32)
    margin = top1_margin_np(pred_a, mask)
    chosen = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if margin[i] >= tau:
            neg = -1e9
            chosen[i] = int(np.where(mask[i], pred_a[i], neg).argmax())
        else:
            chosen[i] = route_topk_residual(
                pred_a[i : i + 1],
                delta_b[i : i + 1],
                mask[i : i + 1],
                has_response[i : i + 1],
                gamma=gamma,
                top_k=top_k,
            )[0]
    return chosen


def route_topk_residual(
    pred_a: np.ndarray,
    delta_b: np.ndarray,
    mask: np.ndarray,
    has_response: np.ndarray,
    *,
    gamma: float,
    top_k: int = 3,
) -> np.ndarray:
    """
    Restrict final argmax to top-k models by Channel A score.
    Arms without response use delta=0 (final = A).
    """
    n, k = pred_a.shape
    final = pred_a + float(gamma) * delta_b * has_response.astype(np.float32)
    neg_inf = -1e9
    chosen = np.zeros(n, dtype=np.int64)
    for i in range(n):
        avail = np.where(mask[i])[0]
        if len(avail) == 0:
            chosen[i] = 0
            continue
        a_masked = np.where(mask[i], pred_a[i], neg_inf)
        if len(avail) <= top_k:
            candidates = avail
        else:
            candidates = avail[np.argpartition(a_masked[avail], -top_k)[-top_k:]]
        sub = final[i, candidates]
        chosen[i] = int(candidates[int(np.argmax(sub))])
    return chosen
