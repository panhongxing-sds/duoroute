"""Per-model behavioral signatures from train response embeddings."""

from __future__ import annotations

import numpy as np
import torch

from duoroute.residual import DEFAULT_RESPONSE_NORM_EPS


def response_cell_mask(
    response_emb: torch.Tensor,
    arm_mask: np.ndarray,
    *,
    norm_eps: float = DEFAULT_RESPONSE_NORM_EPS,
) -> torch.Tensor:
    present = torch.linalg.norm(response_emb, dim=-1) > norm_eps
    return present & torch.from_numpy(arm_mask.astype(bool))


def compute_mean_behavioral_signature(
    response_emb: torch.Tensor,
    arm_mask: np.ndarray,
    *,
    norm_eps: float = DEFAULT_RESPONSE_NORM_EPS,
) -> torch.Tensor:
    """
    behav_sig[m] = mean(response_emb[q,m]) over train cells with has_response.
    Models with no observations get the global mean over all observed cells.
    """
    resp = response_emb.float()
    present = response_cell_mask(resp, arm_mask, norm_eps=norm_eps)
    k, d = resp.shape[1], resp.shape[2]
    sig = torch.zeros(k, d, dtype=resp.dtype)
    counts = torch.zeros(k, dtype=resp.dtype)
    for j in range(k):
        idx = present[:, j]
        if idx.any():
            sig[j] = resp[idx, j].mean(dim=0)
            counts[j] = float(idx.sum())
    observed = present.any(dim=1)
    if observed.any():
        global_mean = resp[observed].mean(dim=0)
        for j in range(k):
            if counts[j] == 0:
                sig[j] = global_mean
    return sig


def compute_per_dataset_signatures(
    response_emb: torch.Tensor,
    arm_mask: np.ndarray,
    dataset_ids: list[str],
    *,
    norm_eps: float = DEFAULT_RESPONSE_NORM_EPS,
) -> dict[str, torch.Tensor]:
    """Per-dataset mean signature; fallback to global mean signature per model."""
    global_sig = compute_mean_behavioral_signature(response_emb, arm_mask, norm_eps=norm_eps)
    datasets = sorted(set(dataset_ids))
    k, d = response_emb.shape[1], response_emb.shape[2]
    out: dict[str, torch.Tensor] = {}
    present = response_cell_mask(response_emb, arm_mask, norm_eps=norm_eps)
    for ds in datasets:
        sig = global_sig.clone()
        q_idx = [i for i, ds_id in enumerate(dataset_ids) if ds_id == ds]
        if not q_idx:
            out[ds] = sig
            continue
        for j in range(k):
            cells = [response_emb[i, j] for i in q_idx if present[i, j]]
            if cells:
                sig[j] = torch.stack(cells, dim=0).mean(dim=0)
        out[ds] = sig
    return out
