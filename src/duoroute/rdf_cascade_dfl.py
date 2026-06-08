"""Cascade-DFL: STOP-action unified modeling with recourse utility regret loss.

Stage 1 picks m1 (AP-balance or RegretRouter).  Stage 2 outputs softmax over
{STOP, model_0, ..., model_{M-1}} and is trained with regret on recourse ΔU.

Recourse utility (m1 cost already sunk):
  U_stop = perf_{m1} - λ*c_{m1}
  ΔU_STOP = 0
  ΔU_m   = perf_m - perf_{m1} - λ*cost_m

Final utility if STOP: U_stop.
Final utility if reroute to m: perf_m - λ*(cost_{m1} + cost_m).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from duoroute.model import MLP
from duoroute.r_dfl_ap import (
    RDFLAPTrainConfig,
    _forward_init_kwargs,
    _model_forward,
    infer_probs,
    make_router,
    train_r_dfl_ap,
)
from duoroute.rdf_router import masked_softmax


def _one_hot(indices: torch.Tensor, n_models: int) -> torch.Tensor:
    out = torch.zeros(indices.size(0), n_models, device=indices.device, dtype=torch.float32)
    out.scatter_(1, indices.unsqueeze(1).clamp(min=0), 1.0)
    return out


def _masked_entropy(probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    p = probs.clamp(min=1e-8) * mask.float()
    p = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return -(p * p.log()).sum(dim=-1)


def _stage1_margin(logits: torch.Tensor, m1: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_logits = logits.masked_fill(~mask, -1e9)
    top_logits, _ = masked_logits.max(dim=-1)
    m1_logits = logits.gather(1, m1.unsqueeze(1)).squeeze(1)
    return (top_logits - m1_logits).unsqueeze(-1)


# ---------------------------------------------------------------------------
# Recourse utility
# ---------------------------------------------------------------------------


def compute_delta_u_np(
    perf: np.ndarray,
    cost: np.ndarray,
    m1: np.ndarray,
    mask: np.ndarray,
    *,
    lambda_cost: float,
) -> np.ndarray:
    """ΔU vector [N, M+1]: index 0 = STOP (0), indices 1..M = model gains."""
    n, k = perf.shape
    idx = np.arange(n)
    perf_m1 = perf[idx, m1]
    delta = np.zeros((n, k + 1), dtype=np.float32)
    for m in range(k):
        delta[:, m + 1] = perf[:, m] - perf_m1 - float(lambda_cost) * cost[:, m]
    model_slots = delta[:, 1:]
    model_slots[~mask] = -1e9
    delta[:, 1:] = model_slots
    delta[:, 0] = 0.0  # STOP always 0
    return delta


def compute_delta_u_torch(
    perf: torch.Tensor,
    cost: torch.Tensor,
    m1: torch.Tensor,
    mask: torch.Tensor,
    *,
    lambda_cost: float,
) -> torch.Tensor:
    """ΔU [B, M+1]: STOP=0, model slots = perf_m - perf_m1 - λ*cost_m."""
    batch = perf.size(0)
    idx = torch.arange(batch, device=perf.device)
    perf_m1 = perf[idx, m1]
    delta_models = perf - perf_m1.unsqueeze(1) - float(lambda_cost) * cost
    delta_models = delta_models.masked_fill(~mask, -1e9)
    stop_col = torch.zeros(batch, 1, device=perf.device, dtype=perf.dtype)
    return torch.cat([stop_col, delta_models], dim=-1)


def oracle_cascade_dfl_action(
    delta_u: np.ndarray,
    m1: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Oracle over {STOP, model_0..M-1}.
    STOP if max_m ΔU_m <= 0 else argmax_m ΔU_m.
    Returns (m_final, reroute_flags, m2_or_m1).
    """
    n = len(m1)
    m_final = m1.copy()
    reroute = np.zeros(n, dtype=bool)
    # model slots are indices 1..K in delta_u
    model_delta = delta_u[:, 1:].copy()
    model_delta[~mask] = -1e9
    best_gain = model_delta.max(axis=1)
    best_m = model_delta.argmax(axis=1)
    reroute = best_gain > 0
    m_final[reroute] = best_m[reroute]
    return m_final, reroute, best_m


def recourse_regret_loss(
    probs: torch.Tensor,
    delta_u: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    L = max_a ΔU_a - x2^T ΔU  (regret on recourse gains).
    probs: [B, M+1] over {STOP, model_0..M-1}.
    delta_u: [B, M+1].
    """
    # Mask invalid model actions: set their ΔU to -inf
    valid = torch.ones(delta_u.size(1), device=delta_u.device, dtype=torch.bool)
    valid[0] = True  # STOP always valid
    model_valid = mask  # [B, M]
    full_valid = torch.cat([valid[:1].expand(delta_u.size(0), 1), model_valid], dim=-1)
    masked_delta = delta_u.masked_fill(~full_valid, -1e9)
    oracle_best = masked_delta.max(dim=-1).values
    expected = (probs * masked_delta).sum(dim=-1)
    return (oracle_best - expected).mean()


# ---------------------------------------------------------------------------
# Stage-2 network
# ---------------------------------------------------------------------------


class CascadeDFLStage2(nn.Module):
    """MLP on [h, m1_oh, x1, margin, entropy] -> logits over M+1 actions."""

    def __init__(self, query_dim: int, n_models: int, *, hidden_dim: int = 128):
        super().__init__()
        in_dim = query_dim + 2 * n_models + 2
        self.head = MLP(in_dim, n_models + 1, hidden_dim=hidden_dim)
        self.n_models = n_models
        # Mild STOP prior at init.
        with torch.no_grad():
            self.head.net[-1].bias[0] = 0.5

    def forward(
        self,
        h: torch.Tensor,
        x1: torch.Tensor,
        m1: torch.Tensor,
        margin: torch.Tensor,
        entropy: torch.Tensor,
        mask: torch.Tensor,
        *,
        stop_bias: float = 0.0,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m1_oh = _one_hot(m1, self.n_models)
        feat = torch.cat([h, m1_oh, x1, margin, entropy], dim=-1)
        logits = self.head(feat)
        logits[:, 0] = logits[:, 0] + float(stop_bias)
        # Mask invalid model slots (indices 1..M)
        model_logits = logits[:, 1:].masked_fill(~mask, -1e9)
        logits = torch.cat([logits[:, :1], model_logits], dim=-1)
        probs = F.softmax(logits / max(float(temperature), 1e-4), dim=-1)
        return probs, logits


class CascadeDFLRouter(nn.Module):
    """Frozen stage-1 + learned STOP-action stage-2."""

    def __init__(self, stage1: nn.Module, query_dim: int, n_models: int, *, hidden_dim: int = 128):
        super().__init__()
        self.stage1 = stage1
        self.stage2 = CascadeDFLStage2(query_dim, n_models, hidden_dim=hidden_dim)
        self.n_models = n_models

    @torch.no_grad()
    def stage1_forward(
        self,
        h: torch.Tensor,
        mask: torch.Tensor,
        ap: torch.Tensor,
        cfg: RDFLAPTrainConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.stage1.eval()
        inits = _forward_init_kwargs(ap, mask, cfg)
        probs, logits = _model_forward(self.stage1, h, mask, cfg, inits=inits)
        m1 = probs.argmax(dim=-1)
        return probs, logits, m1

    def forward(
        self,
        h: torch.Tensor,
        mask: torch.Tensor,
        ap: torch.Tensor,
        cfg: RDFLAPTrainConfig,
        *,
        probs: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        m1: torch.Tensor | None = None,
        stop_bias: float = 0.0,
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        if probs is None or logits is None or m1 is None:
            probs, logits, m1 = self.stage1_forward(h, mask, ap, cfg)
        margin = _stage1_margin(logits, m1, mask)
        entropy = _masked_entropy(probs, mask).unsqueeze(-1)
        probs2, logits2 = self.stage2(
            h, probs, m1, margin, entropy, mask,
            stop_bias=stop_bias, temperature=temperature,
        )
        return {
            "probs1": probs,
            "logits1": logits,
            "m1": m1,
            "probs2": probs2,
            "logits2": logits2,
            "margin": margin,
            "entropy": entropy,
        }


@dataclass
class CascadeDFLTrainConfig:
    hidden_dim: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    epochs: int = 28
    patience: int = 8
    seed: int = 42
    lambda_cost: float = 0.2
    temperature: float = 1.0
    bce_weight: float = 0.05  # auxiliary CE on oracle action
    regret_weight: float = 1.0


def train_cascade_dfl(
    router: CascadeDFLRouter,
    train_h: torch.Tensor,
    train_perf: torch.Tensor,
    train_cost: torch.Tensor,
    train_mask: torch.Tensor,
    train_ap: torch.Tensor,
    val_h: torch.Tensor,
    val_perf: torch.Tensor,
    val_cost: torch.Tensor,
    val_mask: torch.Tensor,
    val_ap: torch.Tensor,
    stage1_cfg: RDFLAPTrainConfig,
    cfg: CascadeDFLTrainConfig,
    *,
    train_m1: torch.Tensor | None = None,
) -> CascadeDFLRouter:
    """Train stage-2 with regret on recourse ΔU; stage-1 frozen."""
    import random

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    router = router.to(device)
    for p in router.stage1.parameters():
        p.requires_grad = False
    router.stage1.eval()

    opt = torch.optim.AdamW(
        router.stage2.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    n = train_h.shape[0]
    best_val = float("inf")
    best_state = None
    stale = 0
    rng = random.Random(cfg.seed)

    def _batch_loss(h, perf, cost, m, ap, m1_fixed=None):
        with torch.no_grad():
            probs, logits, m1 = router.stage1_forward(h, m, ap, stage1_cfg)
            if m1_fixed is not None:
                m1 = m1_fixed
        out = router(h, m, ap, stage1_cfg, probs=probs, logits=logits, m1=m1,
                     temperature=cfg.temperature)
        delta_u = compute_delta_u_torch(
            perf, cost, m1, m, lambda_cost=cfg.lambda_cost,
        )
        regret = recourse_regret_loss(out["probs2"], delta_u, m)
        loss = cfg.regret_weight * regret
        if cfg.bce_weight > 0:
            model_delta = delta_u[:, 1:]
            y_reroute = model_delta.max(dim=-1).values > 0
            best_m = model_delta.argmax(dim=-1)
            oracle_action = torch.where(y_reroute, best_m + 1, torch.zeros_like(best_m))
            ce = F.cross_entropy(out["logits2"], oracle_action)
            loss = loss + cfg.bce_weight * ce
        return loss, float(regret.item())

    for _ in range(cfg.epochs):
        router.train()
        perm = list(range(n))
        rng.shuffle(perm)
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            if not idx:
                continue
            h = train_h[idx].to(device)
            perf = train_perf[idx].to(device)
            cost = train_cost[idx].to(device)
            m = train_mask[idx].to(device).bool()
            ap = train_ap[idx].to(device)
            m1f = train_m1[idx].to(device) if train_m1 is not None else None
            loss, _ = _batch_loss(h, perf, cost, m, ap, m1f)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.stage2.parameters(), max_norm=1.0)
            opt.step()

        router.eval()
        vsum, vcnt = 0.0, 0
        with torch.no_grad():
            for start in range(0, val_h.shape[0], 512):
                h = val_h[start : start + 512].to(device)
                perf = val_perf[start : start + 512].to(device)
                cost = val_cost[start : start + 512].to(device)
                m = val_mask[start : start + 512].to(device).bool()
                ap = val_ap[start : start + 512].to(device)
                vloss, _ = _batch_loss(h, perf, cost, m, ap)
                vsum += float(vloss.item()) * h.size(0)
                vcnt += h.size(0)
        vloss = vsum / max(vcnt, 1)
        if vloss < best_val - 1e-5:
            best_val = vloss
            best_state = {k: v.cpu().clone() for k, v in router.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    if best_state is not None:
        router.load_state_dict(best_state)
    return router


@torch.no_grad()
def infer_cascade_dfl(
    router: CascadeDFLRouter,
    h: torch.Tensor,
    mask: torch.Tensor,
    ap: torch.Tensor,
    *,
    stage1_cfg: RDFLAPTrainConfig,
    stop_bias: float = 0.0,
    temperature: float = 1.0,
    m1_override: torch.Tensor | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deployable inference: h, m1, x1, entropy, margin only.
    Returns (m_final, m1, reroute_flags).
    m1_override: optional precomputed stage-1 choices (e.g. AP-balance).
    """
    device = next(router.parameters()).device
    router.eval()
    final_parts: list[np.ndarray] = []
    m1_parts: list[np.ndarray] = []
    reroute_parts: list[np.ndarray] = []
    for start in range(0, h.shape[0], 512):
        hb = h[start : start + 512].to(device)
        mb = mask[start : start + 512].to(device).bool()
        apb = ap[start : start + 512].to(device)
        probs, logits, m1 = router.stage1_forward(hb, mb, apb, stage1_cfg)
        if m1_override is not None:
            if isinstance(m1_override, np.ndarray):
                m1 = torch.from_numpy(m1_override[start : start + 512]).long().to(device)
            else:
                m1 = m1_override[start : start + 512].to(device).long()
        out = router(
            hb, mb, apb, stage1_cfg,
            probs=probs, logits=logits, m1=m1,
            stop_bias=stop_bias, temperature=temperature,
        )
        action = out["probs2"].argmax(dim=-1)
        reroute = action > 0
        m2 = action - 1
        m_final = torch.where(reroute, m2, m1)
        final_parts.append(m_final.cpu().numpy())
        m1_parts.append(m1.cpu().numpy())
        reroute_parts.append(reroute.cpu().numpy())
    return (
        np.concatenate(final_parts).astype(np.int64),
        np.concatenate(m1_parts).astype(np.int64),
        np.concatenate(reroute_parts).astype(bool),
    )


def eval_cascade_dfl_row(
    test_perf: np.ndarray,
    test_cost: np.ndarray,
    test_mask: np.ndarray,
    m1: np.ndarray,
    m_final: np.ndarray,
    reroute: np.ndarray,
    *,
    lambda_cost: float,
    test_u: np.ndarray | None = None,
) -> dict:
    """
    Evaluate with cascade_true recourse cost accounting.
    STOP: cost=c(m1), utility=perf(m1)-λ*c(m1).
    Reroute: cost=c(m1)+c(m2), utility=perf(m2)-λ*(c(m1)+c(m2)).
    """
    n = len(m1)
    idx = np.arange(n)
    esc = reroute.astype(bool)
    routed_perf = test_perf[idx, m_final]
    routed_cost = test_cost[idx, m1].copy()
    routed_cost[esc] += test_cost[idx, m_final][esc]
    routed_u = routed_perf - float(lambda_cost) * routed_cost

    avg_acc = float((routed_perf >= 0.5).mean())
    avg_utility = float(routed_u.mean())
    avg_cost = float(routed_cost.mean())
    reroute_rate = float(esc.mean())

    # Stage1-only utility for comparison
    stage1_u = test_perf[idx, m1] - float(lambda_cost) * test_cost[idx, m1]
    delta_utility_vs_stage1 = avg_utility - float(stage1_u.mean())
    stage1_acc = float((test_perf[idx, m1] >= 0.5).mean())
    stage1_cost = float(test_cost[idx, m1].mean())

    neg_inf = -1e9
    if test_u is not None:
        true_m = np.where(test_mask, test_u, neg_inf)
        oracle_u = true_m.max(axis=1)
        gap_at_oracle = float((oracle_u - routed_u).mean())
    else:
        gap_at_oracle = None

    # Utility gain and cost per rescue among rerouted queries
    utility_gain_on_reroute = None
    cost_per_rescue = None
    if esc.any():
        gain = routed_u[esc] - stage1_u[esc]
        utility_gain_on_reroute = float(gain.mean())
        cost_per_rescue = float(routed_cost[esc].mean())

    return {
        "avg_acc": avg_acc,
        "avg_utility": avg_utility,
        "avg_cost": avg_cost,
        "reroute_rate": reroute_rate,
        "n_rerouted": int(esc.sum()),
        "stage1_acc": stage1_acc,
        "stage1_avg_cost": stage1_cost,
        "stage1_avg_utility": float(stage1_u.mean()),
        "delta_utility_vs_stage1": delta_utility_vs_stage1,
        "gap_at_oracle": gap_at_oracle,
        "utility_gain_on_reroute": utility_gain_on_reroute,
        "cost_per_rescue": cost_per_rescue,
    }


def tune_stop_bias_val(
    router: CascadeDFLRouter,
    val_h: torch.Tensor,
    val_perf: np.ndarray,
    val_cost: np.ndarray,
    val_mask: torch.Tensor,
    val_ap: torch.Tensor,
    val_m1: np.ndarray,
    *,
    stage1_cfg: RDFLAPTrainConfig,
    lambda_cost: float,
    target_rates: list[float] | None = None,
    biases: np.ndarray | None = None,
    temperature: float = 1.0,
    m1_override: np.ndarray | None = None,
) -> list[dict]:
    """Sweep STOP logit bias on val; return rows sorted by reroute rate."""
    if biases is None:
        biases = np.linspace(-3.0, 3.0, 61)
    rows: list[dict] = []
    for bias in biases:
        m_final, m1, reroute = infer_cascade_dfl(
            router, val_h, val_mask, val_ap,
            stage1_cfg=stage1_cfg, stop_bias=float(bias), temperature=temperature,
            m1_override=m1_override if m1_override is not None else val_m1,
        )
        row = eval_cascade_dfl_row(
            val_perf, val_cost, val_mask.numpy().astype(bool),
            m1, m_final, reroute, lambda_cost=lambda_cost,
        )
        row["stop_bias"] = float(bias)
        row["temperature"] = float(temperature)
        rows.append(row)
    rows.sort(key=lambda r: r["reroute_rate"])
    return rows


def pick_bias_for_target_rate(
    pareto_rows: list[dict],
    target_rate: float,
    *,
    tolerance: float = 0.03,
) -> dict | None:
    """Pick val row closest to target reroute rate; tie-break by utility."""
    if not pareto_rows:
        return None
    within = [r for r in pareto_rows if abs(r["reroute_rate"] - target_rate) <= tolerance]
    pool = within if within else pareto_rows
    return max(
        pool,
        key=lambda r: (r["avg_utility"], -abs(r["reroute_rate"] - target_rate)),
    )


def train_stage1_rdfl(
    data: dict,
    seed: int,
    stage1_cfg: RDFLAPTrainConfig,
) -> tuple[nn.Module, RDFLAPTrainConfig]:
    """Train RegretRouter stage-1 and return frozen router."""
    d = data["train_h"].shape[1]
    k = len(data["model_names"])
    cost = torch.tensor(data["cost"], dtype=torch.float32)
    stage1 = make_router(d, k, kind="rdfl", cfg=stage1_cfg, cost=cost)
    train_r_dfl_ap(
        stage1,
        torch.from_numpy(data["train_h"]).float(),
        torch.from_numpy(data["train_u"]).float(),
        torch.from_numpy(data["train_mask"]).bool(),
        torch.from_numpy(data["ap_balance_train_idx"]).long(),
        cost,
        torch.from_numpy(data["val_h"]).float(),
        torch.from_numpy(data["val_u"]).float(),
        torch.from_numpy(data["val_mask"]).bool(),
        torch.from_numpy(data["ap_balance_val_idx"]).long(),
        stage1_cfg,
    )
    return stage1, stage1_cfg


def build_cascade_dfl_router(
    stage1: nn.Module,
    data: dict,
    *,
    hidden_dim: int = 128,
) -> CascadeDFLRouter:
    d = data["train_h"].shape[1]
    k = len(data["model_names"])
    return CascadeDFLRouter(stage1, d, k, hidden_dim=hidden_dim)
