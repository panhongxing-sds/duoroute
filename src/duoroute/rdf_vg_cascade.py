"""Verifier-Guided Budgeted Cascade (RegretRouter + Cascade / Budgeted Cascade-DFL).

Stage1 picks m1.  Perfect verifier (Level B): wrong iff perf_{m1} < 0.5.
Among wrong cases, reroute only top-ρ% by learned selector confidence.
Selector has NO STOP action — picks m2 from {0..M-1} only.

Wrong-only recourse utility (perf_{m1}=0 on training wrong subset):
  ΔU_m = perf_m - λ*cost_m

Selector loss (wrong-only train): regret max_m ΔU_m - p^T ΔU_m
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from duoroute.r_dfl_ap import (
    RDFLAPTrainConfig,
    _forward_init_kwargs,
    _model_forward,
    make_router,
    train_r_dfl_ap,
)
from duoroute.rdf_cascade_decomp import (
    RecourseSelector,
    RerouteGate,
    apply_gate_selector,
    build_answer_features,
    compute_delta_u_np,
    eval_cascade_decomp,
    oracle_reroute_labels_np,
    oracle_selector_np,
    perfect_verifier_gate_np,
    strongest_cost_aware_idx,
)
from duoroute.rdf_cascade_dfl import _masked_entropy, _one_hot, _stage1_margin


# ---------------------------------------------------------------------------
# Wrong-only recourse utility
# ---------------------------------------------------------------------------


def wrong_only_delta_u_np(
    perf: np.ndarray,
    cost: np.ndarray,
    mask: np.ndarray,
    *,
    lambda_cost: float,
) -> np.ndarray:
    """ΔU_m = alpha*perf_tilde + (1-alpha)*(1-c_norm); LLMRouterBench wrong-only utility."""
    from duoroute.reward_builder import build_oracle_reward

    delta = build_oracle_reward(perf, cost, lambda_cost=lambda_cost)
    delta = delta.copy()
    delta[~mask] = -1e9
    return delta.astype(np.float32)


def wrong_only_delta_u_torch(
    perf: torch.Tensor,
    cost: torch.Tensor,
    mask: torch.Tensor,
    *,
    lambda_cost: float,
) -> torch.Tensor:
    """LLMRouterBench per-model utility (torch, per-query min-max)."""
    alpha = 1.0 - float(lambda_cost)
    q_min = perf.min(dim=1, keepdim=True).values
    q_max = perf.max(dim=1, keepdim=True).values
    q_tilde = (perf - q_min) / (q_max - q_min + 1e-8)
    pos = cost > 0
    c_max = cost.max(dim=1, keepdim=True).values
    inf = torch.full_like(cost, float("inf"))
    masked_min = torch.where(pos, cost, inf)
    c_min_pos = masked_min.min(dim=1, keepdim=True).values
    c_min_pos = torch.where(torch.isfinite(c_min_pos), c_min_pos, torch.zeros_like(c_min_pos))
    denom = c_max - c_min_pos + 1e-8
    c_norm = torch.where(pos, (cost - c_min_pos) / denom, torch.zeros_like(cost))
    delta = alpha * q_tilde + (1.0 - alpha) * (1.0 - c_norm)
    return delta.masked_fill(~mask, -1e9)


def wrong_only_selector_regret_loss(
    probs: torch.Tensor,
    delta_u: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Regret on model slots only (no STOP): max_m ΔU_m - p^T ΔU_m."""
    model_delta = delta_u.masked_fill(~mask, -1e9)
    oracle_best = model_delta.max(dim=-1).values
    expected = (probs * model_delta).sum(dim=-1)
    return (oracle_best - expected).mean()


def wrong_mask_np(perf: np.ndarray, m1: np.ndarray) -> np.ndarray:
    """True where stage1 answer is wrong (perf_{m1} < 0.5)."""
    idx = np.arange(len(m1))
    return perf[idx, m1] < 0.5


def worthiness_labels_np(
    perf: np.ndarray,
    cost: np.ndarray,
    mask: np.ndarray,
    *,
    lambda_cost: float,
) -> np.ndarray:
    """z = 1[max_m ΔU_m > 0] on wrong-only utility."""
    delta = wrong_only_delta_u_np(perf, cost, mask, lambda_cost=lambda_cost)
    return delta.max(axis=1) > 0


def budgeted_reroute_mask(
    wrong_mask: np.ndarray,
    scores: np.ndarray,
    rho_pct: float,
) -> np.ndarray:
    """
    Among wrong cases, reroute top-ρ% by score (descending).
    ρ=100 reroutes all wrong; ρ=0 reroutes none.
    """
    n = len(wrong_mask)
    reroute = np.zeros(n, dtype=bool)
    wrong_idx = np.where(wrong_mask)[0]
    if wrong_idx.size == 0 or rho_pct <= 0:
        return reroute
    if rho_pct >= 100:
        reroute[wrong_idx] = True
        return reroute
    k = max(1, int(np.ceil(wrong_idx.size * rho_pct / 100.0)))
    order = wrong_idx[np.argsort(-scores[wrong_idx])]
    reroute[order[:k]] = True
    return reroute


def selector_confidence_scores(sel_probs: np.ndarray, sel_logits: np.ndarray) -> np.ndarray:
    """Ranking score: max softmax probability (tie-break by max logit)."""
    max_prob = sel_probs.max(axis=1)
    max_logit = sel_logits.max(axis=1)
    return max_prob + 1e-6 * max_logit


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


class VGCascadeRouter(nn.Module):
    """Frozen stage1 + learned selector (no STOP) + optional worthiness scorer."""

    def __init__(
        self,
        stage1: nn.Module,
        query_dim: int,
        n_models: int,
        *,
        hidden_dim: int = 128,
        answer_feat_dim: int = 0,
        with_worthiness: bool = True,
    ):
        super().__init__()
        self.stage1 = stage1
        self.n_models = n_models
        self.answer_feat_dim = answer_feat_dim
        self.selector = RecourseSelector(
            query_dim, n_models, hidden_dim=hidden_dim, answer_feat_dim=answer_feat_dim,
        )
        self.with_worthiness = with_worthiness
        if with_worthiness:
            self.worthiness = RerouteGate(
                query_dim, n_models, hidden_dim=hidden_dim, answer_feat_dim=answer_feat_dim,
            )

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

    def _derive_features(
        self,
        probs: torch.Tensor,
        logits: torch.Tensor,
        m1: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        margin = _stage1_margin(logits, m1, mask)
        entropy = _masked_entropy(probs, mask).unsqueeze(-1)
        return margin, entropy

    def forward_selector(
        self,
        h: torch.Tensor,
        mask: torch.Tensor,
        ap: torch.Tensor,
        cfg: RDFLAPTrainConfig,
        *,
        probs: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        m1: torch.Tensor | None = None,
        answer_feat: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if probs is None or logits is None or m1 is None:
            probs, logits, m1 = self.stage1_forward(h, mask, ap, cfg)
        margin, entropy = self._derive_features(probs, logits, m1, mask)
        sel_probs, sel_logits = self.selector(
            h, probs, m1, margin, entropy, mask, answer_feat, temperature=temperature,
        )
        return sel_probs, sel_logits, m1

    def forward_worthiness(
        self,
        h: torch.Tensor,
        mask: torch.Tensor,
        ap: torch.Tensor,
        cfg: RDFLAPTrainConfig,
        *,
        probs: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        m1: torch.Tensor | None = None,
        answer_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if probs is None or logits is None or m1 is None:
            probs, logits, m1 = self.stage1_forward(h, mask, ap, cfg)
        margin, entropy = self._derive_features(probs, logits, m1, mask)
        return self.worthiness(
            h, probs, m1, margin, entropy, mask, answer_feat, return_logit=False,
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class VGCascadeTrainConfig:
    hidden_dim: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    epochs: int = 28
    patience: int = 8
    seed: int = 42
    lambda_cost: float = 0.2
    temperature: float = 1.0
    use_pos_weight: bool = True
    focal_gamma: float = 2.0


def _focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma: float = 2.0,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    p = torch.sigmoid(logits)
    pt = targets * p + (1.0 - targets) * (1.0 - p)
    return ((1.0 - pt).clamp(min=1e-8) ** gamma * bce).mean()


def train_vg_selector_wrong_only(
    router: VGCascadeRouter,
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
    cfg: VGCascadeTrainConfig,
    *,
    train_m1: torch.Tensor,
    val_m1: torch.Tensor | None = None,
    train_answer_feat: torch.Tensor | None = None,
    val_answer_feat: torch.Tensor | None = None,
) -> VGCascadeRouter:
    """Train selector on wrong-only train subset with regret on wrong-only ΔU."""
    import random

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    router = router.to(device)
    for p in router.stage1.parameters():
        p.requires_grad = False
    router.stage1.eval()

    train_perf_dev = train_perf.to(device)
    train_m1_dev = train_m1.to(device)
    with torch.no_grad():
        idx = torch.arange(train_perf_dev.size(0), device=device)
        train_wrong = train_perf_dev[idx, train_m1_dev] < 0.5
        wrong_idx = torch.where(train_wrong)[0].cpu().tolist()
    if not wrong_idx:
        return router

    opt = torch.optim.AdamW(
        router.selector.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    best_val = float("inf")
    best_state = None
    stale = 0
    rng = random.Random(cfg.seed)

    def _batch_loss(h, perf, cost, m, ap, m1_fixed, ans):
        with torch.no_grad():
            probs, logits, m1 = router.stage1_forward(h, m, ap, stage1_cfg)
            if m1_fixed is not None:
                m1 = m1_fixed
            idx_b = torch.arange(h.size(0), device=h.device)
            wrong_b = perf[idx_b, m1] < 0.5
        if not wrong_b.any():
            return torch.zeros((), device=h.device)
        delta_u = wrong_only_delta_u_torch(perf, cost, m, lambda_cost=cfg.lambda_cost)
        margin, entropy = router._derive_features(probs, logits, m1, m)
        _, sel_logits = router.selector(
            h, probs, m1, margin, entropy, m, ans, temperature=cfg.temperature,
        )
        sel_probs = F.softmax(sel_logits / max(float(cfg.temperature), 1e-4), dim=-1)
        return wrong_only_selector_regret_loss(sel_probs[wrong_b], delta_u[wrong_b], m[wrong_b])

    val_perf_dev = val_perf.to(device) if val_m1 is not None else None
    val_wrong_idx: list[int] = []
    if val_m1 is not None:
        vm1 = val_m1.to(device)
        with torch.no_grad():
            vidx = torch.arange(val_perf_dev.size(0), device=device)
            val_wrong = val_perf_dev[vidx, vm1] < 0.5
            val_wrong_idx = torch.where(val_wrong)[0].cpu().tolist()

    for _ in range(cfg.epochs):
        router.train()
        perm = wrong_idx.copy()
        rng.shuffle(perm)
        for start in range(0, len(perm), cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            if not idx:
                continue
            h = train_h[idx].to(device)
            perf = train_perf_dev[idx]
            cost = train_cost[idx].to(device)
            m = train_mask[idx].to(device).bool()
            ap = train_ap[idx].to(device)
            m1f = train_m1[idx].to(device)
            ans = train_answer_feat[idx].to(device) if train_answer_feat is not None else None
            loss = _batch_loss(h, perf, cost, m, ap, m1f, ans)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.selector.parameters(), max_norm=1.0)
            opt.step()

        if not val_wrong_idx:
            continue
        router.eval()
        vsum, vcnt = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(val_wrong_idx), 512):
                idx = val_wrong_idx[start : start + 512]
                h = val_h[idx].to(device)
                perf = val_perf_dev[idx]
                cost = val_cost[idx].to(device)
                m = val_mask[idx].to(device).bool()
                ap = val_ap[idx].to(device)
                m1f = val_m1[idx].to(device) if val_m1 is not None else None
                ans = val_answer_feat[idx].to(device) if val_answer_feat is not None else None
                vloss = _batch_loss(h, perf, cost, m, ap, m1f, ans)
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


def train_worthiness_scorer(
    router: VGCascadeRouter,
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
    cfg: VGCascadeTrainConfig,
    *,
    train_m1: torch.Tensor,
    val_m1: torch.Tensor | None = None,
    train_answer_feat: torch.Tensor | None = None,
    val_answer_feat: torch.Tensor | None = None,
) -> VGCascadeRouter:
    """Train worthiness P(profitable recourse) on wrong-only train subset."""
    import random

    if not router.with_worthiness:
        return router

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    router = router.to(device)
    for p in router.stage1.parameters():
        p.requires_grad = False
    router.stage1.eval()

    train_perf_dev = train_perf.to(device)
    train_m1_dev = train_m1.to(device)
    with torch.no_grad():
        idx = torch.arange(train_perf_dev.size(0), device=device)
        train_wrong = train_perf_dev[idx, train_m1_dev] < 0.5
        wrong_idx = torch.where(train_wrong)[0].cpu().tolist()
    if not wrong_idx:
        return router

    pos_weight_tensor: torch.Tensor | None = None
    if cfg.use_pos_weight:
        with torch.no_grad():
            labels: list[torch.Tensor] = []
            for start in range(0, len(wrong_idx), 512):
                idx = wrong_idx[start : start + 512]
                perf0 = train_perf_dev[idx]
                cost0 = train_cost[idx].to(device)
                m0 = train_mask[idx].to(device).bool()
                m1_0 = train_m1[idx].to(device)
                du = wrong_only_delta_u_torch(perf0, cost0, m0, lambda_cost=cfg.lambda_cost)
                y0 = (du.max(dim=-1).values > 0).float()
                labels.append(y0)
            y_all = torch.cat(labels)
            pos = float(y_all.sum().item())
            neg = float(y_all.numel() - pos)
            pos_weight_tensor = torch.tensor(neg / max(pos, 1.0), device=device, dtype=torch.float32)

    opt = torch.optim.AdamW(
        router.worthiness.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    best_val = float("inf")
    best_state = None
    stale = 0
    rng = random.Random(cfg.seed + 1)

    def _batch_loss(h, perf, cost, m, ap, m1_fixed, ans):
        with torch.no_grad():
            probs, logits, m1 = router.stage1_forward(h, m, ap, stage1_cfg)
            if m1_fixed is not None:
                m1 = m1_fixed
            idx_b = torch.arange(h.size(0), device=h.device)
            wrong_b = perf[idx_b, m1] < 0.5
        if not wrong_b.any():
            return torch.zeros((), device=h.device)
        delta_u = wrong_only_delta_u_torch(perf, cost, m, lambda_cost=cfg.lambda_cost)
        y = (delta_u.max(dim=-1).values > 0).float()
        margin, entropy = router._derive_features(probs, logits, m1, m)
        w_logit = router.worthiness(
            h, probs, m1, margin, entropy, m, ans, return_logit=True,
        )
        return _focal_bce_with_logits(
            w_logit[wrong_b], y[wrong_b], gamma=cfg.focal_gamma, pos_weight=pos_weight_tensor,
        )

    val_perf_dev = val_perf.to(device) if val_m1 is not None else None
    val_wrong_idx: list[int] = []
    if val_m1 is not None:
        vm1 = val_m1.to(device)
        with torch.no_grad():
            vidx = torch.arange(val_perf_dev.size(0), device=device)
            val_wrong = val_perf_dev[vidx, vm1] < 0.5
            val_wrong_idx = torch.where(val_wrong)[0].cpu().tolist()

    for _ in range(cfg.epochs):
        router.train()
        perm = wrong_idx.copy()
        rng.shuffle(perm)
        for start in range(0, len(perm), cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            if not idx:
                continue
            h = train_h[idx].to(device)
            perf = train_perf_dev[idx]
            cost = train_cost[idx].to(device)
            m = train_mask[idx].to(device).bool()
            ap = train_ap[idx].to(device)
            m1f = train_m1[idx].to(device)
            ans = train_answer_feat[idx].to(device) if train_answer_feat is not None else None
            loss = _batch_loss(h, perf, cost, m, ap, m1f, ans)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.worthiness.parameters(), max_norm=1.0)
            opt.step()

        if not val_wrong_idx:
            continue
        router.eval()
        vsum, vcnt = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(val_wrong_idx), 512):
                idx = val_wrong_idx[start : start + 512]
                h = val_h[idx].to(device)
                perf = val_perf_dev[idx]
                cost = val_cost[idx].to(device)
                m = val_mask[idx].to(device).bool()
                ap = val_ap[idx].to(device)
                m1f = val_m1[idx].to(device) if val_m1 is not None else None
                ans = val_answer_feat[idx].to(device) if val_answer_feat is not None else None
                vloss = _batch_loss(h, perf, cost, m, ap, m1f, ans)
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
def infer_vg_selector(
    router: VGCascadeRouter,
    h: torch.Tensor,
    mask: torch.Tensor,
    ap: torch.Tensor,
    *,
    stage1_cfg: RDFLAPTrainConfig,
    m1_override: np.ndarray | None = None,
    answer_feat: np.ndarray | None = None,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (m2, sel_probs, sel_logits, m1)."""
    device = next(router.parameters()).device
    router.eval()
    m2_parts: list[np.ndarray] = []
    prob_parts: list[np.ndarray] = []
    logit_parts: list[np.ndarray] = []
    m1_parts: list[np.ndarray] = []
    for start in range(0, h.shape[0], 512):
        hb = h[start : start + 512].to(device)
        mb = mask[start : start + 512].to(device).bool()
        apb = ap[start : start + 512].to(device)
        ans = None
        if answer_feat is not None:
            ans = torch.from_numpy(answer_feat[start : start + 512]).float().to(device)
        probs, logits, m1 = router.stage1_forward(hb, mb, apb, stage1_cfg)
        if m1_override is not None:
            m1 = torch.from_numpy(m1_override[start : start + 512]).long().to(device)
        sel_probs, sel_logits, _ = router.forward_selector(
            hb, mb, apb, stage1_cfg,
            probs=probs, logits=logits, m1=m1, answer_feat=ans, temperature=temperature,
        )
        m2_parts.append(sel_probs.argmax(dim=-1).cpu().numpy())
        prob_parts.append(sel_probs.cpu().numpy())
        logit_parts.append(sel_logits.cpu().numpy())
        m1_parts.append(m1.cpu().numpy())
    return (
        np.concatenate(m2_parts).astype(np.int64),
        np.concatenate(prob_parts),
        np.concatenate(logit_parts),
        np.concatenate(m1_parts).astype(np.int64),
    )


@torch.no_grad()
def infer_worthiness_scores(
    router: VGCascadeRouter,
    h: torch.Tensor,
    mask: torch.Tensor,
    ap: torch.Tensor,
    *,
    stage1_cfg: RDFLAPTrainConfig,
    m1_override: np.ndarray | None = None,
    answer_feat: np.ndarray | None = None,
) -> np.ndarray:
    device = next(router.parameters()).device
    router.eval()
    parts: list[np.ndarray] = []
    for start in range(0, h.shape[0], 512):
        hb = h[start : start + 512].to(device)
        mb = mask[start : start + 512].to(device).bool()
        apb = ap[start : start + 512].to(device)
        ans = None
        if answer_feat is not None:
            ans = torch.from_numpy(answer_feat[start : start + 512]).float().to(device)
        probs, logits, m1 = router.stage1_forward(hb, mb, apb, stage1_cfg)
        if m1_override is not None:
            m1 = torch.from_numpy(m1_override[start : start + 512]).long().to(device)
        w = router.forward_worthiness(
            hb, mb, apb, stage1_cfg, probs=probs, logits=logits, m1=m1, answer_feat=ans,
        )
        parts.append(w.cpu().numpy())
    return np.concatenate(parts).astype(np.float32)


def apply_vg_budgeted_cascade(
    m1: np.ndarray,
    m2: np.ndarray,
    test_perf: np.ndarray,
    *,
    rho_pct: float,
    selector_scores: np.ndarray,
    worthiness_scores: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perfect verifier + budgeted reroute.
    Returns (m_final, reroute_flags).
    """
    wrong = perfect_verifier_gate_np(test_perf, m1)
    if worthiness_scores is not None:
        rank_scores = selector_scores * worthiness_scores
    else:
        rank_scores = selector_scores
    reroute = budgeted_reroute_mask(wrong, rank_scores, rho_pct)
    m_final = apply_gate_selector(m1, reroute, m2)
    return m_final, reroute


def eval_vg_cascade(
    test_perf: np.ndarray,
    test_cost: np.ndarray,
    test_mask: np.ndarray,
    m1: np.ndarray,
    m_final: np.ndarray,
    reroute: np.ndarray,
    *,
    lambda_cost: float,
    test_u: np.ndarray | None = None,
    rho_pct: float | None = None,
    oracle_should_reroute: np.ndarray | None = None,
) -> dict:
    """Extended cascade metrics including wrong-subset reroute rate."""
    row = eval_cascade_decomp(
        test_perf, test_cost, test_mask, m1, m_final, reroute,
        lambda_cost=lambda_cost, test_u=test_u, oracle_should_reroute=oracle_should_reroute,
    )
    wrong = wrong_mask_np(test_perf, m1)
    n_wrong = int(wrong.sum())
    reroute_wrong = reroute.astype(bool) & wrong
    row["n_wrong"] = n_wrong
    row["reroute_rate_wrong_subset"] = float(reroute_wrong.sum() / max(n_wrong, 1))
    row["n_rerouted_wrong"] = int(reroute_wrong.sum())
    if rho_pct is not None:
        row["rho_pct"] = float(rho_pct)
    return row


def build_vg_cascade_router(
    stage1: nn.Module,
    data: dict,
    *,
    hidden_dim: int = 128,
    answer_feat_dim: int = 0,
    with_worthiness: bool = True,
) -> VGCascadeRouter:
    d = data["train_h"].shape[1]
    k = len(data["model_names"])
    return VGCascadeRouter(
        stage1, d, k, hidden_dim=hidden_dim,
        answer_feat_dim=answer_feat_dim, with_worthiness=with_worthiness,
    )


def train_stage1_rdfl(
    data: dict,
    seed: int,
    stage1_cfg: RDFLAPTrainConfig,
) -> tuple[nn.Module, RDFLAPTrainConfig]:
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


__all__ = [
    "VGCascadeRouter",
    "VGCascadeTrainConfig",
    "apply_vg_budgeted_cascade",
    "budgeted_reroute_mask",
    "build_vg_cascade_router",
    "eval_vg_cascade",
    "infer_vg_selector",
    "infer_worthiness_scores",
    "oracle_reroute_labels_np",
    "oracle_selector_np",
    "perfect_verifier_gate_np",
    "selector_confidence_scores",
    "strongest_cost_aware_idx",
    "train_stage1_rdfl",
    "train_vg_selector_wrong_only",
    "train_worthiness_scorer",
    "wrong_mask_np",
    "wrong_only_delta_u_np",
    "worthiness_labels_np",
    "build_answer_features",
    "compute_delta_u_np",
    "apply_gate_selector",
]
