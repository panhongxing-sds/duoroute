"""Cascade gate/selector decomposition: RerouteGate + RecourseSelector (two-head).

Gate: worth rerouting?  Selector: which m2?

Recourse utility (m1 cost sunk):
  U_stop = perf_{m1} - λ*c_{m1}
  ΔU_m   = perf_m - perf_{m1} - λ*cost_m

STOP if gate=0; else m2 = selector(x).
Final reroute cost = c(m1)+c(m2); utility = perf(m2) - λ*(c(m1)+c(m2)).
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
    make_router,
    train_r_dfl_ap,
)
from duoroute.rdf_cascade_dfl import (
    _masked_entropy,
    _one_hot,
    _stage1_margin,
    compute_delta_u_np,
    compute_delta_u_torch,
)


def selector_regret_loss(
    probs: torch.Tensor,
    delta_u: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Regret on model slots only: max_m ΔU_m - p^T ΔU_m (ΔU includes STOP col at 0)."""
    model_delta = delta_u[:, 1:].masked_fill(~mask, -1e9)
    oracle_best = model_delta.max(dim=-1).values
    expected = (probs * model_delta).sum(dim=-1)
    return (oracle_best - expected).mean()


# ---------------------------------------------------------------------------
# Oracle / verifier policies (analysis only unless deployable gate)
# ---------------------------------------------------------------------------


def oracle_reroute_labels_np(delta_u: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """y=1 iff max_m ΔU_m > 0."""
    model_delta = delta_u[:, 1:].copy()
    model_delta[~mask] = -1e9
    return model_delta.max(axis=1) > 0


def perfect_verifier_gate_np(perf: np.ndarray, m1: np.ndarray) -> np.ndarray:
    """Level B upper bound: g = 1[perf_{m1} < 0.5] — knows m1 wrong only."""
    idx = np.arange(len(m1))
    return perf[idx, m1] < 0.5


def oracle_selector_np(
    delta_u: np.ndarray,
    m1: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """argmax_m ΔU_m (oracle appendix)."""
    model_delta = delta_u[:, 1:].copy()
    model_delta[~mask] = -1e9
    return model_delta.argmax(axis=1)


def apply_gate_selector(
    m1: np.ndarray,
    reroute: np.ndarray,
    m2: np.ndarray,
) -> np.ndarray:
    """Combine gate decision with selector choice."""
    m_final = m1.copy()
    esc = reroute.astype(bool)
    m_final[esc] = m2[esc]
    return m_final


def strongest_cost_aware_idx(
    train_perf: np.ndarray,
    train_cost: np.ndarray,
    train_mask: np.ndarray,
    *,
    lambda_cost: float,
) -> int:
    """Train-determined strongest cost-aware model (max mean utility)."""
    k = train_perf.shape[1]
    means = []
    for j in range(k):
        mk = train_mask[:, j]
        if not mk.any():
            means.append(-1e9)
            continue
        u = train_perf[mk, j] - float(lambda_cost) * train_cost[mk, j]
        means.append(float(u.mean()))
    return int(np.argmax(means))


def topk_candidate_mask(
    x1: np.ndarray,
    mask: np.ndarray,
    *,
    k: int,
    extra_idx: int | None = None,
) -> np.ndarray:
    """Boolean [N, M] mask: TopK(x1) ∪ {extra_idx}."""
    n, m = x1.shape
    neg = -1e9
    scored = np.where(mask, x1, neg)
    order = np.argsort(-scored, axis=1)
    out = np.zeros((n, m), dtype=bool)
    for i in range(n):
        avail = order[i][scored[i, order[i]] > neg + 1]
        for j in avail[:k]:
            out[i, j] = True
        if extra_idx is not None and mask[i, extra_idx]:
            out[i, extra_idx] = True
    return out


# ---------------------------------------------------------------------------
# Answer features (Level 1, deployable: m1 response only)
# ---------------------------------------------------------------------------


def build_answer_features(
    response_texts: list[list[str]],
    m1: np.ndarray,
    resp_emb: torch.Tensor | np.ndarray | None = None,
    *,
    emb_proj_dim: int = 32,
) -> tuple[np.ndarray, int]:
    """
    Per-query Level-1 features from m1's response only (deployable).
    Returns (features [N, F], feature_dim).
    """
    n = len(m1)
    lens = np.zeros(n, dtype=np.float32)
    empty = np.zeros(n, dtype=np.float32)
    for i in range(n):
        m = int(m1[i])
        text = (response_texts[i][m] if m < len(response_texts[i]) else "") or ""
        stripped = text.strip()
        lens[i] = np.log1p(len(stripped))
        empty[i] = 1.0 if len(stripped) < 3 else 0.0

    parts = [lens.reshape(-1, 1), empty.reshape(-1, 1)]
    if resp_emb is not None:
        if isinstance(resp_emb, torch.Tensor):
            re = resp_emb.numpy()
        else:
            re = np.asarray(resp_emb, dtype=np.float32)
        idx = np.arange(n)
        m1_emb = re[idx, m1]
        # PCA-free projection: take first emb_proj_dim dims (normalized embeddings).
        d = min(emb_proj_dim, m1_emb.shape[1])
        parts.append(m1_emb[:, :d].astype(np.float32))
    feat = np.concatenate(parts, axis=1).astype(np.float32)
    return feat, feat.shape[1]


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


class RerouteGate(nn.Module):
    """MLP gate: Level0 [h,x1,m1,margin,entropy] or Level1 + answer features."""

    def __init__(
        self,
        query_dim: int,
        n_models: int,
        *,
        hidden_dim: int = 128,
        answer_feat_dim: int = 0,
    ):
        super().__init__()
        in_dim = query_dim + 2 * n_models + 2 + answer_feat_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.answer_feat_dim = answer_feat_dim

    def forward(
        self,
        h: torch.Tensor,
        x1: torch.Tensor,
        m1: torch.Tensor,
        margin: torch.Tensor,
        entropy: torch.Tensor,
        mask: torch.Tensor,
        answer_feat: torch.Tensor | None = None,
        *,
        return_logit: bool = False,
    ) -> torch.Tensor:
        del mask  # features already derived from masked stage1
        m1_oh = _one_hot(m1, x1.size(1))
        parts = [h, x1, m1_oh, margin, entropy]
        if self.answer_feat_dim > 0:
            if answer_feat is None:
                answer_feat = torch.zeros(
                    h.size(0), self.answer_feat_dim, device=h.device, dtype=h.dtype,
                )
            parts.append(answer_feat)
        feat = torch.cat(parts, dim=-1)
        logit = self.net(feat).squeeze(-1)
        if return_logit:
            return logit
        return torch.sigmoid(logit)


class RecourseSelector(nn.Module):
    """MLP selector: regret on ΔU over model candidates (no STOP action)."""

    def __init__(
        self,
        query_dim: int,
        n_models: int,
        *,
        hidden_dim: int = 128,
        answer_feat_dim: int = 0,
    ):
        super().__init__()
        in_dim = query_dim + 2 * n_models + 2 + answer_feat_dim
        self.head = MLP(in_dim, n_models, hidden_dim=hidden_dim)
        self.n_models = n_models
        self.answer_feat_dim = answer_feat_dim

    def forward(
        self,
        h: torch.Tensor,
        x1: torch.Tensor,
        m1: torch.Tensor,
        margin: torch.Tensor,
        entropy: torch.Tensor,
        mask: torch.Tensor,
        answer_feat: torch.Tensor | None = None,
        *,
        candidate_mask: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m1_oh = _one_hot(m1, self.n_models)
        parts = [h, x1, m1_oh, margin, entropy]
        if self.answer_feat_dim > 0:
            if answer_feat is None:
                answer_feat = torch.zeros(
                    h.size(0), self.answer_feat_dim, device=h.device, dtype=h.dtype,
                )
            parts.append(answer_feat)
        feat = torch.cat(parts, dim=-1)
        logits = self.head(feat)
        logits = logits.masked_fill(~mask, -1e9)
        if candidate_mask is not None:
            logits = logits.masked_fill(~candidate_mask, -1e9)
        probs = F.softmax(logits / max(float(temperature), 1e-4), dim=-1)
        return probs, logits


class CascadeDecompRouter(nn.Module):
    """Frozen stage-1 + learned gate + learned selector (two-head)."""

    def __init__(
        self,
        stage1: nn.Module,
        query_dim: int,
        n_models: int,
        *,
        hidden_dim: int = 128,
        gate_level: Literal[0, 1] = 0,
        answer_feat_dim: int = 0,
    ):
        super().__init__()
        self.stage1 = stage1
        self.n_models = n_models
        self.gate_level = gate_level
        self.answer_feat_dim = answer_feat_dim if gate_level >= 1 else 0
        self.gate = RerouteGate(
            query_dim, n_models, hidden_dim=hidden_dim, answer_feat_dim=self.answer_feat_dim,
        )
        self.selector = RecourseSelector(
            query_dim, n_models, hidden_dim=hidden_dim, answer_feat_dim=self.answer_feat_dim,
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
        answer_feat: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
        gate_threshold: float = 0.5,
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        if probs is None or logits is None or m1 is None:
            probs, logits, m1 = self.stage1_forward(h, mask, ap, cfg)
        margin, entropy = self._derive_features(probs, logits, m1, mask)
        gate_logit = self.gate(
            h, probs, m1, margin, entropy, mask, answer_feat, return_logit=True,
        )
        gate_prob = torch.sigmoid(gate_logit)
        sel_probs, sel_logits = self.selector(
            h, probs, m1, margin, entropy, mask, answer_feat,
            candidate_mask=candidate_mask, temperature=temperature,
        )
        m2 = sel_probs.argmax(dim=-1)
        reroute = gate_prob > float(gate_threshold)
        m_final = torch.where(reroute, m2, m1)
        return {
            "probs1": probs,
            "logits1": logits,
            "m1": m1,
            "margin": margin,
            "entropy": entropy,
            "gate_logit": gate_logit,
            "gate_prob": gate_prob,
            "sel_probs": sel_probs,
            "sel_logits": sel_logits,
            "m2": m2,
            "reroute": reroute,
            "m_final": m_final,
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


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


@dataclass
class CascadeDecompTrainConfig:
    hidden_dim: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    epochs: int = 28
    patience: int = 8
    seed: int = 42
    lambda_cost: float = 0.2
    temperature: float = 1.0
    gate_loss_type: Literal["bce", "focal"] = "bce"
    focal_gamma: float = 2.0
    use_pos_weight: bool = True
    gate_threshold: float = 0.5
    selector_topk: int | None = None
    best_single_idx: int | None = None
    train_selector_on_positive_only: bool = False


def train_cascade_decomp(
    router: CascadeDecompRouter,
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
    cfg: CascadeDecompTrainConfig,
    *,
    train_m1: torch.Tensor | None = None,
    train_answer_feat: torch.Tensor | None = None,
    val_answer_feat: torch.Tensor | None = None,
    train_gate_only: bool = False,
    train_selector_only: bool = False,
) -> CascadeDecompRouter:
    """Train gate (BCE/focal) and selector (regret on ΔU)."""
    import random

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    router = router.to(device)
    for p in router.stage1.parameters():
        p.requires_grad = False
    router.stage1.eval()

    if train_gate_only:
        params = list(router.gate.parameters())
    elif train_selector_only:
        params = list(router.selector.parameters())
    else:
        params = list(router.gate.parameters()) + list(router.selector.parameters())

    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    n = train_h.shape[0]
    best_val = float("inf")
    best_state = None
    stale = 0
    rng = random.Random(cfg.seed)

    pos_weight_tensor: torch.Tensor | None = None
    if cfg.use_pos_weight and not train_selector_only:
        with torch.no_grad():
            labels: list[torch.Tensor] = []
            for start in range(0, n, 512):
                h0 = train_h[start : start + 512].to(device)
                perf0 = train_perf[start : start + 512].to(device)
                cost0 = train_cost[start : start + 512].to(device)
                m0 = train_mask[start : start + 512].to(device).bool()
                ap0 = train_ap[start : start + 512].to(device)
                _, _, m1_0 = router.stage1_forward(h0, m0, ap0, stage1_cfg)
                if train_m1 is not None:
                    m1_0 = train_m1[start : start + 512].to(device)
                du = compute_delta_u_torch(perf0, cost0, m1_0, m0, lambda_cost=cfg.lambda_cost)
                y0 = (du[:, 1:].masked_fill(~m0, -1e9).max(dim=-1).values > 0).float()
                labels.append(y0)
            y_all = torch.cat(labels)
            pos = float(y_all.sum().item())
            neg = float(y_all.numel() - pos)
            pos_weight_tensor = torch.tensor(neg / max(pos, 1.0), device=device, dtype=torch.float32)

    def _candidate_mask_batch(probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
        if cfg.selector_topk is None:
            return None
        x1 = probs.detach().cpu().numpy()
        mk = mask.detach().cpu().numpy()
        cm = topk_candidate_mask(
            x1, mk, k=cfg.selector_topk, extra_idx=cfg.best_single_idx,
        )
        return torch.from_numpy(cm).to(device).bool()

    def _batch_loss(h, perf, cost, m, ap, m1_fixed=None, ans=None):
        with torch.no_grad():
            probs, logits, m1 = router.stage1_forward(h, m, ap, stage1_cfg)
            if m1_fixed is not None:
                m1 = m1_fixed
        delta_u = compute_delta_u_torch(perf, cost, m1, m, lambda_cost=cfg.lambda_cost)
        y_reroute = (delta_u[:, 1:].masked_fill(~m, -1e9).max(dim=-1).values > 0).float()

        margin, entropy = router._derive_features(probs, logits, m1, m)
        gate_logit = router.gate(h, probs, m1, margin, entropy, m, ans, return_logit=True)

        loss = torch.zeros((), device=h.device)
        if not train_selector_only:
            if cfg.gate_loss_type == "focal":
                gate_loss = _focal_bce_with_logits(
                    gate_logit, y_reroute, gamma=cfg.focal_gamma, pos_weight=pos_weight_tensor,
                )
            else:
                gate_loss = F.binary_cross_entropy_with_logits(
                    gate_logit, y_reroute, pos_weight=pos_weight_tensor,
                )
            loss = loss + gate_loss

        if not train_gate_only:
            cand = _candidate_mask_batch(probs, m)
            _, sel_logits = router.selector(
                h, probs, m1, margin, entropy, m, ans,
                candidate_mask=cand, temperature=cfg.temperature,
            )
            sel_probs = F.softmax(sel_logits / max(float(cfg.temperature), 1e-4), dim=-1)
            pos_mask = y_reroute > 0.5
            if cfg.train_selector_on_positive_only:
                if pos_mask.any():
                    regret = selector_regret_loss(sel_probs[pos_mask], delta_u[pos_mask], m[pos_mask])
                else:
                    regret = torch.zeros((), device=h.device)
            else:
                regret = selector_regret_loss(sel_probs, delta_u, m)
            loss = loss + regret

        return loss

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
            ans = train_answer_feat[idx].to(device) if train_answer_feat is not None else None
            loss = _batch_loss(h, perf, cost, m, ap, m1f, ans)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
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
                ans = val_answer_feat[start : start + 512].to(device) if val_answer_feat is not None else None
                vloss = _batch_loss(h, perf, cost, m, ap, ans=ans)
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
def infer_cascade_decomp(
    router: CascadeDecompRouter,
    h: torch.Tensor,
    mask: torch.Tensor,
    ap: torch.Tensor,
    *,
    stage1_cfg: RDFLAPTrainConfig,
    m1_override: torch.Tensor | np.ndarray | None = None,
    answer_feat: torch.Tensor | np.ndarray | None = None,
    candidate_mask: np.ndarray | None = None,
    gate_threshold: float = 0.5,
    temperature: float = 1.0,
    gate_override: np.ndarray | None = None,
    selector_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Deployable inference (unless overrides supplied for oracle analysis).
    Returns (m_final, m1, reroute, m2).
    """
    device = next(router.parameters()).device
    router.eval()
    final_parts: list[np.ndarray] = []
    m1_parts: list[np.ndarray] = []
    reroute_parts: list[np.ndarray] = []
    m2_parts: list[np.ndarray] = []
    offset = 0
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

        ans = None
        if answer_feat is not None:
            if isinstance(answer_feat, np.ndarray):
                ans = torch.from_numpy(answer_feat[start : start + 512]).float().to(device)
            else:
                ans = answer_feat[start : start + 512].to(device).float()

        cand_t = None
        if candidate_mask is not None:
            cand_t = torch.from_numpy(candidate_mask[start : start + 512]).to(device).bool()

        out = router(
            hb, mb, apb, stage1_cfg,
            probs=probs, logits=logits, m1=m1,
            answer_feat=ans, candidate_mask=cand_t,
            gate_threshold=gate_threshold, temperature=temperature,
        )
        reroute = out["reroute"].cpu().numpy()
        m2 = out["m2"].cpu().numpy()
        if gate_override is not None:
            reroute = gate_override[start : start + 512].astype(bool)
        if selector_override is not None:
            m2 = selector_override[start : start + 512].astype(np.int64)
        m_final = apply_gate_selector(m1.cpu().numpy(), reroute, m2)
        final_parts.append(m_final)
        m1_parts.append(m1.cpu().numpy())
        reroute_parts.append(reroute)
        m2_parts.append(m2)
        offset += hb.size(0)

    return (
        np.concatenate(final_parts).astype(np.int64),
        np.concatenate(m1_parts).astype(np.int64),
        np.concatenate(reroute_parts).astype(bool),
        np.concatenate(m2_parts).astype(np.int64),
    )


def tune_gate_threshold_val(
    router: CascadeDecompRouter,
    val_h: torch.Tensor,
    val_perf: np.ndarray,
    val_cost: np.ndarray,
    val_mask: torch.Tensor,
    val_ap: torch.Tensor,
    val_m1: np.ndarray,
    *,
    stage1_cfg: RDFLAPTrainConfig,
    lambda_cost: float,
    val_answer_feat: np.ndarray | None = None,
    thresholds: np.ndarray | None = None,
) -> tuple[float, list[dict]]:
    """Tune gate threshold on val utility."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    rows: list[dict] = []
    for tau in thresholds:
        m_final, m1, reroute, _ = infer_cascade_decomp(
            router, val_h, val_mask, val_ap,
            stage1_cfg=stage1_cfg, m1_override=val_m1,
            answer_feat=val_answer_feat, gate_threshold=float(tau),
        )
        row = eval_cascade_decomp(
            val_perf, val_cost, val_mask.numpy().astype(bool),
            m1, m_final, reroute, lambda_cost=lambda_cost,
        )
        row["gate_threshold"] = float(tau)
        rows.append(row)
    best = max(rows, key=lambda r: r["avg_utility"])
    return float(best["gate_threshold"]), rows


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def eval_cascade_decomp(
    test_perf: np.ndarray,
    test_cost: np.ndarray,
    test_mask: np.ndarray,
    m1: np.ndarray,
    m_final: np.ndarray,
    reroute: np.ndarray,
    *,
    lambda_cost: float,
    test_u: np.ndarray | None = None,
    oracle_should_reroute: np.ndarray | None = None,
) -> dict:
    """
    Cascade_true cost accounting + gate/selector decomposition metrics.
    perf from test used ONLY for final metric computation, not routing.

    Utility / Gap@O use ``routed_llmrouterbench_utility`` (perf_tilde + per-query cost bounds,
    same as ``build_oracle_reward`` / ``test_u``). Gap@O compares against
    ``cascade_oracle_utility`` (best single- or two-call strategy given fixed m1).
    ``routed_cost`` is cascade_true: c(m1) plus c(m2) when rerouted.
    """
    from duoroute.reward_builder import cascade_oracle_utility, routed_llmrouterbench_utility

    n = len(m1)
    idx = np.arange(n)
    esc = reroute.astype(bool)
    routed_cost = test_cost[idx, m1].copy()
    routed_cost[esc] += test_cost[idx, m_final][esc]
    routed_u = routed_llmrouterbench_utility(
        test_perf, m_final, routed_cost, test_cost, lambda_cost=lambda_cost,
    )

    stage1_u = routed_llmrouterbench_utility(
        test_perf, m1, test_cost[idx, m1], test_cost, lambda_cost=lambda_cost,
    )
    routed_perf = test_perf[idx, m_final]
    stage1_perf = test_perf[idx, m1]
    m1_ok = stage1_perf >= 0.5

    avg_acc = float((routed_perf >= 0.5).mean())
    avg_utility = float(routed_u.mean())
    avg_cost = float(routed_cost.mean())
    reroute_rate = float(esc.mean())

    # Oracle reroute labels for gate quality (computed from test ΔU if not supplied)
    if oracle_should_reroute is None:
        delta_u = compute_delta_u_np(
            test_perf, test_cost, m1, test_mask, lambda_cost=lambda_cost,
        )
        oracle_should_reroute = oracle_reroute_labels_np(delta_u, test_mask)

    should = oracle_should_reroute.astype(bool)
    tp = int(np.sum(esc & should))
    fp = int(np.sum(esc & ~should))
    fn = int(np.sum(~esc & should))
    reroute_precision = tp / max(tp + fp, 1)
    reroute_recall = tp / max(tp + fn, 1)

    # Rescue / Harm (accuracy-based)
    m2_perf = test_perf[idx, m_final]
    rescue = int(np.sum(~m1_ok & esc & (m2_perf >= 0.5)))
    harm = int(np.sum(m1_ok & esc & (m2_perf < 0.5)))

    utility_gain_on_reroute = None
    cost_per_rescue = None
    delta_u_reroute_subset = None
    if esc.any():
        gain = routed_u[esc] - stage1_u[esc]
        utility_gain_on_reroute = float(gain.mean())
        delta_u_reroute_subset = utility_gain_on_reroute
    rescued_mask = ~m1_ok & esc & (m2_perf >= 0.5)
    if rescued_mask.any():
        cost_per_rescue = float(routed_cost[rescued_mask].mean())

    neg_inf = -1e9
    gap_at_oracle = None
    gap_at_single_call_oracle = None
    cascade_oracle_u = cascade_oracle_utility(
        test_perf, test_cost, test_mask, m1, lambda_cost=lambda_cost,
    )
    gap_at_oracle = float((cascade_oracle_u - routed_u).mean())
    if test_u is not None:
        true_m = np.where(test_mask, test_u, neg_inf)
        single_oracle_u = true_m.max(axis=1)
        gap_at_single_call_oracle = float((single_oracle_u - routed_u).mean())

    return {
        "avg_acc": avg_acc,
        "avg_utility": avg_utility,
        "avg_cost": avg_cost,
        "reroute_rate": reroute_rate,
        "n_rerouted": int(esc.sum()),
        "reroute_precision": float(reroute_precision),
        "reroute_recall": float(reroute_recall),
        "rescue": rescue,
        "harm": harm,
        "cost_per_rescue": cost_per_rescue,
        "delta_utility_reroute_subset": delta_u_reroute_subset,
        "stage1_acc": float(m1_ok.mean()),
        "stage1_avg_cost": float(test_cost[idx, m1].mean()),
        "stage1_avg_utility": float(stage1_u.mean()),
        "delta_utility_vs_stage1": avg_utility - float(stage1_u.mean()),
        "gap_at_oracle": gap_at_oracle,
        "gap_at_single_call_oracle": gap_at_single_call_oracle,
        "utility_gain_on_reroute": utility_gain_on_reroute,
    }


def build_cascade_decomp_router(
    stage1: nn.Module,
    data: dict,
    *,
    hidden_dim: int = 128,
    gate_level: Literal[0, 1] = 0,
    answer_feat_dim: int = 0,
) -> CascadeDecompRouter:
    d = data["train_h"].shape[1]
    k = len(data["model_names"])
    return CascadeDecompRouter(
        stage1, d, k, hidden_dim=hidden_dim,
        gate_level=gate_level, answer_feat_dim=answer_feat_dim,
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
