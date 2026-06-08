"""Recursive Decision-Focused Learning (R-DFL) router for LLM model selection.

Maps the paper's prediction-optimization loop to routing:
  v  = query embedding
  x  = soft routing distribution over K models (simplex)
  F  = predicts per-model utility logits from (v, x)
  G  = masked softmax (differentiable argmax surrogate)
  x_{i+1} = G(F(v, x_i))   for i = 0..K-1

S-DFL baseline uses K=1 with x_0 fixed (uniform / AP init).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from duoroute.model import MLP

FeedbackMode = Literal["full", "none", "detach", "shuffle", "hard", "hard_ste", "query_none", "random"]
PredictorMode = Literal["concat", "two_branch", "interaction"]
StateMode = Literal["evolving", "frozen_init"]
OptLayer = Literal["softmax", "entmax"]


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Softmax over valid models only; invalid arms get 0 mass."""
    temp = max(float(temperature), 1e-4)
    masked = logits.masked_fill(~mask, -1e9)
    w = F.softmax(masked / temp, dim=-1)
    return w * mask.float()


def masked_entmax(logits: torch.Tensor, mask: torch.Tensor, alpha: float = 1.5, n_iter: int = 24) -> torch.Tensor:
    """Masked α-entmax (α=1 → softmax, α=2 → sparsemax)."""
    if alpha <= 1.0 + 1e-6:
        return masked_softmax(logits, mask, temperature=1.0)
    z = logits.masked_fill(~mask, -1e9)
    z = z - z.max(dim=-1, keepdim=True).values
    inv = 1.0 / (float(alpha) - 1.0)
    tau_lo = (z - 1.0).min(dim=-1, keepdim=True).values
    tau_hi = z.max(dim=-1, keepdim=True).values
    for _ in range(n_iter):
        tau_m = (tau_lo + tau_hi) / 2
        p_m = torch.clamp(z - tau_m, min=0).pow(inv) * mask.float()
        f_m = p_m.sum(dim=-1, keepdim=True) - 1.0
        gt = f_m > 0
        tau_lo = torch.where(gt, tau_m, tau_lo)
        tau_hi = torch.where(gt, tau_hi, tau_m)
    p = torch.clamp(z - tau_lo, min=0).pow(inv) * mask.float()
    return p / p.sum(dim=-1, keepdim=True).clamp(min=1e-8)


def apply_opt_layer(
    logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    opt_layer: OptLayer = "softmax",
    temperature: float = 1.0,
    entmax_alpha: float = 1.5,
) -> torch.Tensor:
    if opt_layer == "entmax":
        return masked_entmax(logits, mask, alpha=entmax_alpha)
    return masked_softmax(logits, mask, temperature=temperature)


def hard_choices(probs: torch.Tensor) -> torch.Tensor:
    return probs.argmax(dim=-1)


class SequentialDFLRouter(nn.Module):
    """S-DFL: one-shot utility prediction from query only (no feedback)."""

    def __init__(self, query_dim: int, n_models: int, *, hidden_dim: int = 128):
        super().__init__()
        self.head = MLP(query_dim, n_models, hidden_dim=hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        mask: torch.Tensor,
        *,
        n_steps: int = 1,
        temperature: float = 1.0,
        x_init: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del n_steps, x_init
        logits = self.head(query)
        probs = masked_softmax(logits, mask, temperature=temperature)
        return probs, logits


class RecursiveDFLRouter(nn.Module):
    """R-DFL: K-step bidirectional feedback between routing state and prediction."""

    def __init__(
        self,
        query_dim: int,
        n_models: int,
        *,
        hidden_dim: int = 128,
        feedback_dim: int | None = None,
        feedback_hidden: int = 128,
        default_steps: int = 5,
        feedback_mode: FeedbackMode = "full",
        state_mode: StateMode = "evolving",
        rich_feedback: bool = False,
        query_init: bool = False,
        feedback_scale: float = 0.25,
        predictor_mode: PredictorMode = "concat",
        branch_dim: int = 128,
        opt_layer: OptLayer = "softmax",
        entmax_alpha: float = 1.5,
    ):
        super().__init__()
        self.n_models = n_models
        self.default_steps = default_steps
        self.feedback_mode = feedback_mode
        self.state_mode = state_mode
        self.rich_feedback = rich_feedback
        self.query_init = query_init
        self.feedback_scale = feedback_scale
        self.predictor_mode = predictor_mode
        self.branch_dim = branch_dim
        self.opt_layer = opt_layer
        self.entmax_alpha = entmax_alpha
        fb = feedback_dim or (feedback_hidden if rich_feedback else n_models)
        if query_init:
            self.init_head = MLP(query_dim, n_models, hidden_dim=hidden_dim)
        else:
            self.init_head = None
        if rich_feedback:
            self.feedback_enc = nn.Sequential(
                nn.Linear(n_models * 3, feedback_hidden),
                nn.ReLU(),
                nn.Linear(feedback_hidden, feedback_hidden),
            )
            self.query_branch = MLP(query_dim, n_models, hidden_dim=hidden_dim)
            self.feedback_branch = MLP(feedback_hidden, n_models, hidden_dim=hidden_dim)
            nn.init.zeros_(self.feedback_branch.net[-1].weight)
            nn.init.zeros_(self.feedback_branch.net[-1].bias)
        elif feedback_mode == "none":
            self.feedback_proj = nn.Identity()
            self.head = MLP(query_dim, n_models, hidden_dim=hidden_dim)
            self.phi_h = None
            self.phi_x = None
            self.out_head = None
        elif predictor_mode in ("two_branch", "interaction"):
            self.feedback_proj = nn.Identity()
            self.head = None
            self.phi_h = MLP(query_dim, branch_dim, hidden_dim=hidden_dim)
            self.phi_x = MLP(n_models, branch_dim, hidden_dim=hidden_dim)
            out_in = branch_dim * (3 if predictor_mode == "interaction" else 2)
            self.out_head = nn.Linear(out_in, n_models)
        elif feedback_mode == "query_none":
            self.feedback_proj = nn.Identity()
            self.phi_h = None
            self.phi_x = None
            self.out_head = None
            if fb != n_models:
                self.feedback_proj = nn.Linear(fb, fb)
                in_dim = fb
            else:
                in_dim = n_models
            self.head = MLP(in_dim, n_models, hidden_dim=hidden_dim)
        else:
            self.phi_h = None
            self.phi_x = None
            self.out_head = None
            if fb != n_models:
                self.feedback_proj: nn.Module = nn.Linear(fb, fb)
                in_dim = query_dim + fb
            else:
                self.feedback_proj = nn.Identity()
                in_dim = query_dim + n_models
            self.head = MLP(in_dim, n_models, hidden_dim=hidden_dim)

    def _feedback(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.feedback_mode == "none":
            return x
        if self.feedback_mode == "detach":
            return x.detach()
        if self.feedback_mode == "shuffle":
            perm = torch.randperm(x.size(0), device=x.device)
            return x[perm]
        if self.feedback_mode == "hard":
            idx = x.argmax(dim=-1)
            hard = torch.zeros_like(x)
            hard.scatter_(1, idx.unsqueeze(1), 1.0)
            if mask is not None:
                hard = hard * mask.float()
                hard = hard / hard.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            return hard
        if self.feedback_mode == "hard_ste":
            idx = x.argmax(dim=-1)
            hard = torch.zeros_like(x)
            hard.scatter_(1, idx.unsqueeze(1), 1.0)
            if mask is not None:
                hard = hard * mask.float()
                hard = hard / hard.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            return (hard - x).detach() + x
        if self.feedback_mode == "random":
            probs = torch.rand_like(x)
            if mask is not None:
                probs = probs * mask.float()
            return probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return x

    def _project_feedback(self, x: torch.Tensor, x_prev: torch.Tensor | None, mask: torch.Tensor | None) -> torch.Tensor:
        x_fb = self._feedback(x, mask)
        if self.rich_feedback and mask is not None and x_prev is not None:
            delta = (x_fb - x_prev) * mask.float()
            ent = _masked_entropy(x_fb, mask).unsqueeze(-1).expand_as(x_fb)
            raw = torch.cat([x_fb * mask.float(), delta, ent * mask.float()], dim=-1)
            return self.feedback_enc(raw)
        return self.feedback_proj(x_fb)

    def predict_logits(
        self,
        query: torch.Tensor,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        x_prev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.rich_feedback and self.feedback_mode != "none":
            fb = self._project_feedback(x, x_prev, mask)
            if self.feedback_mode == "query_none":
                return self.feedback_branch(fb)
            return self.query_branch(query) + self.feedback_scale * self.feedback_branch(fb)
        if self.feedback_mode == "none":
            return self.head(query)
        if self.feedback_mode == "query_none":
            x_feat = self._project_feedback(x, x_prev, mask)
            return self.head(x_feat)
        x_feat = self._project_feedback(x, x_prev, mask)
        if self.phi_h is not None and self.phi_x is not None and self.out_head is not None:
            q = self.phi_h(query)
            s = self.phi_x(x_feat)
            if self.predictor_mode == "interaction":
                feat = torch.cat([q, s, q * s], dim=-1)
            else:
                feat = torch.cat([q, s], dim=-1)
            return self.out_head(feat)
        return self.head(torch.cat([query, x_feat], dim=-1))

    def forward(
        self,
        query: torch.Tensor,
        mask: torch.Tensor,
        *,
        n_steps: int = 5,
        temperature: float = 1.0,
        x_init: torch.Tensor | None = None,
        ap_init: torch.Tensor | None = None,
        return_trajectory: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        counts = mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        x_prev = mask.float() / counts
        if ap_init is not None:
            x = ap_init
        elif self.query_init and self.init_head is not None:
            x = masked_softmax(self.init_head(query), mask, temperature=temperature)
        elif x_init is None:
            x = x_prev
        else:
            x = x_init
        x = x * mask.float()
        x = x / x.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        x_anchor = x
        logits_last = None
        trajectory = [x]
        for _ in range(max(1, n_steps)):
            x_in = x_anchor if self.state_mode == "frozen_init" else x
            logits_last = self.predict_logits(query, x_in, mask, x_prev=x_prev)
            x_prev = x
            x = apply_opt_layer(
                logits_last,
                mask,
                opt_layer=self.opt_layer,
                temperature=temperature,
                entmax_alpha=self.entmax_alpha,
            )
            trajectory.append(x)
        assert logits_last is not None
        if return_trajectory:
            return x, logits_last, trajectory[1:]
        return x, logits_last


class RecursiveDFLRouterImplicit(nn.Module):
    """R-DFL-I-lite: fixed-point approximation x* ≈ G(F(h, x*))."""

    def __init__(
        self,
        query_dim: int,
        n_models: int,
        *,
        hidden_dim: int = 128,
        default_steps: int = 20,
        feedback_mode: FeedbackMode = "full",
        predictor_mode: PredictorMode = "concat",
        branch_dim: int = 128,
        mix_alpha: float = 0.5,
        opt_layer: OptLayer = "softmax",
        entmax_alpha: float = 1.5,
    ):
        super().__init__()
        self.core = RecursiveDFLRouter(
            query_dim,
            n_models,
            hidden_dim=hidden_dim,
            default_steps=default_steps,
            feedback_mode=feedback_mode,
            state_mode="evolving",
            predictor_mode=predictor_mode,
            branch_dim=branch_dim,
            opt_layer=opt_layer,
            entmax_alpha=entmax_alpha,
        )
        self.default_steps = default_steps
        self.mix_alpha = mix_alpha
        self.feedback_mode = feedback_mode

    def forward(
        self,
        query: torch.Tensor,
        mask: torch.Tensor,
        *,
        n_steps: int | None = None,
        temperature: float = 1.0,
        x_init: torch.Tensor | None = None,
        ap_init: torch.Tensor | None = None,
        return_trajectory: bool = False,
    ):
        del ap_init
        steps = n_steps or self.default_steps
        counts = mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        x = mask.float() / counts if x_init is None else x_init
        x = x * mask.float()
        x = x / x.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        alpha = self.mix_alpha
        logits_last = None
        trajectory: list[torch.Tensor] = []
        for _ in range(max(1, steps)):
            logits_last = self.core.predict_logits(query, x, mask)
            x_next = apply_opt_layer(
                logits_last,
                mask,
                opt_layer=self.core.opt_layer,
                temperature=temperature,
                entmax_alpha=self.core.entmax_alpha,
            )
            x = alpha * x_next + (1.0 - alpha) * x
            x = x * mask.float()
            x = x / x.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            trajectory.append(x)
        if return_trajectory:
            return x, logits_last, trajectory
        return x, logits_last


def _masked_entropy(probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    p = probs.clamp(min=1e-8) * mask.float()
    p = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return -(p * p.log()).sum(dim=-1)


def _step_temperature(
    step_idx: int,
    n_steps: int,
    *,
    base: float,
    anneal: bool,
    temp_start: float,
    temp_end: float,
) -> float:
    if not anneal or n_steps <= 1:
        return base
    frac = step_idx / max(n_steps - 1, 1)
    return temp_start + (temp_end - temp_start) * frac


class RecursiveDFLRouterV2(nn.Module):
    """R-DFL v2: query-first logits + residual refinement with encoded feedback."""

    def __init__(
        self,
        query_dim: int,
        n_models: int,
        *,
        hidden_dim: int = 128,
        feedback_hidden: int = 128,
        max_steps: int = 8,
        feedback_mode: FeedbackMode = "full",
        state_mode: StateMode = "evolving",
        refine_scale: float = 0.5,
    ):
        super().__init__()
        self.n_models = n_models
        self.max_steps = max_steps
        self.feedback_mode = feedback_mode
        self.state_mode = state_mode
        self.refine_scale = refine_scale
        self.init_head = MLP(query_dim, n_models, hidden_dim=hidden_dim)
        self.query_norm = nn.LayerNorm(query_dim)
        self.step_embed = nn.Embedding(max_steps, hidden_dim)
        fb_in = n_models * 3
        self.feedback_enc = nn.Sequential(
            nn.Linear(fb_in, feedback_hidden),
            nn.ReLU(),
            nn.Linear(feedback_hidden, hidden_dim),
        )
        if feedback_mode == "query_none":
            refine_in = hidden_dim + hidden_dim
        elif feedback_mode == "none":
            refine_in = query_dim + hidden_dim
        else:
            refine_in = query_dim + hidden_dim + hidden_dim
        self.refine_head = MLP(refine_in, n_models, hidden_dim=hidden_dim)
        nn.init.zeros_(self.refine_head.net[-1].weight)
        nn.init.zeros_(self.refine_head.net[-1].bias)

    def _encode_feedback(self, x: torch.Tensor, x_prev: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        delta = (x - x_prev) * mask.float()
        ent = _masked_entropy(x, mask).unsqueeze(-1).expand_as(x)
        raw = torch.cat([x * mask.float(), delta, ent * mask.float()], dim=-1)
        return self.feedback_enc(raw)

    def _refine_delta(
        self,
        query: torch.Tensor,
        x_in: torch.Tensor,
        x_prev: torch.Tensor,
        mask: torch.Tensor,
        step_idx: int,
    ) -> torch.Tensor:
        q = self.query_norm(query)
        fb = self._encode_feedback(x_in, x_prev, mask)
        step_e = self.step_embed(torch.tensor(step_idx, device=query.device)).unsqueeze(0).expand(query.size(0), -1)
        if self.feedback_mode == "query_none":
            delta = self.refine_head(torch.cat([fb, step_e], dim=-1))
        elif self.feedback_mode == "none":
            delta = self.refine_head(torch.cat([q, step_e], dim=-1))
        else:
            delta = self.refine_head(torch.cat([q, fb, step_e], dim=-1))
        return delta * self.refine_scale

    def forward(
        self,
        query: torch.Tensor,
        mask: torch.Tensor,
        *,
        n_steps: int = 3,
        temperature: float = 1.0,
        temp_anneal: bool = False,
        temp_start: float = 2.0,
        temp_end: float = 0.5,
        x_init: torch.Tensor | None = None,
        ap_init: torch.Tensor | None = None,
        return_trajectory: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        del ap_init, x_init
        n_steps = max(1, n_steps)
        counts = mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        x_prev = mask.float() / counts

        tau0 = _step_temperature(0, n_steps, base=temperature, anneal=temp_anneal, temp_start=temp_start, temp_end=temp_end)
        logits = self.init_head(self.query_norm(query))
        x = masked_softmax(logits, mask, temperature=tau0)
        x_anchor = x
        trajectory = [x]
        logits_last = logits

        for step in range(1, n_steps):
            x_in = x_anchor if self.state_mode == "frozen_init" else x
            tau = _step_temperature(
                step, n_steps, base=temperature, anneal=temp_anneal, temp_start=temp_start, temp_end=temp_end
            )
            logits = logits + self._refine_delta(query, x_in, x_prev, mask, step)
            logits_last = logits
            x_prev = x
            x = masked_softmax(logits, mask, temperature=tau)
            trajectory.append(x)

        if return_trajectory:
            return x, logits_last, trajectory
        return x, logits_last


class ModelCardAttentionRouter(nn.Module):
    """R-DFL with bilinear model-card cross-attention scorer and K-step feedback."""

    def __init__(
        self,
        query_dim: int,
        n_models: int,
        model_embeddings: torch.Tensor,
        *,
        hidden_dim: int = 128,
        proj_dim: int = 128,
        default_steps: int = 3,
        feedback_mode: FeedbackMode = "full",
        state_mode: StateMode = "evolving",
        num_heads: int = 4,
        opt_layer: OptLayer = "softmax",
        entmax_alpha: float = 1.5,
    ):
        super().__init__()
        self.n_models = n_models
        self.default_steps = default_steps
        self.feedback_mode = feedback_mode
        self.state_mode = state_mode
        self.opt_layer = opt_layer
        self.entmax_alpha = entmax_alpha
        mdim = model_embeddings.shape[1]
        self.model_emb = nn.Parameter(model_embeddings[:n_models].clone(), requires_grad=False)
        self.query_proj = nn.Linear(query_dim, proj_dim)
        self.model_proj = nn.Linear(mdim, proj_dim)
        self.bilinear_w = nn.Parameter(torch.randn(proj_dim, proj_dim) * 0.02)
        self.feedback_proj = nn.Linear(n_models, proj_dim)
        self.attn = nn.MultiheadAttention(proj_dim, num_heads, batch_first=True)

    def _feedback(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.feedback_mode == "none":
            return torch.zeros_like(x)
        if self.feedback_mode == "detach":
            return x.detach()
        if self.feedback_mode == "hard":
            idx = x.argmax(dim=-1)
            hard = torch.zeros_like(x)
            hard.scatter_(1, idx.unsqueeze(1), 1.0)
            if mask is not None:
                hard = hard * mask.float()
                hard = hard / hard.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            return hard
        if self.feedback_mode == "hard_ste":
            idx = x.argmax(dim=-1)
            hard = torch.zeros_like(x)
            hard.scatter_(1, idx.unsqueeze(1), 1.0)
            if mask is not None:
                hard = hard * mask.float()
                hard = hard / hard.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            return (hard - x).detach() + x
        return x

    def predict_logits(self, query: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch = query.size(0)
        q = self.query_proj(query)
        m = self.model_proj(self.model_emb)
        q_w = q @ self.bilinear_w
        scores = torch.einsum("bd,md->bm", q_w, m)
        m_exp = m.unsqueeze(0).expand(batch, -1, -1)
        attn_out, _ = self.attn(q.unsqueeze(1), m_exp, m_exp)
        scores = scores + torch.einsum("bd,md->bm", attn_out.squeeze(1), m)
        if self.feedback_mode != "none":
            fb = self._feedback(x, mask)
            fb_q = self.feedback_proj(fb)
            scores = scores + torch.einsum("bd,md->bm", fb_q, m)
        return scores.masked_fill(~mask, -1e9)

    def forward(
        self,
        query: torch.Tensor,
        mask: torch.Tensor,
        *,
        n_steps: int = 3,
        temperature: float = 1.0,
        x_init: torch.Tensor | None = None,
        ap_init: torch.Tensor | None = None,
        return_trajectory: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        del ap_init
        counts = mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        x_prev = mask.float() / counts
        if x_init is None:
            x = x_prev
        else:
            x = x_init
        x = x * mask.float()
        x = x / x.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        x_anchor = x
        logits_last = None
        trajectory = [x]
        for _ in range(max(1, n_steps)):
            x_in = x_anchor if self.state_mode == "frozen_init" else x
            logits_last = self.predict_logits(query, x_in, mask)
            x_prev = x
            x = apply_opt_layer(
                logits_last,
                mask,
                opt_layer=self.opt_layer,
                temperature=temperature,
                entmax_alpha=self.entmax_alpha,
            )
            trajectory.append(x)
        assert logits_last is not None
        if return_trajectory:
            return x, logits_last, trajectory[1:]
        return x, logits_last


def stepwise_decision_focused_loss(
    trajectory: list[torch.Tensor],
    oracle_reward: torch.Tensor,
    mask: torch.Tensor,
    *,
    mode: Literal["utility", "regret"] = "regret",
    gamma: float = 0.5,
) -> torch.Tensor:
    """Weight later refinement steps more (final routing matters most)."""
    total_w = 0.0
    loss = torch.zeros((), device=oracle_reward.device)
    n = len(trajectory)
    for i, probs in enumerate(trajectory):
        w = gamma ** (n - 1 - i)
        loss = loss + w * decision_focused_loss(probs, oracle_reward, mask, mode=mode)
        total_w += w
    return loss / max(total_w, 1e-8)


def oracle_ce_loss(
    logits: torch.Tensor,
    oracle_reward: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Supervised CE on masked argmax(oracle_reward); not decision-focused."""
    masked_oracle = oracle_reward.masked_fill(~mask, -1e9)
    target = masked_oracle.argmax(dim=-1)
    masked_logits = logits.masked_fill(~mask, -1e9)
    return F.cross_entropy(masked_logits, target)


def utility_kl_loss(
    probs: torch.Tensor,
    oracle_reward: torch.Tensor,
    mask: torch.Tensor,
    *,
    tau_star: float = 0.2,
    beta: float = 0.05,
) -> torch.Tensor:
    """KL(π* || probs) with π* = softmax(U/τ*) over masked oracle utilities."""
    target = masked_softmax(oracle_reward, mask, temperature=tau_star)
    log_probs = torch.where(mask, probs.clamp(min=1e-8).log(), torch.zeros_like(probs))
    ce = -(target * log_probs).sum(dim=-1).mean()
    return beta * ce


def oracle_contrastive_loss(
    logits: torch.Tensor,
    oracle_reward: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float = 0.03,
) -> torch.Tensor:
    """Cross-entropy on argmax oracle utility (contrastive pull toward oracle)."""
    return beta * oracle_ce_loss(logits, oracle_reward, mask)


def decision_focused_loss(
    probs: torch.Tensor,
    oracle_reward: torch.Tensor,
    mask: torch.Tensor,
    *,
    mode: Literal["utility", "regret"] = "utility",
) -> torch.Tensor:
    """
    Differentiable decision loss on soft routing distribution.
    utility: maximize E_{j~probs}[oracle_j]  -> minimize negative
    regret:  E[oracle_max - oracle_j]
    """
    masked_oracle = oracle_reward.masked_fill(~mask, -1e9)
    if mode == "utility":
        expected = (probs * oracle_reward * mask.float()).sum(dim=-1)
        return -expected.mean()
    oracle_best = masked_oracle.max(dim=-1).values
    expected = (probs * oracle_reward * mask.float()).sum(dim=-1)
    regret = oracle_best - expected
    return regret.mean()


@dataclass
class RDFTrainConfig:
    n_steps: int = 5
    hidden_dim: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    epochs: int = 40
    temperature: float = 1.0
    loss_mode: Literal["utility", "regret"] = "utility"
    patience: int = 8
    seed: int = 42


def train_router(
    model: nn.Module,
    train_q: torch.Tensor,
    train_oracle: torch.Tensor,
    train_mask: torch.Tensor,
    val_q: torch.Tensor,
    val_oracle: torch.Tensor,
    val_mask: torch.Tensor,
    cfg: RDFTrainConfig,
) -> nn.Module:
    import random

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n = train_q.shape[0]
    best_val = float("inf")
    best_state = None
    stale = 0
    rng = random.Random(cfg.seed)

    for _epoch in range(cfg.epochs):
        model.train()
        perm = list(range(n))
        rng.shuffle(perm)
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            if not idx:
                continue
            q = train_q[idx].to(device)
            o = train_oracle[idx].to(device)
            m = train_mask[idx].to(device)
            steps = cfg.n_steps if isinstance(model, RecursiveDFLRouter) else 1
            probs, _ = model(q, m, n_steps=steps, temperature=cfg.temperature)
            loss = decision_focused_loss(probs, o, m, mode=cfg.loss_mode)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vloss = 0.0
            bs = 512
            for start in range(0, val_q.shape[0], bs):
                q = val_q[start : start + bs].to(device)
                o = val_oracle[start : start + bs].to(device)
                m = val_mask[start : start + bs].to(device)
                steps = cfg.n_steps if isinstance(model, RecursiveDFLRouter) else 1
                probs, _ = model(q, m, n_steps=steps, temperature=cfg.temperature)
                vloss += float(decision_focused_loss(probs, o, m, mode=cfg.loss_mode).item()) * q.shape[0]
            vloss /= max(val_q.shape[0], 1)
        if vloss < best_val - 1e-5:
            best_val = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def route_hard(
    model: nn.Module,
    query: torch.Tensor,
    mask: torch.Tensor,
    *,
    n_steps: int = 5,
    temperature: float = 1.0,
) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    chosen = []
    bs = 512
    for start in range(0, query.shape[0], bs):
        q = query[start : start + bs].to(device)
        m = mask[start : start + bs].to(device).bool()
        if isinstance(model, RecursiveDFLRouter):
            probs, _ = model(q, m, n_steps=n_steps, temperature=temperature)
        else:
            probs, _ = model(q, m, n_steps=1, temperature=temperature)
        chosen.append(hard_choices(probs).cpu().numpy())
    return np.concatenate(chosen, axis=0).astype(np.int64)
