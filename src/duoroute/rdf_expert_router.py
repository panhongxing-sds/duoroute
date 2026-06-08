"""Expert/domain-aware recursive DFL router with per (query, model) scoring."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from duoroute.rdf_router import FeedbackMode, StateMode, masked_softmax


class ExpertAwareRecursiveDFLRouter(nn.Module):
    """
    s_{q,m} = MLP([q, e_m, q⊙e_m, (d_q, d_q⊙e_m), x_{t-1,m}, cost_m])
    x_t = MaskedSoftmax(s_q)
    """

    def __init__(
        self,
        query_dim: int,
        n_models: int,
        cost: torch.Tensor,
        *,
        expert_dim: int = 64,
        query_proj_dim: int = 128,
        num_domains: int = 0,
        hidden_dim: int = 128,
        use_interaction: bool = True,
        use_cost: bool = True,
        feedback_mode: FeedbackMode = "full",
        state_mode: StateMode = "evolving",
        default_steps: int = 3,
    ):
        super().__init__()
        self.n_models = n_models
        self.expert_dim = expert_dim
        self.query_proj_dim = query_proj_dim
        self.num_domains = num_domains
        self.use_interaction = use_interaction
        self.use_cost = use_cost
        self.feedback_mode = feedback_mode
        self.state_mode = state_mode
        self.default_steps = default_steps

        self.register_buffer("cost", cost.float().view(1, n_models, 1))
        self.query_proj = nn.Linear(query_dim, query_proj_dim)
        self.expert_emb = nn.Embedding(n_models, expert_dim)
        self.q_int_proj = (
            nn.Linear(query_proj_dim, expert_dim) if use_interaction else None
        )

        if num_domains > 0:
            self.domain_proto = nn.Parameter(torch.randn(num_domains, expert_dim) * 0.02)
            self.domain_logits = nn.Linear(query_dim, num_domains)
        else:
            self.domain_proto = None
            self.domain_logits = None

        feat_dim = query_proj_dim + expert_dim
        if use_interaction:
            feat_dim += expert_dim
        if num_domains > 0:
            feat_dim += expert_dim
            if use_interaction:
                feat_dim += expert_dim
        if feedback_mode != "none":
            feat_dim += 1
        if use_cost:
            feat_dim += 1

        self.pair_mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _latent_domain(self, query: torch.Tensor) -> torch.Tensor | None:
        if self.domain_proto is None or self.domain_logits is None:
            return None
        weights = F.softmax(self.domain_logits(query), dim=-1)
        return weights @ self.domain_proto

    def _pair_features(
        self,
        query: torch.Tensor,
        x_prev: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch = query.size(0)
        q = self.query_proj(query)
        e = self.expert_emb.weight
        q_exp = q.unsqueeze(1).expand(batch, self.n_models, self.query_proj_dim)
        e_exp = e.unsqueeze(0).expand(batch, self.n_models, self.expert_dim)
        parts = [q_exp, e_exp]
        if self.use_interaction and self.q_int_proj is not None:
            q_int = self.q_int_proj(q).unsqueeze(1).expand(batch, self.n_models, self.expert_dim)
            parts.append(q_int * e_exp)
        d_q = self._latent_domain(query)
        if d_q is not None:
            d_exp = d_q.unsqueeze(1).expand(batch, self.n_models, self.expert_dim)
            parts.append(d_exp)
            if self.use_interaction:
                parts.append(d_exp * e_exp)
        if self.feedback_mode != "none":
            parts.append(x_prev.unsqueeze(-1))
        if self.use_cost:
            parts.append(self.cost.expand(batch, self.n_models, 1))
        feat = torch.cat(parts, dim=-1)
        return feat * mask.unsqueeze(-1).float()

    def score_logits(self, query: torch.Tensor, x_prev: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.feedback_mode == "query_none":
            x_in = torch.zeros_like(x_prev)
        elif self.feedback_mode == "none":
            x_in = torch.zeros_like(x_prev)
        else:
            x_in = x_prev
        feat = self._pair_features(query, x_in, mask)
        logits = self.pair_mlp(feat).squeeze(-1)
        return logits.masked_fill(~mask, -1e9)

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
        return_logits: bool = False,
    ):
        del x_init, ap_init
        counts = mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        x_prev = mask.float() / counts
        x = x_prev.clone()
        x_anchor = x
        logits_last = None
        trajectory: list[torch.Tensor] = []

        for _ in range(max(1, n_steps)):
            x_in = x_anchor if self.state_mode == "frozen_init" else x
            logits_last = self.score_logits(query, x_in, mask)
            x_prev = x
            x = masked_softmax(logits_last, mask, temperature=temperature)
            trajectory.append(x)

        if return_trajectory and return_logits:
            return x, logits_last, trajectory
        if return_trajectory:
            return x, logits_last, trajectory
        if return_logits:
            return x, logits_last
        return x, logits_last


def rare_expert_weights(
    performance: torch.Tensor,
    mask: torch.Tensor,
    *,
    mode: str = "mid",
) -> torch.Tensor:
    """Per-query sample weights from count of models with perf >= 0.5."""
    correct = ((performance >= 0.5) & mask).sum(dim=-1).float()
    w = torch.ones_like(correct)
    if mode == "none":
        return w
    if mode == "light":
        w = torch.where(correct == 0, 0.5, w)
        w = torch.where(correct == 1, 2.0, w)
        w = torch.where((correct >= 2) & (correct <= 3), 1.5, w)
        return w
    if mode == "strong":
        w = torch.where(correct == 0, 0.5, w)
        w = torch.where(correct == 1, 4.0, w)
        w = torch.where((correct >= 2) & (correct <= 3), 2.5, w)
        return w
    # mid (default)
    w = torch.where(correct == 0, 0.5, w)
    w = torch.where(correct == 1, 3.0, w)
    w = torch.where((correct >= 2) & (correct <= 3), 2.0, w)
    return w


def pairwise_ranking_loss(
    logits: torch.Tensor,
    oracle_reward: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float = 0.05,
) -> torch.Tensor:
    if beta <= 0:
        return torch.zeros((), device=logits.device)
    star = oracle_reward.masked_fill(~mask, -1e9).argmax(dim=-1, keepdim=True)
    u_star = oracle_reward.gather(1, star)
    gap = (u_star - oracle_reward).clamp(min=0.0) * mask.float()
    gap.scatter_(1, star, 0.0)
    diff = logits.gather(1, star) - logits
    per_pair = gap * F.softplus(-diff)
    denom = gap.gt(0).sum().clamp(min=1).float()
    return beta * per_pair.sum() / denom


def batch_usage_entropy_reg(probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Encourage diverse batch-level routing; return value to subtract from loss."""
    valid = mask.float()
    denom = valid.sum(dim=0).clamp(min=1.0)
    mean_p = (probs * valid).sum(dim=0) / denom
    mean_p = mean_p / mean_p.sum().clamp(min=1e-8)
    ent = -(mean_p * mean_p.clamp(min=1e-8).log()).sum()
    max_ent = torch.log(torch.tensor(float(mask.shape[1]), device=probs.device))
    return ent / max_ent.clamp(min=1e-8)


def topk_oracle_recall_loss(
    logits: torch.Tensor,
    oracle_reward: torch.Tensor,
    mask: torch.Tensor,
    *,
    k: int = 3,
    gamma: float = 0.05,
) -> torch.Tensor:
    """Penalize when utility-oracle is outside top-k logits."""
    if gamma <= 0:
        return torch.zeros((), device=logits.device)
    star = oracle_reward.masked_fill(~mask, -1e9).argmax(dim=-1)
    neg = logits.masked_fill(~mask, -1e9)
    topk = neg.topk(min(k, mask.shape[1]), dim=-1).indices
    hit = (topk == star.unsqueeze(1)).any(dim=1)
    return gamma * (~hit).float().mean()


class TopKReranker(nn.Module):
    """Rerank within base-router top-k candidates."""

    def __init__(
        self,
        query_dim: int,
        n_models: int,
        cost: torch.Tensor,
        *,
        top_k: int = 3,
        hidden_dim: int = 64,
        query_proj_dim: int = 64,
    ):
        super().__init__()
        self.n_models = n_models
        self.top_k = top_k
        self.register_buffer("cost", cost.float().view(n_models))
        self.query_proj = nn.Linear(query_dim, query_proj_dim)
        self.rank_emb = nn.Embedding(top_k, 16)
        feat_dim = query_proj_dim + 3 + 16  # base_prob, base_logit, cost, rank
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def score_candidates(
        self,
        query: torch.Tensor,
        mask: torch.Tensor,
        base_probs: torch.Tensor,
        base_logits: torch.Tensor,
        cand_idx: torch.Tensor,
        cand_rank: torch.Tensor,
    ) -> torch.Tensor:
        batch, k = cand_idx.shape
        q = self.query_proj(query).unsqueeze(1).expand(batch, k, -1)
        bp = base_probs.gather(1, cand_idx).unsqueeze(-1)
        bl = base_logits.gather(1, cand_idx).unsqueeze(-1)
        c = self.cost[cand_idx].unsqueeze(-1)
        r = self.rank_emb(cand_rank)
        feat = torch.cat([q, bp, bl, c, r], dim=-1)
        return self.head(feat).squeeze(-1)

    def forward(
        self,
        query: torch.Tensor,
        mask: torch.Tensor,
        base_probs: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        neg = base_logits.masked_fill(~mask, -1e9)
        k = min(self.top_k, mask.shape[1])
        topv, topi = neg.topk(k, dim=-1)
        ranks = torch.arange(k, device=query.device).unsqueeze(0).expand(query.size(0), k)
        scores = self.score_candidates(query, mask, base_probs, base_logits, topi, ranks)
        pick = scores.argmax(dim=-1)
        chosen = topi.gather(1, pick.unsqueeze(1)).squeeze(1)
        full = torch.full_like(base_probs, -1e9)
        full.scatter_(1, topi, scores)
        return chosen, full


def rerank_topk_loss(
    reranker: TopKReranker,
    query: torch.Tensor,
    mask: torch.Tensor,
    base_probs: torch.Tensor,
    base_logits: torch.Tensor,
    oracle_reward: torch.Tensor,
) -> torch.Tensor:
    neg = base_logits.masked_fill(~mask, -1e9)
    k = min(reranker.top_k, mask.shape[1])
    topi = neg.topk(k, dim=-1).indices
    ranks = torch.arange(k, device=query.device).unsqueeze(0).expand(query.size(0), k)
    scores = reranker.score_candidates(query, mask, base_probs, base_logits, topi, ranks)
    star = oracle_reward.masked_fill(~mask, -1e9).argmax(dim=-1)
    loss = torch.zeros((), device=query.device)
    n = 0
    for b in range(query.size(0)):
        s = int(star[b].item())
        cand = topi[b].tolist()
        if s not in cand:
            continue
        pos = cand.index(s)
        for j, cj in enumerate(cand):
            if j == pos:
                continue
            gap = (oracle_reward[b, s] - oracle_reward[b, cj]).clamp(min=0.0)
            if gap.item() <= 0:
                continue
            loss = loss + gap * F.softplus(-(scores[b, pos] - scores[b, j]))
            n += 1
    if n == 0:
        return torch.zeros((), device=query.device)
    return loss / n


def weighted_decision_focused_loss(
    probs: torch.Tensor,
    oracle_reward: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
    *,
    mode: str = "regret",
) -> torch.Tensor:
    from duoroute.rdf_router import decision_focused_loss

    masked_oracle = oracle_reward.masked_fill(~mask, -1e9)
    if mode == "utility":
        per = -(probs * oracle_reward * mask.float()).sum(dim=-1)
    else:
        best = masked_oracle.max(dim=-1).values
        expected = (probs * oracle_reward * mask.float()).sum(dim=-1)
        per = best - expected
    w = weights / weights.sum().clamp(min=1e-8) * weights.numel()
    return (w * per).mean()
