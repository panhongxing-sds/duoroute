"""Wrong-only query×model recourse selector for RegretRouter + Cascade Stage2.

Score each (query, model) pair:
  s_{q,m} = f_θ(h_q, e_m, h_q ⊙ e_m, c_m, m_1)

Trained only on verified-wrong samples with wrong-only recourse utility:
  ΔU_m = perf_m - λ * cost_m

Loss: L = L_regret + β * L_rank (pairwise margin on ΔU gaps).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from duoroute.encoders import build_model_embeddings
from duoroute.rdf_cascade_dfl import _one_hot
from duoroute.rdf_expert_router import pairwise_ranking_loss
from duoroute.rdf_vg_cascade import wrong_only_delta_u_torch, wrong_only_selector_regret_loss
from duoroute.utils import project_root


class QueryModelRecourseSelector(nn.Module):
    """Per (query, model) scorer: MLP([q, e_m, q⊙e_m, c_m, m_1])."""

    def __init__(
        self,
        query_dim: int,
        n_models: int,
        model_emb: torch.Tensor,
        cost: torch.Tensor,
        *,
        hidden_dim: int = 128,
        query_proj_dim: int = 128,
        use_cost: bool = True,
        use_m1: bool = True,
    ):
        super().__init__()
        self.n_models = n_models
        self.use_cost = use_cost
        self.use_m1 = use_m1
        self.query_proj = nn.Linear(query_dim, query_proj_dim)
        mdim = int(model_emb.shape[1])
        self.register_buffer("model_emb", model_emb.float().clone())
        self.model_proj = nn.Linear(mdim, query_proj_dim)
        self.register_buffer("cost", cost.float().view(1, n_models, 1))
        feat_dim = query_proj_dim * 3 + (1 if use_cost else 0) + (n_models if use_m1 else 0)
        self.pair_mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _pair_features(
        self,
        h: torch.Tensor,
        m1: torch.Tensor,
    ) -> torch.Tensor:
        batch = h.size(0)
        q = self.query_proj(h)
        e = self.model_proj(self.model_emb)
        q_exp = q.unsqueeze(1).expand(batch, self.n_models, q.size(-1))
        e_exp = e.unsqueeze(0).expand(batch, self.n_models, e.size(-1))
        inter = q_exp * e_exp
        parts = [q_exp, e_exp, inter]
        if self.use_cost:
            parts.append(self.cost.expand(batch, self.n_models, 1))
        if self.use_m1:
            m1_oh = _one_hot(m1, self.n_models)
            parts.append(m1_oh.unsqueeze(1).expand(batch, self.n_models, self.n_models))
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        h: torch.Tensor,
        mask: torch.Tensor,
        m1: torch.Tensor,
        *,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self._pair_features(h, m1)
        logits = self.pair_mlp(feat).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e9)
        probs = F.softmax(logits / max(float(temperature), 1e-4), dim=-1)
        return probs, logits


class QueryModelTop5Reranker(nn.Module):
    """Stage2 reranker within selector top-k C_q.

    Per-candidate features:
      [h_q, e_m, h_q⊙e_m, c_m?, s_{q,m}, rank_m, m_1_onehot,
       e_{m_1}?, e_m-e_{m_1}?, e_m⊙e_{m_1}?]
    """

    def __init__(
        self,
        query_dim: int,
        n_models: int,
        model_emb: torch.Tensor,
        cost: torch.Tensor,
        *,
        top_k: int = 5,
        hidden_dim: int = 128,
        query_proj_dim: int = 128,
        use_cost: bool = True,
        use_failed_emb: bool = False,
    ):
        super().__init__()
        self.n_models = n_models
        self.top_k = top_k
        self.use_cost = use_cost
        self.use_failed_emb = use_failed_emb
        self.query_proj = nn.Linear(query_dim, query_proj_dim)
        mdim = int(model_emb.shape[1])
        self.register_buffer("model_emb", model_emb.float().clone())
        self.model_proj = nn.Linear(mdim, query_proj_dim)
        self.register_buffer("cost", cost.float().view(n_models))
        feat_dim = query_proj_dim * 3 + 2 + n_models
        if use_cost:
            feat_dim += 1
        if use_failed_emb:
            feat_dim += query_proj_dim * 3
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _model_emb_proj(self) -> torch.Tensor:
        return self.model_proj(self.model_emb)

    def topk_candidates(
        self,
        mask: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        neg = base_logits.masked_fill(~mask, -1e9)
        k = min(self.top_k, mask.shape[1])
        topi = neg.topk(k, dim=-1).indices
        ranks = torch.arange(k, device=base_logits.device).unsqueeze(0).expand(topi.size(0), k)
        return topi, ranks

    def score_candidates(
        self,
        h: torch.Tensor,
        m1: torch.Tensor,
        base_probs: torch.Tensor,
        base_logits: torch.Tensor,
        cand_idx: torch.Tensor,
        cand_rank: torch.Tensor,
    ) -> torch.Tensor:
        batch, k = cand_idx.shape
        q = self.query_proj(h)
        e = self._model_emb_proj()
        q_exp = q.unsqueeze(1).expand(batch, k, q.size(-1))
        e_sel = e[cand_idx]
        inter = q_exp * e_sel
        s_qm = base_logits.gather(1, cand_idx).unsqueeze(-1)
        rank_m = cand_rank.float().unsqueeze(-1) / max(self.top_k - 1, 1)
        parts: list[torch.Tensor] = [q_exp, e_sel, inter]
        if self.use_cost:
            parts.append(self.cost[cand_idx].unsqueeze(-1))
        m1_oh = _one_hot(m1, self.n_models)
        parts.extend([s_qm, rank_m, m1_oh.unsqueeze(1).expand(batch, k, self.n_models)])
        if self.use_failed_emb:
            e_m1 = e[m1].unsqueeze(1).expand(batch, k, e.size(-1))
            parts.extend([e_m1, e_sel - e_m1, e_sel * e_m1])
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)

    def forward(
        self,
        h: torch.Tensor,
        m1: torch.Tensor,
        mask: torch.Tensor,
        base_probs: torch.Tensor,
        base_logits: torch.Tensor,
        *,
        lambda_cost: float = 0.0,
        cost_posthoc: bool = False,
        conservative: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        topi, ranks = self.topk_candidates(mask, base_logits)
        p_hat = self.score_candidates(h, m1, base_probs, base_logits, topi, ranks)
        pick_scores = p_hat
        if cost_posthoc and lambda_cost > 0:
            pick_scores = p_hat - lambda_cost * self.cost[topi]
        if conservative:
            sel_top = pick_scores[:, 0]
            rerank_top = pick_scores.max(dim=-1).values
            rerank_idx = pick_scores.argmax(dim=-1)
            pick = torch.where(
                rerank_top > sel_top,
                rerank_idx,
                torch.zeros_like(rerank_idx),
            )
            chosen = topi.gather(1, pick.unsqueeze(1)).squeeze(1)
            full = torch.full_like(base_logits, -1e9)
            full.scatter_(1, topi, p_hat)
            return chosen, full, topi
        pick = pick_scores.argmax(dim=-1)
        chosen = topi.gather(1, pick.unsqueeze(1)).squeeze(1)
        full = torch.full_like(base_logits, -1e9)
        full.scatter_(1, topi, p_hat)
        return chosen, full, topi


@dataclass
class QueryModelSelectorTrainConfig:
    hidden_dim: int = 128
    query_proj_dim: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    epochs: int = 28
    patience: int = 8
    seed: int = 42
    lambda_cost: float = 0.2
    temperature: float = 1.0
    ranking_beta: float = 0.05


@dataclass
class QueryModelRerankTrainConfig:
    hidden_dim: int = 128
    query_proj_dim: int = 128
    lr: float = 5e-4
    weight_decay: float = 1e-5
    batch_size: int = 256
    epochs: int = 28
    patience: int = 10
    seed: int = 42
    lambda_cost: float = 0.2
    top_k: int = 5
    loss_mode: str = "listwise"  # listwise | pairwise
    listwise_temperature: float = 0.1
    regret_weight: float = 0.0
    use_cost: bool = True
    use_failed_emb: bool = False
    train_on_perf: bool = False


def load_model_embedding_table(data: dict, *, seed: int = 42) -> torch.Tensor:
    """Load frozen model-card embeddings aligned with pool model order."""
    from duoroute.model_cards import cards_for_models, load_model_cards

    data_dir = project_root() / "data/seed42_flagship"
    embed_path = data_dir / "model_embeddings.pth"
    if not embed_path.exists():
        embed_path = data_dir / "model_embeddings.f16.npz"
    cards = load_model_cards(cards_path=str(data_dir / "model_cards.json"))
    model_cards = cards_for_models(list(data["model_names"]), cards)
    return build_model_embeddings(model_cards, embed_path=str(embed_path), seed=seed)


def build_query_model_selector(
    data: dict,
    *,
    seed: int = 42,
    use_cost: bool = True,
    use_m1: bool = True,
) -> QueryModelRecourseSelector:
    d = data["train_h"].shape[1]
    k = len(data["model_names"])
    cost = torch.tensor(data["cost"], dtype=torch.float32)
    model_emb = load_model_embedding_table(data, seed=seed)
    if model_emb.shape[0] != k:
        model_emb = model_emb[:k]
    return QueryModelRecourseSelector(
        d, k, model_emb, cost, use_cost=use_cost, use_m1=use_m1,
    )


def build_query_model_reranker(
    data: dict,
    *,
    seed: int = 42,
    top_k: int = 5,
    use_cost: bool = True,
    use_failed_emb: bool = False,
) -> QueryModelTop5Reranker:
    d = data["train_h"].shape[1]
    k = len(data["model_names"])
    cost = torch.tensor(data["cost"], dtype=torch.float32)
    model_emb = load_model_embedding_table(data, seed=seed)
    if model_emb.shape[0] != k:
        model_emb = model_emb[:k]
    return QueryModelTop5Reranker(
        d, k, model_emb, cost,
        top_k=top_k, use_cost=use_cost, use_failed_emb=use_failed_emb,
    )


def _rerank_topk_scores(
    reranker: QueryModelTop5Reranker,
    h: torch.Tensor,
    m1: torch.Tensor,
    mask: torch.Tensor,
    base_probs: torch.Tensor,
    base_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    topi, ranks = reranker.topk_candidates(mask, base_logits)
    scores = reranker.score_candidates(h, m1, base_probs, base_logits, topi, ranks)
    return scores, topi


def listwise_topk_loss(
    scores: torch.Tensor,
    oracle_reward: torch.Tensor,
    topi: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    """L_list = -Σ y_m log p_m, y_m=softmax(ΔU/T), p_m=softmax(r_m) within top-k."""
    target = oracle_reward.gather(1, topi)
    y_m = F.softmax(target / max(float(temperature), 1e-4), dim=-1)
    p_m = F.softmax(scores, dim=-1)
    return -(y_m * p_m.clamp(min=1e-8).log()).sum(dim=-1).mean()


def topk_oracle_ce_loss(
    scores: torch.Tensor,
    oracle_reward: torch.Tensor,
    topi: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy on argmax ΔU position within C_q."""
    star = oracle_reward.argmax(dim=-1)
    target_pos = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
    for b in range(scores.size(0)):
        s = int(star[b].item())
        cand = topi[b].tolist()
        if s in cand:
            target_pos[b] = cand.index(s)
        else:
            target_pos[b] = int(oracle_reward[b, topi[b]].argmax().item())
    return F.cross_entropy(scores, target_pos)


def topk_regret_loss(
    scores: torch.Tensor,
    oracle_reward: torch.Tensor,
    topi: torch.Tensor,
) -> torch.Tensor:
    """Regret within C_q: max_{m∈C_q} ΔU_m - softmax(r)^T ΔU_m."""
    delta_cand = oracle_reward.gather(1, topi)
    probs = F.softmax(scores, dim=-1)
    oracle_best = delta_cand.max(dim=-1).values
    expected = (probs * delta_cand).sum(dim=-1)
    return (oracle_best - expected).mean()


def pairwise_topk_loss(
    scores: torch.Tensor,
    oracle_reward: torch.Tensor,
    topi: torch.Tensor,
) -> torch.Tensor:
    """Pairwise margin on ΔU gaps within selector top-k."""
    star = oracle_reward.argmax(dim=-1)
    pair = torch.zeros((), device=scores.device)
    n = 0
    for b in range(scores.size(0)):
        s = int(star[b].item())
        cand = topi[b].tolist()
        if s not in cand:
            continue
        pos = cand.index(s)
        for j in range(len(cand)):
            if j == pos:
                continue
            gap = (oracle_reward[b, s] - oracle_reward[b, cand[j]]).clamp(min=0.0)
            if gap.item() <= 0:
                continue
            pair = pair + gap * F.softplus(-(scores[b, pos] - scores[b, j]))
            n += 1
    if n > 0:
        pair = pair / n
    return pair


def _rerank_topk_loss(
    reranker: QueryModelTop5Reranker,
    h: torch.Tensor,
    m1: torch.Tensor,
    mask: torch.Tensor,
    base_probs: torch.Tensor,
    base_logits: torch.Tensor,
    oracle_reward: torch.Tensor,
    *,
    loss_mode: str = "listwise",
    listwise_temperature: float = 0.1,
    regret_weight: float = 1.0,
) -> torch.Tensor:
    scores, topi = _rerank_topk_scores(reranker, h, m1, mask, base_probs, base_logits)
    target = oracle_reward.gather(1, topi)
    pointwise = F.mse_loss(scores, target)
    if loss_mode == "pairwise":
        rank = pairwise_topk_loss(scores, oracle_reward, topi)
    else:
        rank = listwise_topk_loss(scores, oracle_reward, topi, temperature=listwise_temperature)
        rank = rank + 0.5 * pairwise_topk_loss(scores, oracle_reward, topi)
    regret = topk_regret_loss(scores, oracle_reward, topi) if regret_weight > 0 else 0.0
    return pointwise + rank + regret_weight * regret


def train_query_model_selector_wrong_only(
    selector: QueryModelRecourseSelector,
    train_h: torch.Tensor,
    train_perf: torch.Tensor,
    train_cost: torch.Tensor,
    train_mask: torch.Tensor,
    val_h: torch.Tensor,
    val_perf: torch.Tensor,
    val_cost: torch.Tensor,
    val_mask: torch.Tensor,
    cfg: QueryModelSelectorTrainConfig,
    *,
    train_m1: torch.Tensor,
    val_m1: torch.Tensor | None = None,
) -> QueryModelRecourseSelector:
    """Wrong-only regret (+ optional pairwise ranking) on ΔU_m = perf_m - λ*cost_m."""
    import random

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selector = selector.to(device)
    train_perf_dev = train_perf.to(device)
    train_m1_dev = train_m1.to(device)
    with torch.no_grad():
        idx = torch.arange(train_perf_dev.size(0), device=device)
        wrong_idx = torch.where(train_perf_dev[idx, train_m1_dev] < 0.5)[0].cpu().tolist()
    if not wrong_idx:
        return selector

    opt = torch.optim.AdamW(selector.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val = float("inf")
    best_state = None
    stale = 0
    rng = random.Random(cfg.seed)

    def _batch_loss(h, perf, cost, m, m1_fixed):
        idx_b = torch.arange(h.size(0), device=h.device)
        wrong_b = perf[idx_b, m1_fixed] < 0.5
        if not wrong_b.any():
            return torch.zeros((), device=h.device)
        delta_u = wrong_only_delta_u_torch(perf, cost, m, lambda_cost=cfg.lambda_cost)
        _, sel_logits = selector(h, m, m1_fixed, temperature=cfg.temperature)
        sel_probs = F.softmax(sel_logits / max(float(cfg.temperature), 1e-4), dim=-1)
        regret = wrong_only_selector_regret_loss(sel_probs[wrong_b], delta_u[wrong_b], m[wrong_b])
        rank = pairwise_ranking_loss(
            sel_logits[wrong_b], delta_u[wrong_b], m[wrong_b], beta=cfg.ranking_beta,
        )
        return regret + rank

    val_wrong_idx: list[int] = []
    val_perf_dev = val_perf.to(device) if val_m1 is not None else None
    if val_m1 is not None:
        vm1 = val_m1.to(device)
        with torch.no_grad():
            vidx = torch.arange(val_perf_dev.size(0), device=device)
            val_wrong_idx = torch.where(val_perf_dev[vidx, vm1] < 0.5)[0].cpu().tolist()

    for _ in range(cfg.epochs):
        selector.train()
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
            m1f = train_m1[idx].to(device)
            loss = _batch_loss(h, perf, cost, m, m1f)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(), max_norm=1.0)
            opt.step()

        if not val_wrong_idx:
            continue
        selector.eval()
        vsum, vcnt = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(val_wrong_idx), 512):
                idx = val_wrong_idx[start : start + 512]
                h = val_h[idx].to(device)
                perf = val_perf_dev[idx]
                cost = val_cost[idx].to(device)
                m = val_mask[idx].to(device).bool()
                m1f = val_m1[idx].to(device)
                vloss = _batch_loss(h, perf, cost, m, m1f)
                vsum += float(vloss.item()) * h.size(0)
                vcnt += h.size(0)
        vloss = vsum / max(vcnt, 1)
        if vloss < best_val - 1e-5:
            best_val = vloss
            best_state = {k: v.cpu().clone() for k, v in selector.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    if best_state is not None:
        selector.load_state_dict(best_state)
    return selector


def train_query_model_reranker_wrong_only(
    reranker: QueryModelTop5Reranker,
    selector: QueryModelRecourseSelector,
    train_h: torch.Tensor,
    train_perf: torch.Tensor,
    train_cost: torch.Tensor,
    train_mask: torch.Tensor,
    val_h: torch.Tensor,
    val_perf: torch.Tensor,
    val_cost: torch.Tensor,
    val_mask: torch.Tensor,
    cfg: QueryModelRerankTrainConfig,
    *,
    train_m1: torch.Tensor,
    val_m1: torch.Tensor | None = None,
) -> QueryModelTop5Reranker:
    """Train top-k reranker on wrong-only subset with pairwise margin loss."""
    import random

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reranker = reranker.to(device)
    selector = selector.to(device)
    selector.eval()
    for p in selector.parameters():
        p.requires_grad = False

    train_perf_dev = train_perf.to(device)
    train_m1_dev = train_m1.to(device)
    with torch.no_grad():
        idx = torch.arange(train_perf_dev.size(0), device=device)
        wrong_idx = torch.where(train_perf_dev[idx, train_m1_dev] < 0.5)[0].cpu().tolist()
    if not wrong_idx:
        return reranker

    opt = torch.optim.AdamW(reranker.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val = float("inf")
    best_state = None
    last_state = None
    stale = 0
    rng = random.Random(cfg.seed)

    def _batch_loss(h, perf, cost, m, m1_fixed):
        idx_b = torch.arange(h.size(0), device=h.device)
        wrong_b = perf[idx_b, m1_fixed] < 0.5
        if not wrong_b.any():
            return torch.zeros((), device=h.device)
        delta_u = wrong_only_delta_u_torch(perf, cost, m, lambda_cost=cfg.lambda_cost)
        oracle = perf if cfg.train_on_perf else delta_u
        with torch.no_grad():
            base_probs, base_logits = selector(h, m, m1_fixed)
        return _rerank_topk_loss(
            reranker, h[wrong_b], m1_fixed[wrong_b], m[wrong_b],
            base_probs[wrong_b], base_logits[wrong_b], oracle[wrong_b],
            loss_mode=cfg.loss_mode,
            listwise_temperature=cfg.listwise_temperature,
            regret_weight=cfg.regret_weight,
        )

    val_wrong_idx: list[int] = []
    val_perf_dev = val_perf.to(device) if val_m1 is not None else None
    if val_m1 is not None:
        vm1 = val_m1.to(device)
        with torch.no_grad():
            vidx = torch.arange(val_perf_dev.size(0), device=device)
            val_wrong_idx = torch.where(val_perf_dev[vidx, vm1] < 0.5)[0].cpu().tolist()

    for _ in range(cfg.epochs):
        reranker.train()
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
            m1f = train_m1[idx].to(device)
            loss = _batch_loss(h, perf, cost, m, m1f)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reranker.parameters(), max_norm=1.0)
            opt.step()
        last_state = {k: v.cpu().clone() for k, v in reranker.state_dict().items()}

        if not val_wrong_idx:
            continue
        reranker.eval()
        vsum, vcnt = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(val_wrong_idx), 512):
                idx = val_wrong_idx[start : start + 512]
                h = val_h[idx].to(device)
                perf = val_perf_dev[idx]
                cost = val_cost[idx].to(device)
                m = val_mask[idx].to(device).bool()
                m1f = val_m1[idx].to(device)
                vloss = _batch_loss(h, perf, cost, m, m1f)
                vsum += float(vloss.item()) * h.size(0)
                vcnt += h.size(0)
        vloss = vsum / max(vcnt, 1)
        if vloss < best_val - 1e-5:
            best_val = vloss
            best_state = last_state
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    if best_state is not None:
        reranker.load_state_dict(best_state)
    elif last_state is not None:
        reranker.load_state_dict(last_state)
    return reranker


@torch.no_grad()
def infer_query_model_selector(
    selector: QueryModelRecourseSelector,
    h: torch.Tensor,
    mask: torch.Tensor,
    m1: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (m2, sel_probs, sel_logits)."""
    device = next(selector.parameters()).device
    selector.eval()
    m2_parts: list[np.ndarray] = []
    prob_parts: list[np.ndarray] = []
    logit_parts: list[np.ndarray] = []
    for start in range(0, h.shape[0], 512):
        hb = h[start : start + 512].to(device)
        mb = mask[start : start + 512].to(device).bool()
        m1b = m1[start : start + 512].to(device)
        sel_probs, sel_logits = selector(hb, mb, m1b, temperature=temperature)
        m2_parts.append(sel_probs.argmax(dim=-1).cpu().numpy())
        prob_parts.append(sel_probs.cpu().numpy())
        logit_parts.append(sel_logits.cpu().numpy())
    return (
        np.concatenate(m2_parts).astype(np.int64),
        np.concatenate(prob_parts),
        np.concatenate(logit_parts),
    )


@torch.no_grad()
def infer_query_model_reranker(
    reranker: QueryModelTop5Reranker,
    selector: QueryModelRecourseSelector,
    h: torch.Tensor,
    mask: torch.Tensor,
    m1: torch.Tensor,
    *,
    lambda_cost: float = 0.2,
    cost_posthoc: bool = False,
    conservative: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (m2, p_hat scores full [N,M]; decision may apply cost-posthoc)."""
    device = next(reranker.parameters()).device
    reranker.eval()
    selector.eval()
    m2_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    for start in range(0, h.shape[0], 512):
        hb = h[start : start + 512].to(device)
        mb = mask[start : start + 512].to(device).bool()
        m1b = m1[start : start + 512].to(device)
        base_probs, base_logits = selector(hb, mb, m1b)
        chosen, full, _ = reranker(
            hb, m1b, mb, base_probs, base_logits,
            lambda_cost=lambda_cost, cost_posthoc=cost_posthoc, conservative=conservative,
        )
        m2_parts.append(chosen.cpu().numpy())
        score_parts.append(full.cpu().numpy())
    return (
        np.concatenate(m2_parts).astype(np.int64),
        np.concatenate(score_parts),
    )


def oracle_topk_rerank_np(
    base_logits: np.ndarray,
    delta_u: np.ndarray,
    mask: np.ndarray,
    *,
    k: int = 5,
) -> np.ndarray:
    """Diagnostic upper bound: argmax ΔU within selector top-k."""
    neg = -1e9
    masked = np.where(mask, base_logits, neg)
    order = np.argsort(-masked, axis=1)
    n, m = masked.shape
    m2 = np.zeros(n, dtype=np.int64)
    for i in range(n):
        cands = [j for j in order[i, :k] if mask[i, j]]
        if not cands:
            m2[i] = int(order[i, 0])
            continue
        best_j = max(cands, key=lambda j: delta_u[i, j])
        m2[i] = int(best_j)
    return m2


def selector_recall_metrics(
    scores: np.ndarray,
    oracle_m2: np.ndarray,
    mask: np.ndarray,
    subset: np.ndarray,
    *,
    ks: tuple[int, ...] = (1, 2, 3, 5),
    test_perf: np.ndarray | None = None,
) -> dict[str, float]:
    """Top-k recourse recall on subset: is oracle m2 in selector top-k?"""
    neg = -1e9
    masked = np.where(mask, scores, neg)
    order = np.argsort(-masked, axis=1)
    idx = np.where(subset)[0]
    n = len(idx)
    if n == 0:
        return {f"top{k}_recourse_recall": 0.0 for k in ks} | {"n_subset": 0.0}
    out: dict[str, float] = {"n_subset": float(n)}
    for k in ks:
        hits = 0
        for i in idx:
            if int(oracle_m2[i]) in set(order[i, :k].tolist()):
                hits += 1
        out[f"top{k}_recourse_recall"] = hits / n
    out["top1_recourse_hit"] = out["top1_recourse_recall"]
    if test_perf is not None:
        m2_pred = order[:, 0]
        rescue = float((test_perf[idx, m2_pred[idx]] >= 0.5).mean())
        out["selector_precision_on_wrong"] = rescue
    return out


__all__ = [
    "QueryModelRecourseSelector",
    "QueryModelSelectorTrainConfig",
    "QueryModelTop5Reranker",
    "QueryModelRerankTrainConfig",
    "build_query_model_selector",
    "build_query_model_reranker",
    "infer_query_model_selector",
    "infer_query_model_reranker",
    "listwise_topk_loss",
    "load_model_embedding_table",
    "oracle_topk_rerank_np",
    "pairwise_topk_loss",
    "topk_oracle_ce_loss",
    "topk_regret_loss",
    "selector_recall_metrics",
    "train_query_model_selector_wrong_only",
    "train_query_model_reranker_wrong_only",
]
