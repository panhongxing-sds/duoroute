"""RegretRouter: K-step recursive decision-focused router (final-step regret).

Default x0 is uniform over available models; AP-balance is an external baseline, not part of the method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from duoroute.rdf_expert_router import (
    ExpertAwareRecursiveDFLRouter,
    TopKReranker,
    batch_usage_entropy_reg,
    pairwise_ranking_loss,
    rare_expert_weights,
    rerank_topk_loss,
    topk_oracle_recall_loss,
    weighted_decision_focused_loss,
)
from duoroute.rdf_router import (
    FeedbackMode,
    ModelCardAttentionRouter,
    OptLayer,
    PredictorMode,
    RDFTrainConfig,
    RecursiveDFLRouter,
    RecursiveDFLRouterImplicit,
    RecursiveDFLRouterV2,
    SequentialDFLRouter,
    StateMode,
    decision_focused_loss,
    oracle_ce_loss,
    oracle_contrastive_loss,
    route_hard,
    stepwise_decision_focused_loss,
    train_router,
    utility_kl_loss,
)

InitMode = Literal["apinit", "uniform", "flat", "random"]
FeedbackMode = FeedbackMode
LossMode = Literal["regret", "utility", "ce"]
RouterVersion = Literal["v1", "v2", "expert", "expert_domain", "implicit", "model_card"]
RareWeightMode = Literal["none", "light", "mid", "strong"]

OURS_METHOD_NAME = "RegretRouter"
# Deployable cascade main method (Stage1 + PV gate + Q×M selector + top5 rerank).
CASCADE_METHOD_NAME = "RegretRouter + Cascade"


def ap_init_distribution(
    ap_idx: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 0.05,
) -> torch.Tensor:
    """x0 = (1-eps) * one_hot(m_AP) + eps / M (masked, renormalized)."""
    B, M = mask.shape
    device = mask.device
    x = torch.full((B, M), float(eps) / M, device=device, dtype=torch.float32)
    valid = mask.gather(1, ap_idx.unsqueeze(1)).squeeze(1)
    x[torch.arange(B, device=device), ap_idx] += (1.0 - float(eps)) * valid.float()
    x = x * mask.float()
    x = x / x.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return x


def uniform_init_distribution(mask: torch.Tensor) -> torch.Tensor:
    counts = mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
    return mask.float() / counts


def flat_init_distribution(mask: torch.Tensor, *, eps: float = 0.05) -> torch.Tensor:
    """x0 = eps/M only (no AP one-hot peak)."""
    counts = mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
    x = mask.float() * (float(eps) / counts)
    return x / x.sum(dim=-1, keepdim=True).clamp(min=1e-8)


def random_init_distribution(
    mask: torch.Tensor,
    *,
    sample_seeds: torch.Tensor | None = None,
) -> torch.Tensor:
    """Uniform random point on masked simplex; optional per-row seeds for deterministic eval."""
    batch, _ = mask.shape
    device = mask.device
    if sample_seeds is not None:
        out = torch.empty(batch, mask.shape[1], device=device, dtype=torch.float32)
        for row in range(batch):
            gen = torch.Generator(device=device)
            gen.manual_seed(int(sample_seeds[row].item()))
            probs = torch.rand(mask.shape[1], device=device, generator=gen, dtype=torch.float32)
            probs = probs * mask[row].float()
            out[row] = probs / probs.sum().clamp(min=1e-8)
        return out
    probs = torch.rand(batch, mask.shape[1], device=device, dtype=torch.float32)
    probs = probs * mask.float()
    return probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)


# Aliases for benchmark method names
RDFLAPRouter = RecursiveDFLRouter
SDFLAPRouter = SequentialDFLRouter


@dataclass
class RDFLAPTrainConfig:
    K: int = 3
    hidden_dim: int = 128
    temperature: float = 1.0
    ap_init_eps: float = 0.05
    init_mode: InitMode = "uniform"
    feedback_mode: FeedbackMode = "full"
    state_mode: StateMode = "evolving"
    loss_mode: LossMode = "regret"
    router_version: RouterVersion = "v1"
    predictor_mode: PredictorMode = "concat"
    branch_dim: int = 128
    implicit_steps: int = 20
    implicit_alpha: float = 0.5
    step_loss: bool = False
    step_loss_gamma: float = 0.5
    temp_anneal: bool = False
    temp_start: float = 2.0
    temp_end: float = 0.5
    rich_feedback: bool = False
    query_init: bool = False
    feedback_scale: float = 0.25
    expert_dim: int = 64
    query_proj_dim: int = 128
    num_domains: int = 0
    use_cost_feat: bool = True
    use_interaction: bool = True
    rare_weight_mode: RareWeightMode = "none"
    rank_beta: float = 0.0
    usage_reg_alpha: float = 0.0
    recall_k: int = 0
    recall_gamma: float = 0.05
    rerank_k: int = 0
    rerank_epochs: int = 12
    kl_beta: float = 0.0
    kl_tau_star: float = 0.2
    ctr_beta: float = 0.0
    opt_layer: OptLayer = "softmax"
    entmax_alpha: float = 1.5
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    epochs: int = 30
    patience: int = 8
    seed: int = 42


def _is_recursive(model: nn.Module) -> bool:
    return isinstance(
        model,
        (
            RecursiveDFLRouter,
            RecursiveDFLRouterV2,
            RecursiveDFLRouterImplicit,
            ExpertAwareRecursiveDFLRouter,
            ModelCardAttentionRouter,
        ),
    )


def _model_forward(
    model: nn.Module,
    h: torch.Tensor,
    m: torch.Tensor,
    cfg: RDFLAPTrainConfig,
    *,
    inits: dict[str, torch.Tensor | None],
    return_trajectory: bool = False,
):
    steps = cfg.K if _is_recursive(model) else 1
    if isinstance(model, RecursiveDFLRouterV2):
        return model(
            h,
            m,
            n_steps=steps,
            temperature=cfg.temperature,
            temp_anneal=cfg.temp_anneal,
            temp_start=cfg.temp_start,
            temp_end=cfg.temp_end,
            return_trajectory=return_trajectory,
        )
    if isinstance(model, (RecursiveDFLRouter, ModelCardAttentionRouter)):
        return model(h, m, n_steps=steps, temperature=cfg.temperature, return_trajectory=return_trajectory, **inits)
    if isinstance(model, RecursiveDFLRouterImplicit):
        return model(
            h,
            m,
            n_steps=cfg.implicit_steps if cfg.router_version == "implicit" else steps,
            temperature=cfg.temperature,
            return_trajectory=return_trajectory,
            **inits,
        )
    if isinstance(model, ExpertAwareRecursiveDFLRouter):
        return model(
            h,
            m,
            n_steps=steps,
            temperature=cfg.temperature,
            return_trajectory=return_trajectory,
            **inits,
        )
    return model(h, m, n_steps=1, temperature=cfg.temperature)


def _routing_loss(
    out,
    u: torch.Tensor,
    m: torch.Tensor,
    cfg: RDFLAPTrainConfig,
    *,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if cfg.loss_mode == "ce":
        logits = out[1]
        base = oracle_ce_loss(logits, u, m)
    elif cfg.step_loss and len(out) > 2:
        if sample_weights is not None and cfg.rare_weight_mode != "none":
            traj = out[2]
            losses = []
            gamma = cfg.step_loss_gamma
            for t, probs_t in enumerate(traj):
                w_t = gamma ** (len(traj) - 1 - t)
                losses.append(
                    w_t
                    * weighted_decision_focused_loss(
                        probs_t, u, m, sample_weights, mode=cfg.loss_mode
                    )
                )
            base = sum(losses)
        else:
            base = stepwise_decision_focused_loss(
                out[2], u, m, mode=cfg.loss_mode, gamma=cfg.step_loss_gamma
            )
    elif sample_weights is not None and cfg.rare_weight_mode != "none":
        base = weighted_decision_focused_loss(out[0], u, m, sample_weights, mode=cfg.loss_mode)
    else:
        base = decision_focused_loss(out[0], u, m, mode=cfg.loss_mode)
    if cfg.rank_beta > 0 and cfg.loss_mode != "ce":
        base = base + pairwise_ranking_loss(out[1], u, m, beta=cfg.rank_beta)
    if cfg.recall_k > 0 and cfg.loss_mode != "ce":
        base = base + topk_oracle_recall_loss(
            out[1], u, m, k=cfg.recall_k, gamma=cfg.recall_gamma
        )
    if cfg.usage_reg_alpha > 0 and cfg.loss_mode != "ce":
        base = base - cfg.usage_reg_alpha * batch_usage_entropy_reg(out[0], m)
    if cfg.kl_beta > 0 and cfg.loss_mode != "ce":
        base = base + utility_kl_loss(
            out[0], u, m, tau_star=cfg.kl_tau_star, beta=cfg.kl_beta
        )
    if cfg.ctr_beta > 0 and cfg.loss_mode != "ce":
        base = base + oracle_contrastive_loss(out[1], u, m, beta=cfg.ctr_beta)
    return base


def _forward_init_kwargs(
    ap: torch.Tensor,
    mask: torch.Tensor,
    cfg: RDFLAPTrainConfig,
    *,
    sample_seeds: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | None]:
    if cfg.init_mode == "apinit":
        x0 = ap_init_distribution(ap, mask, eps=cfg.ap_init_eps)
        return {"ap_init": x0, "x_init": None}
    if cfg.init_mode == "flat":
        return {"ap_init": None, "x_init": flat_init_distribution(mask, eps=cfg.ap_init_eps)}
    if cfg.init_mode == "random":
        return {"ap_init": None, "x_init": random_init_distribution(mask, sample_seeds=sample_seeds)}
    if cfg.init_mode == "uniform":
        return {"ap_init": None, "x_init": None}
    raise ValueError(f"unknown init_mode: {cfg.init_mode}")


def train_regretrouter(
    model: nn.Module,
    train_h: torch.Tensor,
    train_u: torch.Tensor,
    train_mask: torch.Tensor,
    train_ap: torch.Tensor,
    cost: torch.Tensor,
    val_h: torch.Tensor,
    val_u: torch.Tensor,
    val_mask: torch.Tensor,
    val_ap: torch.Tensor,
    cfg: RDFLAPTrainConfig,
    *,
    train_perf: torch.Tensor | None = None,
) -> nn.Module:
    del cost  # utilities already cost-aware in oracle_reward
    use_rare = cfg.rare_weight_mode != "none"
    if use_rare and train_perf is None:
        raise ValueError("train_perf required when rare_weight_mode != 'none'")
    import random

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n = train_h.shape[0]
    best_val = float("inf")
    best_state = None
    stale = 0
    rng = random.Random(cfg.seed)
    want_traj = cfg.step_loss and cfg.loss_mode != "ce"

    for _ in range(cfg.epochs):
        model.train()
        perm = list(range(n))
        rng.shuffle(perm)
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            if not idx:
                continue
            h = train_h[idx].to(device)
            u = train_u[idx].to(device)
            m = train_mask[idx].to(device).bool()
            ap = train_ap[idx].to(device)
            inits = _forward_init_kwargs(ap, m, cfg)
            out = _model_forward(model, h, m, cfg, inits=inits, return_trajectory=want_traj)
            sw = None
            if use_rare:
                sw = rare_expert_weights(train_perf[idx].to(device), m, mode=cfg.rare_weight_mode)
            loss = _routing_loss(out, u, m, cfg, sample_weights=sw)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        vsum, vcnt = 0.0, 0
        with torch.no_grad():
            for start in range(0, val_h.shape[0], 512):
                h = val_h[start : start + 512].to(device)
                u = val_u[start : start + 512].to(device)
                m = val_mask[start : start + 512].to(device).bool()
                ap = val_ap[start : start + 512].to(device)
                inits = _forward_init_kwargs(ap, m, cfg)
                out = _model_forward(model, h, m, cfg, inits=inits, return_trajectory=want_traj)
                if want_traj:
                    vloss = decision_focused_loss(out[0], u, m, mode=cfg.loss_mode)
                else:
                    vloss = _routing_loss(out, u, m, cfg)
                vsum += float(vloss.item()) * h.size(0)
                vcnt += h.size(0)
        vloss = vsum / max(vcnt, 1)
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


def train_reranker(
    base_model: nn.Module,
    reranker: TopKReranker,
    train_h: torch.Tensor,
    train_u: torch.Tensor,
    train_mask: torch.Tensor,
    train_ap: torch.Tensor,
    val_h: torch.Tensor,
    val_u: torch.Tensor,
    val_mask: torch.Tensor,
    val_ap: torch.Tensor,
    cfg: RDFLAPTrainConfig,
) -> TopKReranker:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = base_model.to(device).eval()
    for p in base_model.parameters():
        p.requires_grad = False
    reranker = reranker.to(device)
    opt = torch.optim.AdamW(reranker.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n = train_h.shape[0]
    best_val = float("inf")
    best_state = None
    stale = 0
    import random

    rng = random.Random(cfg.seed + 17)
    for _ in range(cfg.rerank_epochs):
        reranker.train()
        perm = list(range(n))
        rng.shuffle(perm)
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            if not idx:
                continue
            h = train_h[idx].to(device)
            u = train_u[idx].to(device)
            m = train_mask[idx].to(device).bool()
            ap = train_ap[idx].to(device)
            with torch.no_grad():
                inits = _forward_init_kwargs(ap, m, cfg)
                probs, logits = _model_forward(base_model, h, m, cfg, inits=inits)
            loss = rerank_topk_loss(reranker, h, m, probs, logits, u)
            opt.zero_grad()
            loss.backward()
            opt.step()

        reranker.eval()
        vsum, vcnt = 0.0, 0
        with torch.no_grad():
            for start in range(0, val_h.shape[0], 512):
                h = val_h[start : start + 512].to(device)
                u = val_u[start : start + 512].to(device)
                m = val_mask[start : start + 512].to(device).bool()
                ap = val_ap[start : start + 512].to(device)
                inits = _forward_init_kwargs(ap, m, cfg)
                probs, logits = _model_forward(base_model, h, m, cfg, inits=inits)
                vloss = rerank_topk_loss(reranker, h, m, probs, logits, u)
                vsum += float(vloss.item()) * h.size(0)
                vcnt += h.size(0)
        vloss = vsum / max(vcnt, 1)
        if vloss < best_val - 1e-5:
            best_val = vloss
            best_state = {k: v.cpu().clone() for k, v in reranker.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= max(4, cfg.patience // 2):
                break
    if best_state is not None:
        reranker.load_state_dict(best_state)
    return reranker


@torch.no_grad()
def infer_choices(
    model: nn.Module,
    h: torch.Tensor,
    mask: torch.Tensor,
    cost: torch.Tensor,
    ap_idx: torch.Tensor,
    *,
    cfg: RDFLAPTrainConfig,
    reranker: TopKReranker | None = None,
) -> np.ndarray:
    del cost
    device = next(model.parameters()).device
    model.eval()
    if reranker is not None:
        reranker.eval()
    parts: list[np.ndarray] = []
    for start in range(0, h.shape[0], 512):
        hb = h[start : start + 512].to(device)
        mb = mask[start : start + 512].to(device).bool()
        apb = ap_idx[start : start + 512].to(device)
        if _is_recursive(model):
            bs = hb.size(0)
            seeds = cfg.seed * 1_000_003 + torch.arange(start, start + bs, device=device, dtype=torch.long)
            inits = _forward_init_kwargs(apb, mb, cfg, sample_seeds=seeds)
            probs, logits = _model_forward(model, hb, mb, cfg, inits=inits)
            if reranker is not None:
                chosen, _ = reranker(hb, mb, probs, logits)
                parts.append(chosen.cpu().numpy())
            else:
                parts.append(probs.argmax(dim=-1).cpu().numpy())
        else:
            parts.append(route_hard(model, hb, mb, n_steps=1, temperature=cfg.temperature))
    return np.concatenate(parts).astype(np.int64)


@torch.no_grad()
def infer_probs(
    model: nn.Module,
    h: torch.Tensor,
    mask: torch.Tensor,
    ap_idx: torch.Tensor,
    *,
    cfg: RDFLAPTrainConfig,
) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    parts: list[np.ndarray] = []
    for start in range(0, h.shape[0], 512):
        hb = h[start : start + 512].to(device)
        mb = mask[start : start + 512].to(device).bool()
        apb = ap_idx[start : start + 512].to(device)
        if _is_recursive(model):
            bs = hb.size(0)
            seeds = cfg.seed * 1_000_003 + torch.arange(start, start + bs, device=device, dtype=torch.long)
            inits = _forward_init_kwargs(apb, mb, cfg, sample_seeds=seeds)
            probs, _ = _model_forward(model, hb, mb, cfg, inits=inits)
            parts.append(probs.cpu().numpy())
        else:
            logits = model(hb, mb, n_steps=1, temperature=cfg.temperature)[1]
            from duoroute.rdf_router import masked_softmax

            parts.append(masked_softmax(logits, mb, temperature=cfg.temperature).cpu().numpy())
    return np.concatenate(parts, axis=0).astype(np.float32)


def routing_potential(performance: np.ndarray, mask: np.ndarray, ap_chosen: np.ndarray) -> dict:
    n = len(ap_chosen)
    idx = np.arange(n)
    ap_ok = performance[idx, ap_chosen] >= 0.5
    oracle_ok = (performance[idx] * mask).max(axis=1) >= 0.5
    gap = oracle_ok.astype(float) - ap_ok.astype(float)
    return {
        "ap_acc": float(ap_ok.mean()),
        "oracle_acc": float(oracle_ok.mean()),
        "potential_pp": float(gap.mean() * 100),
        "n_rescueable": int(gap.sum()),
    }


def make_router(
    embed_dim: int,
    num_models: int,
    *,
    kind: Literal["sdf", "rdfl"],
    cfg: RDFLAPTrainConfig,
    cost: torch.Tensor | None = None,
    model_embeddings: torch.Tensor | None = None,
) -> nn.Module:
    if kind == "sdf":
        return SequentialDFLRouter(embed_dim, num_models, hidden_dim=cfg.hidden_dim)
    if cfg.router_version in ("expert", "expert_domain"):
        if cost is None:
            raise ValueError("cost tensor required for expert router")
        num_domains = cfg.num_domains if cfg.router_version == "expert_domain" else 0
        return ExpertAwareRecursiveDFLRouter(
            embed_dim,
            num_models,
            cost,
            expert_dim=cfg.expert_dim,
            query_proj_dim=cfg.query_proj_dim,
            num_domains=num_domains,
            hidden_dim=cfg.hidden_dim,
            use_interaction=cfg.use_interaction,
            use_cost=cfg.use_cost_feat,
            feedback_mode=cfg.feedback_mode,
            state_mode=cfg.state_mode,
            default_steps=cfg.K,
        )
    if cfg.router_version == "implicit":
        return RecursiveDFLRouterImplicit(
            embed_dim,
            num_models,
            hidden_dim=cfg.hidden_dim,
            default_steps=cfg.implicit_steps,
            feedback_mode=cfg.feedback_mode,
            predictor_mode=cfg.predictor_mode,
            branch_dim=cfg.branch_dim,
            mix_alpha=cfg.implicit_alpha,
            opt_layer=cfg.opt_layer,
            entmax_alpha=cfg.entmax_alpha,
        )
    if cfg.router_version == "v2":
        return RecursiveDFLRouterV2(
            embed_dim,
            num_models,
            hidden_dim=cfg.hidden_dim,
            max_steps=max(cfg.K, 8),
            feedback_mode=cfg.feedback_mode,
            state_mode=cfg.state_mode,
        )
    if cfg.router_version == "model_card":
        if model_embeddings is None:
            raise ValueError("model_embeddings required for router_version=model_card")
        return ModelCardAttentionRouter(
            embed_dim,
            num_models,
            model_embeddings,
            hidden_dim=cfg.hidden_dim,
            proj_dim=cfg.query_proj_dim,
            default_steps=cfg.K,
            feedback_mode=cfg.feedback_mode,
            state_mode=cfg.state_mode,
            opt_layer=cfg.opt_layer,
            entmax_alpha=cfg.entmax_alpha,
        )
    return RecursiveDFLRouter(
        embed_dim,
        num_models,
        hidden_dim=cfg.hidden_dim,
        default_steps=cfg.K,
        feedback_mode=cfg.feedback_mode,
        state_mode=cfg.state_mode,
        rich_feedback=cfg.rich_feedback,
        query_init=cfg.query_init,
        feedback_scale=cfg.feedback_scale,
        predictor_mode=cfg.predictor_mode,
        branch_dim=cfg.branch_dim,
        opt_layer=cfg.opt_layer,
        entmax_alpha=cfg.entmax_alpha,
    )


RegretRouterTrainConfig = RDFLAPTrainConfig
train_r_dfl_ap = train_regretrouter
