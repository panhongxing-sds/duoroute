"""Response / residual VAE pipelines, model embeddings, router training, diagnostics."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset

from duoroute.behavioral_signature import response_cell_mask
from duoroute.data import DuoRouteGroupedData
from duoroute.encoders import (
    build_model_embeddings,
    build_response_embeddings,
    load_embedding_dim,
    load_or_build_query_embeddings,
)
from duoroute.evaluator import evaluate_predictions
from duoroute.inference import predict_channel_a
from duoroute.losses import duoroute_loss
from duoroute.model import DuoRouteModel
from duoroute.model_cards import ModelCard, cards_for_models, load_model_cards
from duoroute.prompt_ids import assign_prompt_ids, build_prompt_id_map
from duoroute.residual import DEFAULT_RESPONSE_NORM_EPS
from duoroute.response_vae import (
    QueryConditionedResponseVAE,
    QueryToResponseLinear,
    ResidualBehaviorVAE,
    normalize_residual,
    vae_loss,
)
from duoroute.trainer import GroupedQueryDataset, assign_prompt_ids_for_grouped
from duoroute.utils import set_seed

EmbedMethod = Literal[
    "vae_kmeans",
    "residual_ae_kmeans",
    "residual_vae_kmeans",
    "avg_response",
    "r2me",
]


@dataclass
class ProjectorTrainConfig:
    epochs: int = 20
    batch_size: int = 512
    lr: float = 1e-3
    seed: int = 42
    device: str = "cuda"


@dataclass
class VAETrainConfig:
    latent_dim: int = 32
    hidden_dim: int = 512
    epochs: int = 40
    batch_size: int = 256
    lr: float = 1e-3
    kl_weight: float = 0.0
    kl_warmup_epochs: int = 1
    seed: int = 42
    device: str = "cuda"
    use_mu_at_inference: bool = True


@dataclass
class ModelEmbedConfig:
    n_clusters: int = 5
    seed: int = 42
    metadata_scale: float = 0.25


@dataclass
class RouterTrainConfig:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    hidden_dim: int = 64
    alpha: float = 0.5
    temperature: float = 0.5
    reward_target: str = "performance"
    seed: int = 42
    device: str = "cuda"


def _metadata_vector(card: ModelCard) -> np.ndarray:
    return np.array(
        [
            float(np.log1p(max(card.context_length, 0))),
            float(card.input_price),
            float(card.output_price),
            float(len(card.capability or [])),
            float(len(card.specialty or [])),
        ],
        dtype=np.float32,
    )


class MetadataEncoder(torch.nn.Module):
    def __init__(self, meta_dim: int, out_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(meta_dim, out_dim),
            torch.nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def collect_response_cells(
    response_emb: torch.Tensor,
    query_emb: torch.Tensor,
    prompt_ids: np.ndarray,
    arm_mask: np.ndarray,
    *,
    norm_eps: float = DEFAULT_RESPONSE_NORM_EPS,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    present = response_cell_mask(response_emb, arm_mask, norm_eps=norm_eps)
    q_idx, m_idx = np.where(present.cpu().numpy())
    if len(q_idx) == 0:
        raise ValueError("No response embedding cells found")
    s = torch.stack([response_emb[i, j] for i, j in zip(q_idx, m_idx)], dim=0).float()
    x = query_emb[torch.from_numpy(prompt_ids[q_idx])]
    return s, x, q_idx, m_idx


def fit_query_to_response_projector(
    s: torch.Tensor,
    x: torch.Tensor,
    *,
    query_dim: int,
    response_dim: int,
    cfg: ProjectorTrainConfig,
    out_path: Path | None = None,
) -> QueryToResponseLinear:
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = QueryToResponseLinear(query_dim, response_dim).to(device)
    loader = DataLoader(TensorDataset(x, s), batch_size=cfg.batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history: list[dict] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for xb, sb in loader:
            xb, sb = xb.to(device), sb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = F.mse_loss(pred, sb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        history.append({"epoch": epoch, "mse": total / max(n, 1)})
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "history": history}, out_path)
    return model


@torch.no_grad()
def compute_normalized_residuals(
    projector: QueryToResponseLinear,
    s: torch.Tensor,
    x: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    projector.eval()
    pred = projector(x.to(device))
    return normalize_residual(s.to(device) - pred)


def train_legacy_response_vae(
    s: torch.Tensor,
    x: torch.Tensor,
    *,
    embed_dim: int,
    cfg: VAETrainConfig,
    out_path: Path | None = None,
) -> QueryConditionedResponseVAE:
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = QueryConditionedResponseVAE(
        embed_dim, latent_dim=cfg.latent_dim, hidden_dim=cfg.hidden_dim
    ).to(device)
    loader = DataLoader(TensorDataset(s, x), batch_size=cfg.batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history: list[dict] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        totals = {"total": 0.0, "recon": 0.0, "kl": 0.0}
        n = 0
        kl_w = cfg.kl_weight * min(1.0, epoch / max(cfg.kl_warmup_epochs, 1))
        for sb, xb in loader:
            sb, xb = sb.to(device), xb.to(device)
            opt.zero_grad()
            recon, mu, logvar, _ = model(sb, xb)
            loss, parts = vae_loss(sb, recon, mu, logvar, kl_weight=kl_w)
            loss.backward()
            opt.step()
            bs = sb.size(0)
            for k in totals:
                totals[k] += parts.get(k, parts["total"] if k == "total" else 0.0) * bs
            n += bs
        row = {k: v / max(n, 1) for k, v in totals.items()}
        row["epoch"] = epoch
        row["kl_weight"] = kl_w
        history.append(row)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "config": asdict(cfg), "history": history}, out_path)
    return model


def train_residual_behavior_model(
    r: torch.Tensor,
    x: torch.Tensor,
    *,
    residual_dim: int,
    query_dim: int,
    cfg: VAETrainConfig,
    out_path: Path | None = None,
) -> ResidualBehaviorVAE:
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = ResidualBehaviorVAE(
        residual_dim, query_dim, latent_dim=cfg.latent_dim, hidden_dim=cfg.hidden_dim
    ).to(device)
    loader = DataLoader(TensorDataset(r, x), batch_size=cfg.batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    is_ae = cfg.kl_weight <= 0.0
    history: list[dict] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        totals = {"total": 0.0, "recon": 0.0, "kl": 0.0}
        n = 0
        kl_w = 0.0 if is_ae else cfg.kl_weight * min(1.0, epoch / max(cfg.kl_warmup_epochs, 1))
        for rb, xb in loader:
            rb, xb = rb.to(device), xb.to(device)
            opt.zero_grad()
            recon, mu, logvar, _ = model(rb, xb, sample=not is_ae and kl_w > 0)
            loss, parts = vae_loss(rb, recon, mu, logvar, kl_weight=kl_w)
            loss.backward()
            opt.step()
            bs = rb.size(0)
            for k in totals:
                totals[k] += parts.get(k, parts["total"] if k == "total" else 0.0) * bs
            n += bs
        row = {k: v / max(n, 1) for k, v in totals.items()}
        row["epoch"] = epoch
        row["kl_weight"] = kl_w
        history.append(row)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "config": asdict(cfg),
                "history": history,
                "is_ae": is_ae,
            },
            out_path,
        )
    return model


def _kmeans_centers(vectors: np.ndarray, k: int, seed: int) -> np.ndarray:
    if vectors.shape[0] == 0:
        raise ValueError("empty vectors for KMeans")
    k = min(k, vectors.shape[0])
    if k <= 1:
        return vectors.mean(axis=0, keepdims=True)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    km.fit(vectors)
    return km.cluster_centers_


def build_metadata_bias_latent(
    cards: list[ModelCard],
    latent_dim: int,
    *,
    scale: float,
    device: torch.device,
) -> torch.Tensor:
    vecs = np.stack([_metadata_vector(c) for c in cards], axis=0)
    meta = MetadataEncoder(vecs.shape[1], latent_dim).to(device)
    with torch.no_grad():
        return (meta(torch.from_numpy(vecs).to(device)) * scale).cpu()


@torch.no_grad()
def encode_legacy_latents(
    vae: QueryConditionedResponseVAE,
    response_emb: torch.Tensor,
    query_emb: torch.Tensor,
    prompt_ids: np.ndarray,
    arm_mask: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vae.eval()
    s, x, q_idx, m_idx = collect_response_cells(response_emb, query_emb, prompt_ids, arm_mask)
    z_parts: list[np.ndarray] = []
    batch = 512
    for start in range(0, s.size(0), batch):
        sb = s[start : start + batch].to(device)
        xb = x[start : start + batch].to(device)
        z_parts.append(vae.encode_latent(sb, xb, use_mu=True).cpu().numpy())
    return np.concatenate(z_parts, axis=0), q_idx, m_idx


@torch.no_grad()
def encode_residual_latents(
    model: ResidualBehaviorVAE,
    projector: QueryToResponseLinear,
    response_emb: torch.Tensor,
    query_emb: torch.Tensor,
    prompt_ids: np.ndarray,
    arm_mask: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    s, x, q_idx, m_idx = collect_response_cells(response_emb, query_emb, prompt_ids, arm_mask)
    r = compute_normalized_residuals(projector, s, x, device)
    z_parts: list[np.ndarray] = []
    batch = 512
    for start in range(0, r.size(0), batch):
        rb = r[start : start + batch]
        xb = x[start : start + batch].to(device)
        z_parts.append(model.encode_latent(rb, xb, use_mu=True).cpu().numpy())
    return np.concatenate(z_parts, axis=0), q_idx, m_idx


def latent_diagnostics(
    z: np.ndarray,
    q_idx: np.ndarray,
    m_idx: np.ndarray,
    grouped: DuoRouteGroupedData,
) -> dict:
    """Model separation and same-query cross-model latent stats."""
    by_q: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for qi, mj, zi in zip(q_idx, m_idx, z):
        by_q[int(qi)].append((int(mj), zi))

    same_query_dists: list[float] = []
    perf_gaps: list[float] = []
    for qi, items in by_q.items():
        if len(items) < 2:
            continue
        vecs = np.stack([v for _, v in items], axis=0)
        models = [mj for mj, _ in items]
        perfs = [float(grouped.performance[qi, mj]) for mj in models]
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                same_query_dists.append(float(np.linalg.norm(vecs[a] - vecs[b])))
                perf_gaps.append(abs(perfs[a] - perfs[b]))

    # Global: mean distance between model centroids
    k_models = len(grouped.model_names)
    centroids = []
    for j in range(k_models):
        sel = m_idx == j
        if sel.any():
            centroids.append(z[sel].mean(axis=0))
    inter_model_centroid_dists: list[float] = []
    for a in range(len(centroids)):
        for b in range(a + 1, len(centroids)):
            inter_model_centroid_dists.append(float(np.linalg.norm(centroids[a] - centroids[b])))

    return {
        "n_shared_queries": sum(1 for items in by_q.values() if len(items) >= 2),
        "mean_same_query_cross_model_dist": float(np.mean(same_query_dists)) if same_query_dists else 0.0,
        "mean_pairwise_perf_gap": float(np.mean(perf_gaps)) if perf_gaps else 0.0,
        "corr_same_query_dist_perf_gap": float(np.corrcoef(same_query_dists, perf_gaps)[0, 1])
        if len(same_query_dists) > 2
        else 0.0,
        "mean_inter_model_centroid_dist": float(np.mean(inter_model_centroid_dists))
        if inter_model_centroid_dists
        else 0.0,
        # alias for backward compat with V1 logs
        "mean_pairwise_latent_dist": float(np.mean(same_query_dists)) if same_query_dists else 0.0,
        "corr_dist_perf_gap": float(np.corrcoef(same_query_dists, perf_gaps)[0, 1])
        if len(same_query_dists) > 2
        else 0.0,
    }


def build_latent_model_embeddings(
    z: np.ndarray,
    m_idx: np.ndarray,
    *,
    k_models: int,
    cards: list[ModelCard],
    me_cfg: ModelEmbedConfig,
    latent_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """e_j = mean(KMeans centers in latent) + metadata bias (latent space)."""
    meta_bias = build_metadata_bias_latent(cards, latent_dim, scale=me_cfg.metadata_scale, device=device)
    table = torch.zeros(k_models, latent_dim, dtype=torch.float32)
    for j in range(k_models):
        sel = m_idx == j
        if not sel.any():
            table[j] = meta_bias[j]
            continue
        centers = _kmeans_centers(z[sel], me_cfg.n_clusters, me_cfg.seed)
        table[j] = torch.from_numpy(centers.mean(axis=0).astype(np.float32)) + meta_bias[j]
    return table


def build_model_embeddings_for_method(
    method: EmbedMethod,
    *,
    grouped: DuoRouteGroupedData,
    response_emb: torch.Tensor,
    query_emb: torch.Tensor,
    prompt_ids: np.ndarray,
    cards: list[ModelCard],
    me_cfg: ModelEmbedConfig,
    device: torch.device,
    legacy_vae: QueryConditionedResponseVAE | None = None,
    residual_model: ResidualBehaviorVAE | None = None,
    projector: QueryToResponseLinear | None = None,
    latent_dim: int = 32,
    embed_dim: int = 2048,
) -> tuple[torch.Tensor, dict]:
    k_models = len(grouped.model_names)

    if method == "avg_response":
        from duoroute.behavioral_signature import compute_mean_behavioral_signature

        beh = compute_mean_behavioral_signature(response_emb, grouped.mask)
        meta = build_metadata_bias_latent(cards, embed_dim, scale=me_cfg.metadata_scale, device=device)
        return beh.float() + meta, {}

    if method == "vae_kmeans":
        if legacy_vae is None:
            raise ValueError("legacy_vae required")
        z, q_idx, m_idx = encode_legacy_latents(
            legacy_vae, response_emb, query_emb, prompt_ids, grouped.mask, device
        )
        diag = latent_diagnostics(z, q_idx, m_idx, grouped)
        meta_bias = build_metadata_bias_latent(cards, embed_dim, scale=me_cfg.metadata_scale, device=device)
        legacy_vae.eval()
        table = torch.zeros(k_models, embed_dim, dtype=torch.float32)
        for j in range(k_models):
            sel = m_idx == j
            if not sel.any():
                table[j] = meta_bias[j]
                continue
            centers = _kmeans_centers(z[sel], me_cfg.n_clusters, me_cfg.seed)
            pooled_z = torch.from_numpy(centers.mean(axis=0).astype(np.float32)).to(device)
            q_mean = query_emb[torch.from_numpy(prompt_ids[q_idx[sel]])].mean(dim=0).to(device)
            decoded = legacy_vae.decode(pooled_z.unsqueeze(0), q_mean.unsqueeze(0)).squeeze(0).cpu()
            table[j] = decoded + meta_bias[j]
        return table, diag

    if method in ("residual_ae_kmeans", "residual_vae_kmeans", "r2me"):
        method = "residual_ae_kmeans" if method == "r2me" else method
    if method in ("residual_ae_kmeans", "residual_vae_kmeans"):
        if residual_model is None or projector is None:
            raise ValueError("residual model and projector required")
        z, q_idx, m_idx = encode_residual_latents(
            residual_model, projector, response_emb, query_emb, prompt_ids, grouped.mask, device
        )
        diag = latent_diagnostics(z, q_idx, m_idx, grouped)
        table = build_latent_model_embeddings(
            z, m_idx, k_models=k_models, cards=cards, me_cfg=me_cfg, latent_dim=latent_dim, device=device
        )
        return table, diag

    raise ValueError(f"Unknown method: {method}")


def train_channel_a_router(
    *,
    query_emb: torch.Tensor,
    model_emb: torch.Tensor,
    train: DuoRouteGroupedData,
    val: DuoRouteGroupedData,
    test: DuoRouteGroupedData,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    test_ids: np.ndarray,
    train_resp: torch.Tensor,
    cfg: RouterTrainConfig,
    return_test_choices: bool = False,
) -> dict:
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    k = len(train.model_names)
    test_resp = torch.zeros(test.oracle_reward.shape[0], k, train_resp.shape[-1])

    model = DuoRouteModel(
        query_emb,
        model_emb,
        hidden_dim=cfg.hidden_dim,
        query_dim=int(query_emb.shape[1]),
        response_dim=int(train_resp.shape[-1]),
        model_dim=int(model_emb.shape[1]),
        use_id_fallback=False,
    ).to(device)

    loader = DataLoader(
        GroupedQueryDataset(train, train_ids, train_resp, reward_target=cfg.reward_target),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    best_regret = float("inf")
    best_state = None

    for _epoch in range(1, cfg.epochs + 1):
        model.train()
        for batch in loader:
            opt.zero_grad()
            prompt_ids = batch["prompt_id"].to(device)
            response_emb = batch["response_emb"].to(device)
            oracle = batch["oracle"].to(device)
            mask = batch["mask"].to(device)
            pred_a, pred_b = model(prompt_ids, response_emb)
            out = duoroute_loss(
                pred_a, pred_b, oracle, mask, alpha=cfg.alpha, beta=0.0, temperature=cfg.temperature
            )
            out.total.backward()
            opt.step()

        pred_val = predict_channel_a(model, torch.from_numpy(val_ids).to(device))
        val_oracle = val.performance if cfg.reward_target == "performance" else val.oracle_reward
        val_m = evaluate_predictions(
            val_oracle, pred_val, val.mask, performance=val.performance, cost=val.cost, random_seed=cfg.seed
        )
        if val_m.routing_regret < best_regret:
            best_regret = val_m.routing_regret
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    pred_test = predict_channel_a(model, torch.from_numpy(test_ids).to(device))
    test_oracle = test.performance if cfg.reward_target == "performance" else test.oracle_reward
    test_m = evaluate_predictions(
        test_oracle, pred_test, test.mask, performance=test.performance, cost=test.cost, random_seed=cfg.seed
    )
    out = test_m.to_dict()
    if return_test_choices:
        from duoroute.flip_metrics import route_choices

        out["test_choices"] = route_choices(pred_test, test.mask).tolist()
    return out


def load_pool_tensors(data_dir: Path, cards_path: str | None = None) -> dict:
    train = DuoRouteGroupedData.load(data_dir / "train")
    val = DuoRouteGroupedData.load(data_dir / "val")
    test = DuoRouteGroupedData.load(data_dir / "test")
    all_texts = train.prompt_texts + val.prompt_texts + test.prompt_texts
    text_to_pid = build_prompt_id_map(all_texts)
    train_ids = assign_prompt_ids_for_grouped(train, text_to_pid)
    val_ids = assign_prompt_ids_for_grouped(val, text_to_pid)
    test_ids = assign_prompt_ids_for_grouped(test, text_to_pid)

    dim = load_embedding_dim(data_dir, fallback=2048)
    embed_path = data_dir / "question_embeddings.pth"
    query_emb = load_or_build_query_embeddings(
        sorted(text_to_pid.keys()),
        embed_path=str(embed_path) if embed_path.exists() else None,
        dim=dim,
    )

    cards = load_model_cards(
        cards_path=cards_path or str(data_dir / "model_cards.json"),
        model_names=train.model_names,
    )
    card_list = cards_for_models(train.model_names, cards)
    card_emb = build_model_embeddings(
        card_list,
        embed_path=str(data_dir / "model_embeddings.pth"),
        dim=dim,
    )

    def _resp(split: str, grouped: DuoRouteGroupedData) -> torch.Tensor:
        p = data_dir / split / "response_embeddings.pth"
        return build_response_embeddings(
            grouped.response_texts,
            embed_path=str(p) if p.exists() else None,
            dim=dim,
        )

    return {
        "train": train,
        "val": val,
        "test": test,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "query_emb": query_emb,
        "card_emb": card_emb,
        "cards": card_list,
        "train_resp": _resp("train", train),
        "val_resp": _resp("val", val),
        "test_resp": _resp("test", test),
        "embed_dim": dim,
    }


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
