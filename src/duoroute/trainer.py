"""Grouped query dataset and DuoRoute trainer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from duoroute.data import DuoRouteGroupedData
from duoroute.encoders import (
    build_model_embeddings,
    build_response_embeddings,
    load_embedding_dim,
    load_or_build_query_embeddings,
    zero_response_embeddings,
)
from duoroute.evaluator import evaluate_predictions
from duoroute.inference import predict_channel_a, predict_channel_b
from duoroute.losses import duoroute_loss
from duoroute.model import DuoRouteModel
from duoroute.model_cards import ModelCard, cards_for_models, load_model_cards
from duoroute.utils import set_seed


from duoroute.prompt_ids import assign_prompt_ids, build_prompt_id_map, load_global_prompt_map


def assign_prompt_ids_for_grouped(grouped: DuoRouteGroupedData, text_to_pid: dict[str, int]) -> np.ndarray:
    return assign_prompt_ids(grouped.prompt_texts, text_to_pid)


class GroupedQueryDataset(Dataset):
    def __init__(
        self,
        data: DuoRouteGroupedData,
        prompt_ids: np.ndarray,
        response_emb: torch.Tensor,
        *,
        reward_target: str = "oracle_reward",
    ):
        if reward_target == "performance":
            oracle = data.performance
        elif reward_target in {"utility", "oracle_reward"}:
            oracle = data.oracle_reward
        else:
            raise ValueError(reward_target)
        self.oracle = torch.from_numpy(oracle.astype(np.float32))
        self.mask = torch.from_numpy(data.mask)
        self.prompt_ids = torch.from_numpy(prompt_ids.astype(np.int64))
        self.response_emb = response_emb.float()

    def __len__(self) -> int:
        return self.oracle.shape[0]

    def __getitem__(self, index: int):
        return {
            "prompt_id": self.prompt_ids[index],
            "response_emb": self.response_emb[index],
            "oracle": self.oracle[index],
            "mask": self.mask[index],
        }


@dataclass
class TrainConfig:
    data_dir: str
    output_dir: Optional[str] = None
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    hidden_dim: int = 64
    embed_dim: int = 256
    alpha: float = 0.5
    beta: float = 0.1
    temperature: float = 0.5
    reward_target: str = "oracle_reward"
    distill_warmup_epochs: int = 3
    device: str = "cuda"
    seed: int = 42
    max_samples: Optional[int] = None
    use_id_fallback: bool = False
    embed_path: Optional[str] = None
    model_embed_path: Optional[str] = None
    model_cards_path: Optional[str] = None
    llmbench_graphrouter_config: Optional[str] = None
    llmbench_collector_config: Optional[str] = None


def _maybe_subsample(data: DuoRouteGroupedData, max_samples: Optional[int]) -> DuoRouteGroupedData:
    if max_samples is None:
        return data
    n = min(max_samples, data.performance.shape[0])
    if n >= data.performance.shape[0]:
        return data
    return DuoRouteGroupedData(
        model_names=data.model_names,
        prompt_keys=data.prompt_keys[:n],
        dataset_ids=data.dataset_ids[:n],
        performance=data.performance[:n],
        cost=data.cost[:n],
        utility=data.utility[:n],
        oracle_reward=data.oracle_reward[:n],
        mask=data.mask[:n],
        prompt_texts=data.prompt_texts[:n],
        response_texts=data.response_texts[:n],
        model_cards=data.model_cards,
    )


def _load_split_embeddings(data_dir: Path, split: str, grouped: DuoRouteGroupedData, embed_dim: int, seed: int):
    resp_path = data_dir / split / "response_embeddings.pth"
    return build_response_embeddings(
        grouped.response_texts,
        embed_path=str(resp_path) if resp_path.exists() else None,
        dim=embed_dim,
        seed=seed,
    )


def train_duoroute(cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)
    data_dir = Path(cfg.data_dir)
    train_data = _maybe_subsample(DuoRouteGroupedData.load(data_dir / "train"), cfg.max_samples)
    val_data = _maybe_subsample(DuoRouteGroupedData.load(data_dir / "val"), cfg.max_samples)
    test_data = _maybe_subsample(DuoRouteGroupedData.load(data_dir / "test"), cfg.max_samples)

    all_texts = train_data.prompt_texts + val_data.prompt_texts + test_data.prompt_texts
    text_to_pid = build_prompt_id_map(all_texts)
    train_ids = assign_prompt_ids_for_grouped(train_data, text_to_pid)
    val_ids = assign_prompt_ids_for_grouped(val_data, text_to_pid)
    test_ids = assign_prompt_ids_for_grouped(test_data, text_to_pid)

    embed_path = cfg.embed_path or str(data_dir / "question_embeddings.pth")
    hash_dim = load_embedding_dim(data_dir, fallback=cfg.embed_dim)
    query_emb = load_or_build_query_embeddings(
        sorted(text_to_pid.keys()),
        embed_path=embed_path if Path(embed_path).exists() else None,
        dim=hash_dim,
        seed=cfg.seed,
    )

    cards = load_model_cards(
        cards_path=cfg.model_cards_path or str(data_dir / "model_cards.json"),
        llmbench_graphrouter_config=cfg.llmbench_graphrouter_config,
        llmbench_collector_config=cfg.llmbench_collector_config,
        model_names=train_data.model_names,
    )
    model_emb = build_model_embeddings(
        cards_for_models(train_data.model_names, cards),
        embed_path=cfg.model_embed_path or str(data_dir / "model_embeddings.pth"),
        dim=hash_dim,
        seed=cfg.seed,
    )

    train_resp = _load_split_embeddings(data_dir, "train", train_data, hash_dim, cfg.seed)
    k = len(train_data.model_names)
    test_resp = zero_response_embeddings(test_data.oracle_reward.shape[0], k, hash_dim)

    model = DuoRouteModel(
        query_emb,
        model_emb,
        hidden_dim=cfg.hidden_dim,
        query_dim=int(query_emb.shape[1]),
        response_dim=int(train_resp.shape[-1]),
        model_dim=int(model_emb.shape[1]),
        use_id_fallback=cfg.use_id_fallback,
        num_models=len(train_data.model_names),
    ).to(cfg.device)

    loader = DataLoader(
        GroupedQueryDataset(train_data, train_ids, train_resp, reward_target=cfg.reward_target),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    device = torch.device(cfg.device)

    tag = f"DuoRoute-a{cfg.alpha:g}_b{cfg.beta:g}_T{cfg.temperature:g}"
    out_dir = Path(cfg.output_dir or data_dir / f"runs/{tag}_seed{cfg.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_regret = float("inf")
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "l_reg": 0.0, "l_rank": 0.0, "l_distill": 0.0}
        count = 0
        beta = 0.0 if epoch <= cfg.distill_warmup_epochs else cfg.beta
        for batch in loader:
            optimizer.zero_grad()
            prompt_ids = batch["prompt_id"].to(device)
            response_emb = batch["response_emb"].to(device)
            oracle = batch["oracle"].to(device)
            mask = batch["mask"].to(device)
            pred_a, pred_b = model(prompt_ids, response_emb)
            out = duoroute_loss(
                pred_a,
                pred_b,
                oracle,
                mask,
                alpha=cfg.alpha,
                beta=beta,
                temperature=cfg.temperature,
            )
            out.total.backward()
            optimizer.step()
            bs = prompt_ids.size(0)
            totals["loss"] += out.total.item() * bs
            totals["l_reg"] += out.l_reg.item() * bs
            totals["l_rank"] += out.l_rank.item() * bs
            totals["l_distill"] += out.l_distill.item() * bs
            count += bs
        train_stats = {k: v / max(count, 1) for k, v in totals.items()}

        pred_val_a = predict_channel_a(model, torch.from_numpy(val_ids).to(device))
        val_metrics = evaluate_predictions(
            val_data.oracle_reward,
            pred_val_a,
            val_data.mask,
            performance=val_data.performance,
            cost=val_data.cost,
            random_seed=cfg.seed,
        )
        row = {"epoch": epoch, "beta_active": beta, **{f"train_{k}": v for k, v in train_stats.items()}}
        row.update({f"val_{k}": v for k, v in val_metrics.to_dict().items()})
        history.append(row)

        if val_metrics.routing_regret < best_val_regret:
            best_val_regret = val_metrics.routing_regret
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_names": train_data.model_names,
                    "config": cfg.__dict__,
                },
                out_dir / "best.pt",
            )

        print(
            f"epoch {epoch}/{cfg.epochs} loss={train_stats['loss']:.4f} "
            f"reg={train_stats['l_reg']:.4f} rank={train_stats['l_rank']:.4f} "
            f"distill={train_stats['l_distill']:.4f} beta={beta:.3f} "
            f"val_regret={val_metrics.routing_regret:.4f}"
        )

    ckpt = torch.load(out_dir / "best.pt", map_location=cfg.device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    pred_test_a = predict_channel_a(model, torch.from_numpy(test_ids).to(device))
    pred_test_b = predict_channel_b(
        model,
        torch.from_numpy(test_ids).to(device),
        test_resp.to(device),
    )
    test_metrics = evaluate_predictions(
        test_data.oracle_reward,
        pred_test_a,
        test_data.mask,
        performance=test_data.performance,
        cost=test_data.cost,
        random_seed=cfg.seed,
    )

    results = {
        "method": tag,
        "test_query_only": test_metrics.to_dict(),
        "history": history,
        "model_names": train_data.model_names,
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        import json

        json.dump(results, f, indent=2)
    np.save(out_dir / "test_pred_a.npy", pred_test_a)
    np.save(out_dir / "test_pred_b.npy", pred_test_b)
    return results
