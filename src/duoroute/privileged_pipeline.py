"""Shared loaders for response-teacher / query-student distillation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from duoroute.data import DuoRouteGroupedData
from duoroute.encoders import (
    build_model_embeddings,
    build_response_embeddings,
    load_embedding_dim,
    load_or_build_query_embeddings,
)
from duoroute.model import DuoRouteModel
from duoroute.model_cards import cards_for_models, load_model_cards
from duoroute.prompt_ids import build_prompt_id_map
from duoroute.residual import response_present_mask
from duoroute.trainer import GroupedQueryDataset, assign_prompt_ids_for_grouped
from duoroute.utils import load_yaml, project_root


def load_subset_datasets(pool_name: str, config_path: Path | None = None) -> dict[str, list[str]]:
    path = config_path or (project_root() / "configs/subset_datasets.yaml")
    raw = load_yaml(path)
    block = raw.get(pool_name, {})
    out = {
        split: list(block.get(split, []))
        for split in ("train", "val", "test")
        if block.get(split)
    }
    if out:
        train_set = set(out.get("train", []))
        test_set = set(out.get("test", []))
        if train_set and test_set and train_set != test_set:
            raise ValueError(
                f"subset_datasets for {pool_name}: train {sorted(train_set)} "
                f"!= test {sorted(test_set)}; use the same dataset list on both splits"
            )
        val_set = set(out.get("val", []))
        if val_set and test_set and val_set != test_set:
            raise ValueError(
                f"subset_datasets for {pool_name}: val {sorted(val_set)} "
                f"!= test {sorted(test_set)}; keep val aligned with test for distillation"
            )
    return out


def filter_grouped_by_datasets(
    grouped: DuoRouteGroupedData,
    prompt_ids: np.ndarray,
    response_emb: torch.Tensor,
    allowed: list[str],
) -> tuple[DuoRouteGroupedData, np.ndarray, torch.Tensor]:
    """Keep only queries whose dataset_id is in allowed."""
    if not allowed:
        return grouped, prompt_ids, response_emb
    allow = set(allowed)
    idx = np.array([i for i, ds in enumerate(grouped.dataset_ids) if ds in allow], dtype=np.int64)
    if len(idx) == 0:
        raise ValueError(f"No queries left after filtering to datasets={allowed}")

    oracle_exp = grouped.oracle_reward_exp[idx] if grouped.oracle_reward_exp is not None else None
    filtered = DuoRouteGroupedData(
        model_names=grouped.model_names,
        prompt_keys=[grouped.prompt_keys[i] for i in idx],
        dataset_ids=[grouped.dataset_ids[i] for i in idx],
        performance=grouped.performance[idx],
        cost=grouped.cost[idx],
        utility=grouped.utility[idx],
        oracle_reward=grouped.oracle_reward[idx],
        mask=grouped.mask[idx],
        prompt_texts=[grouped.prompt_texts[i] for i in idx],
        response_texts=[grouped.response_texts[i] for i in idx],
        model_cards=grouped.model_cards,
        oracle_reward_exp=oracle_exp,
    )
    return filtered, prompt_ids[idx], response_emb[idx]


@dataclass
class PoolBundle:
    data_dir: Path
    cfg: dict
    train: DuoRouteGroupedData
    val: DuoRouteGroupedData
    test: DuoRouteGroupedData
    train_ids: np.ndarray
    val_ids: np.ndarray
    test_ids: np.ndarray
    train_resp: torch.Tensor
    val_resp: torch.Tensor
    test_resp: torch.Tensor
    model: DuoRouteModel
    device: torch.device


def load_pool_bundle(
    data_dir: Path,
    config_path: Path,
    checkpoint_path: Path,
    *,
    device: torch.device,
    freeze_channel_a: bool = True,
    zero_init_delta: bool = True,
    subset_datasets: dict[str, list[str]] | None = None,
) -> PoolBundle:
    cfg_yaml = load_yaml(config_path).get("duoroute", {})
    train = DuoRouteGroupedData.load(data_dir / "train")
    val = DuoRouteGroupedData.load(data_dir / "val")
    test = DuoRouteGroupedData.load(data_dir / "test")

    all_texts = train.prompt_texts + val.prompt_texts + test.prompt_texts
    text_to_pid = build_prompt_id_map(all_texts)
    train_ids = assign_prompt_ids_for_grouped(train, text_to_pid)
    val_ids = assign_prompt_ids_for_grouped(val, text_to_pid)
    test_ids = assign_prompt_ids_for_grouped(test, text_to_pid)

    hash_dim = load_embedding_dim(data_dir, fallback=int(cfg_yaml.get("embed_dim", 2048)))
    query_emb = load_or_build_query_embeddings(
        sorted(text_to_pid.keys()),
        embed_path=str(data_dir / "question_embeddings.pth"),
    )
    cards = load_model_cards(
        cards_path=str(data_dir / "model_cards.json"),
        model_names=train.model_names,
    )
    model_emb = build_model_embeddings(
        cards_for_models(train.model_names, cards),
        embed_path=str(data_dir / "model_embeddings.pth"),
    )

    train_resp = build_response_embeddings(
        train.response_texts,
        embed_path=str(data_dir / "train" / "response_embeddings.pth"),
        dim=hash_dim,
        seed=42,
        zero_if_missing=True,
    )
    val_resp = build_response_embeddings(
        val.response_texts,
        embed_path=str(data_dir / "val" / "response_embeddings.pth"),
        dim=hash_dim,
        seed=42,
        zero_if_missing=True,
    )
    test_resp = build_response_embeddings(
        test.response_texts,
        embed_path=str(data_dir / "test" / "response_embeddings.pth"),
        dim=hash_dim,
        seed=42,
        zero_if_missing=True,
    )

    if subset_datasets:
        if subset_datasets.get("train"):
            train, train_ids, train_resp = filter_grouped_by_datasets(
                train, train_ids, train_resp, subset_datasets["train"]
            )
        if subset_datasets.get("val"):
            val, val_ids, val_resp = filter_grouped_by_datasets(
                val, val_ids, val_resp, subset_datasets["val"]
            )
        if subset_datasets.get("test"):
            test, test_ids, test_resp = filter_grouped_by_datasets(
                test, test_ids, test_resp, subset_datasets["test"]
            )

    model = DuoRouteModel(
        query_emb,
        model_emb,
        hidden_dim=int(cfg_yaml.get("hidden_dim", 64)),
        response_dim=hash_dim,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])

    if freeze_channel_a:
        for name, param in model.named_parameters():
            if "channel_b" in name or "response_projector" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    if zero_init_delta:
        last = model.channel_b.net[-1]
        if isinstance(last, torch.nn.Linear):
            torch.nn.init.zeros_(last.weight)
            torch.nn.init.zeros_(last.bias)

    return PoolBundle(
        data_dir=data_dir,
        cfg=cfg_yaml,
        train=train,
        val=val,
        test=test,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        train_resp=train_resp,
        val_resp=val_resp,
        test_resp=test_resp,
        model=model,
        device=device,
    )


def make_train_loader(
    bundle: PoolBundle,
    *,
    reward_target: str,
    batch_size: int | None = None,
) -> DataLoader:
    bs = batch_size or int(bundle.cfg.get("batch_size", 64))
    ds = GroupedQueryDataset(
        bundle.train, bundle.train_ids, bundle.train_resp, reward_target=reward_target
    )
    return DataLoader(ds, batch_size=bs, shuffle=True)


def has_response_query_mask(
    response_emb: torch.Tensor,
    arm_mask: np.ndarray,
) -> np.ndarray:
    """Queries with at least one arm having non-zero response embedding."""
    resp = response_present_mask(response_emb).cpu().numpy() & arm_mask
    return resp.any(axis=1)


def response_coverage_stats(
    response_emb: torch.Tensor,
    arm_mask: np.ndarray,
) -> dict[str, float]:
    resp = response_present_mask(response_emb).cpu().numpy() & arm_mask
    n_cell = float(arm_mask.sum())
    n_resp = float(resp.sum())
    return {
        "cell_coverage": n_resp / max(n_cell, 1.0),
        "query_coverage": float(resp.any(axis=1).mean()),
    }
