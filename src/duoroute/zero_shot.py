"""Leave-one-model-out and unseen model generalization helpers."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from duoroute.data import DuoRouteGroupedData
from duoroute.model_cards import ModelCard, cards_for_models, load_model_cards
from duoroute.encoders import build_model_embeddings


def mask_out_model(data: DuoRouteGroupedData, model_idx: int) -> DuoRouteGroupedData:
    mask = data.mask.copy()
    mask[:, model_idx] = False
    return DuoRouteGroupedData(
        model_names=data.model_names,
        prompt_keys=data.prompt_keys,
        dataset_ids=data.dataset_ids,
        performance=data.performance,
        cost=data.cost,
        utility=data.utility,
        oracle_reward=data.oracle_reward,
        mask=mask,
        prompt_texts=data.prompt_texts,
        response_texts=data.response_texts,
        model_cards=data.model_cards,
    )


def build_zero_shot_model_embeddings(
    model_names: List[str],
    held_out_model: str,
    cards: dict[str, ModelCard],
    *,
    embed_path: Optional[str],
    dim: int,
    seed: int,
) -> torch.Tensor:
    """
    Training excludes held-out model column; at test time only description embedding is available.
    """
    visible_cards = cards_for_models(model_names, cards)
    return build_model_embeddings(visible_cards, embed_path=embed_path, dim=dim, seed=seed)


def held_out_model_index(model_names: List[str], held_out_model: str) -> int:
    try:
        return model_names.index(held_out_model)
    except ValueError as exc:
        raise ValueError(f"Model {held_out_model} not in pool {model_names}") from exc
