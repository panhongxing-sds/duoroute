#!/usr/bin/env python3
"""Evaluate query-only DuoRoute routing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from duoroute.data import DuoRouteGroupedData
from duoroute.encoders import build_model_embeddings, load_embedding_dim, load_or_build_query_embeddings
from duoroute.evaluator import evaluate_predictions
from duoroute.inference import predict_channel_a, route_query_only
from duoroute.model import DuoRouteModel
from duoroute.model_cards import cards_for_models, load_model_cards
from duoroute.prompt_ids import assign_prompt_ids, load_global_prompt_map
from duoroute.utils import load_yaml, project_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(project_root() / "configs/default.yaml"))
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    cfg = load_yaml(args.config).get("duoroute", {})
    data_dir = Path(args.data_dir)
    grouped = DuoRouteGroupedData.load(data_dir / args.split)
    text_to_pid = load_global_prompt_map(data_dir)
    prompt_ids = assign_prompt_ids(grouped.prompt_texts, text_to_pid)

    cards = load_model_cards(cards_path=str(data_dir / "model_cards.json"), model_names=grouped.model_names)
    query_emb = load_or_build_query_embeddings(sorted(text_to_pid.keys()), embed_path=str(data_dir / "question_embeddings.pth"))
    model_emb = build_model_embeddings(cards_for_models(grouped.model_names, cards), embed_path=str(data_dir / "model_embeddings.pth"))
    response_dim = load_embedding_dim(data_dir, fallback=int(cfg.get("embed_dim", 2048)))

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = DuoRouteModel(query_emb, model_emb, hidden_dim=64, response_dim=response_dim)
    model.load_state_dict(ckpt["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    pred_a = predict_channel_a(model, torch.from_numpy(prompt_ids).to(device))
    metrics = evaluate_predictions(grouped.oracle_reward, pred_a, grouped.mask, performance=grouped.performance, cost=grouped.cost)
    decision = route_query_only(pred_a, grouped.mask)

    out = {
        "mode": "query_only",
        "split": args.split,
        "metrics": metrics.to_dict(),
        "chosen_models": [grouped.model_names[i] for i in decision.chosen[:10]],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(project_root() / "src"))
    main()
