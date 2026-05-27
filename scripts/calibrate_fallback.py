#!/usr/bin/env python3
"""Calibrate fallback threshold on dev split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from duoroute.calibration import calibrate_fallback_threshold
from duoroute.data import DuoRouteGroupedData
from duoroute.encoders import build_model_embeddings, build_response_embeddings, load_embedding_dim, load_or_build_query_embeddings
from duoroute.inference import predict_channel_a, predict_channel_b
from duoroute.model import DuoRouteModel
from duoroute.model_cards import cards_for_models, load_model_cards
from duoroute.prompt_ids import assign_prompt_ids, load_global_prompt_map
from duoroute.utils import load_yaml, project_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(project_root() / "configs/default.yaml"))
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--calibration-split", default="train", help="split with response embeddings for Channel B")
    args = parser.parse_args()

    cfg = load_yaml(args.config).get("duoroute", {})
    data_dir = Path(args.data_dir)
    cal_split = args.calibration_split
    grouped = DuoRouteGroupedData.load(data_dir / cal_split)
    text_to_pid = load_global_prompt_map(data_dir)
    prompt_ids = assign_prompt_ids(grouped.prompt_texts, text_to_pid)

    cards = load_model_cards(cards_path=str(data_dir / "model_cards.json"), model_names=grouped.model_names)
    query_emb = load_or_build_query_embeddings(sorted(text_to_pid.keys()), embed_path=str(data_dir / "question_embeddings.pth"))
    model_emb = build_model_embeddings(cards_for_models(grouped.model_names, cards), embed_path=str(data_dir / "model_embeddings.pth"))
    embed_dim = load_embedding_dim(data_dir, fallback=int(cfg.get("embed_dim", 2048)))
    resp_path = data_dir / cal_split / "response_embeddings.pth"
    if not resp_path.exists():
        raise FileNotFoundError(
            f"Missing {resp_path}. Response embeddings are built for train only; "
            "use --calibration-split train."
        )
    response_emb = build_response_embeddings(grouped.response_texts, embed_path=str(resp_path), dim=embed_dim)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = DuoRouteModel(query_emb, model_emb, hidden_dim=64, response_dim=embed_dim)
    model.load_state_dict(ckpt["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    pred_a = predict_channel_a(model, torch.from_numpy(prompt_ids).to(device))
    pred_b = predict_channel_b(model, torch.from_numpy(prompt_ids).to(device), response_emb.to(device))
    rows = calibrate_fallback_threshold(
        pred_a,
        pred_b,
        grouped.oracle_reward,
        grouped.mask,
        performance=grouped.performance,
        cost=grouped.cost,
        thresholds=cfg.get("fallback_thresholds"),
    )
    out_path = Path(args.checkpoint).parent / "fallback_calibration.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(project_root() / "src"))
    main()
