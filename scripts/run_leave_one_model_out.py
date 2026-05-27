#!/usr/bin/env python3
"""Leave-one-model-out training/evaluation skeleton."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from duoroute.trainer import TrainConfig, train_duoroute
from duoroute.zero_shot import held_out_model_index, mask_out_model
from duoroute.data import DuoRouteGroupedData
from duoroute.utils import project_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--held-out-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train = DuoRouteGroupedData.load(data_dir / "train")
    val = DuoRouteGroupedData.load(data_dir / "val")
    test = DuoRouteGroupedData.load(data_dir / "test")
    idx = held_out_model_index(train.model_names, args.held_out_model)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mask_out_model(train, idx).save(out / "train")
    val.save(out / "val")
    test.save(out / "test")
    if (data_dir / "model_cards.json").exists():
        import shutil

        shutil.copy(data_dir / "model_cards.json", out / "model_cards.json")

    results = train_duoroute(
        TrainConfig(
            data_dir=str(out),
            output_dir=str(out / "run"),
            epochs=args.epochs,
            max_samples=512,
        )
    )
    summary = {
        "held_out_model": args.held_out_model,
        "held_out_index": idx,
        "note": "Train masks held-out model column; evaluate with description embedding only.",
        "results": results,
    }
    with open(out / "lomo_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(project_root() / "src"))
    main()
