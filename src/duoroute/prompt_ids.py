"""Prompt id helpers shared by train/eval/inference."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from duoroute.data import DuoRouteGroupedData


def build_prompt_id_map(all_texts: Iterable[str]) -> dict[str, int]:
    return {text: idx for idx, text in enumerate(sorted(set(all_texts)))}


def assign_prompt_ids(prompt_texts: list[str], text_to_pid: dict[str, int]) -> np.ndarray:
    return np.array([text_to_pid[text] for text in prompt_texts], dtype=np.int64)


def load_global_prompt_map(data_dir: str | Path) -> dict[str, int]:
    data_dir = Path(data_dir)
    texts: list[str] = []
    for split in ("train", "val", "test"):
        grouped = DuoRouteGroupedData.load(data_dir / split)
        texts.extend(grouped.prompt_texts)
    return build_prompt_id_map(texts)
