"""Benchmark record schema for DuoRoute data ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BenchRecord:
    dataset_id: str
    split: str
    model_name: str
    record_index: int
    origin_query: str
    prompt: str
    prediction: str
    raw_output: Any
    ground_truth: str
    score: float
    prompt_tokens: int
    completion_tokens: int
    cost: float
