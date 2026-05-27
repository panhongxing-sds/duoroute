"""Model card loading compatible with LLMRouterBench GraphRouter descriptions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import json

from duoroute.utils import load_yaml


@dataclass
class ModelCard:
    name: str
    feature: str
    input_price: float = 0.0
    output_price: float = 0.0
    capability: Optional[List[str]] = None
    specialty: Optional[List[str]] = None
    provider: str = ""
    context_length: int = 0

    def to_embedding_text(self) -> str:
        parts = [self.feature.strip()]
        if self.capability:
            parts.append(f"Capabilities: {', '.join(self.capability)}.")
        if self.specialty:
            parts.append(f"Specialties: {', '.join(self.specialty)}.")
        if self.provider:
            parts.append(f"Provider: {self.provider}.")
        if self.context_length:
            parts.append(f"Context length: {self.context_length}.")
        if self.input_price or self.output_price:
            parts.append(
                f"Pricing per 1M tokens: input {self.input_price}, output {self.output_price}."
            )
        return " ".join(parts)


def _parse_card(name: str, payload: dict[str, Any]) -> ModelCard:
    return ModelCard(
        name=name,
        feature=str(payload.get("feature") or payload.get("description") or f"{name} language model."),
        input_price=float(payload.get("input_price") or payload.get("cost_per_1m_input_tokens") or 0.0),
        output_price=float(payload.get("output_price") or payload.get("cost_per_1m_output_tokens") or 0.0),
        capability=payload.get("capability"),
        specialty=payload.get("specialty"),
        provider=str(payload.get("provider") or ""),
        context_length=int(payload.get("context_length") or 0),
    )


def _load_graphrouter_cards(path: Path) -> Dict[str, ModelCard]:
    cfg = load_yaml(path)
    block = cfg.get("graphrouter", cfg)
    raw = block.get("llm_descriptions") or block.get("model_cards") or {}
    return {name: _parse_card(name, payload) for name, payload in raw.items()}


def _load_collector_pricing(path: Path) -> Dict[str, tuple[float, float]]:
    cfg = load_yaml(path)
    pricing: Dict[str, tuple[float, float]] = {}
    for model in cfg.get("models", []):
        name = model.get("name")
        prices = model.get("pricing") or {}
        if not name:
            continue
        pricing[name] = (
            float(prices.get("prompt_price_per_million") or 0.0),
            float(prices.get("completion_price_per_million") or 0.0),
        )
    return pricing


def _load_cards_file(path: Path) -> Dict[str, dict[str, Any]]:
    if path.suffix.lower() == ".json":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = load_yaml(path)
    if isinstance(raw, dict) and "model_cards" in raw:
        raw = raw["model_cards"]
    return raw if isinstance(raw, dict) else {}


def load_model_cards(
    *,
    cards_path: Optional[str] = None,
    llmbench_graphrouter_config: Optional[str] = None,
    llmbench_collector_config: Optional[str] = None,
    model_names: Optional[Iterable[str]] = None,
) -> Dict[str, ModelCard]:
    cards: Dict[str, ModelCard] = {}

    if cards_path and Path(cards_path).exists():
        raw = _load_cards_file(Path(cards_path))
        cards.update({name: _parse_card(name, payload) for name, payload in raw.items()})

    if llmbench_graphrouter_config and Path(llmbench_graphrouter_config).exists():
        graph_cards = _load_graphrouter_cards(Path(llmbench_graphrouter_config))
        for name, card in graph_cards.items():
            if name not in cards:
                cards[name] = card

    pricing: Dict[str, tuple[float, float]] = {}
    if llmbench_collector_config and Path(llmbench_collector_config).exists():
        pricing = _load_collector_pricing(Path(llmbench_collector_config))

    for name, (inp, out) in pricing.items():
        if name in cards:
            if not cards[name].input_price:
                cards[name].input_price = inp
            if not cards[name].output_price:
                cards[name].output_price = out
        else:
            cards[name] = ModelCard(
                name=name,
                feature=f"{name} is an instruction-tuned language model.",
                input_price=inp,
                output_price=out,
            )

    if model_names:
        for name in model_names:
            if name not in cards:
                cards[name] = ModelCard(
                    name=name,
                    feature=f"{name} is an open-weight instruction-tuned language model for general reasoning and coding.",
                )

    return cards


def cards_for_models(model_names: List[str], cards: Dict[str, ModelCard]) -> List[ModelCard]:
    return [cards[name] for name in model_names]
