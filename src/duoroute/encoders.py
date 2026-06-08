"""Embedding model configuration and encoding."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from duoroute.embedding_io import load_embedding_tensor, resolve_embedding_path
from duoroute.model_cards import ModelCard
from duoroute.utils import load_yaml, project_root, resolve_project_path


@dataclass
class EmbeddingConfig:
    backend: str = "sentence_transformers"
    model: str = "Qwen/Qwen3-Embedding-8B"
    batch_size: int = 4
    device: str = "cuda:0"
    max_seq_length: int = 8192
    trust_remote_code: bool = True
    torch_dtype: str = "float16"
    use_flash_attention: bool = False
    normalize: bool = True
    query_prompt_name: Optional[str] = "query"
    document_prompt_name: Optional[str] = None
    base_url: str = ""
    api_key: str = ""
    api_key_env: str = "DUOROUTE_EMBED_API_KEY"
    dimensions: int = 2048
    max_chars: int = 30000
    max_tokens: int = 8000
    request_timeout: int = 120

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EmbeddingConfig":
        return cls(
            backend=str(raw.get("backend", cls.backend)),
            model=str(raw.get("model", cls.model)),
            batch_size=int(raw.get("batch_size", cls.batch_size)),
            device=str(raw.get("device", cls.device)),
            max_seq_length=int(raw.get("max_seq_length", cls.max_seq_length)),
            trust_remote_code=bool(raw.get("trust_remote_code", True)),
            torch_dtype=str(raw.get("torch_dtype", cls.torch_dtype)),
            use_flash_attention=bool(raw.get("use_flash_attention", False)),
            normalize=bool(raw.get("normalize", True)),
            query_prompt_name=raw.get("query_prompt_name", "query"),
            document_prompt_name=raw.get("document_prompt_name"),
            base_url=str(raw.get("base_url", cls.base_url)),
            api_key=str(raw.get("api_key", cls.api_key)),
            api_key_env=str(raw.get("api_key_env", cls.api_key_env)),
            dimensions=int(raw.get("dimensions", cls.dimensions)),
            max_chars=int(raw.get("max_chars", cls.max_chars)),
            max_tokens=int(raw.get("max_tokens", cls.max_tokens)),
            request_timeout=int(raw.get("request_timeout", cls.request_timeout)),
        )


def load_embedding_config(path: str | Path | None = None) -> EmbeddingConfig:
    if path is None:
        path = project_root() / "configs/embedding.yaml"
    else:
        path = resolve_project_path(path) or Path(path)
    path = Path(path)
    raw = load_yaml(path).get("embedding", {})
    cfg = EmbeddingConfig.from_dict(raw)

    local_path = path.parent / "embedding_api.local.yaml"
    if local_path.exists():
        local_raw = load_yaml(local_path).get("embedding", {})
        cfg = EmbeddingConfig.from_dict({**raw, **local_raw})

    if cfg.backend == "openai_api" and not cfg.api_key:
        cfg = replace(cfg, api_key=os.environ.get(cfg.api_key_env, ""))
    if cfg.backend == "openai_api" and not cfg.api_key:
        raise ValueError(
            "OpenAI embedding backend requires api_key in embedding_api.local.yaml "
            f"or env {cfg.api_key_env}"
        )
    return cfg


def _torch_dtype(name: str):
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(name.lower(), torch.float16)


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars]


def _truncate_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return text
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        return _truncate_text(text, max_tokens * 3)
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


def _prepare_openai_text(text: str, cfg: EmbeddingConfig) -> str:
    text = str(text)
    text = _truncate_text(text, cfg.max_chars)
    if cfg.max_tokens > 0:
        text = _truncate_tokens(text, cfg.max_tokens)
    return text


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / (norms + 1e-8)


def create_sentence_transformer(cfg: EmbeddingConfig):
    from sentence_transformers import SentenceTransformer

    model_kwargs: dict[str, Any] = {}
    tokenizer_kwargs: dict[str, Any] = {}
    dtype = _torch_dtype(cfg.torch_dtype)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if cfg.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        tokenizer_kwargs["padding_side"] = "left"

    logger.info(
        f"Loading embedding model {cfg.model} "
        f"(max_seq_length={cfg.max_seq_length}, dtype={cfg.torch_dtype})"
    )
    st_kwargs: dict[str, Any] = {
        "device": cfg.device,
        "trust_remote_code": cfg.trust_remote_code,
    }
    if model_kwargs:
        st_kwargs["model_kwargs"] = model_kwargs
    if tokenizer_kwargs:
        st_kwargs["tokenizer_kwargs"] = tokenizer_kwargs
    model = SentenceTransformer(cfg.model, **st_kwargs)
    model.max_seq_length = int(cfg.max_seq_length)
    return model


def _encode_openai_api(
    texts: Sequence[str],
    cfg: EmbeddingConfig,
    *,
    kind: str,
    checkpoint_path: Path | None = None,
    save_every_batches: int = 0,
    on_checkpoint: Callable[[int, int], None] | None = None,
) -> torch.Tensor:
    from openai import OpenAI

    client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
    prepared = [_prepare_openai_text(text, cfg) for text in texts]
    rows: list[list[float]] = []
    max_retries = 8
    resume_from = 0

    if checkpoint_path is not None and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if int(ckpt.get("total", len(prepared))) != len(prepared):
            logger.warning(
                f"Checkpoint total mismatch ({ckpt.get('total')} vs {len(prepared)}); ignoring checkpoint"
            )
        else:
            rows = [list(row) for row in ckpt["rows"]]
            resume_from = len(rows)
            logger.info(f"Resuming {kind} embeddings from checkpoint: {resume_from}/{len(prepared)}")

    logger.info(
        f"OpenAI API embedding model={cfg.model} dim={cfg.dimensions} "
        f"texts={len(prepared)} batch={cfg.batch_size} kind={kind}"
    )

    batch_starts = list(range(0, len(prepared), cfg.batch_size))
    for batch_idx, start in enumerate(
        tqdm(batch_starts, desc=f"{kind} embeddings", initial=resume_from // cfg.batch_size)
    ):
        if start < resume_from:
            continue
        batch = prepared[start : start + cfg.batch_size]
        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(
                    model=cfg.model,
                    input=batch,
                    dimensions=cfg.dimensions,
                    timeout=cfg.request_timeout,
                )
                ordered = sorted(response.data, key=lambda item: item.index)
                rows.extend([item.embedding for item in ordered])
                break
            except Exception as exc:
                wait_s = min(2 ** attempt, 60)
                logger.warning(
                    f"Embedding batch failed ({start}:{start + len(batch)}), "
                    f"retry {attempt + 1}/{max_retries}: {exc}"
                )
                if attempt + 1 >= max_retries:
                    raise
                time.sleep(wait_s)

        if (
            checkpoint_path is not None
            and save_every_batches > 0
            and (batch_idx + 1) % save_every_batches == 0
        ):
            torch.save(
                {"rows": rows, "total": len(prepared), "kind": kind},
                checkpoint_path,
            )
            if on_checkpoint is not None:
                on_checkpoint(len(rows), len(prepared))

    if checkpoint_path is not None and rows:
        torch.save({"rows": rows, "total": len(prepared), "kind": kind}, checkpoint_path)

    arr = np.asarray(rows, dtype=np.float32)
    if cfg.normalize:
        arr = _normalize_rows(arr)
    logger.info(f"Encoded {len(texts)} {kind} texts -> {arr.shape}")
    return torch.from_numpy(arr)


def _embedding_dim(cfg: EmbeddingConfig) -> int:
    if cfg.backend == "openai_api":
        return cfg.dimensions
    return cfg.max_seq_length  # unused placeholder for empty-only batches


def _encode_nonempty(
    texts: Sequence[str],
    cfg: EmbeddingConfig,
    *,
    kind: str,
    checkpoint_path: Path | None = None,
    save_every_batches: int = 0,
    on_checkpoint: Callable[[int, int], None] | None = None,
) -> torch.Tensor:
    if cfg.backend == "openai_api":
        return _encode_openai_api(
            texts,
            cfg,
            kind=kind,
            checkpoint_path=checkpoint_path,
            save_every_batches=save_every_batches,
            on_checkpoint=on_checkpoint,
        )

    model = create_sentence_transformer(cfg)
    prompt_name: Optional[str] = None
    if kind == "query" and cfg.query_prompt_name:
        prompt_name = cfg.query_prompt_name
    elif kind == "document" and cfg.document_prompt_name:
        prompt_name = cfg.document_prompt_name

    encode_kwargs: dict[str, Any] = {
        "batch_size": cfg.batch_size,
        "show_progress_bar": True,
        "convert_to_numpy": True,
        "normalize_embeddings": cfg.normalize,
    }
    if prompt_name:
        encode_kwargs["prompt_name"] = prompt_name

    emb = model.encode(list(texts), **encode_kwargs)
    return torch.from_numpy(np.asarray(emb, dtype=np.float32))


def encode_texts(
    texts: Sequence[str],
    cfg: EmbeddingConfig,
    *,
    kind: str = "document",
    skip_empty: bool = False,
    checkpoint_path: Path | None = None,
    save_every_batches: int = 0,
    on_checkpoint: Callable[[int, int], None] | None = None,
) -> torch.Tensor:
    if not texts:
        return torch.empty(0, _embedding_dim(cfg))

    texts_list = [str(text) for text in texts]
    if not skip_empty:
        emb = _encode_nonempty(
            texts_list,
            cfg,
            kind=kind,
            checkpoint_path=checkpoint_path,
            save_every_batches=save_every_batches,
            on_checkpoint=on_checkpoint,
        )
        logger.info(f"Encoded {len(texts_list)} {kind} texts -> {tuple(emb.shape)}")
        return emb

    dim = _embedding_dim(cfg)
    result = torch.zeros(len(texts_list), dim, dtype=torch.float32)
    nonempty_indices = [idx for idx, text in enumerate(texts_list) if text.strip()]
    if nonempty_indices:
        nonempty_texts = [texts_list[idx] for idx in nonempty_indices]
        encoded = _encode_nonempty(
            nonempty_texts,
            cfg,
            kind=kind,
            checkpoint_path=checkpoint_path,
            save_every_batches=save_every_batches,
            on_checkpoint=on_checkpoint,
        )
        if encoded.shape[0] != len(nonempty_indices):
            raise RuntimeError(
                f"Embedding count mismatch: expected {len(nonempty_indices)}, got {encoded.shape[0]}"
            )
        for row, idx in enumerate(nonempty_indices):
            result[idx] = encoded[row]

    empty_count = len(texts_list) - len(nonempty_indices)
    logger.info(
        f"Encoded {len(nonempty_indices)}/{len(texts_list)} non-empty {kind} texts "
        f"({empty_count} zero-filled) -> {tuple(result.shape)}"
    )
    return result


def save_embedding_meta(path: Path, cfg: EmbeddingConfig, dim: int, *, extra: dict | None = None) -> None:
    import json

    meta = {
        "backend": cfg.backend,
        "model": cfg.model,
        "dim": dim,
        "dimensions": cfg.dimensions if cfg.backend == "openai_api" else dim,
        "max_seq_length": cfg.max_seq_length,
        "query_prompt_name": cfg.query_prompt_name,
        "normalize": cfg.normalize,
    }
    if extra:
        meta.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_embedding_dim(data_dir: Path, *, fallback: int = 2048) -> int:
    meta_path = data_dir / "embedding_meta.json"
    if meta_path.exists():
        import json

        with open(meta_path, encoding="utf-8") as f:
            return int(json.load(f)["dim"])
    return fallback


# --- hash fallback helpers (smoke tests only) ---


def _hash_embedding(text: str, dim: int, seed: int) -> np.ndarray:
    digest = abs(hash(text)) % (2**31)
    rng = np.random.default_rng(digest + seed)
    vec = rng.normal(size=dim).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-8
    return vec


def build_hash_embeddings(texts: Sequence[str], *, dim: int = 2048, seed: int = 42) -> torch.Tensor:
    rows = np.stack([_hash_embedding(text, dim, seed) for text in texts], axis=0)
    return torch.from_numpy(rows)


def load_or_build_query_embeddings(
    prompt_texts: List[str],
    *,
    embed_path: Optional[str] = None,
    dim: int = 2048,
    seed: int = 42,
) -> torch.Tensor:
    resolved = resolve_embedding_path(embed_path) if embed_path else None
    if resolved and resolved.exists():
        emb = load_embedding_tensor(resolved)
        if emb.shape[0] == len(prompt_texts):
            logger.info(f"Loaded query embeddings from {resolved} shape={tuple(emb.shape)}")
            return emb
        logger.warning("Query embedding count mismatch; rebuilding hash embeddings")

    logger.warning(f"Built hash query embeddings N={len(prompt_texts)} dim={dim} (smoke-test only)")
    return build_hash_embeddings(prompt_texts, dim=dim, seed=seed)


def zero_response_embeddings(n: int, k: int, dim: int) -> torch.Tensor:
    return torch.zeros(n, k, dim, dtype=torch.float32)


def build_response_embeddings(
    response_texts: List[List[str]],
    *,
    embed_path: Optional[str] = None,
    dim: int = 2048,
    seed: int = 42,
    zero_if_missing: bool = False,
) -> torch.Tensor:
    n = len(response_texts)
    k = len(response_texts[0]) if n else 0
    resolved = resolve_embedding_path(embed_path) if embed_path else None
    if resolved and resolved.exists():
        emb = load_embedding_tensor(resolved)
        if emb.shape[:2] == (n, k):
            logger.info(f"Loaded response embeddings from {resolved} shape={tuple(emb.shape)}")
            return emb
        logger.warning("Response embedding shape mismatch; rebuilding hash embeddings")

    if zero_if_missing:
        logger.info(f"Using zero response embeddings shape=({n}, {k}, {dim})")
        return zero_response_embeddings(n, k, dim)

    flat = [text for row in response_texts for text in row]
    flat_emb = build_hash_embeddings(flat, dim=dim, seed=seed + 1)
    return flat_emb.view(n, k, dim)


def build_model_embeddings(
    model_cards: Sequence[ModelCard],
    *,
    embed_path: Optional[str] = None,
    dim: int = 2048,
    seed: int = 42,
) -> torch.Tensor:
    texts = [card.to_embedding_text() for card in model_cards]
    resolved = resolve_embedding_path(embed_path) if embed_path else None
    if resolved and resolved.exists():
        emb = load_embedding_tensor(resolved)
        if emb.shape[0] == len(texts):
            logger.info(f"Loaded model embeddings from {resolved} shape={tuple(emb.shape)}")
            return emb
        logger.warning("Model embedding count mismatch; rebuilding hash embeddings")

    logger.warning(f"Built hash model-card embeddings K={len(texts)} dim={dim} (smoke-test only)")
    return build_hash_embeddings(texts, dim=dim, seed=seed + 2)


def encode_with_sentence_transformer(
    texts: Sequence[str],
    *,
    model_name: str,
    batch_size: int = 32,
    device: str = "cuda:0",
) -> torch.Tensor:
    cfg = EmbeddingConfig(model=model_name, batch_size=batch_size, device=device)
    return encode_texts(texts, cfg, kind="document")
