"""Load/save embedding tensors with optional float16 NPZ compression."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from loguru import logger

COMPRESSED_SUFFIX = ".f16.npz"


def compressed_path_for(pth_path: Path | str) -> Path:
    path = Path(pth_path)
    if path.name.endswith(COMPRESSED_SUFFIX):
        return path
    return path.with_name(f"{path.stem}{COMPRESSED_SUFFIX}")


def resolve_embedding_path(embed_path: Path | str) -> Path:
    """Prefer compressed `.f16.npz` when present, else original `.pth`."""
    path = Path(embed_path)
    compressed = compressed_path_for(path)
    if compressed.exists():
        return compressed
    return path


def load_embedding_tensor(embed_path: Path | str) -> torch.Tensor:
    """Load embeddings as float32; supports `.pth` and compressed `.f16.npz`."""
    path = resolve_embedding_path(embed_path)
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {embed_path} (resolved: {path})")

    if path.name.endswith(COMPRESSED_SUFFIX):
        with np.load(path) as data:
            arr = data["embeddings"]
        logger.debug(f"Loaded compressed embeddings from {path} shape={arr.shape} dtype={arr.dtype}")
        return torch.from_numpy(np.asarray(arr)).float()

    emb = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(emb, dict) and "embeddings" in emb:
        emb = emb["embeddings"]
    return emb.float()


def save_compressed_embedding(tensor: torch.Tensor, pth_path: Path | str) -> Path:
    """Write float16 NPZ next to the canonical `.pth` path."""
    out = compressed_path_for(pth_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = tensor.detach().cpu().half().numpy()
    np.savez_compressed(out, embeddings=arr)
    return out


def decompress_embedding(pth_path: Path | str, *, overwrite: bool = False) -> Path:
    """Restore `.pth` float32 tensor from compressed `.f16.npz`."""
    path = Path(pth_path)
    compressed = compressed_path_for(path)
    if not compressed.exists():
        raise FileNotFoundError(f"Compressed embedding not found: {compressed}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")

    tensor = load_embedding_tensor(compressed)
    torch.save(tensor, path)
    return path
