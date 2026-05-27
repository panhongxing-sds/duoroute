#!/usr/bin/env python3
"""Compress DuoRoute embedding `.pth` files to float16 `.f16.npz` (gzip inside NPZ)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from duoroute.embedding_io import COMPRESSED_SUFFIX, compressed_path_for, save_compressed_embedding


def _human(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{n} B"


def compress_file(pth_path: Path, *, dry_run: bool = False) -> tuple[int, int] | None:
    out = compressed_path_for(pth_path)
    if out.exists():
        print(f"skip (exists): {out}")
        return None
    if not pth_path.exists():
        print(f"skip (missing): {pth_path}")
        return None

    before = pth_path.stat().st_size
    if dry_run:
        print(f"would compress: {pth_path} -> {out.name}")
        return before, 0

    tensor = torch.load(pth_path, map_location="cpu", weights_only=False)
    if isinstance(tensor, dict) and "embeddings" in tensor:
        tensor = tensor["embeddings"]
    out_path = save_compressed_embedding(tensor, pth_path)
    after = out_path.stat().st_size
    ratio = before / after if after else float("inf")
    print(
        f"compressed: {pth_path.relative_to(ROOT)} "
        f"{_human(before)} -> {_human(after)} ({ratio:.2f}x) -> {out_path.name}"
    )
    return before, after


def discover_embedding_files(data_dir: Path) -> list[Path]:
    files: list[Path] = []
    for name in ("question_embeddings.pth", "model_embeddings.pth"):
        p = data_dir / name
        if p.exists():
            files.append(p)
    for split in ("train", "val", "test"):
        p = data_dir / split / "response_embeddings.pth"
        if p.exists():
            files.append(p)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dirs",
        nargs="+",
        default=["data/seed42_small", "data/seed42_flagship"],
        help="dataset roots containing embedding files",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    total_before = 0
    total_after = 0
    count = 0
    for rel in args.data_dirs:
        data_dir = (ROOT / rel).resolve()
        if not data_dir.is_dir():
            print(f"skip missing dir: {data_dir}")
            continue
        print(f"\n=== {data_dir.relative_to(ROOT)} ===")
        for pth in discover_embedding_files(data_dir):
            result = compress_file(pth, dry_run=args.dry_run)
            if result is None:
                continue
            b, a = result
            total_before += b
            total_after += a
            count += 1

    if count:
        saved = total_before - total_after
        print(
            f"\nDone: {count} file(s), "
            f"{_human(total_before)} -> {_human(total_after)} "
            f"(saved {_human(saved)}, {total_before / max(total_after, 1):.2f}x)"
        )
        print(f"Compressed files use suffix {COMPRESSED_SUFFIX!r}; loaders prefer them automatically.")
    else:
        print("\nNo files compressed.")


if __name__ == "__main__":
    main()
