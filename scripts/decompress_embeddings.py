#!/usr/bin/env python3
"""Restore float32 `.pth` embedding files from compressed `.f16.npz`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from duoroute.embedding_io import COMPRESSED_SUFFIX, compressed_path_for, decompress_embedding


def discover_compressed_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.rglob(f"*{COMPRESSED_SUFFIX}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dirs",
        nargs="+",
        default=["data/seed42_small", "data/seed42_flagship"],
        help="dataset roots to scan for compressed embeddings",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing `.pth` files")
    args = parser.parse_args()

    restored = 0
    for rel in args.data_dirs:
        data_dir = (ROOT / rel).resolve()
        if not data_dir.is_dir():
            print(f"skip missing dir: {data_dir}")
            continue
        for compressed in discover_compressed_files(data_dir):
            stem = compressed.name[: -len(COMPRESSED_SUFFIX)]
            pth = compressed.with_name(f"{stem}.pth")
            try:
                out = decompress_embedding(pth, overwrite=args.overwrite)
            except FileExistsError:
                print(f"skip (exists): {pth}")
                continue
            print(f"restored: {out.relative_to(ROOT)}")
            restored += 1

    print(f"\nRestored {restored} file(s).")


if __name__ == "__main__":
    main()
