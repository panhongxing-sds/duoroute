#!/usr/bin/env python3
"""Gzip split meta.json files for Git-friendly storage (prefer meta.json.gz at load time)."""

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _human(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{n} B"


def compress_meta(path: Path, *, force: bool = False, dry_run: bool = False) -> bool:
    gz_path = path.with_suffix(path.suffix + ".gz")
    if gz_path.exists() and not force:
        print(f"skip (exists): {gz_path.relative_to(ROOT)}")
        return False
    if not path.exists():
        print(f"skip (missing): {path.relative_to(ROOT)}")
        return False

    before = path.stat().st_size
    if dry_run:
        print(f"would gzip: {path.relative_to(ROOT)} -> {gz_path.name}")
        return True

    with open(path, "rb") as src, gzip.open(gz_path, "wb", compresslevel=6) as dst:
        shutil.copyfileobj(src, dst)
    after = gz_path.stat().st_size
    ratio = before / after if after else float("inf")
    print(
        f"gzipped: {path.relative_to(ROOT)} "
        f"{_human(before)} -> {_human(after)} ({ratio:.2f}x) -> {gz_path.name}"
    )
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[
            ROOT / "data" / "seed42_small",
            ROOT / "data" / "seed42_flagship",
        ],
        help="Dataset roots to scan for **/meta.json",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing .gz files")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    count = 0
    for root in args.roots:
        root = root.resolve()
        if not root.is_dir():
            print(f"skip (not a dir): {root}")
            continue
        for meta_path in sorted(root.rglob("meta.json")):
            if compress_meta(meta_path, force=args.force, dry_run=args.dry_run):
                count += 1

    print(f"done: {count} meta.json file(s) processed under {len(args.roots)} root(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
