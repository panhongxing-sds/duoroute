#!/usr/bin/env python3
import json
from pathlib import Path

import torch
from loguru import logger

from duoroute.data import DuoRouteGroupedData
from duoroute.encoders import (
    encode_texts,
    load_embedding_config,
    save_embedding_meta,
)
from duoroute.model_cards import cards_for_models, load_model_cards
from duoroute.utils import load_yaml, project_root, resolve_project_path


def _load_dataset_order(data_dir: Path) -> list[str]:
    config_json = data_dir / "config.json"
    if not config_json.exists():
        return []
    raw = json.loads(config_json.read_text(encoding="utf-8"))
    bench_path = Path(str(raw.get("bench_config", "")))
    if not bench_path.is_absolute():
        bench_path = project_root() / bench_path
    if not bench_path.exists():
        return []
    bench = load_yaml(bench_path).get("baseline", {})
    datasets = bench.get("filters", {}).get("datasets")
    return list(datasets) if datasets else []


def _ordered_datasets(dataset_ids: list[str], preferred: list[str]) -> list[str]:
    present = set(dataset_ids)
    ordered = [ds for ds in preferred if ds in present]
    for ds in sorted(present - set(ordered)):
        ordered.append(ds)
    return ordered


def _save_response_checkpoint(
    *,
    flat: list[str],
    nonempty_indices: list[int],
    encoded_rows: list[list[float]],
    n: int,
    k: int,
    out: Path,
) -> None:
    dim = len(encoded_rows[0]) if encoded_rows else 0
    result = torch.zeros(len(flat), dim, dtype=torch.float32)
    for row_idx, flat_idx in enumerate(nonempty_indices[: len(encoded_rows)]):
        result[flat_idx] = torch.tensor(encoded_rows[row_idx], dtype=torch.float32)
    torch.save(result.view(n, k, -1), out)
    print(f"Checkpoint saved {out} ({len(encoded_rows)} non-empty encoded)")


def _remove_old_embeddings(
    data_dir: Path,
    *,
    target: str,
    response_splits: tuple[str, ...],
    resume: bool,
) -> None:
    if resume:
        return
    if target in {"query", "all"}:
        for path in (data_dir / "question_embeddings.pth", data_dir / "embedding_meta.json"):
            if path.exists():
                path.unlink()
                print(f"Removed {path}")
    if target in {"response", "all"}:
        for split in response_splits:
            split_dir = data_dir / split
            for path in split_dir.glob("response_embed*"):
                path.unlink()
                print(f"Removed {path}")
            for name in ("response_embeddings.pth",):
                path = split_dir / name
                if path.exists():
                    path.unlink()
                    print(f"Removed {path}")
    if target in {"model", "all"}:
        for path in (data_dir / "model_embeddings.pth", data_dir / "embedding_meta.json"):
            if path.exists():
                path.unlink()
                print(f"Removed {path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build query/response/model embeddings")
    parser.add_argument("--config", default="configs/default.yaml", help="duoroute runtime config")
    parser.add_argument(
        "--embedding-config",
        default=None,
        help="embedding model config (default: configs/embedding.yaml)",
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--target", choices=["query", "response", "model", "all"], default="all")
    parser.add_argument(
        "--response-splits",
        default="train",
        help="comma-separated splits for response embeddings (default: train only)",
    )
    parser.add_argument(
        "--skip-empty-responses",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="skip API calls for empty response texts and store zero vectors",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from API checkpoint files instead of deleting outputs",
    )
    parser.add_argument(
        "--checkpoint-every-batches",
        type=int,
        default=64,
        help="save API checkpoint every N batches (default: 64, ~2048 texts)",
    )
    parser.add_argument(
        "--by-dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="encode/save response embeddings dataset-by-dataset in config order",
    )
    parser.add_argument(
        "--stop-after-dataset",
        default=None,
        help="stop after this dataset completes (by-dataset mode only)",
    )
    parser.add_argument(
        "--only-datasets",
        default=None,
        help="comma-separated dataset ids to encode (by-dataset mode only)",
    )
    args = parser.parse_args()

    only_datasets: set[str] | None = None
    if args.only_datasets:
        only_datasets = {part.strip() for part in args.only_datasets.split(",") if part.strip()}
        if not only_datasets:
            raise ValueError("--only-datasets must list at least one dataset id")

    response_splits = tuple(part.strip() for part in args.response_splits.split(",") if part.strip())
    if not response_splits:
        raise ValueError("At least one split is required for --response-splits")

    duoroute_cfg = load_yaml(resolve_project_path(args.config)).get("duoroute", {})
    embed_cfg_path = args.embedding_config or duoroute_cfg.get("embedding_config", "configs/embedding.yaml")
    embed_cfg = load_embedding_config(embed_cfg_path)

    data_dir = Path(args.data_dir)
    _remove_old_embeddings(
        data_dir,
        target=args.target,
        response_splits=response_splits,
        resume=args.resume,
    )
    cards = load_model_cards(
        cards_path=str(data_dir / "model_cards.json")
        if (data_dir / "model_cards.json").exists()
        else duoroute_cfg.get("model_cards"),
        llmbench_graphrouter_config=duoroute_cfg.get("llmbench_graphrouter_config"),
        llmbench_collector_config=duoroute_cfg.get("llmbench_collector_config"),
        model_names=DuoRouteGroupedData.load(data_dir / "train").model_names,
    )

    embed_dim: int | None = None
    response_stats: dict[str, dict[str, int]] = {}

    if args.target in {"query", "all"}:
        texts: list[str] = []
        for split in ("train", "val", "test"):
            grouped = DuoRouteGroupedData.load(data_dir / split)
            texts.extend(grouped.prompt_texts)
        unique = sorted(set(texts))
        query_emb = encode_texts(unique, embed_cfg, kind="query")
        embed_dim = int(query_emb.shape[1])
        torch.save(query_emb, data_dir / "question_embeddings.pth")
        print(f"Saved question_embeddings.pth shape={tuple(query_emb.shape)}")

    if args.target in {"response", "all"}:
        for split in response_splits:
            grouped = DuoRouteGroupedData.load(data_dir / split)
            n, k = grouped.performance.shape
            out = data_dir / split / "response_embeddings.pth"
            progress_path = data_dir / split / "response_embed_progress.json"
            dataset_order = _ordered_datasets(grouped.dataset_ids, _load_dataset_order(data_dir))
            if only_datasets is not None:
                dataset_order = [ds for ds in dataset_order if ds in only_datasets]

            if args.by_dataset:
                dim = embed_cfg.dimensions if embed_cfg.backend == "openai_api" else embed_cfg.max_seq_length
                completed: set[str] = set()
                if args.resume and progress_path.exists():
                    progress = json.loads(progress_path.read_text(encoding="utf-8"))
                    completed = set(progress.get("completed_datasets", []))
                if args.resume and out.exists() and completed:
                    result = torch.load(out, map_location="cpu", weights_only=False).float()
                else:
                    result = torch.zeros(n, k, dim, dtype=torch.float32)

                total_nonempty = 0
                for ds in dataset_order:
                    if ds in completed:
                        logger.info(f"Skip completed dataset {ds}")
                        continue
                    row_indices = [i for i, ds_id in enumerate(grouped.dataset_ids) if ds_id == ds]
                    flat = [text for i in row_indices for text in grouped.response_texts[i]]
                    api_ckpt = data_dir / split / f"response_embed_api_ckpt_{ds}.pt"
                    logger.info(
                        f"Encoding dataset {ds}: prompts={len(row_indices)} cells={len(flat)} "
                        f"order={dataset_order.index(ds) + 1}/{len(dataset_order)}"
                    )

                    resp_flat = encode_texts(
                        flat,
                        embed_cfg,
                        kind="document",
                        skip_empty=args.skip_empty_responses,
                        checkpoint_path=api_ckpt,
                        save_every_batches=max(1, args.checkpoint_every_batches),
                    )
                    cursor = 0
                    for row_idx in row_indices:
                        result[row_idx] = resp_flat[cursor : cursor + k]
                        cursor += k
                    embed_dim = int(resp_flat.shape[1])
                    torch.save(result, out)
                    completed.add(ds)
                    progress_payload: dict = {
                        "completed_datasets": [d for d in dataset_order if d in completed],
                        "dataset_order": dataset_order,
                    }
                    if args.stop_after_dataset and ds == args.stop_after_dataset:
                        progress_payload["paused_after"] = ds
                    progress_path.write_text(
                        json.dumps(progress_payload, indent=2),
                        encoding="utf-8",
                    )
                    if api_ckpt.exists():
                        api_ckpt.unlink()
                    ds_nonempty = sum(1 for text in flat if str(text).strip())
                    total_nonempty += ds_nonempty
                    print(f"Saved dataset {ds} -> {out} ({len(row_indices)} prompts, {ds_nonempty} non-empty)")
                    if args.stop_after_dataset and ds == args.stop_after_dataset:
                        logger.info(f"Stopped after dataset {ds} (--stop-after-dataset)")
                        break

                embed_dim = int(result.shape[-1])
                response_stats[split] = {
                    "total": n * k,
                    "nonempty": total_nonempty,
                    "empty_zero_filled": n * k - total_nonempty,
                    "dataset_order": dataset_order,
                }
                continue

            flat = [text for row in grouped.response_texts for text in row]
            api_ckpt = data_dir / split / "response_embed_api_ckpt.pt"
            nonempty_indices = [idx for idx, text in enumerate(flat) if str(text).strip()]

            def _on_checkpoint(done: int, total: int) -> None:
                if not api_ckpt.exists():
                    return
                ckpt = torch.load(api_ckpt, map_location="cpu", weights_only=False)
                _save_response_checkpoint(
                    flat=flat,
                    nonempty_indices=nonempty_indices,
                    encoded_rows=ckpt["rows"],
                    n=n,
                    k=k,
                    out=out,
                )

            resp_emb = encode_texts(
                flat,
                embed_cfg,
                kind="document",
                skip_empty=args.skip_empty_responses,
                checkpoint_path=api_ckpt,
                save_every_batches=max(1, args.checkpoint_every_batches),
                on_checkpoint=_on_checkpoint,
            )
            embed_dim = int(resp_emb.shape[1])
            torch.save(resp_emb.view(n, k, -1), out)
            if api_ckpt.exists():
                api_ckpt.unlink()
            nonempty = sum(1 for text in flat if str(text).strip())
            response_stats[split] = {
                "total": len(flat),
                "nonempty": nonempty,
                "empty_zero_filled": len(flat) - nonempty,
            }
            print(
                f"Saved {out} shape={(n, k, embed_dim)} "
                f"(api_calls~={(nonempty + embed_cfg.batch_size - 1) // embed_cfg.batch_size} batches)"
            )

    if args.target in {"model", "all"}:
        train_grouped = DuoRouteGroupedData.load(data_dir / "train")
        model_cards = cards_for_models(train_grouped.model_names, cards)
        texts = [card.to_embedding_text() for card in model_cards]
        model_emb = encode_texts(texts, embed_cfg, kind="document")
        embed_dim = int(model_emb.shape[1])
        torch.save(model_emb, data_dir / "model_embeddings.pth")
        print(f"Saved model_embeddings.pth shape={tuple(model_emb.shape)}")

    if embed_dim is not None:
        save_embedding_meta(
            data_dir / "embedding_meta.json",
            embed_cfg,
            embed_dim,
            extra={
                "response_splits": list(response_splits),
                "skip_empty_responses": args.skip_empty_responses,
                "response_stats": response_stats,
            },
        )
        print(f"Saved embedding_meta.json model={embed_cfg.model} dim={embed_dim}")


if __name__ == "__main__":
    main()
