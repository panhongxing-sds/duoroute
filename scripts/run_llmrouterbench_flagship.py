#!/usr/bin/env python3
"""Train LLMRouterBench router SOTA on DuoRoute flagship 851 pool and export test choices."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from duoroute.reward_builder import rebuild_grouped_rewards
from duoroute.utils import project_root

DEFAULT_LB_ROOT = Path(os.environ.get("LLMROUTERBENCH_ROOT", "/home/phx/LLMRouterBench"))
DEFAULT_DATA = project_root() / "data/seed42_flagship"
DEFAULT_OUT = project_root() / "outputs/llmrouterbench_flagship"
DEFAULT_LAMBDA_TRAIN = 0.2

# Display names used in export_main_tables MAIN_METHOD_ORDER
METHOD_GRAPHROUTER = "GraphRouter-lite"
METHOD_EMBEDLLM = "EmbedLLM"
METHOD_ROUTERDC = "RouterDC-lite"
METHOD_ROUTELLM = "RouteLLM"
METHOD_HYBRIDLLM = "HybridLLM-lite"

SOTA_METHODS = [
    METHOD_GRAPHROUTER,
    METHOD_EMBEDLLM,
    METHOD_ROUTERDC,
    METHOD_ROUTELLM,
    METHOD_HYBRIDLLM,
]


def _masked_argmax(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    neg_inf = -1e9
    masked = np.where(mask, values, neg_inf)
    return masked.argmax(axis=1)


def _llmb_root(path: Path | None) -> Path:
    root = path or DEFAULT_LB_ROOT
    if not (root / "baselines" / "spo").is_dir():
        raise FileNotFoundError(
            f"LLMRouterBench not found at {root}. Set LLMROUTERBENCH_ROOT or clone LLMRouterBench."
        )
    return root


def _ensure_lb_path(lb_root: Path) -> None:
    if str(lb_root) not in sys.path:
        sys.path.insert(0, str(lb_root))


def lb_root_from_run(run_dir: Path) -> Path:
    for p in [run_dir] + list(run_dir.parents):
        if (p / "baselines" / "spo").is_dir():
            return p
    return DEFAULT_LB_ROOT


def flagship_data_dir(seed: int, *, pool_root: Path | None = None) -> Path:
    """Flagship grouped splits; currently built under seed42_flagship for all seeds."""
    del seed
    base = pool_root or project_root() / "data"
    p = base / "seed42_flagship"
    if not (p / "test" / "grouped.npz").exists():
        raise FileNotFoundError(f"Missing flagship pool at {p}")
    return p


def patch_split_utilities(data_dir: Path, lambda_cost: float) -> None:
    """Rewrite utility in grouped.npz to match DuoRoute λ protocol (in-place)."""
    from baselines.spo_data import GroupedRoutingData

    for split in ("train", "val", "test"):
        g = GroupedRoutingData.load(data_dir / split)
        g.utility = rebuild_grouped_rewards(g.performance, g.cost, lambda_cost=lambda_cost)
        g.save(data_dir / split)


def run_subprocess(cmd: list[str], *, lb_root: Path, gpu: str | None) -> None:
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    print(">>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(lb_root), env=env, check=True)


def train_graphrouter(
    data_dir: Path,
    out_dir: Path,
    *,
    seed: int,
    epochs: int,
    lb_root: Path,
    gpu: str | None,
    embed_path: Path | None,
) -> Path:
    run_dir = out_dir / "runs" / f"graphrouter_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "baselines/spo/train_graphrouter_lite.py",
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(run_dir),
        "--seed",
        str(seed),
        "--train-epoch",
        str(epochs),
        "--spo-loss-weight",
        "0.0",
        "--bce-warmup-epochs",
        "0",
    ]
    if embed_path and embed_path.exists():
        cmd.extend(["--embed-path", str(embed_path)])
    run_subprocess(cmd, lb_root=lb_root, gpu=gpu)
    return run_dir


def train_embedllm(
    data_dir: Path,
    out_dir: Path,
    *,
    seed: int,
    epochs: int,
    lb_root: Path,
    gpu: str | None,
    embed_path: Path | None,
) -> Path:
    run_dir = out_dir / "runs" / f"embedllm_bce_grouped_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "baselines/spo/train_embedllm.py",
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(run_dir),
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--objective",
        "bce_grouped",
        "--aligned",
        "--baseline-name",
        "EmbedLLM",
        "--device",
        "cuda:0",
    ]
    if embed_path and embed_path.exists():
        cmd.extend(["--embed-path", str(embed_path)])
    run_subprocess(cmd, lb_root=lb_root, gpu=gpu)
    return run_dir


def train_routellm(
    data_dir: Path,
    out_dir: Path,
    *,
    seed: int,
    epochs: int,
    lb_root: Path,
    gpu: str | None,
    embed_path: Path | None,
) -> Path:
    """RouteLLM aligned proxy: same K-head trainer as EmbedLLM (LLMRouterBench spo convention)."""
    run_dir = out_dir / "runs" / f"routellm_bce_grouped_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "baselines/spo/train_embedllm.py",
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(run_dir),
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--objective",
        "bce_grouped",
        "--aligned",
        "--baseline-name",
        "RouteLLM-proxy",
        "--device",
        "cuda:0",
    ]
    if embed_path and embed_path.exists():
        cmd.extend(["--embed-path", str(embed_path)])
    run_subprocess(cmd, lb_root=lb_root, gpu=gpu)
    return run_dir


def train_hybridllm(
    data_dir: Path,
    out_dir: Path,
    *,
    seed: int,
    epochs: int,
    lb_root: Path,
    gpu: str | None,
) -> Path:
    run_dir = out_dir / "runs" / f"hybridllm_lite_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "baselines/spo/train_hybridllm_lite.py",
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(run_dir),
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--spo-loss-weight",
        "0.0",
        "--loss-mode",
        "additive",
        "--batch-size",
        "4",
        "--device",
        "cuda:0",
    ]
    run_subprocess(cmd, lb_root=lb_root, gpu=gpu)
    return run_dir


def train_routerdc(
    data_dir: Path,
    out_dir: Path,
    *,
    seed: int,
    epochs: int,
    lb_root: Path,
    gpu: str | None,
) -> Path:
    run_dir = out_dir / "runs" / f"routerdc_lite_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "baselines/spo/train_routerdc_lite.py",
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(run_dir),
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--spo-loss-weight",
        "0.0",
        "--device",
        "cuda:0",
    ]
    run_subprocess(cmd, lb_root=lb_root, gpu=gpu)
    return run_dir


def export_graphrouter_choices(
    data_dir: Path,
    run_dir: Path,
    *,
    seed: int,
    device: str,
    embed_path: Path | None,
    **_,
) -> np.ndarray:
    import torch

    lb = lb_root_from_run(run_dir)
    _ensure_lb_path(lb)
    from baselines.spo.embeddings import load_or_build_query_embeddings
    from baselines.spo.train_embedllm import _build_prompt_id_map
    from baselines.spo.train_graphrouter_lite import (
        build_unified_graph,
        predict_test_utilities,
    )
    from baselines.spo_data import GroupedRoutingData

    gr_dir = lb / "baselines" / "GraphRouter" / "model"
    if str(gr_dir) not in sys.path:
        sys.path.insert(0, str(gr_dir))
    from graph_nn import EncoderDecoderNet  # noqa: E402

    splits = {
        "train": GroupedRoutingData.load(data_dir / "train"),
        "val": GroupedRoutingData.load(data_dir / "val"),
        "test": GroupedRoutingData.load(data_dir / "test"),
    }
    k = len(splits["train"].model_names)
    all_texts = sum([g.prompt_texts for g in splits.values()], [])
    text_to_pid = _build_prompt_id_map(all_texts)
    ep = embed_path or data_dir / "question_embeddings.pth"
    q_all = load_or_build_query_embeddings(
        sorted(text_to_pid.keys()),
        embed_path=str(ep) if Path(ep).exists() else None,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    llm_dim = 64
    llm_emb = rng.normal(size=(k, llm_dim)).astype(np.float32)
    task_emb = rng.normal(size=(1, llm_dim)).astype(np.float32)
    graph = build_unified_graph(splits, q_all, text_to_pid, llm_emb, task_emb, device)
    model = EncoderDecoderNet(
        query_feature_dim=q_all.shape[1],
        llm_feature_dim=llm_dim,
        hidden_features=llm_dim,
        in_edges=3,
    ).to(device)
    ckpt = run_dir / "best_gnn.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"GraphRouter checkpoint missing: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    pred = predict_test_utilities(model, graph)
    return _masked_argmax(pred, splits["test"].mask)


def export_embedllm_choices(
    data_dir: Path,
    run_dir: Path,
    *,
    seed: int,
    device: str,
    embed_path: Path | None,
    **_,
) -> np.ndarray:
    import torch
    from baselines.spo.embeddings import load_or_build_query_embeddings
    from baselines.spo.models import QueryKHeadRouter
    from baselines.spo.train_embedllm import _assign_ids, _build_prompt_id_map, predict_utilities
    from baselines.spo_data import GroupedRoutingData

    train_g = GroupedRoutingData.load(data_dir / "train")
    val_g = GroupedRoutingData.load(data_dir / "val")
    test_g = GroupedRoutingData.load(data_dir / "test")
    all_texts = train_g.prompt_texts + val_g.prompt_texts + test_g.prompt_texts
    text_to_pid = _build_prompt_id_map(all_texts)
    ep = embed_path or data_dir / "question_embeddings.pth"
    q_all = load_or_build_query_embeddings(
        sorted(text_to_pid.keys()),
        embed_path=str(ep) if Path(ep).exists() else None,
        seed=seed,
    )
    test_ids = _assign_ids(test_g, text_to_pid)
    model = QueryKHeadRouter(
        q_all,
        num_models=len(train_g.model_names),
        model_dim=64,
        text_dim=int(q_all.shape[1]),
        alpha_noise=0.0,
    )
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        alt = next(run_dir.glob("**/best.pt"), None)
        if alt is None:
            raise FileNotFoundError(f"EmbedLLM checkpoint missing under {run_dir}")
        ckpt_path = alt
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    pred = predict_utilities(model, test_g, test_ids, torch.device(device))
    return _masked_argmax(pred, test_g.mask)


def export_routerdc_choices(
    data_dir: Path,
    run_dir: Path,
    *,
    device: str,
    batch_size: int = 16,
    max_len: int = 256,
    **_,
) -> np.ndarray:
    import torch
    from baselines.spo.routerdc_module import RouterModule
    from baselines.spo.train_routerdc_lite import predict_scores
    from baselines.spo_data import GroupedRoutingData
    from transformers import AutoModel, AutoTokenizer

    test_g = GroupedRoutingData.load(data_dir / "test")
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"RouterDC checkpoint missing: {ckpt_path}")
    model_name = "cross-encoder/nli-deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    backbone = AutoModel.from_pretrained(model_name)
    hidden = backbone.config.hidden_size
    k = len(test_g.model_names)
    router = RouterModule(backbone, hidden_state_dim=hidden, node_size=k, similarity_function="cos")
    router.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    router.to(device)
    pred = predict_scores(router, tokenizer, test_g, device, batch_size, max_len)
    return _masked_argmax(pred, test_g.mask)


def export_hybridllm_choices(
    data_dir: Path,
    run_dir: Path,
    *,
    device: str,
    batch_size: int = 8,
    max_len: int = 256,
    **_,
) -> np.ndarray:
    import torch
    from baselines.spo.train_hybridllm_lite import HybridLLMRouter, predict_scores
    from baselines.spo_data import GroupedRoutingData
    from transformers import AutoModel, AutoTokenizer

    test_g = GroupedRoutingData.load(data_dir / "test")
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"HybridLLM-lite checkpoint missing: {ckpt_path}")
    model_name = "cross-encoder/nli-deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    backbone = AutoModel.from_pretrained(model_name)
    k = len(test_g.model_names)
    router = HybridLLMRouter(backbone, backbone.config.hidden_size, k)
    router.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    router.to(device)
    pred = predict_scores(router, tokenizer, test_g, device, batch_size, max_len)
    return _masked_argmax(pred, test_g.mask)


EXPORTERS: dict[str, Callable[..., np.ndarray]] = {
    METHOD_GRAPHROUTER: export_graphrouter_choices,
    METHOD_EMBEDLLM: export_embedllm_choices,
    METHOD_ROUTERDC: export_routerdc_choices,
    METHOD_ROUTELLM: export_embedllm_choices,
    METHOD_HYBRIDLLM: export_hybridllm_choices,
}

TRAINERS: dict[str, Callable[..., Path]] = {
    METHOD_GRAPHROUTER: train_graphrouter,
    METHOD_EMBEDLLM: train_embedllm,
    METHOD_ROUTERDC: train_routerdc,
    METHOD_ROUTELLM: train_routellm,
    METHOD_HYBRIDLLM: train_hybridllm,
}


def run_method(
    method: str,
    *,
    data_dir: Path,
    out_root: Path,
    seed: int,
    epochs: int,
    lb_root: Path,
    gpu: str | None,
    train: bool,
    export_only: bool,
    lambda_train: float,
) -> dict:
    seed_dir = out_root / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    choices_path = seed_dir / f"{method.replace(' ', '_')}_test_choices.npy"
    meta_path = seed_dir / f"{method.replace(' ', '_')}_meta.json"
    embed_path = data_dir / "question_embeddings.pth"
    device = "cuda:0" if gpu is not None else ("cuda" if __import__("torch").cuda.is_available() else "cpu")

    run_dir: Path | None = None
    if train and not export_only:
        if lambda_train != DEFAULT_LAMBDA_TRAIN:
            print(
                f"  note: training utilities should match λ={lambda_train}; "
                f"flagship config uses λ={DEFAULT_LAMBDA_TRAIN}",
                flush=True,
            )
        train_kwargs: dict = {
            "seed": seed,
            "epochs": epochs,
            "lb_root": lb_root,
            "gpu": gpu,
        }
        if method in (METHOD_GRAPHROUTER, METHOD_EMBEDLLM, METHOD_ROUTELLM):
            train_kwargs["embed_path"] = embed_path
        run_dir = TRAINERS[method](data_dir, seed_dir, **train_kwargs)
    else:
        runs = seed_dir / "runs"
        run_subdirs = {
            METHOD_GRAPHROUTER: f"graphrouter_seed{seed}",
            METHOD_EMBEDLLM: f"embedllm_bce_grouped_seed{seed}",
            METHOD_ROUTERDC: f"routerdc_lite_seed{seed}",
            METHOD_ROUTELLM: f"routellm_bce_grouped_seed{seed}",
            METHOD_HYBRIDLLM: f"hybridllm_lite_seed{seed}",
        }
        run_dir = runs / run_subdirs[method]

    if not choices_path.exists() or train:
        _ensure_lb_path(lb_root)
        chosen = EXPORTERS[method](
            data_dir,
            run_dir,
            seed=seed,
            device=device,
            embed_path=embed_path
            if method in (METHOD_GRAPHROUTER, METHOD_EMBEDLLM, METHOD_ROUTELLM)
            else None,
        )
        np.save(choices_path, chosen)
    else:
        chosen = np.load(choices_path)

    meta = {
        "method": method,
        "seed": seed,
        "data_dir": str(data_dir),
        "run_dir": str(run_dir) if run_dir else None,
        "choices_path": str(choices_path),
        "test_n": int(len(chosen)),
        "lambda_train": lambda_train,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_sota_routes(cache_dir: Path, seed: int, methods: list[str] | None = None) -> dict[str, np.ndarray] | None:
    methods = methods or SOTA_METHODS
    out: dict[str, np.ndarray] = {}
    seed_dir = cache_dir / f"seed{seed}"
    for m in methods:
        p = seed_dir / f"{m.replace(' ', '_')}_test_choices.npy"
        if not p.exists():
            return None
        out[m] = np.load(p)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="LLMRouterBench SOTA on DuoRoute flagship 851 pool")
    p.add_argument("--llmbench-root", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--methods", nargs="+", default=SOTA_METHODS, choices=SOTA_METHODS)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--gpu", default="0")
    p.add_argument("--lambda-train", type=float, default=DEFAULT_LAMBDA_TRAIN)
    p.add_argument("--patch-utility", action="store_true", help="Rewrite grouped utility to λ before train")
    p.add_argument("--export-only", action="store_true", help="Skip training; export choices from checkpoints")
    p.add_argument(
        "--quick",
        action="store_true",
        help="seed 42, 3 epochs, all selected methods (smoke)",
    )
    p.add_argument("--skip-routerdc", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.seeds = [42]
        args.epochs = 3

    if args.skip_routerdc and METHOD_ROUTERDC in args.methods:
        args.methods = [m for m in args.methods if m != METHOD_ROUTERDC]

    lb_root = _llmb_root(args.llmbench_root)
    data_dir = args.data_dir or flagship_data_dir(args.seeds[0])
    out_root = project_root() / args.output_dir if not args.output_dir.is_absolute() else args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    if args.patch_utility:
        _ensure_lb_path(lb_root)
        print(f"Patching utilities at λ={args.lambda_train} in {data_dir}...", flush=True)
        patch_split_utilities(data_dir, args.lambda_train)

    summary = {"seeds": args.seeds, "methods": args.methods, "epochs": args.epochs, "runs": []}
    for seed in args.seeds:
        print(f"\n=== LLMRouterBench SOTA seed={seed} ===", flush=True)
        for method in args.methods:
            print(f"  {method}...", flush=True)
            meta = run_method(
                method,
                data_dir=data_dir,
                out_root=out_root,
                seed=seed,
                epochs=args.epochs,
                lb_root=lb_root,
                gpu=args.gpu,
                train=not args.export_only,
                export_only=args.export_only,
                lambda_train=args.lambda_train,
            )
            summary["runs"].append(meta)

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
