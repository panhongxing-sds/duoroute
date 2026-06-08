#!/usr/bin/env python3
"""Mechanism ablations: remove one in-method component (same pipeline, λ=0.2, N=851)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from duoroute.main_table import (
    _train_rdfl,
    aggregate_rows,
    eval_at_lambda,
    fmt_mean_std,
    load_pool_851,
)
from duoroute.pool_data import DEFAULT_LAMBDA, enrich_pool_data, rebuild_utilities
from duoroute.regretrouter import RDFLAPTrainConfig
from duoroute.utils import set_seed

AP_ANCHOR_METHOD = "Official-AvengersPro-balance"

OURS_CFG = dict(
    router_version="v1",
    predictor_mode="concat",
    K=3,
    init_mode="uniform",
    feedback_mode="full",
    state_mode="evolving",
    loss_mode="regret",
    step_loss=False,
    weight_decay=1e-5,
    temperature=1.0,
)

MECHANISM_ABLATIONS: list[tuple[str, str, dict]] = [
    ("Ours", "ours", OURS_CFG),
    ("w/o feedback", "wo_feedback", {**OURS_CFG, "feedback_mode": "none"}),
    ("w/o query", "wo_query", {**OURS_CFG, "feedback_mode": "query_none"}),
    ("w/o regret (CE)", "wo_regret_ce", {**OURS_CFG, "loss_mode": "ce"}),
    ("w/o iterative state", "wo_iter_state", {**OURS_CFG, "state_mode": "frozen_init"}),
]


def run_row(data: dict, seed: int, epochs: int, label: str, cfg_kwargs: dict) -> tuple[str, object]:
    base = enrich_pool_data(rebuild_utilities(data, lambda_cost=DEFAULT_LAMBDA), seed, lambda_cost=DEFAULT_LAMBDA)
    if label == "AP-simple (ref.)":
        return "AP-simple", base["ap_balance_test_idx"]
    cfg = RDFLAPTrainConfig(seed=seed, epochs=epochs, **cfg_kwargs)
    return label, _train_rdfl(base, seed, cfg)


def main() -> None:
    p = argparse.ArgumentParser(description="RegretRouter mechanism ablations")
    p.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    p.add_argument("--epochs", type=int, default=28)
    p.add_argument("--output-dir", default="outputs/r_dfl_ap")
    args = p.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows: list[dict] = []

    for seed in args.seeds:
        set_seed(seed)
        data = load_pool_851(seed)
        print(f"\n=== seed={seed} ===", flush=True)
        label, chosen = run_row(data, seed, args.epochs, "AP-simple (ref.)", {})
        row = eval_at_lambda(data, label, chosen, lambda_eval=DEFAULT_LAMBDA, seed=seed)
        row["ablation"] = "AP-simple (ref.)"
        rows.append(row)

        for label, _tag, cfg_kwargs in MECHANISM_ABLATIONS:
            print(f"  {label}...", flush=True)
            method, chosen = run_row(data, seed, args.epochs, label, cfg_kwargs)
            row = eval_at_lambda(data, method, chosen, lambda_eval=DEFAULT_LAMBDA, seed=seed)
            row["ablation"] = label
            rows.append(row)

    agg = aggregate_rows(rows, ["ablation", "method"])
    order = ["AP-simple (ref.)"] + [a[0] for a in MECHANISM_ABLATIONS]
    order_map = {n: i for i, n in enumerate(order)}
    agg.sort(key=lambda a: order_map.get(a.get("ablation", ""), 99))

    lines = [
        f"# Mechanism Ablations (851, λ={DEFAULT_LAMBDA}, Δ vs {AP_ANCHOR_METHOD})",
        "",
        "Remove one in-method component; Ours = RegretRouter: K=3, uniform x0, full feedback, final-step regret.",
        "w/o regret (CE) = same architecture, CE on argmax(oracle_reward) instead of regret.",
        f"Seeds: {args.seeds}.",
        "",
        "| Ablation | Acc | Gain@B | Gap@O | AvgCost | ΔAcc(pp) | ΔGap@O |",
        "|----------|----:|-------:|------:|--------:|---------:|--------:|",
    ]
    for a in agg:
        lines.append(
            f"| {a.get('ablation', a['method'])} | "
            f"{fmt_mean_std(a['avg_acc'], a.get('avg_acc_std', 0))} | "
            f"{fmt_mean_std(a['gain_at_best_single'], a.get('gain_at_best_single_std', 0))} | "
            f"{fmt_mean_std(a['gap_at_oracle'], a.get('gap_at_oracle_std', 0))} | "
            f"{fmt_mean_std(a['avg_cost'], a.get('avg_cost_std', 0), nd=6)} | "
            f"{fmt_mean_std(a.get('delta_acc_pp', 0), a.get('delta_acc_pp_std', 0), nd=2)} | "
            f"{fmt_mean_std(a.get('delta_gap_at_oracle', 0), a.get('delta_gap_at_oracle_std', 0))} |"
        )

    payload = {
        "lambda": DEFAULT_LAMBDA,
        "pool": "851",
        "seeds": args.seeds,
        "epochs": args.epochs,
        "ablations": [{"label": a[0], "tag": a[1], "cfg": a[2]} for a in MECHANISM_ABLATIONS],
        "rows": rows,
        "aggregated": agg,
        "elapsed_sec": time.time() - t0,
    }
    (out_dir / "mechanism_ablation_table.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "mechanism_ablation_table.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nDone in {payload['elapsed_sec']:.0f}s -> {out_dir / 'mechanism_ablation_table.md'}")


if __name__ == "__main__":
    main()
