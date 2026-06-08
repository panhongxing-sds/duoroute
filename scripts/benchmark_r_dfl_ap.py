#!/usr/bin/env python3
"""Benchmark RegretRouter on flagship N=851 (and optional RouterBench ablations)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from duoroute.pool_data import (  # noqa: E402
    DEFAULT_LAMBDA,
    enrich_pool_data,
    eval_row,
    load_flagship,
    load_routerbench,
    rebuild_utilities,
    routing_potential_bench,
)
from duoroute.regretrouter import (  # noqa: E402
    RDFLAPTrainConfig,
    infer_choices,
    make_router,
    routing_potential,
    train_regretrouter,
)
from duoroute.bench_metrics import best_single_idx_by_train
from duoroute.utils import project_root, set_seed

METHOD_ALIASES = {
    "AP-simple": "ap_simple",
    "S-DFL-MLP": "s_dfl_mlp",
    "RegretRouter-uniform": "regretrouter_uniform",
    "RegretRouter-APinit": "regretrouter_apinit",
    "RegretRouter-no-feedback": "regretrouter_no_feedback",
    "RegretRouter-shuffled-x": "regretrouter_shuffled_x",
    "RegretRouter-detach-x": "regretrouter_detach_x",
    "RegretRouter-K5-APinit": "regretrouter_k5_apinit",
}


def run_suite(data: dict, seed: int, out_dir: Path) -> tuple[list[dict], dict]:
    d = data["train_h"].shape[1]
    k = len(data["model_names"])
    cost = torch.tensor(data["cost"], dtype=torch.float32)
    th = torch.from_numpy(data["train_h"]).float()
    tu = torch.from_numpy(data["train_u"]).float()
    tm = torch.from_numpy(data["train_mask"]).bool()
    vh = torch.from_numpy(data["val_h"]).float()
    vu = torch.from_numpy(data["val_u"]).float()
    vm = torch.from_numpy(data["val_mask"]).bool()
    xh = torch.from_numpy(data["test_h"]).float()
    xm = torch.from_numpy(data["test_mask"]).bool()

    data = enrich_pool_data(data, seed, lambda_cost=DEFAULT_LAMBDA)
    bs_idx = best_single_idx_by_train(
        data["train_perf"],
        data["train_mask"],
        by="oracle_reward",
        train_cost=data["train_cost"],
        lambda_cost=DEFAULT_LAMBDA,
    )
    ap_train = data["ap_balance_train_idx"]
    ap_val = data["ap_balance_val_idx"]
    ap_test = data["ap_balance_test_idx"]

    pot = routing_potential(data["test_perf"], data["test_mask"], ap_test)
    pot.update(
        routing_potential_bench(
            data["test_u"],
            data["test_perf"],
            data["test_mask"],
            ap_test,
            best_single_idx=bs_idx,
        )
    )
    pot["dataset"] = data["name"]

    base = RDFLAPTrainConfig(seed=seed, epochs=28, ap_init_eps=0.05, hidden_dim=128, temperature=1.0)

    experiments: list[tuple[str, str, dict]] = [
        ("AP-simple", "ap", {}),
        ("S-DFL-MLP", "sdf", {"init_mode": "apinit"}),
        ("RegretRouter-uniform", "rdfl", {"K": 3, "init_mode": "uniform", "feedback_mode": "full"}),
        ("RegretRouter-APinit", "rdfl", {"K": 3, "init_mode": "apinit", "feedback_mode": "full"}),
        ("RegretRouter-no-feedback", "rdfl", {"K": 3, "init_mode": "apinit", "feedback_mode": "none"}),
        ("RegretRouter-shuffled-x", "rdfl", {"K": 3, "init_mode": "apinit", "feedback_mode": "shuffle"}),
        ("RegretRouter-detach-x", "rdfl", {"K": 3, "init_mode": "apinit", "feedback_mode": "detach"}),
        ("RegretRouter-K5-APinit", "rdfl", {"K": 5, "init_mode": "apinit", "feedback_mode": "full"}),
    ]

    rows: list[dict] = []
    ap_t = torch.from_numpy(ap_train).long()
    ap_v = torch.from_numpy(ap_val).long()
    ap_te = torch.from_numpy(ap_test).long()

    for name, kind, kw in experiments:
        method_id = METHOD_ALIASES.get(name, name)
        if kind == "ap":
            chosen = ap_test
            row = {
                "method": name,
                "method_id": method_id,
                "dataset": data["name"],
                **eval_row(
                    data["test_perf"],
                    data["test_u"],
                    data["test_mask"],
                    data["test_cost"],
                    ap_test,
                    chosen,
                    best_single_idx=bs_idx,
                ),
            }
        else:
            cfg = RDFLAPTrainConfig(
                seed=seed,
                epochs=base.epochs,
                ap_init_eps=base.ap_init_eps,
                hidden_dim=base.hidden_dim,
                temperature=base.temperature,
                K=kw.get("K", 3),
                init_mode=kw.get("init_mode", "apinit"),
                feedback_mode=kw.get("feedback_mode", "full"),
            )
            model = make_router(d, k, kind="sdf" if kind == "sdf" else "rdfl", cfg=cfg)
            train_regretrouter(model, th, tu, tm, ap_t, cost, vh, vu, vm, ap_v, cfg)
            chosen = infer_choices(model, xh, xm, cost, ap_te, cfg=cfg)
            row = {
                "method": name,
                "method_id": method_id,
                "dataset": data["name"],
                **eval_row(
                    data["test_perf"],
                    data["test_u"],
                    data["test_mask"],
                    data["test_cost"],
                    ap_test,
                    chosen,
                    best_single_idx=bs_idx,
                ),
            }
        rows.append(row)
        print(f"[{data['name']}] {name}: acc={row['avg_acc']:.4f} net={row['net_gain']:+d} ov={row['override_n']}")

    report = {"dataset": data["name"], "test_n": len(ap_test), "routing_potential": pot, "methods": rows}
    (out_dir / f"{data['name']}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return rows, pot


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="RegretRouter benchmark (flagship N=851)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="outputs/regretrouter")
    p.add_argument("--routerbench", action="store_true")
    p.add_argument(
        "--full-test",
        action="store_true",
        help="Use all four-dataset test rows (N≈851) instead of oracle-gap subset (N=502)",
    )
    args = p.parse_args()
    set_seed(args.seed)
    out = project_root() / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    pots: list[dict] = []
    if args.full_test:
        suites = [load_flagship(args.seed, filter_four=False)]
    else:
        suites = [
            load_flagship(args.seed, filter_four=True),
            load_flagship(args.seed, filter_four=False),
        ]
    for data in suites:
        rows_f, pot_f = run_suite(data, args.seed, out)
        all_rows.extend(rows_f)
        pots.append(pot_f)

    if args.routerbench:
        rows_r, pot_r = run_suite(load_routerbench(args.seed), args.seed, out)
        all_rows.extend(rows_r)
        pots.append(pot_r)

    md = [
        f"# RegretRouter benchmark (seed={args.seed})",
        "",
        "Method: **RegretRouter** = K=3 uniform init + final-step regret + masked softmax.",
        "",
        "Pools: flagship **N=502** (four-dataset filter) and **N=851** (full flagship test); RouterBench optional.",
        "",
        "## Routing potential (Acc_oracle − Acc_AP)",
        "",
        "| dataset | AP acc | Oracle acc | potential (pp) | rescueable n |",
        "|---------|-------:|-----------:|---------------:|-------------:|",
    ]
    for pot in pots:
        md.append(
            f"| {pot['dataset']} | {pot['ap_acc']:.4f} | {pot['oracle_acc']:.4f} | "
            f"{pot['potential_pp']:.2f} | {pot['n_rescueable']} |"
        )
    md.extend(
        [
            "",
            "## Main + ablation (vs AP)",
            "",
            "| dataset | method_id | acc | Δpp | net | rescued | harmed | overrides |",
            "|---------|-----------|----:|----:|----:|--------:|-------:|----------:|",
        ]
    )
    for r in all_rows:
        md.append(
            f"| {r['dataset']} | {r['method_id']} | {r['avg_acc']:.4f} | {r['lift_pp']:+.2f} | "
            f"**{r['net_gain']:+d}** | {r['rescued']} | {r['harmed']} | {r['override_n']} |"
        )
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    (out / "report.json").write_text(json.dumps({"rows": all_rows, "potentials": pots}, indent=2), encoding="utf-8")
    print(f"\nWrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
