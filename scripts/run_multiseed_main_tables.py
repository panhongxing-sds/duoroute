#!/usr/bin/env python3
"""Multi-seed (41/42/43) main tables: RegretRouter one-shot + RegretRouter + Cascade.

Outputs:
  outputs/cascade/TABLES_DATA.json   — per-seed rows + 3-seed mean/std
  outputs/cascade/TABLES_LATEX.tex    — ICLR-style tables (3-seed mean)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from duoroute.bench_metrics import (  # noqa: E402
    best_single_idx_by_train,
    compute_cascade_paper_metrics,
    pareto_distance,
    pareto_frontier_points,
    summarize_method_frontier,
)
from duoroute.main_table import (
    DEFAULT_LAMBDAS,
    aggregate_rows,
    best_single_idx,
    eval_at_lambda,
    fmt_mean_std,
    load_pool_851,
    _train_rdfl,
)
from duoroute.pool_data import DEFAULT_LAMBDA, enrich_pool_data, per_dataset_breakdown, rebuild_utilities
from duoroute.regretrouter import CASCADE_METHOD_NAME, OURS_METHOD_NAME, RDFLAPTrainConfig, infer_choices
from duoroute.rdf_query_model_selector import (  # noqa: E402
    QueryModelRerankTrainConfig,
    QueryModelSelectorTrainConfig,
    build_query_model_reranker,
    build_query_model_selector,
    infer_query_model_reranker,
    infer_query_model_selector,
    train_query_model_reranker_wrong_only,
    train_query_model_selector_wrong_only,
)
from duoroute.rdf_vg_cascade import (  # noqa: E402
    apply_gate_selector,
    eval_vg_cascade,
    perfect_verifier_gate_np,
    train_stage1_rdfl,
)
from duoroute.utils import set_seed


STAGE1_CFG = dict(
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

METHOD_ONESHOT = OURS_METHOD_NAME
METHOD_CASCADE = CASCADE_METHOD_NAME
CASCADE_ALIAS = f"{CASCADE_METHOD_NAME} (Ours)"
SELECTOR_BETA = 0.0  # best on seed42 matrix; ranking betas tie at β=0.05

METRIC_FIELDS = [
    "avg_acc",
    "gain_at_random",
    "gain_at_best_single",
    "gap_at_oracle",
    "regret_at_oracle",
    "avg_cost",
]

OUTPUT_DIR = ROOT / "outputs" / "cascade"
CACHE_DIR = OUTPUT_DIR / "multiseed"


def _prepare_data(seed: int, lambda_cost: float = DEFAULT_LAMBDA) -> dict:
    raw = load_pool_851(seed)
    return enrich_pool_data(
        rebuild_utilities(raw, lambda_cost=lambda_cost), seed, lambda_cost=lambda_cost,
    )


def _infer_m1_all(stage1: torch.nn.Module, base: dict, stage1_cfg: RDFLAPTrainConfig) -> dict[str, np.ndarray]:
    cost = torch.tensor(base["cost"], dtype=torch.float32)
    out: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        out[split] = infer_choices(
            stage1,
            torch.from_numpy(base[f"{split}_h"]).float(),
            torch.from_numpy(base[f"{split}_mask"]).bool(),
            cost,
            torch.from_numpy(base[f"ap_balance_{split}_idx"]).long(),
            cfg=stage1_cfg,
        )
    return out


def _cache_path(seed: int, name: str) -> Path:
    d = CACHE_DIR / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def _cascade_cache_path(seed: int, lam: float | None) -> Path:
    name = f"cascade_lam{lam}_routing.npz" if lam is not None else "cascade_routing.npz"
    return _cache_path(seed, name)


def _oneshot_cache_path(seed: int, lam: float | None) -> Path:
    name = f"oneshot_lam{lam}_chosen.npy" if lam is not None else "oneshot_chosen.npy"
    return _cache_path(seed, name)


def _save_routing_cache(
    seed: int,
    m1: dict[str, np.ndarray],
    m_final: np.ndarray,
    pv_gate: np.ndarray,
    *,
    lam: float | None = None,
) -> None:
    np.savez(
        _cascade_cache_path(seed, lam),
        m1_train=m1["train"],
        m1_val=m1["val"],
        m1_test=m1["test"],
        m_final=m_final,
        pv_gate=pv_gate,
    )


def _load_routing_cache(seed: int, lam: float | None = None) -> dict[str, np.ndarray] | None:
    p = _cascade_cache_path(seed, lam)
    if not p.exists():
        return None
    z = np.load(p)
    return {
        "m1": {"train": z["m1_train"], "val": z["m1_val"], "test": z["m1_test"]},
        "m_final": z["m_final"],
        "pv_gate": z["pv_gate"],
    }


def _save_oneshot_cache(seed: int, chosen: np.ndarray, *, lam: float | None = None) -> None:
    np.save(_oneshot_cache_path(seed, lam), chosen)


def _load_oneshot_cache(seed: int, lam: float | None = None) -> np.ndarray | None:
    p = _oneshot_cache_path(seed, lam)
    return np.load(p) if p.exists() else None


def _train_cascade_routing(
    base: dict,
    seed: int,
    epochs: int,
    lam: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    stage1_cfg = RDFLAPTrainConfig(seed=seed, epochs=epochs, **STAGE1_CFG)
    stage1, stage1_cfg = train_stage1_rdfl(base, seed, stage1_cfg)
    m1 = _infer_m1_all(stage1, base, stage1_cfg)
    m1_test = m1["test"]

    selector = build_query_model_selector(base, seed=seed, use_cost=True, use_m1=True)
    scfg = QueryModelSelectorTrainConfig(
        seed=seed, epochs=epochs, lambda_cost=lam, ranking_beta=SELECTOR_BETA,
    )
    selector = train_query_model_selector_wrong_only(
        selector,
        torch.from_numpy(base["train_h"]).float(),
        torch.from_numpy(base["train_perf"]).float(),
        torch.from_numpy(base["train_cost"]).float(),
        torch.from_numpy(base["train_mask"]).bool(),
        torch.from_numpy(base["val_h"]).float(),
        torch.from_numpy(base["val_perf"]).float(),
        torch.from_numpy(base["val_cost"]).float(),
        torch.from_numpy(base["val_mask"]).bool(),
        scfg,
        train_m1=torch.from_numpy(m1["train"]).long(),
        val_m1=torch.from_numpy(m1["val"]).long(),
    )
    reranker = build_query_model_reranker(base, seed=seed, top_k=5)
    rcfg = QueryModelRerankTrainConfig(seed=seed, lambda_cost=lam, epochs=epochs)
    reranker = train_query_model_reranker_wrong_only(
        reranker,
        selector,
        torch.from_numpy(base["train_h"]).float(),
        torch.from_numpy(base["train_perf"]).float(),
        torch.from_numpy(base["train_cost"]).float(),
        torch.from_numpy(base["train_mask"]).bool(),
        torch.from_numpy(base["val_h"]).float(),
        torch.from_numpy(base["val_perf"]).float(),
        torch.from_numpy(base["val_cost"]).float(),
        torch.from_numpy(base["val_mask"]).bool(),
        rcfg,
        train_m1=torch.from_numpy(m1["train"]).long(),
        val_m1=torch.from_numpy(m1["val"]).long(),
    )
    m2, _ = infer_query_model_reranker(
        reranker,
        selector,
        torch.from_numpy(base["test_h"]).float(),
        torch.from_numpy(base["test_mask"]).bool(),
        torch.from_numpy(m1_test).long(),
    )
    pv_gate = perfect_verifier_gate_np(base["test_perf"], m1_test)
    m_final = apply_gate_selector(m1_test, pv_gate, m2)
    return m1, m1_test, m_final, pv_gate


def run_oneshot_seed(
    seed: int,
    epochs: int,
    lambdas: list[float],
    *,
    reuse: bool = False,
    per_lambda_retrain: bool = False,
) -> list[dict]:
    rows: list[dict] = []
    if per_lambda_retrain:
        for lam in lambdas:
            set_seed(seed)
            base = _prepare_data(seed, lambda_cost=lam)
            chosen = _load_oneshot_cache(seed, lam) if reuse else None
            if chosen is None:
                cfg = RDFLAPTrainConfig(seed=seed, epochs=epochs, **STAGE1_CFG)
                chosen = _train_rdfl(base, seed, cfg)
                _save_oneshot_cache(seed, chosen, lam=lam)
            row = eval_at_lambda(base, METHOD_ONESHOT, chosen, lambda_eval=lam, seed=seed)
            row["method"] = METHOD_ONESHOT
            row["lambda_train"] = lam
            rows.append(row)
        return rows

    set_seed(seed)
    base = _prepare_data(seed)
    chosen = _load_oneshot_cache(seed) if reuse else None
    if chosen is None:
        cfg = RDFLAPTrainConfig(seed=seed, epochs=epochs, **STAGE1_CFG)
        chosen = _train_rdfl(base, seed, cfg)
        _save_oneshot_cache(seed, chosen)

    for lam in lambdas:
        row = eval_at_lambda(base, METHOD_ONESHOT, chosen, lambda_eval=lam, seed=seed)
        row["method"] = METHOD_ONESHOT
        row["lambda_train"] = DEFAULT_LAMBDA
        rows.append(row)
    return rows


def _cascade_eval_row(
    base: dict,
    m1_test: np.ndarray,
    m_final: np.ndarray,
    pv_gate: np.ndarray,
    *,
    lam: float,
    seed: int,
    lambda_train: float,
) -> dict:
    d = rebuild_utilities(base, lambda_cost=lam)
    paper = compute_cascade_paper_metrics(
        performance=d["test_perf"],
        cost=d["test_cost"],
        mask=d["test_mask"],
        m1=m1_test,
        m_final=m_final,
        reroute=pv_gate,
        lambda_cost=lam,
        random_seed=seed,
    )
    cas = eval_vg_cascade(
        d["test_perf"],
        d["test_cost"],
        d["test_mask"],
        m1_test,
        m_final,
        pv_gate,
        lambda_cost=lam,
        test_u=d["test_u"],
        rho_pct=100.0,
    )
    return {
        "method": METHOD_CASCADE,
        "lambda": lam,
        "lambda_train": lambda_train,
        "pool": "851",
        "seed": seed,
        "avg_acc": paper["avg_acc"],
        "gain_at_random": paper["gain_at_random"],
        "gain_at_best_single": paper["gain_at_best_single"],
        "gap_at_oracle": paper["gap_at_oracle"],
        "regret_at_oracle": paper["regret_at_oracle"],
        "regret_at_oracle_cascade": paper["regret_at_oracle_cascade"],
        "avg_cost": paper["avg_cost"],
        "avg_utility": cas["avg_utility"],
        "reroute_rate": cas["reroute_rate"],
        "gap_at_oracle_legacy": cas.get("gap_at_oracle"),
    }


def run_cascade_seed(
    seed: int,
    epochs: int,
    lambdas: list[float],
    *,
    reuse_stage1: bool = False,
    per_lambda_retrain: bool = False,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    rows: list[dict] = []
    last_routing: dict[str, np.ndarray] = {}

    if per_lambda_retrain:
        for lam in lambdas:
            set_seed(seed)
            base = _prepare_data(seed, lambda_cost=lam)
            cached = _load_routing_cache(seed, lam) if reuse_stage1 else None
            if cached is not None:
                m1_test = cached["m1"]["test"]
                m_final = cached["m_final"]
                pv_gate = cached["pv_gate"]
            else:
                m1, m1_test, m_final, pv_gate = _train_cascade_routing(base, seed, epochs, lam)
                _save_routing_cache(seed, m1, m_final, pv_gate, lam=lam)
            rows.append(_cascade_eval_row(
                base, m1_test, m_final, pv_gate, lam=lam, seed=seed, lambda_train=lam,
            ))
            last_routing = {"m1_test": m1_test, "m_final": m_final, "pv_gate": pv_gate}
        return rows, last_routing

    set_seed(seed)
    base = _prepare_data(seed)
    cached = _load_routing_cache(seed) if reuse_stage1 else None

    if cached is not None:
        m1_test = cached["m1"]["test"]
        m_final = cached["m_final"]
        pv_gate = cached["pv_gate"]
    else:
        m1, m1_test, m_final, pv_gate = _train_cascade_routing(base, seed, epochs, DEFAULT_LAMBDA)
        _save_routing_cache(seed, m1, m_final, pv_gate)

    for lam in lambdas:
        rows.append(_cascade_eval_row(
            base, m1_test, m_final, pv_gate, lam=lam, seed=seed, lambda_train=DEFAULT_LAMBDA,
        ))
    return rows, {"m1_test": m1_test, "m_final": m_final, "pv_gate": pv_gate}


def per_dataset_gap_oneshot(data: dict, chosen: np.ndarray, seed: int) -> list[dict]:
    data = rebuild_utilities(data, lambda_cost=DEFAULT_LAMBDA)
    data["best_single_idx"] = best_single_idx(data, DEFAULT_LAMBDA)
    ap = data["ap_balance_test_idx"]
    out: list[dict] = []
    for drow in per_dataset_breakdown(data, chosen, ap):
        out.append(
            {
                "seed": seed,
                "method": METHOD_ONESHOT,
                "dataset_id": drow["dataset_id"],
                "n": drow["n"],
                "gap_at_oracle": drow["gap_at_oracle"],
            }
        )
    return out


def per_dataset_gap_cascade(
    data: dict,
    m1_test: np.ndarray,
    m_final: np.ndarray,
    pv_gate: np.ndarray,
    seed: int,
) -> list[dict]:
    data = rebuild_utilities(data, lambda_cost=DEFAULT_LAMBDA)
    ds = data.get("test_dataset_ids")
    if ds is None:
        return []
    ds = np.asarray(ds)
    test_u = data["test_u"]
    out: list[dict] = []
    for d in sorted(set(ds.tolist())):
        m = ds == d
        if not m.any():
            continue
        sub_u = test_u[m]
        sub_mask = data["test_mask"][m]
        sub_m1 = m1_test[m]
        sub_final = m_final[m]
        sub_gate = pv_gate[m]
        sub_perf = data["test_perf"][m]
        sub_cost = data["test_cost"][m]
        cas = eval_vg_cascade(
            sub_perf,
            sub_cost,
            sub_mask,
            sub_m1,
            sub_final,
            sub_gate,
            lambda_cost=DEFAULT_LAMBDA,
            test_u=sub_u,
            rho_pct=100.0,
        )
        out.append(
            {
                "seed": seed,
                "method": METHOD_CASCADE,
                "dataset_id": d,
                "n": int(m.sum()),
                "gap_at_oracle": cas["gap_at_oracle"],
            }
        )
    return out


def _mean_std(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"mean": float("nan"), "std": 0.0}
    return {"mean": mean(vals), "std": pstdev(vals) if len(vals) > 1 else 0.0}


def aggregate_method_lambda(rows: list[dict]) -> list[dict]:
    return aggregate_rows(rows, ["method", "lambda"])


def compute_value_metrics(
    oneshot_rows: list[dict],
    cascade_rows: list[dict],
    seed: int,
) -> dict[str, dict]:
    """PerfGain / CostSave / ParetoDist at λ=0.2 using λ-sweep as config points."""
    data = load_pool_851(seed, lambda_cost=DEFAULT_LAMBDA)
    _, bs_acc, bs_cost, _ = _bs_oracle_reward(data)

    def _points(rows: list[dict], method: str) -> list[tuple[float, float, float]]:
        pts = []
        for r in rows:
            if r["method"] != method or r["seed"] != seed:
                continue
            pts.append((float(r["avg_acc"]), float(r["avg_cost"]), float(r["lambda"])))
        return sorted(pts, key=lambda x: x[2])

    all_raw: list[tuple[str, float, float, float]] = []
    specs = [
        ("oneshot", METHOD_ONESHOT, _points(oneshot_rows, METHOD_ONESHOT)),
        ("cascade", METHOD_CASCADE, _points(cascade_rows, METHOD_CASCADE)),
    ]
    for sid, _, pts in specs:
        for acc, cost, w in pts:
            all_raw.append((sid, w, acc, cost))
    frontier = pareto_frontier_points([(p[3], p[2]) for p in all_raw])

    out: dict[str, dict] = {}
    for sid, method, pts in specs:
        configs = [(a, c, w) for a, c, w in pts]
        summary = summarize_method_frontier(
            configs, bs_acc=bs_acc, bs_cost=bs_cost, global_frontier=frontier,
        )
        primary = next(r for r in (oneshot_rows if sid == "oneshot" else cascade_rows)
                       if r["method"] == method and r["seed"] == seed and abs(r["lambda"] - DEFAULT_LAMBDA) < 1e-9)
        out[sid] = {
            "method": method,
            "perf_gain": summary.perf_gain,
            "cost_save": summary.cost_save,
            "pareto_dist_mean": summary.pareto_dist,
            "pareto_dist_primary": pareto_distance(
                float(primary["avg_cost"]), float(primary["avg_acc"]), frontier,
            ),
            "reroute_rate": primary.get("reroute_rate", 0.0),
            "primary_acc": float(primary["avg_acc"]),
            "primary_cost": float(primary["avg_cost"]),
            "primary_utility": float(primary["avg_utility"]),
            "gap_at_oracle": float(primary["gap_at_oracle"]),
        }
    return out


def _bs_oracle_reward(data: dict) -> tuple[int, float, float, str]:
    bs_idx = best_single_idx_by_train(
        data["train_perf"],
        data["train_mask"],
        by="oracle_reward",
        train_cost=data["train_cost"],
        lambda_cost=DEFAULT_LAMBDA,
    )
    acc = float(data["test_perf"][:, bs_idx].mean())
    mk = data["test_mask"][:, bs_idx]
    cost = float(data["test_cost"][mk, bs_idx].mean())
    return bs_idx, acc, cost, data["model_names"][bs_idx]


def _fmt_lam_cell(vals: dict, nd_gap: int = 3, nd_acc: int = 3) -> str:
    return (
        f"{vals['gap_at_oracle']:.{nd_gap}f} & {vals['avg_acc']:.{nd_acc}f} & "
        f"{vals['avg_cost']:.3f} & {vals['avg_utility']:.3f} & {vals['gain_at_best_single']:.3f}"
    )


def render_latex(
    agg: list[dict],
    value_agg: dict[str, dict],
    per_ds_table: list[dict],
    datasets: list[str],
    seeds: list[int],
) -> str:
    lam_cols = DEFAULT_LAMBDAS
    by_key: dict[tuple, dict] = {}
    for a in agg:
        by_key[(a["method"], a["lambda"])] = a

    def row_cells(method: str) -> str:
        parts = []
        for lam in lam_cols:
            a = by_key.get((method, lam))
            if a is None:
                parts.append("— & — & — & — & —")
            else:
                parts.append(_fmt_lam_cell(a))
        return " & ".join(parts)

    lines = [
        r"\begin{table*}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{2.0pt}",
        r"  \renewcommand{\arraystretch}{1.12}",
        r"  \caption{",
        r"    \textbf{Main Routing Performance (RegretRouter vs RegretRouter + Cascade).}",
        f"    Mean across seeds {'/'.join(str(s) for s in seeds)}.",
        r"    Evaluated on flagship pool ($N=851$).",
        r"    Cascade: PV gate + Query$\times$Model selector + top-5 pairwise rerank ($\rho=100\%$).",
        r"  }",
        r"  \label{tab:routing-main-broad}",
        r"  \resizebox{\textwidth}{!}{",
        r"  \begin{tabular}{l|ccccc|ccccc|ccccc|ccccc|ccccc}",
        r"    \toprule",
        r"    \rowcolor{headerpink}",
        r"    & \multicolumn{5}{c|}{$\lambda=0.0$}",
        r"    & \multicolumn{5}{c|}{$\lambda=0.1$}",
        r"    & \multicolumn{5}{c|}{$\lambda=0.2$}",
        r"    & \multicolumn{5}{c|}{$\lambda=0.5$}",
        r"    & \multicolumn{5}{c}{$\lambda=0.8$} \\",
        r"    \rowcolor{headerpink}",
        r"    Method",
        r"    & Gap@O $\downarrow$ & Acc $\uparrow$ & AvgCost & AvgUtility & Gain@B",
        r"    & Gap@O $\downarrow$ & Acc $\uparrow$ & AvgCost & AvgUtility & Gain@B",
        r"    & Gap@O $\downarrow$ & Acc $\uparrow$ & AvgCost & AvgUtility & Gain@B",
        r"    & Gap@O $\downarrow$ & Acc $\uparrow$ & AvgCost & AvgUtility & Gain@B",
        r"    & Gap@O $\downarrow$ & Acc $\uparrow$ & AvgCost & AvgUtility & Gain@B \\",
        r"    \midrule",
        f"    {METHOD_ONESHOT} & {row_cells(METHOD_ONESHOT)} \\\\",
        r"\rowcolor{rowgray}",
        f"    \\textbf{{{CASCADE_ALIAS}}} & {row_cells(METHOD_CASCADE)} \\\\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  }",
        r"\end{table*}",
        "",
        r"\begin{table*}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{4.0pt}",
        r"  \renewcommand{\arraystretch}{1.12}",
        r"  \caption{",
        r"    \textbf{Cost--Performance Value Metrics ($\lambda=0.2$, mean seeds "
        + "/".join(str(s) for s in seeds)
        + r").}",
        r"    PerfGain/CostSave vs performance-oriented Best Single; ParetoDist vs global $\lambda$-sweep frontier.",
        r"  }",
        r"  \label{tab:cascade-value}",
        r"  \begin{tabular}{l|cccccc|cc}",
        r"    \toprule",
        r"    \rowcolor{headerpink}",
        r"    Method & Acc & AvgUtility & AvgCost & PerfGain & CostSave & ParetoDist $\downarrow$ & RerouteRate & Gap@O \\",
        r"    \midrule",
    ]
    for key, label in [("oneshot", METHOD_ONESHOT), ("cascade", CASCADE_ALIAS)]:
        v = value_agg[key]
        bold = "\\textbf{" if key == "cascade" else ""
        bold_end = "}" if key == "cascade" else ""
        rr = f"{v['reroute_rate']:.3f}" if key == "cascade" else "0.000"
        lines.append(
            f"    {bold}{label}{bold_end} & {v['primary_acc']:.3f} & {v['primary_utility']:.3f} & "
            f"{v['primary_cost']:.3f} & {v['perf_gain']:+.3f} & {v['cost_save']:+.3f} & "
            f"{v['pareto_dist_mean']:.3f} & {rr} & {v['gap_at_oracle']:.3f} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table*}",
        "",
        r"\begin{table*}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{5.0pt}",
        r"  \renewcommand{\arraystretch}{1.12}",
        r"  \caption{",
        r"    \textbf{Per-Dataset Gap@Oracle ($\lambda=0.2$, mean seeds "
        + "/".join(str(s) for s in seeds)
        + r", lower is better).}",
        r"  }",
        r"  \label{tab:per-dataset-cascade}",
        r"  \resizebox{\textwidth}{!}{",
        r"  \begin{tabular}{l|" + "c" * len(datasets) + "}",
        r"    \toprule",
        r"    \rowcolor{headerpink}",
        r"    Method & " + " & ".join(datasets) + r" \\",
        r"    \midrule",
    ]
    for method, gray in [(METHOD_ONESHOT, False), (CASCADE_ALIAS, True)]:
        prefix = r"\rowcolor{rowgray}" + "\n    " if gray else "    "
        cells = []
        row = next((r for r in per_ds_table if r["method"] == method), None)
        for ds in datasets:
            v = row.get(ds) if row else None
            cells.append(f"{v:.3f}" if v is not None else "—")
        label = f"\\textbf{{{method}}}" if gray else method
        lines.append(prefix + f"{label} & " + " & ".join(cells) + r" \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  }",
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines)


def aggregate_per_dataset(cells: list[dict]) -> tuple[list[dict], list[str]]:
    buckets: dict[tuple, list[float]] = defaultdict(list)
    datasets: set[str] = set()
    for c in cells:
        key = (c["method"], c["dataset_id"])
        buckets[key].append(float(c["gap_at_oracle"]))
        datasets.add(c["dataset_id"])
    ds_order = sorted(datasets)
    table: list[dict] = []
    for method in (METHOD_ONESHOT, METHOD_CASCADE):
        row: dict[str, Any] = {"method": method if method != METHOD_CASCADE else CASCADE_ALIAS}
        for ds in ds_order:
            vals = buckets.get((method, ds), [])
            row[ds] = mean(vals) if vals else None
            row[f"{ds}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
        table.append(row)
    return table, ds_order


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-seed main tables (RegretRouter + Cascade)")
    p.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    p.add_argument("--epochs", type=int, default=28)
    p.add_argument("--lambdas", nargs="+", type=float, default=None)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--reuse-stage1", action="store_true", help="Reuse cached cascade routing per seed")
    p.add_argument("--reuse-oneshot", action="store_true", help="Reuse cached one-shot chosen routes")
    p.add_argument(
        "--per-lambda-retrain",
        action="store_true",
        help="Train RegretRouter/Cascade at each λ (λ_train=λ_eval); default: train once at λ=0.2",
    )
    p.add_argument("--skip-oneshot", action="store_true")
    p.add_argument("--skip-cascade", action="store_true")
    p.add_argument("--quick", action="store_true", help="epochs=3, seed 42 only")
    args = p.parse_args()

    if args.quick:
        args.seeds = [42]
        args.epochs = min(args.epochs, 3)

    lambdas = list(args.lambdas) if args.lambdas else list(DEFAULT_LAMBDAS)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    all_rows: list[dict] = []
    per_ds_cells: list[dict] = []
    value_by_seed: dict[int, dict] = {}
    failures: list[dict] = []

    for seed in args.seeds:
        print(f"\n========== seed={seed} ==========", flush=True)
        oneshot_rows: list[dict] = []
        cascade_rows: list[dict] = []
        routing: dict[str, np.ndarray] | None = None

        if not args.skip_oneshot:
            try:
                print(f"  [oneshot] training/eval...", flush=True)
                oneshot_rows = run_oneshot_seed(
                    seed,
                    args.epochs,
                    lambdas,
                    reuse=args.reuse_oneshot,
                    per_lambda_retrain=args.per_lambda_retrain,
                )
                all_rows.extend(oneshot_rows)
                base = _prepare_data(seed)
                chosen = _load_oneshot_cache(seed)
                if chosen is not None:
                    per_ds_cells.extend(per_dataset_gap_oneshot(base, chosen, seed))
                print(f"  [oneshot] λ=0.2 Gap@O={next(r for r in oneshot_rows if r['lambda']==DEFAULT_LAMBDA)['gap_at_oracle']:.4f}", flush=True)
            except Exception as e:
                failures.append({"seed": seed, "method": METHOD_ONESHOT, "error": str(e)})
                print(f"  [oneshot] FAILED: {e}", flush=True)

        if not args.skip_cascade:
            try:
                print(f"  [cascade] stage1+selector+rerank...", flush=True)
                cascade_rows, routing = run_cascade_seed(
                    seed,
                    args.epochs,
                    lambdas,
                    reuse_stage1=args.reuse_stage1,
                    per_lambda_retrain=args.per_lambda_retrain,
                )
                all_rows.extend(cascade_rows)
                if routing is not None:
                    base = _prepare_data(seed)
                    per_ds_cells.extend(
                        per_dataset_gap_cascade(
                            base,
                            routing["m1_test"],
                            routing["m_final"],
                            routing["pv_gate"],
                            seed,
                        )
                    )
                cr = next(r for r in cascade_rows if r["lambda"] == DEFAULT_LAMBDA)
                print(
                    f"  [cascade] λ=0.2 Acc={cr['avg_acc']:.4f} Gap@O={cr['gap_at_oracle']:.4f} "
                    f"Reroute={cr['reroute_rate']:.4f}",
                    flush=True,
                )
            except Exception as e:
                failures.append({"seed": seed, "method": METHOD_CASCADE, "error": str(e)})
                print(f"  [cascade] FAILED: {e}", flush=True)

        if oneshot_rows and cascade_rows:
            value_by_seed[seed] = compute_value_metrics(oneshot_rows, cascade_rows, seed)

    agg = aggregate_method_lambda(all_rows)
    value_agg = {}
    if value_by_seed:
        for key in ("oneshot", "cascade"):
            value_agg[key] = {
                fld: _mean_std([value_by_seed[s][key][fld] for s in value_by_seed])
                for fld in (
                    "perf_gain", "cost_save", "pareto_dist_mean", "pareto_dist_primary",
                    "reroute_rate", "primary_acc", "primary_cost", "primary_utility", "gap_at_oracle",
                )
            }
            value_agg[key]["method"] = (
                METHOD_ONESHOT if key == "oneshot" else METHOD_CASCADE
            )
            # flatten mean for LaTeX
            for fld in list(value_agg[key].keys()):
                if isinstance(value_agg[key][fld], dict) and "mean" in value_agg[key][fld]:
                    value_agg[key][fld] = value_agg[key][fld]["mean"]

    per_ds_table, ds_order = aggregate_per_dataset(per_ds_cells)

    payload = {
        "meta": {
            "seeds": args.seeds,
            "lambdas": lambdas,
            "epochs": args.epochs,
            "train_lambda": "per_lambda" if args.per_lambda_retrain else DEFAULT_LAMBDA,
            "per_lambda_retrain": args.per_lambda_retrain,
            "metric_protocol": "LLMRouterBench acc ratios + raw-perf Regret@O",
            "cascade_pipeline": [
                "RegretRouter Stage1 (frozen after train)",
                "Perfect Verifier gate (perf[m1]<0.5)",
                f"Query×Model selector (β={SELECTOR_BETA}) + top-5 pairwise rerank",
                "ρ=100%",
            ],
            "elapsed_sec": time.time() - t0,
            "failures": failures,
            "n_success_seeds": {
                METHOD_ONESHOT: len({r["seed"] for r in all_rows if r["method"] == METHOD_ONESHOT}),
                METHOD_CASCADE: len({r["seed"] for r in all_rows if r["method"] == METHOD_CASCADE}),
            },
        },
        "per_seed_rows": all_rows,
        "aggregated": agg,
        "value_metrics_per_seed": {str(k): v for k, v in value_by_seed.items()},
        "value_metrics_mean": value_agg,
        "per_dataset_cells": per_ds_cells,
        "per_dataset_table": per_ds_table,
    }

    json_path = out_dir / "TABLES_DATA.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    latex_path = out_dir / "TABLES_LATEX.tex"
    if agg and value_agg and per_ds_table:
        latex_path.write_text(
            render_latex(agg, value_agg, per_ds_table, ds_order, args.seeds),
            encoding="utf-8",
        )

    # Console summary
    print(f"\n{'='*60}", flush=True)
    print(f"Done in {time.time()-t0:.0f}s -> {json_path}", flush=True)
    if failures:
        print(f"Failures ({len(failures)}):", flush=True)
        for f in failures:
            print(f"  seed={f['seed']} {f['method']}: {f['error']}", flush=True)
    print("\n3-seed mean @ λ=0.2:", flush=True)
    for method in (METHOD_ONESHOT, METHOD_CASCADE):
        sub = [a for a in agg if a["method"] == method and a["lambda"] == DEFAULT_LAMBDA]
        if sub:
            a = sub[0]
            print(
                f"  {method}: Gap@O={a['gap_at_oracle']:.4f} Acc={a['avg_acc']:.4f} "
                f"Util={a['avg_utility']:.4f} Cost={a['avg_cost']:.4f} "
                f"(n_seeds={a.get('n_seeds', '?')})",
                flush=True,
            )
    if value_agg:
        print("\nValue metrics (mean):", flush=True)
        for key, label in [("oneshot", METHOD_ONESHOT), ("cascade", CASCADE_ALIAS)]:
            v = value_agg[key]
            print(
                f"  {label}: PerfGain={v['perf_gain']:+.3f} CostSave={v['cost_save']:+.3f} "
                f"ParetoDist={v['pareto_dist_mean']:.3f}",
                flush=True,
            )
    print(f"LaTeX: {latex_path}", flush=True)


if __name__ == "__main__":
    main()
