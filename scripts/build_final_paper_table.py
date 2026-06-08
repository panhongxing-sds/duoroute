#!/usr/bin/env python3
"""Build final paper table: LB Acc metrics + Regret@O (matched λ for Ours after per-λ retrain)."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from duoroute.main_table import DEFAULT_LAMBDAS, eval_at_lambda, load_pool_851
from duoroute.pool_data import enrich_pool_data, rebuild_utilities
from duoroute.regretrouter import CASCADE_METHOD_NAME, OURS_METHOD_NAME

from export_main_tables import (  # noqa: E402
    ROUTER_SOTA_METHODS,
    baseline_routes,
    router_sota_routes,
)

LAMBDAS = list(DEFAULT_LAMBDAS)
SEEDS = [41, 42, 43]
METRICS = [
    ("AvgAcc", "avg_acc"),
    ("Gain@R", "gain_at_random"),
    ("Gain@B", "gain_at_best_single"),
    ("Gap@O", "gap_at_oracle"),
    ("Regret@O", "regret_at_oracle"),
    ("AvgCost", "avg_cost"),
]
OUT_MD = ROOT / "outputs/cascade/FINAL_PAPER_TABLE.md"
OUT_TEX = ROOT / "outputs/cascade/FINAL_PAPER_TABLE.tex"
TABLES_DATA = ROOT / "outputs/cascade/TABLES_DATA.json"
SOTA_CACHE = ROOT / "outputs/llmrouterbench_flagship"
FRUGAL_JSON = ROOT / "outputs/baseline_comparison/frugalgpt_lambda_sweep_retrain.json"

BASELINE_METHODS = [
    ("Oracle", "Oracle-utility"),
    ("BestSingle", "BestSingle"),
    ("AvengersPro", "Official-AvengersPro-balance"),
    ("GraphRouter", "GraphRouter-lite"),
    ("EmbedLLM", "EmbedLLM"),
    ("RouterDC", "RouterDC-lite"),
    ("RouteLLM", "RouteLLM"),
    ("HybridLLM", "HybridLLM-lite"),
    ("FrugalGPT", "FrugalGPT-Cascade"),
]


def _fmt(v: float | None, pct: bool = False) -> str:
    if v is None:
        return "---"
    if pct:
        return f"{v * 100:.2f}"
    return f"{v:.3f}"


def _agg(rows: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for _, k in METRICS:
        vals = [float(r[k]) for r in rows if k in r and r[k] is not None]
        if vals:
            out[k] = mean(vals)
    return out


def eval_baselines(seed: int, lam: float) -> list[dict]:
    raw = load_pool_851(seed, lambda_cost=lam)
    base = enrich_pool_data(rebuild_utilities(raw, lambda_cost=lam), seed, lambda_cost=lam)
    rows: list[dict] = []
    static = {
        "Oracle": "Oracle-utility",
        "BestSingle": "BestSingle",
        "AvengersPro": "Official-AvengersPro-balance",
    }
    routes = baseline_routes(base, seed, lambda_cost=lam)
    for display, key in static.items():
        row = eval_at_lambda(base, display, routes[key], lambda_eval=lam, seed=seed)
        row["method"] = display
        rows.append(row)
    try:
        sota = router_sota_routes(
            seed, cache_dir=SOTA_CACHE, methods=ROUTER_SOTA_METHODS, train=False,
            sota_epochs=28, gpu=None, quick_sota=False,
        )
        mapping = {
            "GraphRouter": "GraphRouter-lite",
            "EmbedLLM": "EmbedLLM",
            "RouterDC": "RouterDC-lite",
            "RouteLLM": "RouteLLM",
            "HybridLLM": "HybridLLM-lite",
        }
        for display, key in mapping.items():
            if key in sota:
                row = eval_at_lambda(base, display, sota[key], lambda_eval=lam, seed=seed)
                row["method"] = display
                rows.append(row)
    except Exception as e:
        print(f"  [warn] SOTA cache seed={seed}: {e}", flush=True)
    return rows


def load_frugal() -> dict[float, dict[str, float]]:
    if not FRUGAL_JSON.exists():
        return {}
    data = json.loads(FRUGAL_JSON.read_text(encoding="utf-8"))
    buckets: dict[float, list[dict]] = defaultdict(list)
    for r in data.get("rows", []):
        if r.get("seed") in SEEDS:
            buckets[float(r["lambda"])].append(r)
    out: dict[float, dict[str, float]] = {}
    for lam, rs in buckets.items():
        out[lam] = {
            "avg_acc": mean(float(x.get("avg_acc", x.get("acc", 0))) for x in rs),
            "gain_at_best_single": mean(float(x.get("gain_at_best_single", 0)) for x in rs),
            "gap_at_oracle": mean(float(x["gap_at_oracle"]) for x in rs),
            "regret_at_oracle": mean(
                float(x.get("regret_at_oracle", x.get("routing_regret", 0))) for x in rs
            ),
            "avg_cost": mean(float(x["avg_cost"]) for x in rs),
        }
    return out


def load_ours() -> dict[float, dict[str, dict[str, float]]]:
    if not TABLES_DATA.exists():
        return {}
    data = json.loads(TABLES_DATA.read_text(encoding="utf-8"))
    out: dict[float, dict[str, dict[str, float]]] = defaultdict(dict)
    for r in data.get("aggregated", []):
        lam = float(r["lambda"])
        method = r["method"]
        out[lam][method] = {k: float(r[k]) for _, k in METRICS if k in r}
    return out


def build_table(lam: float = 0.2) -> dict[str, dict[str, float]]:
    ours = load_ours().get(lam, {})
    frugal = load_frugal().get(lam, {})
    table: dict[str, dict[str, float]] = {}

    for seed in SEEDS:
        for row in eval_baselines(seed, lam):
            table.setdefault(row["method"], []).append(row)  # type: ignore[arg-type]

    result: dict[str, dict[str, float]] = {}
    for display, _ in BASELINE_METHODS[:8]:
        rs = table.get(display, [])
        if rs:
            result[display] = _agg(rs)

    for key in (OURS_METHOD_NAME, CASCADE_METHOD_NAME):
        if key in ours:
            result[key] = ours[key]
    if frugal:
        result["FrugalGPT"] = frugal
    return result


def render_md(all_lam: dict[float, dict[str, dict[str, float]]]) -> str:
    lines = [
        "# Final Paper Table (LB Acc metrics + Regret@O)",
        "",
        "Metrics: AvgAcc, Gain@R, Gain@B, Gap@O per LLMRouterBench Sec. 3.2; "
        "Regret@O = raw-perf utility regret; Ours per-λ retrain with matched eval.",
        "",
    ]
    for lam in LAMBDAS:
        rows = all_lam.get(lam, {})
        if not rows:
            continue
        lines += [f"## λ = {lam} (train = eval for Ours)", ""]
        hdr = "| Method | " + " | ".join(m[0] for m in METRICS) + " |"
        sep = "|---|" + "|".join("---:" for _ in METRICS) + "|"
        lines += [hdr, sep]
        order = [d for d, _ in BASELINE_METHODS] + [OURS_METHOD_NAME, CASCADE_METHOD_NAME]
        for method in order:
            if method not in rows:
                continue
            r = rows[method]
            cells = []
            for i, (_, k) in enumerate(METRICS):
                pct = i in (1, 2, 3)
                cells.append(_fmt(r.get(k), pct=pct))
            lines.append("| " + method + " | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def render_tex(all_lam: dict[float, dict[str, dict[str, float]]], lam: float = 0.2) -> str:
    rows = all_lam.get(lam, {})
    order = [d for d, _ in BASELINE_METHODS] + [OURS_METHOD_NAME, CASCADE_METHOD_NAME]
    arrow_up = "$\\uparrow$"
    arrow_down = "$\\downarrow$"
    metric_parts = []
    for name, _ in METRICS:
        if name in ("Gap@O", "Regret@O"):
            metric_parts.append(f"{name} {arrow_down}")
        elif name == "AvgCost":
            metric_parts.append(name)
        else:
            metric_parts.append(f"{name} {arrow_up}")
    metric_hdr = " & ".join(metric_parts)
    lines = [
        "% Final paper table @ matched lambda",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        rf"  \caption{{Main results ($N=851$, seeds 41/42/43, $\lambda={lam}$, matched train/eval for Ours). "
        r"Acc metrics follow LLMRouterBench; Regret@O uses $U=(1-\lambda)\mathrm{perf}+\lambda(1-\tilde c)$.}}",
        r"  \label{tab:final-paper}",
        r"  \begin{tabular}{l|" + "c" * len(METRICS) + "}",
        r"    \toprule",
        f"    Method & {metric_hdr} \\\\",
        r"    \midrule",
    ]
    for i, method in enumerate(order):
        if method not in rows:
            continue
        r = rows[method]
        label = method
        if method in (OURS_METHOD_NAME, CASCADE_METHOD_NAME):
            label = rf"\textbf{{{method}}}"
        vals = []
        for j, (_, k) in enumerate(METRICS):
            v = r.get(k, 0)
            if j in (1, 2, 3):
                vals.append(f"{v*100:.2f}")
            else:
                vals.append(f"{v:.3f}")
        gray = r"\rowcolor{rowgray}" + "\n    " if i % 2 else "    "
        lines.append(gray + f"{label} & " + " & ".join(vals) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    all_lam: dict[float, dict[str, dict[str, float]]] = {}
    for lam in LAMBDAS:
        print(f"Building baselines @ λ={lam}...", flush=True)
        ours = load_ours().get(lam, {})
        base_rows: dict[str, dict[str, float]] = {}
        for seed in SEEDS:
            for row in eval_baselines(seed, lam):
                base_rows.setdefault(row["method"], [])
                base_rows[row["method"]].append(row)  # type: ignore[index]
        merged: dict[str, dict[str, float]] = {}
        for m, rs in base_rows.items():
            merged[m] = _agg(rs)  # type: ignore[arg-type]
        merged.update(ours)
        frugal = load_frugal().get(lam)
        if frugal:
            merged["FrugalGPT"] = frugal
        all_lam[lam] = merged

    OUT_MD.write_text(render_md(all_lam), encoding="utf-8")
    OUT_TEX.write_text(render_tex(all_lam, lam=0.2), encoding="utf-8")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
