"""Flagship pool loading, AP baselines, and LLMRouterBench-aligned eval helpers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from duoroute.bench_metrics import (
    best_single_idx_by_train,
    compute_bench_metrics,
    compute_paper_routing_metrics,
    oracle_route_performance_tiebreak_cost,
    pred_matrix_from_choices,
)
from duoroute.data import DuoRouteGroupedData
from duoroute.encoders import build_hash_embeddings
from duoroute.privileged_pipeline import filter_grouped_by_datasets
from duoroute.response_vae_pipeline import load_pool_tensors
from duoroute.reward_builder import rebuild_grouped_rewards
from duoroute.utils import project_root

FOUR = ["gpqa", "hle", "livecodebench", "mmlupro"]
ACC = 0.5
DEFAULT_LAMBDA = 0.2


def route_ap_cluster(
    train_q: np.ndarray,
    train_util: np.ndarray,
    train_mask: np.ndarray,
    test_q: np.ndarray,
    seed: int,
) -> np.ndarray:
    from collections import defaultdict

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import Normalizer

    k_models = train_util.shape[1]
    normalizer = Normalizer(norm="l2")
    train_norm = normalizer.fit_transform(train_q)
    km = KMeans(n_clusters=16, random_state=seed, n_init=10)
    labels = km.fit_predict(train_norm)
    cluster_data: dict[int, list[int]] = defaultdict(list)
    for i, cid in enumerate(labels):
        cluster_data[int(cid)].append(i)
    rankings: dict[int, np.ndarray] = {}
    for cid, idxs in cluster_data.items():
        scores = np.zeros(k_models, dtype=np.float64)
        counts = np.zeros(k_models, dtype=np.float64)
        for i in idxs:
            m = train_mask[i]
            scores[m] += train_util[i, m]
            counts[m] += 1
        means = np.where(counts > 0, scores / np.maximum(counts, 1), -1e9)
        rankings[cid] = np.argsort(-means)
    test_norm = normalizer.transform(test_q)
    dists = 1.0 - test_norm @ km.cluster_centers_.T
    chosen = np.zeros(len(test_q), dtype=np.int64)
    for i in range(len(test_q)):
        cid = int(np.argmin(dists[i]))
        order = rankings.get(cid, np.arange(k_models))
        for j in order:
            chosen[i] = int(j)
            break
    return chosen


def rebuild_utilities(data: dict, *, lambda_cost: float = DEFAULT_LAMBDA) -> dict:
    out = dict(data)
    perf_keys = {"train": "train_perf", "val": "val_perf", "test": "test_perf"}
    for split, pk in perf_keys.items():
        out[f"{split}_u"] = rebuild_grouped_rewards(
            data[pk], data[f"{split}_cost"], lambda_cost=lambda_cost
        )
    return out


def route_ap_balance_cluster(
    train_q: np.ndarray,
    train_perf: np.ndarray,
    train_cost: np.ndarray,
    train_mask: np.ndarray,
    query_h: np.ndarray,
    seed: int,
    *,
    performance_weight: float = 0.7,
    cost_sensitivity: float = 0.3,
) -> np.ndarray:
    from collections import defaultdict

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import Normalizer

    k_models = train_perf.shape[1]
    normalizer = Normalizer(norm="l2")
    train_norm = normalizer.fit_transform(train_q)
    km = KMeans(n_clusters=16, random_state=seed, n_init=10)
    labels = km.fit_predict(train_norm)
    cluster_data: dict[int, list[int]] = defaultdict(list)
    for i, cid in enumerate(labels):
        cluster_data[int(cid)].append(i)
    rankings: dict[int, np.ndarray] = {}
    for cid, idxs in cluster_data.items():
        metrics = []
        for j in range(k_models):
            m = train_mask[idxs][:, j]
            if not m.any():
                metrics.append((0.0, float("inf"), 0))
                continue
            acc = float(train_perf[idxs][:, j][m].mean())
            c = float(train_cost[idxs][:, j][m].mean())
            metrics.append((acc, c, int(m.sum())))
        valid = [x for x in metrics if x[2] > 0 and x[1] != float("inf")]
        if valid:
            max_cost = max(x[1] for x in valid)
            max_acc = max(x[0] for x in valid)
            min_acc = min(x[0] for x in valid)
        else:
            max_cost = max_acc = min_acc = 1.0
        scores = np.zeros(k_models, dtype=np.float64)
        for j, (acc, c, cnt) in enumerate(metrics):
            if cnt <= 0:
                continue
            acc_range = max_acc - min_acc
            norm_acc = (acc - min_acc) / acc_range if acc_range > 0 else 1.0
            norm_cost = c / max_cost if max_cost > 0 else 0.0
            cost_score = 1.0 - norm_cost
            scores[j] = performance_weight * norm_acc + cost_sensitivity * cost_score
        rankings[cid] = np.argsort(-scores)
    test_norm = normalizer.transform(query_h)
    dists = 1.0 - test_norm @ km.cluster_centers_.T
    chosen = np.zeros(len(query_h), dtype=np.int64)
    for i in range(len(query_h)):
        cid = int(np.argmin(dists[i]))
        order = rankings.get(cid, np.arange(k_models))
        for j in order:
            if train_mask[:, j].any():
                chosen[i] = int(j)
                break
    return chosen


def attach_ap_routes(data: dict, seed: int) -> dict:
    out = dict(data)
    ap_score = out["train_perf"]
    th, tm = out["train_h"], out["train_mask"]
    out["ap_train_idx"] = route_ap_cluster(th, ap_score, tm, th, seed)
    out["ap_val_idx"] = route_ap_cluster(th, ap_score, tm, out["val_h"], seed)
    out["ap_test_idx"] = route_ap_cluster(th, ap_score, tm, out["test_h"], seed)
    for split, hkey in (("train", "train_h"), ("val", "val_h"), ("test", "test_h")):
        out[f"ap_balance_{split}_idx"] = route_ap_balance_cluster(
            th,
            out["train_perf"],
            out["train_cost"],
            tm,
            out[hkey],
            seed,
            performance_weight=0.7,
            cost_sensitivity=0.3,
        )
    out["ap_costfirst_test_idx"] = route_ap_balance_cluster(
        th,
        out["train_perf"],
        out["train_cost"],
        tm,
        out["test_h"],
        seed,
        performance_weight=0.0,
        cost_sensitivity=1.0,
    )
    return out


def enrich_pool_data(data: dict, seed: int, *, lambda_cost: float = DEFAULT_LAMBDA) -> dict:
    if "train_cost" not in data:
        raise KeyError("pool data missing per-query costs (train_cost)")
    d = rebuild_utilities(data, lambda_cost=lambda_cost)
    d["lambda_train"] = lambda_cost
    return attach_ap_routes(d, seed)


def rescue_decomposition(
    test_perf: np.ndarray,
    test_u: np.ndarray,
    test_cost: np.ndarray,
    ap_chosen: np.ndarray,
    chosen: np.ndarray,
) -> dict:
    n = len(chosen)
    idx = np.arange(n)
    ap_ok = test_perf[idx, ap_chosen] >= ACC
    new_ok = test_perf[idx, chosen] >= ACC
    ov = chosen != ap_chosen
    ap_c = test_cost[idx, ap_chosen]
    new_c = test_cost[idx, chosen]
    ap_u = test_u[idx, ap_chosen]
    new_u = test_u[idx, chosen]
    return {
        "acc_rescued": int(np.sum(~ap_ok & ov & new_ok)),
        "acc_harmed": int(np.sum(ap_ok & ov & ~new_ok)),
        "cost_save_win": int(np.sum(ov & (new_c < ap_c - 1e-12))),
        "cost_harm": int(np.sum(ov & (new_c > ap_c + 1e-12))),
        "utility_win": int(np.sum(ov & (new_u > ap_u + 1e-12))),
        "utility_harm": int(np.sum(ov & (new_u < ap_u - 1e-12))),
    }


def eval_row(
    test_perf: np.ndarray,
    test_u: np.ndarray,
    test_mask: np.ndarray,
    test_cost: np.ndarray,
    ap_chosen: np.ndarray,
    chosen: np.ndarray,
    *,
    best_single_idx: int | None = None,
    random_seed: int = 42,
    lambda_cost: float = DEFAULT_LAMBDA,
    train_perf: np.ndarray | None = None,
    train_mask: np.ndarray | None = None,
    train_cost: np.ndarray | None = None,
) -> dict:
    n, k = test_perf.shape
    bs_perf_idx = best_single_idx
    if bs_perf_idx is None:
        if train_perf is not None and train_mask is not None:
            bs_perf_idx = best_single_idx_by_train(train_perf, train_mask, by="performance")
        else:
            bs_perf_idx = best_single_idx_by_train(test_perf, test_mask, by="performance")
    if train_perf is not None and train_mask is not None and train_cost is not None:
        bs_util_idx = best_single_idx_by_train(
            train_perf,
            train_mask,
            by="oracle_reward",
            train_cost=train_cost,
            lambda_cost=lambda_cost,
        )
    else:
        bs_util_idx = bs_perf_idx
    pred_u = pred_matrix_from_choices(n, k, chosen)
    bench = compute_bench_metrics(
        performance=test_perf,
        cost=test_cost,
        mask=test_mask,
        pred_u=pred_u,
        true_u=test_u,
        best_single_idx=bs_util_idx,
        random_seed=random_seed,
    )
    paper = compute_paper_routing_metrics(
        performance=test_perf,
        cost=test_cost,
        mask=test_mask,
        chosen=chosen,
        lambda_cost=lambda_cost,
        random_seed=random_seed,
        best_single_idx=bs_perf_idx,
    )
    idx = np.arange(n)
    ap_ok = test_perf[idx, ap_chosen] >= ACC
    new_ok = test_perf[idx, chosen] >= ACC
    ov = chosen != ap_chosen
    ap_acc = float(ap_ok.mean())
    ap_row = compute_bench_metrics(
        performance=test_perf,
        cost=test_cost,
        mask=test_mask,
        pred_u=pred_matrix_from_choices(n, k, ap_chosen),
        true_u=test_u,
        best_single_idx=bs_util_idx,
        random_seed=random_seed,
    )
    decomp = rescue_decomposition(test_perf, test_u, test_cost, ap_chosen, chosen)
    routed_u = float(test_u[idx, chosen].mean())
    ap_u_mean = float(test_u[idx, ap_chosen].mean())
    return {
        "avg_acc": paper["avg_acc"],
        "gain_at_random": paper["gain_at_random"],
        "gain_at_best_single": paper["gain_at_best_single"],
        "gap_at_oracle": paper["gap_at_oracle"],
        "regret_at_oracle": paper["regret_at_oracle"],
        "avg_cost": paper["avg_cost"],
        "utility": routed_u,
        "avg_utility": routed_u,
        "routing_regret": bench.routing_regret,
        "regret": paper["regret_at_oracle"],
        "utility_regret_legacy": bench.routing_regret,
        "gain_at_best_single_legacy": bench.gain_at_best_single,
        "gap_at_oracle_legacy": bench.gap_at_oracle,
        "lift_pp": (float(new_ok.mean()) - ap_acc) * 100,
        "delta_acc_pp": (paper["avg_acc"] - ap_row.sample_avg_acc) * 100,
        "delta_gap_at_oracle": paper["gap_at_oracle"] - ap_row.gap_at_oracle,
        "delta_avg_cost": paper["avg_cost"] - ap_row.avg_cost,
        "delta_avg_utility": routed_u - ap_u_mean,
        "rescued": decomp["acc_rescued"],
        "harmed": decomp["acc_harmed"],
        "net": decomp["acc_rescued"] - decomp["acc_harmed"],
        "net_gain": decomp["acc_rescued"] - decomp["acc_harmed"],
        "override_n": int(ov.sum()),
        "ap_acc": ap_acc,
        **decomp,
    }


def routing_potential_bench(
    test_u: np.ndarray,
    test_perf: np.ndarray,
    test_mask: np.ndarray,
    ap_chosen: np.ndarray,
    *,
    best_single_idx: int,
) -> dict:
    n, _ = test_perf.shape
    oracle = oracle_route_performance_tiebreak_cost(test_perf, np.zeros_like(test_perf), test_mask)
    neg_inf = -1e9
    true_m = np.where(test_mask, test_u, neg_inf)
    oracle_u = true_m.max(axis=1)
    idx = np.arange(n)
    ap_u = true_m[idx, ap_chosen]
    bs_u = true_m[:, best_single_idx]
    ap_acc = float((test_perf[idx, ap_chosen] >= ACC).mean())
    oracle_acc = float((test_perf[idx, oracle] >= ACC).mean())
    return {
        "ap_acc": ap_acc,
        "oracle_acc": oracle_acc,
        "potential_pp": (oracle_acc - ap_acc) * 100,
        "n_rescueable": int(np.sum((test_perf[idx, oracle] >= ACC) & (test_perf[idx, ap_chosen] < ACC))),
        "gap_at_oracle": float((oracle_u - ap_u).mean()),
        "gain_at_best_single": float(ap_u.mean() - bs_u.mean()),
    }


def per_dataset_breakdown(data: dict, chosen: np.ndarray, ap_chosen: np.ndarray) -> list[dict]:
    ds = data.get("test_dataset_ids")
    if ds is None:
        return []
    ds = np.asarray(ds)
    rows = []
    for d in sorted(set(ds.tolist())):
        m = ds == d
        if not m.any():
            continue
        rows.append(
            {
                "dataset_id": d,
                "n": int(m.sum()),
                **eval_row(
                    data["test_perf"][m],
                    data["test_u"][m],
                    data["test_mask"][m],
                    data["test_cost"][m],
                    ap_chosen[m],
                    chosen[m],
                    best_single_idx=data.get("best_single_idx", 0),
                ),
            }
        )
    return rows


def load_flagship(seed: int, *, filter_four: bool = True) -> dict:
    del seed
    data_dir = project_root() / "data/seed42_flagship"
    bundle = load_pool_tensors(data_dir)
    if filter_four:
        for split in ("train", "val", "test"):
            bundle[split], bundle[f"{split}_ids"], _ = filter_grouped_by_datasets(
                bundle[split], bundle[f"{split}_ids"], bundle[f"{split}_resp"], FOUR
            )
    qe = bundle["query_emb"]
    emb = qe.float() if isinstance(qe, torch.Tensor) else torch.from_numpy(np.asarray(qe)).float()

    def rows(g, pids):
        return emb[torch.as_tensor(pids, dtype=torch.long)].numpy().astype(np.float32)

    train, val, test = bundle["train"], bundle["val"], bundle["test"]
    train_perf = train.performance.astype(np.float32)
    cost = np.array(
        [
            train.cost[train.mask[:, j], j].mean() if train.mask[:, j].any() else 0.0
            for j in range(len(train.model_names))
        ],
        dtype=np.float32,
    )

    def split_costs(g):
        return g.cost.astype(np.float32)

    return {
        "name": "flagship_four" if filter_four else "flagship_four_full851",
        "train_h": rows(train, bundle["train_ids"]),
        "val_h": rows(val, bundle["val_ids"]),
        "test_h": rows(test, bundle["test_ids"]),
        "train_u": train.oracle_reward.astype(np.float32),
        "train_perf": train_perf,
        "val_perf": val.performance.astype(np.float32),
        "val_u": val.oracle_reward.astype(np.float32),
        "test_u": test.oracle_reward.astype(np.float32),
        "train_mask": train.mask.astype(bool),
        "val_mask": val.mask.astype(bool),
        "test_perf": test.performance.astype(np.float32),
        "test_mask": test.mask.astype(bool),
        "train_cost": split_costs(train),
        "val_cost": split_costs(val),
        "test_cost": split_costs(test),
        "cost": cost,
        "model_names": list(test.model_names),
        "train_dataset_ids": train.dataset_ids,
        "val_dataset_ids": val.dataset_ids,
        "test_dataset_ids": test.dataset_ids,
    }


def load_routerbench(seed: int) -> dict:
    data_dir = project_root() / "data/routerbench_0shot"
    if not (data_dir / "test/grouped.npz").exists():
        script = project_root() / "scripts/build_routerbench_duoroute.py"
        subprocess.check_call([sys.executable, str(script)], cwd=str(project_root()))
    train_g = DuoRouteGroupedData.load(data_dir / "train")
    val_g = DuoRouteGroupedData.load(data_dir / "val")
    test_g = DuoRouteGroupedData.load(data_dir / "test")
    texts = list(dict.fromkeys(train_g.prompt_texts + val_g.prompt_texts + test_g.prompt_texts))
    text_to_i = {t: i for i, t in enumerate(texts)}
    emb_np = build_hash_embeddings(texts, dim=2048, seed=seed).numpy().astype(np.float32)

    def rows(g):
        pids = np.array([text_to_i[t] for t in g.prompt_texts], dtype=np.int64)
        return emb_np[pids]

    cost = np.array(
        [
            train_g.cost[train_g.mask[:, j], j].mean() if train_g.mask[:, j].any() else 0.0
            for j in range(len(train_g.model_names))
        ],
        dtype=np.float32,
    )
    return {
        "name": "routerbench_0shot",
        "train_h": rows(train_g),
        "val_h": rows(val_g),
        "test_h": rows(test_g),
        "train_u": train_g.oracle_reward.astype(np.float32),
        "val_u": val_g.oracle_reward.astype(np.float32),
        "test_u": test_g.oracle_reward.astype(np.float32),
        "train_mask": train_g.mask.astype(bool),
        "val_mask": val_g.mask.astype(bool),
        "test_perf": test_g.performance.astype(np.float32),
        "test_mask": test_g.mask.astype(bool),
        "train_cost": train_g.cost.astype(np.float32),
        "val_cost": val_g.cost.astype(np.float32),
        "test_cost": test_g.cost.astype(np.float32),
        "val_perf": val_g.performance.astype(np.float32),
        "train_perf": train_g.performance.astype(np.float32),
        "cost": cost,
        "model_names": list(test_g.model_names),
        "train_dataset_ids": train_g.dataset_ids,
        "val_dataset_ids": val_g.dataset_ids,
        "test_dataset_ids": test_g.dataset_ids,
    }


def load_pool_dir(data_dir: Path, *, seed: int, name: str | None = None) -> dict:
    del seed
    data_dir = Path(data_dir)
    bundle = load_pool_tensors(data_dir)
    qe = bundle["query_emb"]
    emb = qe.float() if isinstance(qe, torch.Tensor) else torch.from_numpy(np.asarray(qe)).float()

    def rows(g, pids):
        return emb[torch.as_tensor(pids, dtype=torch.long)].numpy().astype(np.float32)

    train, val, test = bundle["train"], bundle["val"], bundle["test"]
    train_perf = train.performance.astype(np.float32)
    cost = np.array(
        [
            train.cost[train.mask[:, j], j].mean() if train.mask[:, j].any() else 0.0
            for j in range(len(train.model_names))
        ],
        dtype=np.float32,
    )
    return {
        "name": name or data_dir.name,
        "train_h": rows(train, bundle["train_ids"]),
        "val_h": rows(val, bundle["val_ids"]),
        "test_h": rows(test, bundle["test_ids"]),
        "train_u": train.oracle_reward.astype(np.float32),
        "train_perf": train_perf,
        "val_perf": val.performance.astype(np.float32),
        "val_u": val.oracle_reward.astype(np.float32),
        "test_u": test.oracle_reward.astype(np.float32),
        "train_mask": train.mask.astype(bool),
        "val_mask": val.mask.astype(bool),
        "test_perf": test.performance.astype(np.float32),
        "test_mask": test.mask.astype(bool),
        "train_cost": train.cost.astype(np.float32),
        "val_cost": val.cost.astype(np.float32),
        "test_cost": test.cost.astype(np.float32),
        "cost": cost,
        "model_names": list(test.model_names),
        "train_dataset_ids": train.dataset_ids,
        "val_dataset_ids": val.dataset_ids,
        "test_dataset_ids": test.dataset_ids,
    }


def load_pool_851(seed: int, lambda_cost: float = DEFAULT_LAMBDA) -> dict:
    raw = load_flagship(seed, filter_four=False)
    data = enrich_pool_data(raw, seed, lambda_cost=lambda_cost)
    data["name"] = "851"
    return data
