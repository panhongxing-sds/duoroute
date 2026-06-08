import numpy as np
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from duoroute.losses import duoroute_loss
from duoroute.model import DuoRouteModel
from duoroute.reward_builder import (
    build_oracle_reward,
    build_routed_oracle_utility,
    cascade_oracle_utility,
    raw_perf_cascade_oracle_utility,
    raw_perf_utility_matrix,
    routed_llmrouterbench_utility,
)
from duoroute.bench_metrics import compute_paper_routing_metrics


def test_routed_utility_matches_oracle_reward() -> None:
    perf = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    cost = np.array([[0.1, 0.5, 0.2], [0.3, 0.4, 0.1]], dtype=np.float32)
    chosen = np.array([1, 2], dtype=np.int64)
    lam = 0.2
    oracle_u = build_oracle_reward(perf, cost, lambda_cost=lam)
    routed = routed_llmrouterbench_utility(
        perf, chosen, cost[np.arange(2), chosen], cost, lambda_cost=lam,
    )
    assert np.allclose(routed, oracle_u[np.arange(2), chosen])


def test_cascade_oracle_at_least_stop() -> None:
    perf = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    cost = np.array([[0.1, 0.5, 0.2], [0.3, 0.4, 0.1]], dtype=np.float32)
    mask = np.ones_like(perf, dtype=bool)
    m1 = np.array([0, 0], dtype=np.int64)
    lam = 0.2
    cascade = cascade_oracle_utility(perf, cost, mask, m1, lambda_cost=lam)
    stop = routed_llmrouterbench_utility(
        perf, m1, cost[np.arange(2), m1], cost, lambda_cost=lam,
    )
    assert np.all(cascade + 1e-6 >= stop)


def test_build_routed_oracle_utility_frugal_path() -> None:
    routed_q = np.array([1.0, 0.0], dtype=np.float32)
    routed_c = np.array([0.2, 0.1], dtype=np.float32)
    cost_mx = np.array([[0.1, 0.5], [0.3, 0.4]], dtype=np.float32)
    u = build_routed_oracle_utility(
        routed_c, cost_mx, lambda_cost=0.2, routed_quality=routed_q,
    )
    assert u.shape == (2,)
    assert 0.0 <= u[0] <= 1.0


def test_raw_perf_utility_and_paper_metrics() -> None:
    perf = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.0]], dtype=np.float32)
    cost = np.array([[0.1, 0.5, 0.2], [0.3, 0.4, 0.1]], dtype=np.float32)
    mask = np.ones_like(perf, dtype=bool)
    lam = 0.2
    u = raw_perf_utility_matrix(perf, cost, lambda_cost=lam)
    assert u.shape == perf.shape
    assert np.all(u[mask] >= 0.0) and np.all(u[mask] <= 1.0 + 1e-5)

    chosen = np.array([0, 1], dtype=np.int64)
    m = compute_paper_routing_metrics(
        performance=perf, cost=cost, mask=mask, chosen=chosen, lambda_cost=lam, random_seed=0,
    )
    assert 0.0 <= m["avg_acc"] <= 1.0
    assert m["gap_at_oracle"] >= 0.0
    assert m["regret_at_oracle"] >= 0.0

    m1 = np.array([2, 0], dtype=np.int64)
    cas = raw_perf_cascade_oracle_utility(perf, cost, mask, m1, lambda_cost=lam)
    assert cas.shape == (2,)
    assert np.all(cas >= -1e-6)


def test_forward_and_backward() -> None:
    b, k, d = 4, 5, 16
    query_emb = torch.randn(8, d)
    model_emb = torch.randn(k, d)
    model = DuoRouteModel(query_emb, model_emb, hidden_dim=16, query_dim=d, response_dim=d, model_dim=d)
    prompt_ids = torch.randint(0, 8, (b,))
    response_emb = torch.randn(b, k, d)
    oracle = torch.rand(b, k)
    mask = torch.ones(b, k, dtype=torch.bool)

    pred_a, pred_b = model(prompt_ids, response_emb)
    out = duoroute_loss(pred_a, pred_b, oracle, mask)
    out.total.backward()

    assert pred_a.shape == (b, k)
    assert pred_b.shape == (b, k)
    assert model.query_projector.net[0].weight.grad is not None
    print("ok")


if __name__ == "__main__":
    test_routed_utility_matches_oracle_reward()
    test_cascade_oracle_at_least_stop()
    test_build_routed_oracle_utility_frugal_path()
    test_forward_and_backward()
