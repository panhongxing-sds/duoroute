import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from duoroute.losses import duoroute_loss
from duoroute.model import DuoRouteModel


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
    test_forward_and_backward()
