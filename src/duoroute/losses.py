"""DuoRoute training losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float()
    return (values * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    neg_inf = torch.finfo(logits.dtype).min / 2
    return F.softmax(logits.masked_fill(~mask, neg_inf), dim=dim)


def dense_reward_regression_loss(
    r_hat_a: torch.Tensor,
    r_hat_b: torch.Tensor,
    oracle: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    err_a = (r_hat_a - oracle).pow(2)
    err_b = (r_hat_b - oracle).pow(2)
    return _masked_mean(err_a + err_b, mask)


def listwise_ranking_distillation_loss(
    r_hat: torch.Tensor,
    oracle: torch.Tensor,
    mask: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    teacher = _masked_softmax(oracle / temperature, mask, dim=-1)
    neg_inf = torch.finfo(r_hat.dtype).min / 2
    student_log = F.log_softmax(r_hat.masked_fill(~mask, neg_inf) / temperature, dim=-1)
    kl = F.kl_div(student_log, teacher, reduction="none").sum(dim=-1)
    valid = mask.any(dim=1)
    if not valid.any():
        return r_hat.sum() * 0.0
    return kl[valid].mean()


def cross_channel_distillation_loss(
    r_hat_a: torch.Tensor,
    r_hat_b: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return _masked_mean((r_hat_a - r_hat_b.detach()).pow(2), mask)


@dataclass
class DuoRouteLossOutput:
    total: torch.Tensor
    l_reg: torch.Tensor
    l_rank: torch.Tensor
    l_distill: torch.Tensor


def duoroute_loss(
    r_hat_a: torch.Tensor,
    r_hat_b: torch.Tensor,
    oracle: torch.Tensor,
    mask: torch.Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    temperature: float = 1.0,
) -> DuoRouteLossOutput:
    l_reg = dense_reward_regression_loss(r_hat_a, r_hat_b, oracle, mask)
    l_rank = (
        listwise_ranking_distillation_loss(r_hat_a, oracle, mask, temperature=temperature)
        + listwise_ranking_distillation_loss(r_hat_b, oracle, mask, temperature=temperature)
    )
    l_distill = cross_channel_distillation_loss(r_hat_a, r_hat_b, mask)
    total = l_reg + alpha * l_rank + beta * l_distill
    return DuoRouteLossOutput(total=total, l_reg=l_reg, l_rank=l_rank, l_distill=l_distill)
