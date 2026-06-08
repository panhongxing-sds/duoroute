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


@dataclass
class ResidualLossOutput:
    total: torch.Tensor
    l_route_reg: torch.Tensor
    l_route_rank: torch.Tensor
    l_delta: torch.Tensor
    l_quality: torch.Tensor


def _top_quartile_quality_labels(performance: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-query top-quartile among masked arms (arm-level BCE target)."""
    labels = torch.zeros_like(performance)
    for i in range(performance.shape[0]):
        m = mask[i]
        if not m.any():
            continue
        vals = performance[i, m]
        if vals.numel() < 2:
            labels[i, m] = (vals >= 0.5).float()
            continue
        q75 = torch.quantile(vals.float(), 0.75)
        labels[i, m] = (performance[i, m] >= q75).float()
    return labels


def residual_verifier_loss(
    pred_a: torch.Tensor,
    delta_b: torch.Tensor,
    oracle: torch.Tensor,
    mask: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    gamma: float = 0.5,
    alpha: float = 0.5,
    temperature: float = 0.5,
    eta: float = 0.1,
    performance: torch.Tensor | None = None,
) -> ResidualLossOutput:
    """
    Frozen-A style training for delta head:
    score_final = pred_a.detach() + gamma * delta * response_mask
    Delta supervised only on response_mask; routing loss on full masked arms.
    """
    resp_f = response_mask.float()
    pred_final = pred_a.detach() + float(gamma) * delta_b * resp_f

    l_route_reg = _masked_mean((pred_final - oracle).pow(2), mask)
    l_route_rank = listwise_ranking_distillation_loss(
        pred_final, oracle, mask, temperature=temperature
    )
    target_delta = (oracle - pred_a.detach()) * resp_f
    l_delta = _masked_mean((delta_b - target_delta).pow(2), response_mask)

    l_quality = pred_final.sum() * 0.0
    if eta > 0 and performance is not None:
        labels = _top_quartile_quality_labels(performance, mask)
        # proxy: delta should be positive when arm is high quality
        logits = delta_b
        l_quality = F.binary_cross_entropy_with_logits(
            logits[response_mask],
            labels[response_mask],
            reduction="mean",
        )

    total = l_route_reg + alpha * l_route_rank + l_delta + float(eta) * l_quality
    return ResidualLossOutput(
        total=total,
        l_route_reg=l_route_reg,
        l_route_rank=l_route_rank,
        l_delta=l_delta,
        l_quality=l_quality,
    )


@dataclass
class TeacherLossOutput:
    total: torch.Tensor
    l_route_reg: torch.Tensor
    l_route_rank: torch.Tensor
    l_pairwise: torch.Tensor
    l_quality: torch.Tensor


def pairwise_delta_loss(
    delta: torch.Tensor,
    oracle: torch.Tensor,
    mask: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    margin: float = 0.1,
    reward_eps: float = 1e-6,
) -> torch.Tensor:
    """Same-query pairs with both arms having response: higher-reward arm should have larger delta."""
    pair_mask = mask & response_mask
    total = delta.sum() * 0.0
    count = 0
    for i in range(delta.shape[0]):
        arms = torch.where(pair_mask[i])[0]
        if arms.numel() < 2:
            continue
        for ai in range(arms.numel()):
            for bi in range(ai + 1, arms.numel()):
                a, b = int(arms[ai]), int(arms[bi])
                ra, rb = oracle[i, a], oracle[i, b]
                da, db = delta[i, a], delta[i, b]
                if ra > rb + reward_eps:
                    total = total + F.relu(float(margin) - delta[i, a] + delta[i, b])
                    count += 1
                elif rb > ra + reward_eps:
                    total = total + F.relu(float(margin) - delta[i, b] + delta[i, a])
                    count += 1
    if count == 0:
        return delta.sum() * 0.0
    return total / count


def teacher_residual_loss(
    pred_a: torch.Tensor,
    delta: torch.Tensor,
    oracle: torch.Tensor,
    mask: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    gamma: float,
    alpha: float = 0.5,
    temperature: float = 0.5,
    mu: float = 0.5,
    eta: float = 0.0,
    performance: torch.Tensor | None = None,
) -> TeacherLossOutput:
    """Teacher: score_T = A.detach() + gamma * delta * response_mask; route + pairwise on delta."""
    resp_f = response_mask.float()
    score_t = pred_a.detach() + float(gamma) * delta * resp_f

    l_route_reg = _masked_mean((score_t - oracle).pow(2), mask)
    l_route_rank = listwise_ranking_distillation_loss(
        score_t, oracle, mask, temperature=temperature
    )
    l_pair = pairwise_delta_loss(delta, oracle, mask, response_mask)
    l_quality = score_t.sum() * 0.0
    if eta > 0 and performance is not None:
        labels = _top_quartile_quality_labels(performance, mask)
        l_quality = F.binary_cross_entropy_with_logits(
            delta[response_mask], labels[response_mask], reduction="mean"
        )

    total = l_route_reg + alpha * l_route_rank + float(mu) * l_pair + float(eta) * l_quality
    return TeacherLossOutput(
        total=total,
        l_route_reg=l_route_reg,
        l_route_rank=l_route_rank,
        l_pairwise=l_pair,
        l_quality=l_quality,
    )


@dataclass
class StudentDistillLossOutput:
    total: torch.Tensor
    l_route_reg: torch.Tensor
    l_route_rank: torch.Tensor
    l_kd: torch.Tensor
    l_margin: torch.Tensor


def advantage_filtered_distill_loss(
    pred_student: torch.Tensor,
    score_teacher: torch.Tensor,
    oracle: torch.Tensor,
    mask: torch.Tensor,
    chosen_a0: torch.Tensor,
    chosen_teacher: torch.Tensor,
    *,
    alpha_kd: float = 0.1,
    beta_margin: float = 0.1,
    route_alpha: float = 0.5,
    temperature: float = 1.0,
    margin: float = 0.1,
    binary_advantage: bool = True,
    advantage_mode: str = "binary",
) -> StudentDistillLossOutput:
    """
    Student is query-only scores. Teacher scores precomputed with response (frozen).
    Only distill queries where teacher oracle reward > A0 oracle reward.
    """
    idx = torch.arange(pred_student.shape[0], device=pred_student.device)
    ra = oracle[idx, chosen_a0]
    rt = oracle[idx, chosen_teacher]
    if advantage_mode == "none":
        w = torch.ones(pred_student.shape[0], device=pred_student.device)
    elif advantage_mode == "continuous":
        w = (rt - ra).clamp_min(0.0)
    elif binary_advantage:
        w = (rt > ra + 1e-8).float()
    else:
        w = (rt - ra).clamp_min(0.0)

    l_route_reg = _masked_mean((pred_student - oracle).pow(2), mask)
    l_route_rank = listwise_ranking_distillation_loss(
        pred_student, oracle, mask, temperature=temperature
    )

    l_kd = pred_student.sum() * 0.0
    if alpha_kd > 0 and w.sum() > 0:
        teacher_p = _masked_softmax(score_teacher / temperature, mask, dim=-1)
        neg_inf = torch.finfo(pred_student.dtype).min / 2
        student_log = F.log_softmax(
            pred_student.masked_fill(~mask, neg_inf) / temperature, dim=-1
        )
        kl = F.kl_div(student_log, teacher_p, reduction="none").sum(dim=-1)
        l_kd = (w * kl).sum() / w.sum().clamp_min(1.0)

    l_margin = pred_student.sum() * 0.0
    if beta_margin > 0 and w.sum() > 0:
        sa = pred_student[idx, chosen_a0]
        st = pred_student[idx, chosen_teacher]
        m = F.relu(float(margin) - sa + st)
        l_margin = (w * m).sum() / w.sum().clamp_min(1.0)

    total = l_route_reg + float(route_alpha) * l_route_rank + float(alpha_kd) * l_kd + float(beta_margin) * l_margin
    return StudentDistillLossOutput(
        total=total,
        l_route_reg=l_route_reg,
        l_route_rank=l_route_rank,
        l_kd=l_kd,
        l_margin=l_margin,
    )
