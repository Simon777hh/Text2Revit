"""Losses for the plan-level Flow Matching training pipeline."""

from __future__ import annotations

import torch


def quantize_corner_targets(xy):
    """Convert coordinates to HD-style 8+8 bit targets in {-1, +1}."""
    clamped = xy.clamp(-1.0, 1.0)
    scaled = (
        (clamped / 2.0 + 0.5) * 256.0
    ).round().int().clamp(0, 255)
    bits = 2 ** torch.arange(
        7,
        -1,
        -1,
        device=scaled.device,
    )
    binary = (
        scaled.unsqueeze(-1).bitwise_and(bits).ne(0)
    ).to(xy.dtype)
    return (
        binary.reshape(xy.shape[0], xy.shape[1], 16)
        * 2.0
        - 1.0
    )


def flow_matching_loss(
    velocity_prediction,
    x0_xy,
    x1_xy,
    corner_valid,
):
    """Standard conditional flow-matching velocity regression."""
    velocity_target = x1_xy - x0_xy
    squared = (
        (velocity_prediction - velocity_target) ** 2
    ).sum(dim=-1)
    if not corner_valid.any():
        return velocity_prediction.new_zeros(())
    return squared[corner_valid].mean()


def discrete_bit_loss(
    discrete_logits,
    x0_xy,
    corner_valid,
    corner_edge_valid,
    sigma,
    threshold=0.06,
):
    """HD-style MSE over low-sigma corner bit targets."""
    if discrete_logits is None:
        return x0_xy.new_zeros(())
    target = quantize_corner_targets(x0_xy)
    per_bit = (discrete_logits - target) ** 2
    low_sigma = sigma[:, None] < threshold
    per_corner = per_bit.mean(dim=-1)
    corner_mask = corner_valid & corner_edge_valid & low_sigma
    per_sample = (
        (per_corner * corner_mask.to(per_corner.dtype)).sum(dim=-1)
        / corner_mask.sum(dim=-1).clamp(min=1.0)
    )
    sample_mask = corner_valid.any(dim=-1) & (
        sigma < threshold
    )
    active_count = sample_mask.sum()
    if int(active_count.item()) == 0:
        return discrete_logits.new_zeros(())
    # This mirrors HD's t_weights = 1[t < T] * B / count followed by
    # per_sample.mean(). The B/count factor and the final /B cancel exactly.
    batch_reweight = (
        sample_mask.to(per_sample.dtype)
        * per_sample.shape[0]
        / active_count.to(per_sample.dtype)
    )
    return (per_sample * batch_reweight).mean()


def discrete_bit_accuracy(
    discrete_logits,
    x0_xy,
    corner_valid,
    corner_edge_valid,
    sigma,
    threshold=0.06,
):
    """Bit accuracy over valid low-sigma corner bits."""
    if discrete_logits is None:
        return corner_valid.new_zeros((), dtype=torch.float32)
    target = quantize_corner_targets(x0_xy)
    mask = (
        corner_valid
        & corner_edge_valid
        & (sigma[:, None] < threshold)
    )
    if not mask.any():
        return discrete_logits.new_zeros(())
    correct = (
        (discrete_logits > 0.0) == (target > 0.0)
    ).to(discrete_logits.dtype)
    return correct[mask].mean()


def compute_flow_matching_loss(
    velocity_prediction,
    discrete_logits,
    x0_xy,
    x1_xy,
    sigma,
    corner_valid,
    corner_edge_valid,
    flow_weight=1.0,
    discrete_weight=1.0,
    discrete_threshold=0.12,
):
    """Combine flow-matching and discrete-bit losses."""
    loss_flow = flow_matching_loss(
        velocity_prediction,
        x0_xy,
        x1_xy,
        corner_valid,
    )
    loss_discrete = discrete_bit_loss(
        discrete_logits,
        x0_xy,
        corner_valid,
        corner_edge_valid,
        sigma,
        threshold=discrete_threshold,
    )
    bit_accuracy = discrete_bit_accuracy(
        discrete_logits,
        x0_xy,
        corner_valid,
        corner_edge_valid,
        sigma,
        threshold=discrete_threshold,
    )

    total = (
        flow_weight * loss_flow
        + discrete_weight * loss_discrete
    )
    return {
        "total": total,
        "flow": loss_flow,
        "discrete": loss_discrete,
        "bit_accuracy": bit_accuracy,
    }
