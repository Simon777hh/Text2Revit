"""Shared deterministic Heun solver for training-time and standalone inference."""

from __future__ import annotations

import math

import torch

from model import (
    flow_x0_with_discrete_refinement,
)


def time_shift(mu, sigmas):
    return math.exp(mu) / (math.exp(mu) + (1.0 / sigmas - 1.0))


def make_sigma_schedule(
    steps,
    device,
    fixed_low_sigma=False,
    low_sigma_threshold=0.12,
    low_sigma_step=0.01,
    low_sigma_steps=12,
):
    if steps < 1:
        raise ValueError("solver steps must be positive")
    sigmas = torch.linspace(
        1.0,
        1.0 / steps,
        steps,
        device=device,
    )
    sigmas = time_shift(1.0, sigmas)
    if fixed_low_sigma:
        low_grid = torch.arange(
            0.0,
            low_sigma_threshold + 0.5 * low_sigma_step,
            low_sigma_step,
            device=device,
        ).flip(0)
        high = sigmas[sigmas > low_sigma_threshold]
        if high.numel() == 0:
            high = torch.ones(1, device=device)
        if low_grid.numel() != int(low_sigma_steps) + 1:
            low_grid = torch.linspace(
                low_sigma_threshold,
                0.0,
                int(low_sigma_steps) + 1,
                device=device,
            )
        return torch.cat([high, low_grid])
    return torch.cat([sigmas, torch.zeros(1, device=device)])


@torch.no_grad()
def solve_heun(
    forward_fn,
    source_xy,
    corner_valid,
    steps=100,
    discrete_threshold=0.1,
    discrete_refine_steps=0,
    discrete_update="replace",
    discrete_hard=False,
    low_sigma_refine=False,
    fixed_low_sigma=False,
    low_sigma_threshold=0.12,
    low_sigma_step=0.01,
    low_sigma_steps=12,
    project_final=None,
    final_refine=False,
    final_hard_decode=True,
    return_trajectory=False,
):
    """Run continuous Heun and optionally project the final geometry."""
    device = source_xy.device
    x_t = source_xy.clone()
    valid_float = corner_valid.unsqueeze(-1).to(x_t.dtype)
    sigmas = make_sigma_schedule(
        steps,
        device,
        fixed_low_sigma=fixed_low_sigma,
        low_sigma_threshold=low_sigma_threshold,
        low_sigma_step=low_sigma_step,
        low_sigma_steps=low_sigma_steps,
    )
    num_steps = sigmas.numel() - 1
    trajectory = []
    if return_trajectory:
        trajectory.append((1.0, x_t.detach().cpu()))

    for step_index in range(num_steps):
        sigma = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]
        dt = sigma_next - sigma

        _, velocity, logits = forward_fn(
            x_t,
            sigma.view(1),
        )
        velocity = velocity * valid_float
        xy_tilde = x_t + dt * velocity

        if sigma_next > 0.0:
            tokens_next, velocity_next, logits_next = forward_fn(
                xy_tilde,
                sigma_next.view(1),
            )
            velocity_next = velocity_next * valid_float
            x_next = x_t + 0.5 * dt * (
                velocity + velocity_next
            )
            if (
                low_sigma_refine
                and sigma_next > 0.0
                and (
                    (
                        discrete_refine_steps > 0
                        and step_index
                        >= num_steps - discrete_refine_steps
                    )
                    or (
                        discrete_refine_steps <= 0
                        and sigma_next < discrete_threshold
                    )
                )
            ):
                x0_refined = flow_x0_with_discrete_refinement(
                    tokens_next,
                    velocity_next,
                    sigma_next.view(1),
                    logits_next,
                    threshold=discrete_threshold,
                    soft=not discrete_hard,
                    hard_at_zero=True,
                )
                if discrete_update == "replace":
                    x_next = (
                        (1.0 - sigma_next) * x0_refined
                        + sigma_next * source_xy
                    )
                elif discrete_update == "velocity":
                    refined_velocity = (
                        tokens_next[:, :, :2] - x0_refined
                    ) / sigma_next.clamp(min=1e-5)
                    x_next = x_t + dt * refined_velocity
                elif discrete_update in (
                    "x0_projection",
                    "posterior",
                ):
                    x0_hard = flow_x0_with_discrete_refinement(
                        x_t,
                        velocity,
                        sigma.view(1),
                        logits,
                        threshold=1.0,
                        soft=False,
                        hard_at_zero=True,
                    )
                    sigma_safe = sigma.clamp(min=1e-5)
                    x1_hat = (
                        x_t
                        - (1.0 - sigma_safe) * x0_hard
                    ) / sigma_safe
                    x_next = (
                        (1.0 - sigma_next) * x0_hard
                        + sigma_next * x1_hat
                    )
                else:
                    raise ValueError(
                        f"unknown discrete_update: {discrete_update}"
                    )
                x_next = x_next * valid_float
        else:
            x_next = x_t + dt * velocity
        x_t = x_next
        if return_trajectory:
            trajectory.append(
                (float(sigma_next), x_t.detach().cpu())
            )

    final_sigma = torch.zeros(1, device=device)
    final_tokens, final_velocity, final_logits = forward_fn(
        x_t,
        final_sigma,
    )
    if final_refine:
        x0 = flow_x0_with_discrete_refinement(
            final_tokens,
            final_velocity,
            final_sigma,
            final_logits,
            threshold=1.0,
            soft=True,
            hard_at_zero=final_hard_decode,
        )
    else:
        x0 = (
            final_tokens[:, :, :2]
            - final_sigma[:, None, None] * final_velocity
        )
    x0 = x0 * valid_float
    if project_final is not None:
        x0 = project_final(x0)
        x0 = x0 * valid_float
    if return_trajectory:
        trajectory[-1] = (0.0, x0.detach().cpu())
        return x0, trajectory
    return x0
