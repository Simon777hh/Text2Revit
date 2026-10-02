"""Helpers for static corner-edge mapping and nine-point AU expansion."""

from __future__ import annotations

import torch


ROOM_START = 4
ROOM_END = 24
CORNER_START = 24
CORNER_END = 52
NUM_ROOMS = 20
MAX_CORNERS_PER_ROOM = 28
AU_POINTS_PER_EDGE = 9
AU_COORD_DIM = AU_POINTS_PER_EDGE * 2
AU_BIT_DIM = AU_COORD_DIM * 8


def build_corner_edge_mapping(corner_features):
    """Return the outgoing edge mask and next-corner token index."""
    device = corner_features.device
    batch_size, sequence_length, _ = corner_features.shape
    room_onehot = corner_features[:, :, ROOM_START:ROOM_END]
    corner_onehot = corner_features[:, :, CORNER_START:CORNER_END]
    valid = room_onehot.sum(dim=-1) > 0.5
    room_ids = room_onehot.argmax(dim=-1)
    corner_ids = corner_onehot.argmax(dim=-1)

    room_corner_counts = room_onehot.sum(dim=1)
    token_room_corner_count = (
        room_onehot * room_corner_counts[:, None, :]
    ).sum(dim=-1)

    index_map = torch.full(
        (batch_size, NUM_ROOMS, MAX_CORNERS_PER_ROOM),
        -1,
        dtype=torch.long,
        device=device,
    )
    batch_ids = torch.arange(
        batch_size,
        device=device,
    ).unsqueeze(1).expand(batch_size, sequence_length)
    valid_batch = batch_ids[valid]
    valid_rooms = room_ids[valid]
    valid_corners = corner_ids[valid]
    valid_tokens = torch.arange(
        sequence_length,
        device=device,
    ).unsqueeze(0).expand(batch_size, sequence_length)[valid]
    index_map[valid_batch, valid_rooms, valid_corners] = valid_tokens

    next_corner_ids = torch.where(
        corner_ids + 1 < token_room_corner_count,
        corner_ids + 1,
        torch.zeros_like(corner_ids),
    )
    safe_corner_ids = torch.where(
        valid,
        next_corner_ids,
        torch.zeros_like(next_corner_ids),
    )
    next_index = index_map[batch_ids, room_ids, safe_corner_ids]
    edge_valid = valid & (token_room_corner_count >= 2)
    return edge_valid, next_index


def build_edge_au_features(
    xy,
    next_index,
    edge_valid,
    num_points: int = AU_POINTS_PER_EDGE,
):
    """Return nine uniformly sampled wall points per valid corner token."""
    batch_size, sequence_length, _ = xy.shape
    device = xy.device
    batch_ids = torch.arange(
        batch_size,
        device=device,
    ).unsqueeze(1)
    end_xy = xy[
        batch_ids,
        next_index.clamp(min=0, max=sequence_length - 1),
    ]
    fractions = torch.linspace(
        0.0,
        1.0,
        num_points,
        device=device,
        dtype=xy.dtype,
    )
    wall_points = (
        (1.0 - fractions)[None, None, :, None] * xy.unsqueeze(2)
        + fractions[None, None, :, None] * end_xy.unsqueeze(2)
    )
    wall_points = wall_points * edge_valid.unsqueeze(-1).unsqueeze(-1).to(
        wall_points.dtype
    )
    return wall_points.reshape(
        batch_size,
        sequence_length,
        num_points * 2,
    )
