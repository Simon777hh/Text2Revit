"""Shared differentiable geometry helpers for Flow Matching train/infer."""

from __future__ import annotations

import math

import torch


EPS = 1e-6
MAX_CORNERS_PER_ROOM = 28
D4_MATRICES = (
    (1.0, 0.0, 0.0, 1.0),
    (0.0, -1.0, 1.0, 0.0),
    (-1.0, 0.0, 0.0, -1.0),
    (0.0, 1.0, -1.0, 0.0),
    (-1.0, 0.0, 0.0, 1.0),
    (0.0, -1.0, -1.0, 0.0),
    (1.0, 0.0, 0.0, -1.0),
    (0.0, 1.0, 1.0, 0.0),
)


def _apply_d4(xy, transforms):
    matrix = xy.new_tensor(D4_MATRICES).reshape(
        len(D4_MATRICES),
        2,
        2,
    )[transforms]
    return torch.einsum(
        "bij,bnj->bni",
        matrix,
        xy,
    )


def augment_d4_targets(
    x0_xy,
    corner_layout,
    transforms=None,
):
    """Apply an independent D4 transform to the GT target geometry.

    The source prior is intentionally left in its original frame. Token slots
    are preserved. Corner ids, ring phase, and next-edge mapping are
    recomputed from the transformed target geometry.
    """
    batch_size = x0_xy.shape[0]
    device = x0_xy.device
    if transforms is None:
        transforms = torch.randint(
            0,
            len(D4_MATRICES),
            (batch_size,),
            device=device,
        )

    x0_xy = _apply_d4(x0_xy, transforms)

    valid = corner_layout["corner_valid"].bool()
    room_ids = corner_layout["room_ids"].long()
    old_ids = corner_layout["corner_ids"].long()
    counts = x0_xy.new_zeros((batch_size, 20))
    for room_id in range(20):
        counts[:, room_id] = (
            valid & (room_ids == room_id)
        ).sum(dim=1)

    token_index = torch.arange(
        x0_xy.shape[1],
        device=device,
    ).unsqueeze(0).expand(batch_size, -1)
    dummy_token = x0_xy.shape[1]
    dummy_flat = 20 * MAX_CORNERS_PER_ROOM
    old_flat_room = room_ids * MAX_CORNERS_PER_ROOM + old_ids
    safe_old_flat_room = torch.where(
        valid,
        old_flat_room,
        torch.full_like(old_flat_room, dummy_flat),
    )
    old_index_map = torch.full(
        (batch_size, dummy_flat + 1),
        -1,
        dtype=torch.long,
        device=device,
    )
    old_index_map.scatter_(1, safe_old_flat_room, token_index)

    id_slots = torch.arange(
        MAX_CORNERS_PER_ROOM,
        device=device,
    ).view(1, -1)
    new_ids_buffer = torch.full(
        (batch_size, x0_xy.shape[1] + 1),
        -1,
        dtype=torch.long,
        device=device,
    )
    for room_id in range(20):
        count = counts[:, room_id]
        count_long = count.long().unsqueeze(1)
        slot_mask = id_slots < count_long
        gather_flat = (
            room_id * MAX_CORNERS_PER_ROOM + id_slots
        )
        token_positions = torch.gather(
            old_index_map,
            1,
            gather_flat.expand(batch_size, -1),
        )
        safe_positions = torch.where(
            slot_mask,
            token_positions,
            torch.zeros_like(token_positions),
        ).clamp(min=0, max=x0_xy.shape[1] - 1)
        ordered_xy = torch.gather(
            x0_xy,
            1,
            safe_positions.unsqueeze(-1).expand(-1, -1, 2),
        )
        ordered_xy = torch.where(
            slot_mask.unsqueeze(-1),
            ordered_xy,
            torch.zeros_like(ordered_xy),
        )
        next_positions = (
            id_slots + 1
        ) % count.clamp(min=1.0).long().unsqueeze(1)
        next_xy = torch.gather(
            ordered_xy,
            1,
            next_positions.unsqueeze(-1).expand(-1, -1, 2),
        )
        signed_area = 0.5 * (
            ordered_xy[..., 0] * next_xy[..., 1]
            - next_xy[..., 0] * ordered_xy[..., 1]
        ).sum(dim=1)
        reverse_positions = (
            count.long().unsqueeze(1) - 1 - id_slots
        ).clamp(min=0)
        reversed_xy = torch.gather(
            ordered_xy,
            1,
            reverse_positions.unsqueeze(-1).expand(-1, -1, 2),
        )
        reverse_mask = (
            (signed_area < 0.0) & (count >= 3.0)
        )
        reversed_tokens = torch.gather(
            token_positions,
            1,
            reverse_positions,
        )
        oriented_xy = torch.where(
            reverse_mask[:, None, None],
            reversed_xy,
            ordered_xy,
        )
        oriented_tokens = torch.where(
            reverse_mask[:, None],
            reversed_tokens,
            token_positions,
        )

        x_key = torch.where(
            slot_mask,
            oriented_xy[..., 0],
            torch.full_like(oriented_xy[..., 0], float("inf")),
        )
        y_key = torch.where(
            slot_mask,
            oriented_xy[..., 1],
            torch.full_like(oriented_xy[..., 1], float("inf")),
        )
        order_by_x = torch.argsort(x_key, dim=1, stable=True)
        y_in_x_order = torch.gather(y_key, 1, order_by_x)
        order_within = torch.argsort(
            y_in_x_order,
            dim=1,
            stable=True,
        )
        lex_order = torch.gather(order_by_x, 1, order_within)
        start_position = lex_order[:, :1]
        rotate_positions = (
            id_slots + start_position
        ) % count.clamp(min=1.0).long().unsqueeze(1)
        canonical_tokens = torch.gather(
            oriented_tokens,
            1,
            rotate_positions,
        )
        scatter_tokens = torch.where(
            slot_mask,
            canonical_tokens.clamp(min=0, max=dummy_token),
            torch.full_like(canonical_tokens, dummy_token),
        )
        scatter_values = torch.where(
            slot_mask,
            id_slots.expand_as(slot_mask),
            torch.full_like(slot_mask, -1, dtype=torch.long),
        )
        new_ids_buffer.scatter_(
            1,
            scatter_tokens,
            scatter_values,
        )

    new_ids = new_ids_buffer[:, : x0_xy.shape[1]]
    new_ids = torch.where(
        valid,
        new_ids,
        torch.zeros_like(new_ids),
    )

    token_counts = torch.gather(
        counts,
        1,
        room_ids.clamp(min=0, max=19),
    )
    safe_token_counts = token_counts.clamp(min=1.0)
    angle = (
        2.0
        * math.pi
        * new_ids.to(x0_xy.dtype)
        / safe_token_counts
    )
    ring_phase = torch.stack(
        [
            torch.cos(angle),
            torch.sin(angle),
            torch.cos(2.0 * angle),
            torch.sin(2.0 * angle),
        ],
        dim=-1,
    )
    corner_id_norm = (
        new_ids.to(x0_xy.dtype)
        / (token_counts - 1.0).clamp(min=1.0)
    ).unsqueeze(-1)

    flat_room = room_ids * MAX_CORNERS_PER_ROOM + new_ids
    dummy_index = 20 * MAX_CORNERS_PER_ROOM
    safe_flat_room = torch.where(
        valid,
        flat_room,
        torch.full_like(flat_room, dummy_index),
    )
    index_map = torch.full(
        (batch_size, dummy_index + 1),
        -1,
        dtype=torch.long,
        device=device,
    )
    token_index = torch.arange(
        x0_xy.shape[1],
        device=device,
    ).unsqueeze(0).expand(batch_size, -1)
    index_map.scatter_(1, safe_flat_room, token_index)
    next_ids = torch.where(
        new_ids + 1 < token_counts.long(),
        new_ids + 1,
        torch.zeros_like(new_ids),
    )
    next_flat_room = room_ids * MAX_CORNERS_PER_ROOM + next_ids
    next_index = torch.gather(
        index_map,
        1,
        next_flat_room.clamp(min=0, max=dummy_index),
    )
    next_index = torch.where(
        valid & (token_counts >= 2.0),
        next_index,
        torch.zeros_like(next_index),
    )
    edge_valid = valid & (token_counts >= 2.0)

    id_map = torch.full(
        (batch_size, dummy_index + 1),
        -1,
        dtype=torch.long,
        device=device,
    )
    id_map.scatter_(
        1,
        safe_old_flat_room,
        new_ids,
    )
    corner_id_map = id_map[:, :dummy_index].reshape(
        batch_size,
        20,
        MAX_CORNERS_PER_ROOM,
    )

    transformed_layout = {
        "room_ids": room_ids,
        "corner_ids": new_ids,
        "ring_phase": ring_phase,
        "corner_id_norm": corner_id_norm,
        "corner_valid": valid,
        "room_corner_counts": corner_layout.get(
            "room_corner_counts"
        ),
        "room_slot_valid": corner_layout.get("room_slot_valid"),
    }
    return (
        x0_xy,
        transformed_layout,
        next_index,
        edge_valid,
        corner_id_map,
    )


def remap_shared_wall_edges(
    wall_rooms,
    wall_edges,
    corner_id_map,
    wall_valid=None,
):
    """Map canonical wall edge ids to the current transformed ring ids."""
    batch_size, wall_count = wall_rooms.shape[:2]
    safe_rooms = wall_rooms.long().clamp(min=0, max=19)
    safe_edges = wall_edges.long().clamp(
        min=0,
        max=MAX_CORNERS_PER_ROOM - 1,
    )
    keys = (
        safe_rooms.unsqueeze(-1) * MAX_CORNERS_PER_ROOM
        + safe_edges
    )
    flat_map = corner_id_map.reshape(batch_size, -1)
    remapped = torch.gather(
        flat_map,
        1,
        keys.reshape(batch_size, -1),
    ).reshape(batch_size, wall_count, 2, 2)
    if wall_valid is not None:
        remapped = torch.where(
            wall_valid.bool().unsqueeze(-1).unsqueeze(-1),
            remapped,
            torch.zeros_like(remapped),
        )
    return remapped


def remap_shared_wall_t(
    wall_rooms,
    wall_edges,
    wall_t,
    corner_id_map,
    room_corner_counts,
):
    """Keep normalized wall parameters under D4 remapping.

    ``remap_shared_wall_edges`` maps each endpoint independently. If a
    reflection reverses a room ring, the mapped start/end pair already
    describes the reflected segment in the same parametric direction, so
    applying an extra ``1 - t`` flip changes the wall segment.
    """
    return wall_t.clone()


def replace_corner_xy(base_features, xy, corner_layout):
    """Clone static corner features and replace coordinates plus room centers."""
    valid = corner_layout["corner_valid"].bool()
    room_ids = corner_layout["room_ids"].long()
    safe_room_ids = torch.where(
        valid,
        room_ids,
        torch.zeros_like(room_ids),
    )
    safe_xy = torch.where(
        valid.unsqueeze(-1),
        xy,
        torch.zeros_like(xy),
    )

    result = base_features.clone()
    result[:, :, :2] = safe_xy

    batch_size, num_rooms = base_features.shape[0], 20
    centers = result.new_zeros((batch_size, num_rooms, 2))
    counts = result.new_zeros((batch_size, num_rooms, 1))
    centers.scatter_add_(
        1,
        safe_room_ids.unsqueeze(-1).expand(-1, -1, 2),
        safe_xy,
    )
    counts.scatter_add_(
        1,
        safe_room_ids.unsqueeze(-1),
        valid.to(result.dtype).unsqueeze(-1),
    )
    centers = centers / counts.clamp(min=1.0)
    result[:, :, 2:4] = torch.gather(
        centers,
        1,
        safe_room_ids.unsqueeze(-1).expand(
            -1,
            -1,
            centers.shape[-1],
        ),
    )
    return result


def resolve_shared_wall_tokens(
    corner_layout,
    wall_rooms,
    wall_edges,
):
    """Resolve room/edge wall descriptions to current token slots."""
    if wall_rooms is None or wall_edges is None:
        return None
    room_ids = corner_layout["room_ids"].long()
    corner_ids = corner_layout["corner_ids"].long()
    valid = corner_layout["corner_valid"].bool()
    batch_size, token_count = room_ids.shape
    lookup = torch.full(
        (batch_size, 20 * MAX_CORNERS_PER_ROOM + 1),
        -1,
        dtype=torch.long,
        device=room_ids.device,
    )
    token_index = torch.arange(
        token_count,
        device=room_ids.device,
    ).view(1, -1).expand(batch_size, -1)
    safe_room = room_ids.clamp(
        min=0,
        max=19,
    )
    safe_corner = corner_ids.clamp(
        min=0,
        max=MAX_CORNERS_PER_ROOM - 1,
    )
    safe_key = (
        safe_room * MAX_CORNERS_PER_ROOM + safe_corner
    )
    dummy_key = 20 * MAX_CORNERS_PER_ROOM
    lookup.scatter_(
        1,
        torch.where(
            valid,
            safe_key,
            torch.full_like(safe_key, dummy_key),
        ),
        token_index,
    )
    wall_rooms = wall_rooms.long().clamp(min=0, max=19)
    wall_edges = wall_edges.long().clamp(
        min=0,
        max=MAX_CORNERS_PER_ROOM - 1,
    )
    keys = torch.stack(
        [
            wall_rooms[:, :, 0]
            * MAX_CORNERS_PER_ROOM
            + wall_edges[:, :, 0, 0],
            wall_rooms[:, :, 0]
            * MAX_CORNERS_PER_ROOM
            + wall_edges[:, :, 0, 1],
            wall_rooms[:, :, 1]
            * MAX_CORNERS_PER_ROOM
            + wall_edges[:, :, 1, 0],
            wall_rooms[:, :, 1]
            * MAX_CORNERS_PER_ROOM
            + wall_edges[:, :, 1, 1],
        ],
        dim=-1,
    )
    wall_count = keys.shape[1]
    resolved = torch.gather(
        lookup,
        1,
        keys.reshape(batch_size, -1),
    )
    return resolved.reshape(batch_size, wall_count, 4)


def project_shared_walls(
    xy,
    corner_layout,
    wall_rooms,
    wall_edges,
    wall_valid,
    quantize=False,
    strength=1.0,
    max_distance=None,
):
    """Make reciprocal GT wall edges share exactly one segment.

    The static wall file stores the source room and local edge ids. Resolving
    those ids through ``corner_layout`` keeps the projection correct after D4
    augmentation, where canonical corner ids and token slots are reordered.
    """
    if (
        wall_rooms is None
        or wall_edges is None
        or wall_valid is None
    ):
        return xy
    batch_size, token_count, _ = xy.shape
    wall_count = wall_valid.shape[1]
    if wall_count == 0:
        return xy

    indices = resolve_shared_wall_tokens(
        corner_layout,
        wall_rooms,
        wall_edges,
    )
    if indices is None:
        return xy
    indices = indices.clamp(
        min=0,
        max=token_count - 1,
    )

    def gather_token(position):
        return torch.gather(
            xy,
            1,
            indices[:, :, position].unsqueeze(-1).expand(
                -1,
                -1,
                2,
            ),
        )

    point_0 = gather_token(0)
    point_1 = gather_token(1)
    point_2 = gather_token(2)
    point_3 = gather_token(3)

    direct_start = (point_0 - point_2).square().sum(dim=-1)
    direct_end = (point_1 - point_3).square().sum(dim=-1)
    reversed_start = (point_0 - point_3).square().sum(dim=-1)
    reversed_end = (point_1 - point_2).square().sum(dim=-1)
    direct_max = torch.maximum(direct_start, direct_end)
    reversed_max = torch.maximum(reversed_start, reversed_end)
    reversed_pair = reversed_max < direct_max
    match_distance = torch.minimum(
        direct_max,
        reversed_max,
    ).clamp(min=0.0).sqrt()
    point_2_matched = torch.where(
        reversed_pair.unsqueeze(-1),
        point_3,
        point_2,
    )
    point_3_matched = torch.where(
        reversed_pair.unsqueeze(-1),
        point_2,
        point_3,
    )

    target_start = 0.5 * (point_0 + point_2_matched)
    target_end = 0.5 * (point_1 + point_3_matched)
    point_2_target = torch.where(
        reversed_pair.unsqueeze(-1),
        target_end,
        target_start,
    )
    point_3_target = torch.where(
        reversed_pair.unsqueeze(-1),
        target_start,
        target_end,
    )
    if strength != 1.0:
        target_start = (
            (1.0 - strength) * point_0
            + strength * target_start
        )
        target_end = (
            (1.0 - strength) * point_1
            + strength * target_end
        )
        point_2_target = (
            (1.0 - strength) * point_2
            + strength * point_2_target
        )
        point_3_target = (
            (1.0 - strength) * point_3
            + strength * point_3_target
        )
    targets = torch.stack(
        [
            target_start,
            target_end,
            point_2_target,
            point_3_target,
        ],
        dim=2,
    )

    pair_valid = wall_valid.bool()
    if max_distance is not None:
        pair_valid = pair_valid & (
            match_distance <= float(max_distance)
        )
    valid = pair_valid.unsqueeze(-1).expand(
        -1,
        -1,
        4,
    )
    flat_indices = indices.reshape(batch_size, -1)
    flat_valid = valid.reshape(batch_size, -1)
    flat_targets = targets.reshape(batch_size, -1, 2)
    safe_indices = torch.where(
        flat_valid,
        flat_indices,
        torch.zeros_like(flat_indices),
    )
    weights = flat_valid.to(flat_targets.dtype)
    accumulated = xy.new_zeros((batch_size, token_count, 2))
    counts = xy.new_zeros((batch_size, token_count))
    accumulated.scatter_add_(
        1,
        safe_indices.unsqueeze(-1).expand(-1, -1, 2),
        flat_targets * weights.unsqueeze(-1),
    )
    counts.scatter_add_(1, safe_indices, weights)
    has_projection = counts > 0.0
    average = accumulated / counts.clamp(min=1.0).unsqueeze(-1)
    projected = torch.where(
        has_projection.unsqueeze(-1),
        average,
        xy,
    )
    if quantize:
        projected = torch.round(
            (projected.clamp(-1.0, 1.0) / 2.0 + 0.5) * 255.0
        ) / 255.0 * 2.0 - 1.0
    return projected


def room_polygon_vertices(xy, corner_layout, counts=None):
    """Return per-room vertex grids and valid corner counts."""
    valid = corner_layout["corner_valid"].bool()
    room_ids = corner_layout["room_ids"].long()
    corner_ids = corner_layout["corner_ids"].long()
    batch_size = xy.shape[0]
    num_rooms = 20

    safe_room_ids = torch.where(
        valid,
        room_ids,
        torch.zeros_like(room_ids),
    ).clamp(min=0, max=num_rooms - 1)
    safe_corner_ids = torch.where(
        valid,
        corner_ids,
        torch.zeros_like(corner_ids),
    ).clamp(min=0, max=MAX_CORNERS_PER_ROOM - 1)
    safe_xy = torch.where(
        valid.unsqueeze(-1),
        xy,
        torch.zeros_like(xy),
    )

    grid = xy.new_zeros(
        (batch_size, num_rooms, MAX_CORNERS_PER_ROOM, 2)
    )
    flat_index = safe_room_ids * MAX_CORNERS_PER_ROOM + safe_corner_ids
    grid.view(batch_size, num_rooms * MAX_CORNERS_PER_ROOM, 2).scatter_add_(
        1,
        flat_index.unsqueeze(-1).expand(-1, -1, 2),
        safe_xy,
    )

    if counts is None:
        counts = xy.new_zeros((batch_size, num_rooms))
        counts.scatter_add_(
            1,
            safe_room_ids,
            valid.to(xy.dtype),
        )
    else:
        counts = counts.to(dtype=xy.dtype)
    return grid, counts


def room_polygon_areas(xy, corner_layout):
    """Return differentiable polygon areas for every room in every batch."""
    grid, counts = room_polygon_vertices(
        xy,
        corner_layout,
        corner_layout.get("room_corner_counts"),
    )
    safe_counts = counts.clamp(min=1.0)
    corner_slots = torch.arange(
        MAX_CORNERS_PER_ROOM,
        device=xy.device,
    ).view(1, 1, -1)
    next_slots = (corner_slots + 1) % safe_counts.long().unsqueeze(-1)
    next_xy = torch.gather(
        grid,
        2,
        next_slots.unsqueeze(-1).expand(
            -1,
            -1,
            -1,
            grid.shape[-1],
        ),
    )
    cross = grid[..., 0] * next_xy[..., 1] - next_xy[..., 0] * grid[
        ..., 1
    ]
    areas = 0.5 * cross.sum(dim=-1).abs()
    return torch.where(
        counts >= 3,
        areas,
        torch.zeros_like(areas),
    )


def room_area_ratios(areas, functional_mask):
    """Normalize predicted room areas to a per-plan functional ratio."""
    if areas.shape != functional_mask.shape:
        raise ValueError("areas and functional_mask must have the same shape")
    masked_areas = areas * functional_mask.to(areas.dtype)
    total = masked_areas.sum(dim=1, keepdim=True)
    ratios = masked_areas / total.clamp(min=EPS)
    return torch.where(
        functional_mask,
        ratios,
        torch.zeros_like(ratios),
    )
