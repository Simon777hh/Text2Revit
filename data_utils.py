"""Data contracts for plan-level normalized Flow Matching experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent / "data" / "sources"
FLOW_ROOT = Path(__file__).resolve().parent
_NESTED_DATA_DIR = FLOW_ROOT / "data"
DATA_DIR = (
    _NESTED_DATA_DIR
    if (_NESTED_DATA_DIR / "plan_norm_gt.npz").exists()
    or not (FLOW_ROOT / "plan_norm_gt.npz").exists()
    else FLOW_ROOT
)
VIZ_DIR = FLOW_ROOT / "outputs"

DEFAULT_RAW_GT = DATA_DIR / "raw_gt_tokens.npz"
DEFAULT_RAW_TOPOLOGY = DATA_DIR / "raw_topology.npz"
DEFAULT_RAW_EDGES = DATA_DIR / "raw_edges.npz"
DEFAULT_RAW_MANIFEST = DATA_DIR / "raw_manifest.json"
DEFAULT_OUT_GT = DATA_DIR / "plan_norm_gt.npz"
DEFAULT_OUT_META = DATA_DIR / "plan_norm_meta.npz"

NUM_ROOMS = 20
NUM_CORNERS = 128
ROOM_START = 4
ROOM_END = 24
CORNER_START = 24
CORNER_END = 52
CANVAS_AREA = 4.0


def valid_corner_mask(features: np.ndarray) -> np.ndarray:
    """Return a boolean mask for non-padding corner tokens.

    Supports both a single plan ``[128, 52]`` and a batch
    ``[N, 128, 52]``.
    """
    if features.ndim == 2:
        room_onehot = features[:, ROOM_START:ROOM_END]
    elif features.ndim == 3:
        room_onehot = features[:, :, ROOM_START:ROOM_END]
    else:
        raise ValueError(f"expected a 2D or 3D feature array: {features.shape}")
    return room_onehot.sum(axis=-1) > 0.5


def canonicalize_corner_rings(features: np.ndarray) -> np.ndarray:
    """Reorder each room ring to CCW starting at minimum y, then minimum x."""
    if features.ndim != 3 or features.shape[1:] != (NUM_CORNERS, 52):
        raise ValueError(
            f"expected features [{NUM_CORNERS}, 52], got {features.shape}"
        )
    result = features.astype(np.float32, copy=True)
    for plan_index in range(len(features)):
        valid = valid_corner_mask(features[plan_index])
        room_ids = features[
            plan_index, :, ROOM_START:ROOM_END
        ].argmax(axis=-1)
        corner_ids = features[
            plan_index, :, CORNER_START:CORNER_END
        ].argmax(axis=-1)
        for room_id in np.unique(room_ids[valid]):
            token_indices = np.flatnonzero(
                valid & (room_ids == room_id)
            )
            if token_indices.size < 3:
                continue
            token_indices = token_indices[
                np.argsort(corner_ids[token_indices])
            ]
            coords = features[plan_index, token_indices, :2].astype(
                np.float64
            )
            signed = 0.5 * (
                coords[:, 0] * np.roll(coords[:, 1], -1)
                - np.roll(coords[:, 0], -1) * coords[:, 1]
            ).sum()
            order = np.arange(len(token_indices))
            if signed < 0.0:
                order = order[::-1]
            start = int(
                np.lexsort(
                    (
                        coords[order, 0],
                        coords[order, 1],
                    )
                )[0]
            )
            order = np.roll(order, -start)
            reordered = features[
                plan_index,
                token_indices[order],
            ].copy()
            reordered[:, CORNER_START:CORNER_END] = 0.0
            reordered[
                np.arange(len(reordered)),
                CORNER_START + np.arange(len(reordered)),
            ] = 1.0
            result[plan_index, token_indices] = reordered
    return result


def build_corner_layout(features):
    """Precompute static room/corner layout features once."""
    if features.ndim != 3 or features.shape[1:] != (NUM_CORNERS, 52):
        raise ValueError(
            f"expected features [{NUM_CORNERS}, 52], got {features.shape}"
        )
    room_onehot = features[:, :, ROOM_START:ROOM_END]
    valid = room_onehot.sum(axis=-1) > 0.5
    room_ids = room_onehot.argmax(axis=-1).astype(np.int16)
    corner_ids = features[
        :, :, CORNER_START:CORNER_END
    ].argmax(axis=-1).astype(np.int16)
    room_counts = room_onehot.sum(axis=1).astype(np.int16)
    token_room_counts = room_counts[
        np.arange(len(room_counts))[:, None],
        room_ids,
    ]
    safe_counts = np.maximum(token_room_counts, 1)
    angle = (
        2.0
        * np.pi
        * corner_ids.astype(np.float32)
        / safe_counts.astype(np.float32)
    )
    ring_phase = np.stack(
        [
            np.cos(angle),
            np.sin(angle),
            np.cos(2.0 * angle),
            np.sin(2.0 * angle),
        ],
        axis=-1,
    ).astype(np.float32)
    corner_count_norm = (
        token_room_counts.astype(np.float32)
        / 28.0
    )[:, :, None].astype(np.float32)
    corner_id_norm = (
        corner_ids.astype(np.float32)
        / np.maximum(token_room_counts - 1, 1).astype(np.float32)
    )[:, :, None].astype(np.float32)
    return {
        "room_ids": room_ids,
        "corner_ids": corner_ids,
        "room_corner_counts": room_counts,
        "room_corner_count_norm": (
            room_counts.astype(np.float32) / 28.0
        )[:, :, None].astype(np.float32),
        "ring_phase": ring_phase,
        "corner_count_norm": corner_count_norm,
        "corner_id_norm": corner_id_norm,
        "corner_valid": valid,
    }


def room_polygon_areas(
    features: np.ndarray,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Return per-room polygon areas in the current coordinate space."""
    if features.ndim != 3 or features.shape[1:] != (NUM_CORNERS, 52):
        raise ValueError(
            f"expected features [{NUM_CORNERS}, 52], got {features.shape}"
        )
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    room_areas = np.zeros(
        (len(features), NUM_ROOMS),
        dtype=np.float32,
    )
    corner_slots = np.arange(28, dtype=np.int64)
    for start in range(0, len(features), chunk_size):
        end = min(start + chunk_size, len(features))
        chunk = features[start:end]
        valid = valid_corner_mask(chunk)
        room_ids = chunk[:, :, ROOM_START:ROOM_END].argmax(axis=-1)
        corner_ids = chunk[:, :, CORNER_START:CORNER_END].argmax(axis=-1)

        grid = np.zeros(
            (len(chunk), NUM_ROOMS, 28, 2),
            dtype=np.float64,
        )
        room_counts = np.zeros(
            (len(chunk), NUM_ROOMS),
            dtype=np.int64,
        )
        batch_ids, token_ids = np.nonzero(valid)
        grid[
            batch_ids,
            room_ids[batch_ids, token_ids],
            corner_ids[batch_ids, token_ids],
        ] = chunk[batch_ids, token_ids, :2]
        np.add.at(
            room_counts,
            (batch_ids, room_ids[batch_ids, token_ids]),
            1,
        )

        safe_counts = np.maximum(room_counts, 1)
        next_slots = (
            corner_slots[None, None, :] + 1
        ) % safe_counts[:, :, None]
        next_xy = np.take_along_axis(
            grid,
            next_slots[..., None],
            axis=2,
        )
        cross = (
            grid[..., 0] * next_xy[..., 1]
            - next_xy[..., 0] * grid[..., 1]
        )
        chunk_areas = 0.5 * np.abs(cross.sum(axis=2))
        chunk_areas[room_counts < 3] = 0.0
        room_areas[start:end] = chunk_areas.astype(np.float32)
    return room_areas


def room_area_ratios(
    areas: np.ndarray,
    functional_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize functional room areas to per-plan ratios."""
    if areas.shape != functional_mask.shape:
        raise ValueError("areas and functional_mask must have the same shape")
    total = (areas * functional_mask).sum(axis=1, keepdims=True)
    ratios = np.divide(
        areas,
        total,
        out=np.zeros_like(areas, dtype=np.float32),
        where=total > 0.0,
    )
    ratios[~functional_mask] = 0.0
    return ratios.astype(np.float32), total[:, 0].astype(np.float32)


def build_area_condition(
    area_ratios: np.ndarray,
    node_size_ids: np.ndarray,
    feature_mask: np.ndarray,
    functional_mask: np.ndarray,
) -> np.ndarray:
    """Build the per-room ratio, discrete level, and text-feature condition."""
    if not (
        area_ratios.shape
        == node_size_ids.shape
        == feature_mask.shape
        == functional_mask.shape
    ):
        raise ValueError("area condition arrays must have the same shape")
    size_normalized = np.zeros_like(
        node_size_ids,
        dtype=np.float32,
    )
    valid_size = functional_mask & (node_size_ids >= 0)
    size_normalized[valid_size] = (
        node_size_ids[valid_size].astype(np.float32) / 2.0
    )
    return np.stack(
        [
            area_ratios.astype(np.float32),
            size_normalized,
            feature_mask.astype(np.float32),
        ],
        axis=-1,
    ).astype(np.float32)


def normalize_plans(
    features: np.ndarray,
    plan_ids: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Apply plan-level max-edge normalization while preserving aspect ratio.

    Coordinates use one shared scale for x and y:

        normalized_xy = (xy - bbox_center) / (0.5 * max(bbox_size))

    Therefore the longest footprint edge maps to [-1, 1] and the shorter
    edge remains proportionally smaller. Absolute plan area and absolute
    footprint size are discarded from the coordinates, but aspect ratio is
    preserved implicitly for the model to learn.
    """
    if features.ndim != 3 or features.shape[1:] != (NUM_CORNERS, 52):
        raise ValueError(
            f"expected features [{NUM_CORNERS}, 52], got {features.shape}"
        )

    normalized = features.astype(np.float32, copy=True)
    valid = valid_corner_mask(features)

    bbox_min = np.full((len(features), 2), np.nan, dtype=np.float32)
    bbox_max = np.full((len(features), 2), np.nan, dtype=np.float32)
    center = np.full((len(features), 2), np.nan, dtype=np.float32)
    half_span = np.full(len(features), np.nan, dtype=np.float32)

    for plan_index in range(len(features)):
        plan_valid = valid[plan_index]
        if not np.any(plan_valid):
            raise ValueError(f"plan index {plan_index} has no valid corners")

        xy = features[plan_index, plan_valid, :2]
        if not np.isfinite(xy).all():
            raise ValueError(
                f"plan index {plan_index} contains non-finite coordinates"
            )

        current_min = xy.min(axis=0)
        current_max = xy.max(axis=0)
        current_size = current_max - current_min
        current_half_span = 0.5 * float(current_size.max())
        if current_half_span < 1e-6:
            raise ValueError(
                f"plan index {plan_index} has a degenerate footprint"
            )

        current_center = 0.5 * (current_min + current_max)
        normalized[plan_index, plan_valid, :2] = (
            xy - current_center
        ) / current_half_span
        normalized[plan_index, plan_valid, 2:4] = (
            features[plan_index, plan_valid, 2:4] - current_center
        ) / current_half_span

        bbox_min[plan_index] = current_min
        bbox_max[plan_index] = current_max
        center[plan_index] = current_center
        half_span[plan_index] = current_half_span

    # Recompute room centers exactly from the normalized ring coordinates.
    room_onehot = normalized[:, :, ROOM_START:ROOM_END]
    room_counts = np.clip(
        room_onehot.sum(axis=1, keepdims=True),
        a_min=1.0,
        a_max=None,
    )
    room_centers = (
        room_onehot.transpose(0, 2, 1)
        @ normalized[:, :, :2]
        / room_counts.transpose(0, 2, 1)
    )
    normalized[:, :, 2:4] = room_onehot @ room_centers

    bbox_size = bbox_max - bbox_min
    width = bbox_size[:, 0]
    height = bbox_size[:, 1]
    metadata = {
        "plan_ids": np.asarray(plan_ids, dtype=np.int64),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "bbox_size": bbox_size,
        "center_offset": center,
        "scale": half_span,
        "aspect_ratio": (width / height).astype(np.float32),
        "orientation_wide": (width >= height).astype(np.bool_),
    }
    return normalized.astype(np.float32), metadata


def normalize_plans_stretched(
    features: np.ndarray,
    plan_ids: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Normalize x and y independently so every plan bbox becomes [-1, 1].

    This is the current main-pipeline contract. It removes aspect ratio by
    applying an anisotropic scale:

        normalized_x = (x - bbox_center_x) / (0.5 * bbox_width)
        normalized_y = (y - bbox_center_y) / (0.5 * bbox_height)
    """
    if features.ndim != 3 or features.shape[1:] != (NUM_CORNERS, 52):
        raise ValueError(
            f"expected features [{NUM_CORNERS}, 52], got {features.shape}"
        )

    stretched = features.astype(np.float32, copy=True)
    valid = valid_corner_mask(features)

    bbox_min = np.full((len(features), 2), np.nan, dtype=np.float32)
    bbox_max = np.full((len(features), 2), np.nan, dtype=np.float32)
    center = np.full((len(features), 2), np.nan, dtype=np.float32)
    half_size = np.full((len(features), 2), np.nan, dtype=np.float32)

    for plan_index in range(len(features)):
        plan_valid = valid[plan_index]
        if not np.any(plan_valid):
            raise ValueError(f"plan index {plan_index} has no valid corners")

        xy = features[plan_index, plan_valid, :2]
        if not np.isfinite(xy).all():
            raise ValueError(
                f"plan index {plan_index} contains non-finite coordinates"
            )

        current_min = xy.min(axis=0)
        current_max = xy.max(axis=0)
        current_size = current_max - current_min
        if np.any(current_size < 1e-6):
            raise ValueError(
                f"plan index {plan_index} has a degenerate footprint"
            )

        current_center = 0.5 * (current_min + current_max)
        current_half_size = 0.5 * current_size
        stretched[plan_index, plan_valid, :2] = (
            xy - current_center
        ) / current_half_size
        stretched[plan_index, plan_valid, 2:4] = (
            features[plan_index, plan_valid, 2:4] - current_center
        ) / current_half_size

        bbox_min[plan_index] = current_min
        bbox_max[plan_index] = current_max
        center[plan_index] = current_center
        half_size[plan_index] = current_half_size

    room_onehot = stretched[:, :, ROOM_START:ROOM_END]
    room_counts = np.clip(
        room_onehot.sum(axis=1, keepdims=True),
        a_min=1.0,
        a_max=None,
    )
    room_centers = (
        room_onehot.transpose(0, 2, 1)
        @ stretched[:, :, :2]
        / room_counts.transpose(0, 2, 1)
    )
    stretched[:, :, 2:4] = room_onehot @ room_centers

    bbox_size = bbox_max - bbox_min
    width = bbox_size[:, 0]
    height = bbox_size[:, 1]
    metadata = {
        "plan_ids": np.asarray(plan_ids, dtype=np.int64),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "bbox_size": bbox_size,
        "center_offset": center,
        "scale_xy": half_size,
        "aspect_ratio": (width / height).astype(np.float32),
        "orientation_wide": (width >= height).astype(np.bool_),
    }
    return stretched.astype(np.float32), metadata


def denormalize_coords(
    coords: np.ndarray,
    plan_index: int,
    metadata: dict[str, np.ndarray],
) -> np.ndarray:
    """Map normalized coordinates back to the combined canvas space."""
    if "scale_xy" in metadata:
        scale = metadata["scale_xy"][plan_index]
    else:
        scale = float(metadata["scale"][plan_index])
    center = metadata["center_offset"][plan_index]
    return (
        np.asarray(coords, dtype=np.float32) * scale + center
    ).astype(np.float32)


def stretch_plans_to_square(
    features: np.ndarray,
) -> np.ndarray:
    """Normalize x and y independently so every bbox becomes [-1, 1].

    This wrapper keeps the comparison utilities independent from plan
    metadata.
    """
    plan_ids = np.arange(len(features), dtype=np.int64)
    stretched, _ = normalize_plans_stretched(features, plan_ids)
    return stretched


def load_plan_order(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def source_labels(
    plan_order: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Return source strings and original source ids in manifest order."""
    sources = np.asarray(
        [row["source"] for row in plan_order],
        dtype="<U16",
    )
    source_ids = np.asarray(
        [row["source_id"] for row in plan_order],
        dtype=np.int64,
    )
    return sources, source_ids
