"""Deterministic geometry cleanup for generated floor-plan polygons.

The postprocessor works on room polygons, not model logits. It is intended
to turn an approximately correct inference result into a clean axis-aligned
room partition suitable for downstream CAD import.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union


ROOM_TYPE_NAMES = {
    0: "living",
    1: "kitchen",
    2: "bedroom",
    3: "bathroom",
    4: "balcony",
    5: "storage",
    6: "stair",
    7: "front_door",
    8: "door",
}

TYPE_PRIORITY = {
    "living": 100,
    "bedroom": 70,
    "kitchen": 60,
    "bathroom": 45,
    "stair": 40,
    "storage": 30,
    "balcony": 20,
    "front_door": 10,
    "door": 5,
}


@dataclass
class PostProcessConfig:
    orthogonalize: bool = True
    use_grid_partition: bool = False
    close_gaps: bool = True
    resolve_overlaps: bool = True
    fill_holes: bool = True
    angle_tolerance_deg: float = 12.0
    coordinate_tolerance: float = 0.015
    gap_tolerance: float = 0.20
    max_edge_move: float = 0.24
    min_shared_length: float = 0.015
    overlap_tolerance: float = 0.00005
    min_overlap_action_area: float = 0.01
    grid_cluster_tolerance: float = 0.02
    wall_segment_tolerance: float = 0.05
    grid_fill_distance: float = 0.08
    max_head_area: float = 0.02
    max_head_width: float = 0.08
    reject_heads: bool = False
    max_hole_area: float = 0.02
    max_hole_span: float = 0.30
    max_self_hole_area: float = 0.15
    max_self_hole_span: float = 0.70
    max_hollow_gap: float = 0.35
    max_hollow_area: float = 0.12
    thin_room_max_short_edge: float = 0.09
    thin_room_min_aspect_ratio: float = 3.5
    thin_room_expand_factor: float = 3.0
    balcony_snap_tolerance: float = 0.30
    seam_snap_tolerance: float = 0.02
    global_seam_tolerance: float = 0.02
    preserve_types: tuple[str, ...] = ("living",)
    grid: float = 0.005
    point_merge_tolerance: float = 0.025
    coordinate_merge_tolerance: float = 0.02
    canonical_iterations: int = 5
    endpoint_snap_tolerance: float = 0.06
    min_door_shared_length: float = 0.15
    max_door_edge_move: float = 0.12
    max_door_edge_area_change: float = 0.035
    door_required_room_types: tuple[str, ...] = (
        "living",
        "bedroom",
        "kitchen",
        "bathroom",
    )


@dataclass
class RoomPolygon:
    room_id: int
    type_name: str
    polygon: Polygon
    name: str = ""
    fixed: bool = False
    source_polygon: Polygon | None = None
    rejected: bool = False
    token_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    source_area: float = 0.0


def _polygon_parts(polygon):
    if polygon is None or polygon.is_empty:
        return []
    if polygon.geom_type == "Polygon":
        return [polygon]
    if polygon.geom_type == "MultiPolygon":
        return [part for part in polygon.geoms if not part.is_empty]
    return []


def _line_parts(geometry):
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]
    if geometry.geom_type == "MultiLineString":
        return [part for part in geometry.geoms if not part.is_empty]
    return []


def _largest_polygon(polygon):
    parts = _polygon_parts(polygon)
    if not parts:
        return None
    return max(parts, key=lambda part: part.area)


def _clean_polygon(polygon):
    if polygon is None or polygon.is_empty:
        return None
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    polygon = _largest_polygon(polygon)
    if polygon is None or polygon.area <= 1e-8:
        return None
    return orient(polygon, sign=1.0)


def _dedupe_points(points, tolerance=1e-7):
    result = []
    for point in points:
        point = np.asarray(point, dtype=float)
        if not result or np.linalg.norm(point - result[-1]) > tolerance:
            result.append(point)
    while (
        len(result) > 1
        and np.linalg.norm(result[0] - result[-1]) <= tolerance
    ):
        result.pop()
    return result


def _remove_collinear(points, tolerance=1e-7):
    if len(points) < 3:
        return points
    result = list(points)
    changed = True
    while changed and len(result) > 3:
        changed = False
        for index in range(len(result)):
            previous = result[index - 1]
            current = result[index]
            following = result[(index + 1) % len(result)]
            first = current - previous
            second = following - previous
            area = abs(
                first[0] * second[1] - first[1] * second[0]
            )
            if area <= tolerance:
                result.pop(index)
                changed = True
                break
    return result


def _polygon_from_points(points):
    points = _dedupe_points(points)
    points = _remove_collinear(points)
    if len(points) < 3:
        return None
    return _clean_polygon(Polygon(points))


def _adds_spurious_head(source, candidate, config):
    if source is None or candidate is None:
        return False
    added = candidate.difference(source)
    for part in _polygon_parts(added):
        min_x, min_y, max_x, max_y = part.bounds
        span_x = max_x - min_x
        span_y = max_y - min_y
        if (
            part.area <= config.max_head_area
            and min(span_x, span_y) <= config.max_head_width
        ):
            return True
    return False


def records_from_xy(
    xy,
    corner_layout,
    node_features,
):
    """Build room polygons from one normalized model output sample."""
    if hasattr(xy, "detach"):
        xy = xy.detach().cpu().numpy()
    else:
        xy = np.asarray(xy)
    if xy.ndim == 3:
        if xy.shape[0] != 1:
            raise ValueError("records_from_xy expects one plan")
        xy = xy[0]

    if hasattr(node_features, "detach"):
        node_features = node_features.detach().cpu().numpy()
    else:
        node_features = np.asarray(node_features)
    if node_features.ndim == 3:
        node_features = node_features[0]

    room_ids = corner_layout["room_ids"]
    corner_ids = corner_layout["corner_ids"]
    valid = corner_layout["corner_valid"]
    if hasattr(room_ids, "detach"):
        room_ids = room_ids.detach().cpu().numpy()
        corner_ids = corner_ids.detach().cpu().numpy()
        valid = valid.detach().cpu().numpy()
    room_ids = np.asarray(room_ids).reshape(-1).astype(np.int64)
    corner_ids = np.asarray(corner_ids).reshape(-1).astype(np.int64)
    valid = np.asarray(valid).reshape(-1).astype(bool)

    type_indices = np.argmax(node_features[:, 20:29], axis=1)
    records = []
    type_counts = {}
    for room_id in np.unique(room_ids[valid]):
        indices = np.flatnonzero(valid & (room_ids == room_id))
        if indices.size < 3:
            continue
        order = np.argsort(corner_ids[indices], kind="stable")
        indices = indices[order]
        polygon = _polygon_from_points(xy[indices])
        if polygon is None:
            continue
        type_name = ROOM_TYPE_NAMES.get(
            int(type_indices[room_id]),
            "unknown",
        )
        count = type_counts.get(type_name, 0)
        type_counts[type_name] = count + 1
        records.append(
            RoomPolygon(
                room_id=int(room_id),
                type_name=type_name,
                polygon=polygon,
                name=f"{type_name}_{count}",
                fixed=type_name in PostProcessConfig.preserve_types,
                source_polygon=polygon,
                token_indices=indices.astype(np.int64),
                source_area=float(polygon.area),
            )
        )
    return records


def adjacency_pairs(adjacency):
    adjacency = np.asarray(adjacency)
    if adjacency.ndim == 3:
        adjacency = adjacency[0]
    pairs = set()
    for first in range(adjacency.shape[0]):
        for second in range(first + 1, adjacency.shape[1]):
            if adjacency[first, second] > 0.5:
                pairs.add((first, second))
    return pairs


def _edge(polygon):
    coords = np.asarray(polygon.exterior.coords[:-1], dtype=float)
    count = len(coords)
    return [
        (coords[index], coords[(index + 1) % count])
        for index in range(count)
    ]


def _set_edge_coordinate(
    record,
    edge_points,
    orientation,
    target,
    config,
    guard_heads=True,
):
    if record.fixed:
        return False
    coords = np.asarray(
        record.polygon.exterior.coords[:-1],
        dtype=float,
    ).copy()
    source = np.asarray(edge_points, dtype=float)
    for index in range(len(coords)):
        following = (index + 1) % len(coords)
        first = coords[index]
        second = coords[following]
        direct = (
            np.linalg.norm(first - source[0]) <= 1e-6
            and np.linalg.norm(second - source[1]) <= 1e-6
        )
        reverse = (
            np.linalg.norm(first - source[1]) <= 1e-6
            and np.linalg.norm(second - source[0]) <= 1e-6
        )
        if not direct and not reverse:
            continue
        coordinate_index = 1 if orientation == "h" else 0
        if (
            abs(coords[index, coordinate_index] - target) <= 1e-9
            and abs(coords[following, coordinate_index] - target) <= 1e-9
        ):
            return False
        coords[index, coordinate_index] = target
        coords[following, coordinate_index] = target
        polygon = _polygon_from_points(coords)
        if polygon is None:
            return False
        if (
            guard_heads
            and _adds_spurious_head(
                record.source_polygon,
                polygon,
                config,
            )
        ):
            return False
        record.polygon = polygon
        return True
    return False


def conservative_orthogonalize(records, config):
    tangent = math.tan(
        math.radians(config.angle_tolerance_deg)
    )
    for record in records:
        coords = np.asarray(
            record.polygon.exterior.coords[:-1],
            dtype=float,
        ).copy()
        for index in range(len(coords)):
            following = (index + 1) % len(coords)
            delta = coords[following] - coords[index]
            dx = abs(delta[0])
            dy = abs(delta[1])
            if dx <= 1e-9 and dy <= 1e-9:
                continue
            if dy <= dx * tangent:
                value = 0.5 * (
                    coords[index, 1] + coords[following, 1]
                )
                coords[index, 1] = value
                coords[following, 1] = value
            elif dx <= dy * tangent:
                value = 0.5 * (
                    coords[index, 0] + coords[following, 0]
                )
                coords[index, 0] = value
                coords[following, 0] = value
        polygon = _polygon_from_points(coords)
        if polygon is not None:
            record.polygon = polygon


def _force_edge_collinear(
    record,
    source_edge,
    target_edge,
    orientation,
    min_shared_length,
    config,
):
    coords = np.asarray(
        record.polygon.exterior.coords[:-1],
        dtype=float,
    ).copy()
    source = np.asarray(source_edge, dtype=float)
    target = np.asarray(target_edge, dtype=float)
    edge_index = None
    for index in range(len(coords)):
        following = (index + 1) % len(coords)
        first = coords[index]
        second = coords[following]
        direct = (
            np.linalg.norm(first - source[0]) <= 1e-6
            and np.linalg.norm(second - source[1]) <= 1e-6
        )
        reverse = (
            np.linalg.norm(first - source[1]) <= 1e-6
            and np.linalg.norm(second - source[0]) <= 1e-6
        )
        if direct or reverse:
            edge_index = index
            break
    if edge_index is None:
        return None
    following = (edge_index + 1) % len(coords)
    if orientation == "h":
        coordinate = float(target[0][1])
        coords[edge_index, 1] = coordinate
        coords[following, 1] = coordinate
        source_low = min(coords[edge_index, 0], coords[following, 0])
        source_high = max(coords[edge_index, 0], coords[following, 0])
        target_low = min(target[0][0], target[1][0])
        target_high = max(target[0][0], target[1][0])
        overlap = min(source_high, target_high) - max(
            source_low,
            target_low,
        )
        if overlap < min_shared_length:
            if source_high < target_low:
                shift = target_low - source_high + min_shared_length
            else:
                shift = target_high - source_low - min_shared_length
            if reverse:
                shift = -shift
            coords[edge_index, 0] += shift
            coords[following, 0] += shift
        for vertex in (edge_index, following):
            value = coords[vertex, 0]
            targets = [target[0][0], target[1][0]]
            nearest = min(targets, key=lambda item: abs(item - value))
            if (
                0.0
                < abs(nearest - value)
                <= config.endpoint_snap_tolerance
            ):
                coords[vertex, 0] = nearest
    else:
        coordinate = float(target[0][0])
        coords[edge_index, 0] = coordinate
        coords[following, 0] = coordinate
        source_low = min(coords[edge_index, 1], coords[following, 1])
        source_high = max(coords[edge_index, 1], coords[following, 1])
        target_low = min(target[0][1], target[1][1])
        target_high = max(target[0][1], target[1][1])
        overlap = min(source_high, target_high) - max(
            source_low,
            target_low,
        )
        if overlap < min_shared_length:
            if source_high < target_low:
                shift = target_low - source_high + min_shared_length
            else:
                shift = target_high - source_low - min_shared_length
            if reverse:
                shift = -shift
            coords[edge_index, 1] += shift
            coords[following, 1] += shift
        for vertex in (edge_index, following):
            value = coords[vertex, 1]
            targets = [target[0][1], target[1][1]]
            nearest = min(targets, key=lambda item: abs(item - value))
            if (
                0.0
                < abs(nearest - value)
                <= config.endpoint_snap_tolerance
            ):
                coords[vertex, 1] = nearest
    return _polygon_from_points(coords)


def enforce_adjacent_collinearity(
    records,
    adjacency,
    config,
):
    by_id = {record.room_id: record for record in records}
    for _ in range(4):
        changed = False
        for first_id, second_id in sorted(adjacency):
            first = by_id.get(int(first_id))
            second = by_id.get(int(second_id))
            if first is None or second is None:
                continue
            if _shared_boundary_length(first, second) >= (
                config.min_shared_length
            ):
                continue
            mover, anchor = _choose_mover(first, second)
            if mover is None:
                continue
            candidate = _parallel_edge_candidate(
                mover,
                anchor,
                max(config.max_edge_move, 0.30),
                config.min_shared_length,
            )
            if candidate is None:
                continue
            (
                _,
                source_edge,
                target_edge,
                orientation,
                _,
                _,
            ) = candidate
            forced = _force_edge_collinear(
                mover,
                source_edge,
                target_edge,
                orientation,
                config.min_shared_length,
                config,
            )
            if forced is None:
                continue
            old_shared = _max_shared_boundary_segment(first, second)
            old_overlap = _non_auxiliary_overlap_score(records)
            old_polygon = mover.polygon
            mover.polygon = forced
            new_shared = _shared_boundary_length(first, second)
            new_overlap = _total_overlap_score(records)
            if (
                new_shared > old_shared + 1e-6
                and new_overlap <= old_overlap + 0.01
            ):
                changed = True
            else:
                mover.polygon = old_polygon
        if not changed:
            break


def _snap_endpoint_on_collinear_edges(
    record,
    source_edge,
    target_edge,
    orientation,
    tolerance,
):
    coords = np.asarray(
        record.polygon.exterior.coords[:-1],
        dtype=float,
    ).copy()
    source = np.asarray(source_edge, dtype=float)
    target = np.asarray(target_edge, dtype=float)
    edge_index = None
    for index in range(len(coords)):
        following = (index + 1) % len(coords)
        first = coords[index]
        second = coords[following]
        direct = (
            np.linalg.norm(first - source[0]) <= 1e-6
            and np.linalg.norm(second - source[1]) <= 1e-6
        )
        reverse = (
            np.linalg.norm(first - source[1]) <= 1e-6
            and np.linalg.norm(second - source[0]) <= 1e-6
        )
        if direct or reverse:
            edge_index = index
            break
    if edge_index is None:
        return None
    following = (edge_index + 1) % len(coords)
    axis = 0 if orientation == "h" else 1
    targets = [target[0][axis], target[1][axis]]
    changed = False
    for vertex in (edge_index, following):
        value = coords[vertex, axis]
        nearest = min(targets, key=lambda item: abs(item - value))
        distance = abs(nearest - value)
        if 0.0 < distance <= tolerance:
            mask = np.isclose(
                coords[:, axis],
                value,
                atol=1e-6,
            )
            coords[mask, axis] = nearest
            changed = True
    if not changed:
        return record.polygon
    return _polygon_from_points(coords)


def _extend_collinear_edge_endpoint(
    record,
    orientation,
    coord,
    low,
    high,
    extend_high,
    distance,
    config,
):
    coords = np.asarray(
        record.polygon.exterior.coords[:-1],
        dtype=float,
    ).copy()
    axis = _edge_axis(orientation)
    varying = 0 if axis == 1 else 1
    tolerance = max(config.coordinate_tolerance, 1e-5)
    source_endpoint = high if extend_high else low
    target_endpoint = (
        source_endpoint + distance
        if extend_high
        else source_endpoint - distance
    )
    matched = False
    for start, end in _edge(record.polygon):
        if _edge_orientation(start, end) != orientation:
            continue
        edge_coord = float(0.5 * (start[axis] + end[axis]))
        if abs(edge_coord - coord) > tolerance:
            continue
        edge_low, edge_high = sorted(
            [float(start[varying]), float(end[varying])]
        )
        if (
            edge_low <= low + tolerance
            and edge_high >= high - tolerance
        ):
            matched = True
            break
    if not matched:
        return None
    mask = np.isclose(
        coords[:, varying],
        source_endpoint,
        atol=tolerance,
    )
    if not mask.any():
        return None
    coords[mask, varying] = target_endpoint
    return _polygon_from_points(coords)


def snap_connected_edge_endpoints(
    records,
    adjacency,
    config,
):
    by_id = {record.room_id: record for record in records}
    for _ in range(3):
        changed = False
        for first_id, second_id in sorted(adjacency):
            first = by_id.get(int(first_id))
            second = by_id.get(int(second_id))
            if first is None or second is None:
                continue
            mover, anchor = _choose_mover(first, second)
            if mover is None:
                continue
            candidate = _parallel_edge_candidate(
                mover,
                anchor,
                max(config.max_edge_move, 0.30),
                config.min_shared_length,
            )
            if candidate is None:
                continue
            (
                _,
                source_edge,
                target_edge,
                orientation,
                _,
                _,
            ) = candidate
            snapped = _snap_endpoint_on_collinear_edges(
                mover,
                source_edge,
                target_edge,
                orientation,
                config.endpoint_snap_tolerance,
            )
            if snapped is None:
                continue
            old_polygon = mover.polygon
            old_shared = _shared_boundary_length(first, second)
            mover.polygon = snapped
            new_shared = _shared_boundary_length(first, second)
            if new_shared + 1e-6 >= old_shared:
                changed = True
            else:
                mover.polygon = old_polygon
        if not changed:
            break


def ensure_doorable_connected_edges(
    records,
    adjacency,
    config,
):
    """Extend short connected wall segments to fit at least one door."""
    by_id = {record.room_id: record for record in records}
    required_types = set(config.door_required_room_types)
    changed_any = False
    for _ in range(4):
        locked = _locked_shared_lengths(
            records,
            adjacency,
            config,
        )
        best = None
        for first_id, second_id in sorted(adjacency):
            first = by_id.get(int(first_id))
            second = by_id.get(int(second_id))
            if first is None or second is None:
                continue
            if first.type_name in {"front_door", "door"}:
                continue
            if second.type_name in {"front_door", "door"}:
                continue
            if (
                first.type_name not in required_types
                and second.type_name not in required_types
            ):
                continue
            old_shared = _shared_boundary_length(first, second)
            if old_shared >= config.min_door_shared_length - 1e-7:
                continue
            mover, anchor = _choose_mover(first, second)
            if mover is None or anchor is None:
                continue
            old_polygon = mover.polygon
            old_area = float(old_polygon.area)
            old_overlap = _non_auxiliary_overlap_score(records)
            old_hole = _hole_area(records)
            for candidate in _parallel_full_edge_pairs(
                mover,
                anchor,
                config,
            ):
                if (
                    candidate["projection"]
                    >= config.min_door_shared_length - 1e-7
                ):
                    continue
                target_length = (
                    candidate["target_high"]
                    - candidate["target_low"]
                )
                if (
                    target_length
                    < config.min_door_shared_length - 1e-7
                ):
                    continue
                source_edge = (
                    _edge_point(
                        candidate["orientation"],
                        candidate["source_coord"],
                        candidate["source_low"],
                    ),
                    _edge_point(
                        candidate["orientation"],
                        candidate["source_coord"],
                        candidate["source_high"],
                    ),
                )
                target_edge = (
                    _edge_point(
                        candidate["orientation"],
                        candidate["target_coord"],
                        candidate["target_low"],
                    ),
                    _edge_point(
                        candidate["orientation"],
                        candidate["target_coord"],
                        candidate["target_high"],
                    ),
                )
                deficit = (
                    config.min_door_shared_length
                    - candidate["projection"]
                    + min(config.grid, 1e-4)
                )
                proposals = []
                if (
                    candidate["source_high"]
                    + deficit
                    <= candidate["target_high"] + 1e-7
                ):
                    extended_high = _extend_collinear_edge_endpoint(
                        mover,
                        candidate["orientation"],
                        candidate["source_coord"],
                        candidate["source_low"],
                        candidate["source_high"],
                        True,
                        deficit,
                        config,
                    )
                    if extended_high is not None:
                        proposals.append(extended_high)
                if (
                    candidate["source_low"]
                    - deficit
                    >= candidate["target_low"] - 1e-7
                ):
                    extended_low = _extend_collinear_edge_endpoint(
                        mover,
                        candidate["orientation"],
                        candidate["source_coord"],
                        candidate["source_low"],
                        candidate["source_high"],
                        False,
                        deficit,
                        config,
                    )
                    if extended_low is not None:
                        proposals.append(extended_low)
                snapped = _snap_endpoint_on_collinear_edges(
                    mover,
                    source_edge,
                    target_edge,
                    candidate["orientation"],
                    max(
                        config.endpoint_snap_tolerance,
                        config.min_door_shared_length,
                        config.max_door_edge_move,
                    ),
                )
                if snapped is not None:
                    proposals.append(snapped)
                for proposal in proposals:
                    area_change = float(
                        proposal.symmetric_difference(old_polygon).area
                    )
                    if (
                        area_change
                        > config.max_door_edge_area_change + 1e-8
                    ):
                        continue
                    old_mover_polygon = mover.polygon
                    mover.polygon = proposal
                    new_shared = _max_shared_boundary_segment(
                        mover,
                        anchor,
                    )
                    valid = (
                        new_shared
                        >= config.min_door_shared_length - 1e-7
                        and _non_auxiliary_overlap_score(records)
                        <= old_overlap + config.overlap_tolerance
                        and _hole_area(records) <= old_hole + 1e-8
                        and _preserves_locked_shared(
                            records,
                            locked,
                            config,
                        )
                        and proposal.is_valid
                        and proposal.area > 1e-8
                    )
                    if valid:
                        score = (
                            area_change,
                            _edge_count(
                                type(mover)(
                                    mover.room_id,
                                    mover.type_name,
                                    proposal,
                                )
                            )
                            - _edge_count(mover),
                            -new_shared,
                            abs(proposal.area - old_area),
                        )
                        if best is None or score < best[0]:
                            best = (
                                score,
                                mover,
                                old_mover_polygon,
                                proposal,
                            )
                    mover.polygon = old_mover_polygon
            mover.polygon = old_polygon
        if best is None:
            chain = _find_partition_chain(
                records,
                adjacency,
                config,
            )
            if chain is None or not _apply_partition_chain(
                records,
                chain,
                config,
            ):
                break
            changed_any = True
            continue
        _, mover, _, snapped = best
        mover.polygon = snapped
        changed_any = True
    return changed_any


def unfit_connected_edges(
    records,
    adjacency,
    config=None,
):
    """Return connected non-door room pairs without a usable door edge."""
    config = config or PostProcessConfig()
    by_id = {record.room_id: record for record in records}
    required_types = set(config.door_required_room_types)
    failures = []
    for first_id, second_id in sorted(adjacency):
        first = by_id.get(int(first_id))
        second = by_id.get(int(second_id))
        if first is None or second is None:
            continue
        if first.type_name in {"front_door", "door"}:
            continue
        if second.type_name in {"front_door", "door"}:
            continue
        if (
            first.type_name not in required_types
            and second.type_name not in required_types
        ):
            continue
        shared_length = _max_shared_boundary_segment(first, second)
        if shared_length >= config.min_door_shared_length - 1e-7:
            continue
        failures.append(
            {
                "first_id": int(first.room_id),
                "first_type": first.type_name,
                "second_id": int(second.room_id),
                "second_type": second.type_name,
                "shared_length": float(shared_length),
                "required_length": float(
                    config.min_door_shared_length
                ),
            }
        )
    return failures


def _partition_edge_rooms(
    records,
    orientation,
    line_coord,
    reference_coord,
    config,
):
    partition_orientation = "v" if orientation == "h" else "h"
    axis = 0 if partition_orientation == "v" else 1
    varying = 1 - axis
    tolerance = max(config.coordinate_tolerance, 1e-5)
    room_ids = set()
    for record in records:
        coords = np.asarray(
            record.polygon.exterior.coords[:-1],
            dtype=float,
        )
        for vertex in coords:
            if (
                abs(float(vertex[axis]) - line_coord) <= tolerance
                and abs(
                    float(vertex[varying]) - reference_coord
                )
                <= tolerance
            ):
                room_ids.add(record.room_id)
                break
        for start, end in _edge(record.polygon):
            if _edge_orientation(start, end) != partition_orientation:
                continue
            edge_coord = float(0.5 * (start[axis] + end[axis]))
            if abs(edge_coord - line_coord) > tolerance:
                continue
            low, high = sorted(
                [float(start[varying]), float(end[varying])]
            )
            if (
                high - reference_coord > tolerance
                or reference_coord - low > tolerance
            ):
                room_ids.add(record.room_id)
                break
    return room_ids


def _partition_opposite_line(
    record,
    orientation,
    line_coord,
    reference_coord,
    direction,
    config,
):
    partition_orientation = "v" if orientation == "h" else "h"
    axis = 0 if partition_orientation == "v" else 1
    varying = 1 - axis
    tolerance = max(config.coordinate_tolerance, 1e-5)
    candidates = []
    for start, end in _edge(record.polygon):
        if _edge_orientation(start, end) != partition_orientation:
            continue
        edge_coord = float(0.5 * (start[axis] + end[axis]))
        distance = direction * (edge_coord - line_coord)
        if distance <= tolerance:
            continue
        low, high = sorted(
            [float(start[varying]), float(end[varying])]
        )
        if (
            high - reference_coord > tolerance
            or reference_coord - low > tolerance
        ):
            candidates.append((distance, edge_coord))
    if not candidates:
        return None
    return min(candidates)[1]


def _partition_chain_steps(
    records,
    orientation,
    reference_coord,
    endpoint_coord,
    direction,
    delta,
    config,
):
    by_id = {record.room_id: record for record in records}
    first_rooms = _partition_edge_rooms(
        records,
        orientation,
        endpoint_coord,
        reference_coord,
        config,
    )
    if not first_rooms:
        return None
    partition_orientation = "v" if orientation == "h" else "h"
    axis = 0 if partition_orientation == "v" else 1
    shrinking = None
    if len(first_rooms) == 1:
        record = by_id[next(iter(first_rooms))]
        if record.fixed or record.type_name == "living":
            return None
        shrinking = (0.0, record)
    for room_id in first_rooms:
        if shrinking is not None:
            break
        record = by_id[room_id]
        center = np.asarray(
            record.polygon.representative_point().coords[0],
            dtype=float,
        )
        distance = direction * (center[axis] - endpoint_coord)
        if distance <= config.coordinate_tolerance:
            continue
        if record.fixed or record.type_name == "living":
            return None
        if shrinking is None or distance < shrinking[0]:
            shrinking = (distance, record)
    if shrinking is None:
        return None

    steps = [(endpoint_coord, set(first_rooms))]
    current_line = endpoint_coord
    current_room = shrinking[1]
    for _ in range(3):
        next_line = _partition_opposite_line(
            current_room,
            orientation,
            current_line,
            reference_coord,
            direction,
            config,
        )
        if next_line is None:
            break
        next_rooms = _partition_edge_rooms(
            records,
            orientation,
            next_line,
            reference_coord,
            config,
        )
        if not next_rooms:
            break
        next_room = None
        for room_id in next_rooms:
            if room_id == current_room.room_id:
                continue
            record = by_id[room_id]
            if record.fixed or record.type_name == "living":
                break
            center = np.asarray(
                record.polygon.representative_point().coords[0],
                dtype=float,
            )
            distance = direction * (center[axis] - next_line)
            if distance <= config.coordinate_tolerance:
                continue
            if next_room is None or distance < next_room[0]:
                next_room = (distance, record)
        if next_room is None:
            break
        steps.append(
            (
                next_line,
                {current_room.room_id} | set(next_rooms),
            )
        )
        current_line = next_line
        current_room = next_room[1]
    step_count = len(steps)
    step_deltas = [
        float(delta) * (step_count - index) / step_count
        for index in range(step_count)
    ]
    return {
        "orientation": orientation,
        "axis": axis,
        "direction": direction,
        "delta": float(delta),
        "steps": steps,
        "step_deltas": step_deltas,
    }


def _apply_partition_chain(records, chain, config):
    by_id = {record.room_id: record for record in records}
    tolerance = max(config.point_merge_tolerance, 1e-5)
    for (line_coord, room_ids), delta in zip(
        chain["steps"],
        chain["step_deltas"],
    ):
        for room_id in room_ids:
            record = by_id[room_id]
            coords = np.asarray(
                record.polygon.exterior.coords[:-1],
                dtype=float,
            ).copy()
            mask = np.isclose(
                coords[:, chain["axis"]],
                line_coord,
                atol=tolerance,
            )
            if not mask.any():
                continue
            coords[mask, chain["axis"]] += (
                chain["direction"] * delta
            )
            polygon = _polygon_from_points(coords)
            if polygon is None:
                return False
            record.polygon = polygon
    return True


def _find_partition_chain(
    records,
    adjacency,
    config,
):
    by_id = {record.room_id: record for record in records}
    required_types = set(config.door_required_room_types)
    best = None
    for first_id, second_id in sorted(adjacency):
        first = by_id.get(int(first_id))
        second = by_id.get(int(second_id))
        if first is None or second is None:
            continue
        if first.type_name in {"front_door", "door"}:
            continue
        if second.type_name in {"front_door", "door"}:
            continue
        if (
            first.type_name not in required_types
            and second.type_name not in required_types
        ):
            continue
        old_shared = _max_shared_boundary_segment(first, second)
        if old_shared >= config.min_door_shared_length - 1e-7:
            continue
        mover, anchor = _choose_mover(first, second)
        if mover is None or anchor is None:
            continue
        for candidate in _parallel_full_edge_pairs(
            mover,
            anchor,
            config,
        ):
            deficit = (
                config.min_door_shared_length
                - candidate["projection"]
                + min(config.grid, 1e-4)
            )
            if deficit <= 0.0 or deficit > config.max_door_edge_move:
                continue
            endpoints = (
                (candidate["source_low"], -1.0),
                (candidate["source_high"], 1.0),
                (candidate["target_low"], -1.0),
                (candidate["target_high"], 1.0),
            )
            for endpoint, direction in endpoints:
                chain = _partition_chain_steps(
                    records,
                    candidate["orientation"],
                    candidate["source_coord"],
                    endpoint,
                    direction,
                    deficit,
                    config,
                )
                if chain is None:
                    continue
                snapshot = [
                    copy.deepcopy(record.polygon)
                    for record in records
                ]
                old_overlap = _non_auxiliary_overlap_score(records)
                old_hole = _hole_area(records)
                old_areas = {
                    record.room_id: float(record.polygon.area)
                    for record in records
                }
                if not _apply_partition_chain(
                    records,
                    chain,
                    config,
                ):
                    _restore_polygon_snapshot(records, snapshot)
                    continue
                new_shared = _max_shared_boundary_segment(
                    mover,
                    anchor,
                )
                area_changes = {
                    record.room_id: max(
                        0.0,
                        old_areas[record.room_id]
                        - float(record.polygon.area),
                    )
                    for record in records
                }
                valid = (
                    new_shared
                    >= config.min_door_shared_length - 1e-7
                    and _non_auxiliary_overlap_score(records)
                    <= old_overlap + config.overlap_tolerance
                    and _hole_area(records) <= old_hole + 1e-8
                    and max(area_changes.values())
                    <= config.max_door_edge_area_change + 1e-8
                    and _preserves_doorable_shared(
                        records,
                        _locked_shared_lengths(
                            [
                                type(record)(
                                    record.room_id,
                                    record.type_name,
                                    polygon,
                                )
                                for record, polygon in zip(
                                    records,
                                    snapshot,
                                )
                            ],
                            adjacency,
                            config,
                        ),
                        config,
                    )
                )
                _restore_polygon_snapshot(records, snapshot)
                if valid:
                    score = (
                        max(area_changes.values()),
                        deficit,
                        len(chain["steps"]),
                    )
                    if best is None or score < best[0]:
                        best = (score, chain)
    if best is None:
        return None
    return best[1]


class _UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[
                self.parent[value]
            ]
            value = self.parent[value]
        return value

    def union(self, first, second):
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def canonicalize_collinearity(records, config):
    """Merge near points, align coordinates, and force rectilinear edges."""
    points = []
    room_vertices = {}
    for room_index, record in enumerate(records):
        coords = np.asarray(
            record.polygon.exterior.coords[:-1],
            dtype=float,
        )
        room_vertices[room_index] = []
        for vertex_index, coordinate in enumerate(coords):
            point_index = len(points)
            points.append(
                {
                    "xy": coordinate.copy(),
                    "fixed": record.fixed,
                    "room_index": room_index,
                    "vertex_index": vertex_index,
                }
            )
            room_vertices[room_index].append(point_index)

    union_find = _UnionFind(len(points))
    for first in range(len(points)):
        for second in range(first + 1, len(points)):
            if (
                np.linalg.norm(
                    points[first]["xy"] - points[second]["xy"]
                )
                <= config.point_merge_tolerance
            ):
                union_find.union(first, second)

    node_members = {}
    for index, point in enumerate(points):
        root = union_find.find(index)
        node_members.setdefault(root, []).append(index)
    point_to_node = {
        index: root
        for root, members in node_members.items()
        for index in members
    }
    node_xy = {}
    node_fixed = {}
    node_room = {}
    for root, members in node_members.items():
        member_points = [points[index] for index in members]
        node_xy[root] = np.mean(
            [point["xy"] for point in member_points],
            axis=0,
        )
        node_fixed[root] = any(
            point["fixed"] for point in member_points
        )
        node_room[root] = member_points[0]["room_index"]

    edges = []
    for room_index, vertices in room_vertices.items():
        for index in range(len(vertices)):
            first = point_to_node[vertices[index]]
            second = point_to_node[
                vertices[(index + 1) % len(vertices)]
            ]
            if first != second:
                edges.append((first, second))

    for _ in range(config.canonical_iterations):
        for first_node, second_node in edges:
            first_xy = node_xy[first_node]
            second_xy = node_xy[second_node]
            delta = second_xy - first_xy
            if abs(delta[0]) >= abs(delta[1]):
                target = 0.5 * (first_xy[1] + second_xy[1])
                if node_fixed[first_node] and not node_fixed[second_node]:
                    node_xy[second_node][1] = first_xy[1]
                elif node_fixed[second_node] and not node_fixed[first_node]:
                    node_xy[first_node][1] = second_xy[1]
                elif not node_fixed[first_node] and not node_fixed[second_node]:
                    node_xy[first_node][1] = target
                    node_xy[second_node][1] = target
            else:
                target = 0.5 * (first_xy[0] + second_xy[0])
                if node_fixed[first_node] and not node_fixed[second_node]:
                    node_xy[second_node][0] = first_xy[0]
                elif node_fixed[second_node] and not node_fixed[first_node]:
                    node_xy[first_node][0] = second_xy[0]
                elif not node_fixed[first_node] and not node_fixed[second_node]:
                    node_xy[first_node][0] = target
                    node_xy[second_node][0] = target

        for axis in (0, 1):
            items = sorted(
                node_xy.items(),
                key=lambda item: item[1][axis],
            )
            groups = []
            current = []
            for item in items:
                if (
                    current
                    and item[1][axis] - current[-1][1][axis]
                    > config.coordinate_merge_tolerance
                ):
                    groups.append(current)
                    current = []
                current.append(item)
            if current:
                groups.append(current)
            for group in groups:
                fixed_values = [
                    item[1][axis]
                    for item in group
                    if node_fixed[item[0]]
                ]
                target = (
                    float(np.median(fixed_values))
                    if fixed_values
                    else float(
                        np.median([item[1][axis] for item in group])
                    )
                )
                for node, coordinate in group:
                    if not node_fixed[node]:
                        coordinate[axis] = target

    for room_index, record in enumerate(records):
        coords = []
        for vertex_index in room_vertices[room_index]:
            node = point_to_node[vertex_index]
            coords.append(node_xy[node].copy())
        polygon = _polygon_from_points(coords)
        if polygon is not None:
            record.polygon = polygon


def merge_near_vertices(records, tolerance=1e-5):
    points = []
    room_vertices = {}
    for room_index, record in enumerate(records):
        coords = np.asarray(
            record.polygon.exterior.coords[:-1],
            dtype=float,
        )
        room_vertices[room_index] = []
        for coordinate in coords:
            room_vertices[room_index].append(len(points))
            points.append(coordinate.copy())
    union_find = _UnionFind(len(points))
    for first in range(len(points)):
        for second in range(first + 1, len(points)):
            if (
                np.linalg.norm(points[first] - points[second])
                <= tolerance
            ):
                union_find.union(first, second)
    members = {}
    for index in range(len(points)):
        root = union_find.find(index)
        members.setdefault(root, []).append(index)
    node_xy = {
        root: np.mean([points[index] for index in group], axis=0)
        for root, group in members.items()
    }
    for room_index, record in enumerate(records):
        coords = [
            node_xy[union_find.find(index)]
            for index in room_vertices[room_index]
        ]
        polygon = _polygon_from_points(coords)
        if polygon is not None:
            record.polygon = polygon


def postprocess_records_canonical(
    records,
    adjacency,
    config=None,
):
    config = config or PostProcessConfig()
    canonicalize_collinearity(records, config)
    conservative_orthogonalize(records, config)
    canonicalize_collinearity(records, config)
    conservative_orthogonalize(records, config)
    enforce_adjacent_collinearity(
        records,
        adjacency,
        config,
    )
    conservative_orthogonalize(records, config)
    snap_connected_edge_endpoints(
        records,
        adjacency,
        config,
    )
    merge_near_vertices(
        records,
        tolerance=config.point_merge_tolerance,
    )
    conservative_orthogonalize(records, config)
    adjacency = {
        (int(first), int(second))
        for first, second in adjacency
        if int(first) != int(second)
    }
    metrics = {
        "rooms": len(records),
        "connected_gaps": _connected_gap_stats(records, adjacency),
        "overlap_areas": _overlap_stats(
            records,
            ignore_auxiliary=True,
        ),
        "holes": len(_hole_polygons(records)),
        "reverted_heads": 0,
    }
    return records, metrics


def _locked_shared_lengths(records, adjacency, config):
    by_id = {record.room_id: record for record in records}
    locked = {}
    for first_id, second_id in adjacency:
        first = by_id.get(int(first_id))
        second = by_id.get(int(second_id))
        if first is None or second is None:
            continue
        if first.type_name in {"front_door", "door"}:
            continue
        if second.type_name in {"front_door", "door"}:
            continue
        length = _shared_boundary_length(first, second)
        if length >= config.min_shared_length:
            locked[(int(first_id), int(second_id))] = length
    return locked


def _preserves_locked_shared(records, locked, config):
    by_id = {record.room_id: record for record in records}
    for (first_id, second_id), length in locked.items():
        first = by_id.get(first_id)
        second = by_id.get(second_id)
        if first is None or second is None:
            return False
        if _shared_boundary_length(first, second) + 1e-6 < length:
            return False
    return True


def _preserves_doorable_shared(records, locked, config):
    by_id = {record.room_id: record for record in records}
    for (first_id, second_id), _ in locked.items():
        first = by_id.get(first_id)
        second = by_id.get(second_id)
        if first is None or second is None:
            return False
        if not _has_doorable_shared_segment(
            first,
            second,
            config,
        ):
            return False
    return True


def _edge_count(record):
    return len(record.polygon.exterior.coords) - 1


def _clip_overlap_with_edge_guard(
    records,
    first,
    second,
    config,
):
    smaller, larger = (
        (first, second)
        if first.polygon.area <= second.polygon.area
        else (second, first)
    )
    # Default: overlap belongs to the smaller room, clip the larger room.
    direct_loser = larger
    direct_winner = smaller
    reverse_loser = smaller
    reverse_winner = larger
    candidates = []
    for loser, winner, reversed_flag in (
        (direct_loser, direct_winner, False),
        (reverse_loser, reverse_winner, True),
    ):
        difference = loser.polygon.difference(winner.polygon)
        difference = _clean_polygon(difference)
        if difference is None:
            continue
        delta = _edge_count(loser)
        after = _edge_count(
            type(loser)(loser.room_id, loser.type_name, difference)
        )
        candidates.append(
            (
                after - delta,
                reversed_flag,
                loser,
                difference,
            )
        )
    if not candidates:
        return False
    direct = [
        candidate
        for candidate in candidates
        if not candidate[1]
    ]
    if direct and direct[0][0] <= 2:
        _, _, loser, difference = direct[0]
        loser.polygon = difference
        return True
    valid = [
        candidate
        for candidate in candidates
        if candidate[0] <= 2
    ]
    if not valid:
        return False
    valid.sort(key=lambda item: item[0])
    _, _, loser, difference = valid[0]
    loser.polygon = difference
    return True


def _overlap_boundary_owner_ratio(first, second, overlap):
    if overlap is None or overlap.is_empty:
        return 0.0, 0.0
    boundary = getattr(overlap, "boundary", None)
    if boundary is None or boundary.is_empty:
        return 0.0, 0.0
    perimeter = boundary.length
    if perimeter <= 1e-9:
        return 0.0, 0.0
    first_contact = boundary.intersection(first.polygon.boundary).length
    second_contact = boundary.intersection(second.polygon.boundary).length
    return first_contact / perimeter, second_contact / perimeter


def _merge_intervals(intervals):
    intervals = sorted(
        (float(low), float(high))
        for low, high in intervals
        if high > low + 1e-9
    )
    if not intervals:
        return []
    result = [intervals[0]]
    for low, high in intervals[1:]:
        previous_low, previous_high = result[-1]
        if low <= previous_high + 1e-6:
            result[-1] = (
                previous_low,
                max(previous_high, high),
            )
        else:
            result.append((low, high))
    return result


def _subtract_intervals(interval, blockers):
    low, high = interval
    result = []
    cursor = low
    for block_low, block_high in _merge_intervals(blockers):
        if block_high <= cursor + 1e-6:
            continue
        if block_low >= high - 1e-6:
            break
        if block_low > cursor + 1e-6:
            result.append((cursor, min(block_low, high)))
        cursor = max(cursor, block_high)
        if cursor >= high - 1e-6:
            break
    if cursor < high - 1e-6:
        result.append((cursor, high))
    return result


def _edge_axis(orientation):
    return 1 if orientation == "h" else 0


def _edge_point(orientation, coord, value):
    if orientation == "h":
        return np.asarray([value, coord], dtype=float)
    return np.asarray([coord, value], dtype=float)


def _free_edge_runs(records, config):
    runs = []
    for record in records:
        coords = np.asarray(
            record.polygon.exterior.coords[:-1],
            dtype=float,
        )
        for index in range(len(coords)):
            start = coords[index]
            end = coords[(index + 1) % len(coords)]
            orientation = _edge_orientation(start, end)
            if orientation is None:
                continue
            axis = _edge_axis(orientation)
            varying = 0 if axis == 1 else 1
            coord = float(0.5 * (start[axis] + end[axis]))
            low, high = sorted(
                [float(start[varying]), float(end[varying])]
            )
            if high - low < config.min_shared_length:
                continue
            line = LineString([start, end])
            blockers = []
            for other in records:
                if other.room_id == record.room_id:
                    continue
                intersection = line.intersection(
                    other.polygon.boundary
                )
                for part in _line_parts(intersection):
                    if part.length < config.min_shared_length:
                        continue
                    part_start, part_end = part.coords[0], part.coords[-1]
                    part_low, part_high = sorted(
                        [
                            float(part_start[varying]),
                            float(part_end[varying]),
                        ]
                    )
                    blockers.append((part_low, part_high))
            for free_low, free_high in _subtract_intervals(
                (low, high),
                blockers,
            ):
                if free_high - free_low < config.min_shared_length:
                    continue
                run_start = _edge_point(
                    orientation,
                    coord,
                    free_low,
                )
                run_end = _edge_point(
                    orientation,
                    coord,
                    free_high,
                )
                runs.append(
                    {
                        "record": record,
                        "room_id": record.room_id,
                        "room_name": record.name,
                        "orientation": orientation,
                        "axis": axis,
                        "varying": varying,
                        "coord": coord,
                        "low": free_low,
                        "high": free_high,
                        "start": run_start,
                        "end": run_end,
                        "line": LineString([run_start, run_end]),
                    }
                )
    return runs


def _hollow_edge_candidates(records, config):
    runs = _free_edge_runs(records, config)
    if not runs:
        return []
    union = unary_union([record.polygon for record in records])
    candidates = {}
    for first_index, first in enumerate(runs):
        for second in runs[first_index + 1 :]:
            if first["room_id"] == second["room_id"]:
                continue
            if first["orientation"] != second["orientation"]:
                continue
            perpendicular = abs(first["coord"] - second["coord"])
            if perpendicular <= 1e-8:
                continue
            if perpendicular > config.max_hollow_gap:
                continue
            projection = min(first["high"], second["high"]) - max(
                first["low"],
                second["low"],
            )
            if projection < 0.02:
                continue
            if first["orientation"] == "h":
                strip = box(
                    max(first["low"], second["low"]),
                    min(first["coord"], second["coord"]),
                    min(first["high"], second["high"]),
                    max(first["coord"], second["coord"]),
                )
            else:
                strip = box(
                    min(first["coord"], second["coord"]),
                    max(first["low"], second["low"]),
                    max(first["coord"], second["coord"]),
                    min(first["high"], second["high"]),
                )
            if strip.area <= 1e-8:
                continue
            void = strip.difference(union)
            for component in _polygon_parts(void):
                area = float(component.area)
                if area < 0.001 or area > config.max_hollow_area:
                    continue
                first_contact = component.boundary.intersection(
                    first["line"]
                ).length
                second_contact = component.boundary.intersection(
                    second["line"]
                ).length
                if first_contact < 0.02 or second_contact < 0.02:
                    continue
                perimeter = component.boundary.length
                if perimeter <= 1e-9:
                    continue
                room_contact = 0.0
                for record in records:
                    room_contact += component.boundary.intersection(
                        record.polygon.boundary
                    ).length
                if room_contact / perimeter < 0.75:
                    continue
                key = (
                    round(area, 8),
                    tuple(
                        round(value, 6)
                        for value in component.bounds
                    ),
                )
                candidate = {
                    "area": area,
                    "component": component,
                    "first": first,
                    "second": second,
                    "projection": projection,
                    "perpendicular": perpendicular,
                }
                previous = candidates.get(key)
                if previous is None or (
                    perpendicular,
                    -projection,
                ) < (
                    previous["perpendicular"],
                    -previous["projection"],
                ):
                    candidates[key] = candidate
    return list(candidates.values())


def _move_polygon_subedge(
    polygon,
    orientation,
    coord,
    low,
    high,
    target_coord,
    config,
):
    coords = np.asarray(
        polygon.exterior.coords[:-1],
        dtype=float,
    )
    axis = _edge_axis(orientation)
    varying = 0 if axis == 1 else 1
    tolerance = max(config.coordinate_tolerance, 1e-5)
    result = []
    matched = False

    for index, vertex in enumerate(coords):
        previous = coords[(index - 1) % len(coords)]
        following = coords[(index + 1) % len(coords)]
        previous_orientation = _edge_orientation(previous, vertex)
        next_orientation = _edge_orientation(vertex, following)
        on_matching_edge = False
        for start, end, edge_orientation in (
            (previous, vertex, previous_orientation),
            (vertex, following, next_orientation),
        ):
            if edge_orientation != orientation:
                continue
            edge_coord = float(0.5 * (start[axis] + end[axis]))
            if abs(edge_coord - coord) > tolerance:
                continue
            edge_low, edge_high = sorted(
                [float(start[varying]), float(end[varying])]
            )
            if (
                low >= edge_low - tolerance
                and high <= edge_high + tolerance
            ):
                on_matching_edge = True
                matched = True
        item = np.asarray(vertex, dtype=float).copy()
        if (
            on_matching_edge
            and low - tolerance
            <= float(vertex[varying])
            <= high + tolerance
        ):
            item[axis] = target_coord
        result.append(item)

        start = vertex
        end = following
        edge_orientation = next_orientation
        if edge_orientation != orientation:
            continue
        edge_coord = float(0.5 * (start[axis] + end[axis]))
        edge_low, edge_high = sorted(
            [float(start[varying]), float(end[varying])]
        )
        if (
            abs(edge_coord - coord) > tolerance
            or low < edge_low - tolerance
            or high > edge_high + tolerance
        ):
            continue
        direction = 1.0 if end[varying] >= start[varying] else -1.0
        boundaries = [low, high]
        boundaries.sort(
            key=lambda value: direction * (value - start[varying])
        )
        segment_start = max(low, edge_low)
        segment_end = min(high, edge_high)
        for value in boundaries:
            if not (
                min(start[varying], end[varying]) + tolerance
                < value
                < max(start[varying], end[varying]) - tolerance
            ):
                continue
            fraction = (
                value - float(start[varying])
            ) / (
                float(end[varying]) - float(start[varying])
            )
            point = start + fraction * (end - start)
            moved = np.asarray(point, dtype=float).copy()
            moved[axis] = target_coord
            if abs(value - segment_start) <= abs(value - segment_end):
                result.append(np.asarray(point, dtype=float))
                result.append(moved)
            else:
                result.append(moved)
                result.append(np.asarray(point, dtype=float))
    if not matched:
        return None
    return _polygon_from_points(result)


def _maximal_collinear_span(
    polygon,
    orientation,
    coord,
    low,
    high,
    config,
):
    axis = _edge_axis(orientation)
    varying = 0 if axis == 1 else 1
    tolerance = max(config.coordinate_tolerance, 1e-5)
    matches = []
    for start, end in _edge(polygon):
        if _edge_orientation(start, end) != orientation:
            continue
        edge_coord = float(0.5 * (start[axis] + end[axis]))
        if abs(edge_coord - coord) > tolerance:
            continue
        edge_low, edge_high = sorted(
            [float(start[varying]), float(end[varying])]
        )
        if (
            edge_low <= low + tolerance
            and edge_high >= high - tolerance
        ):
            matches.append((edge_low, edge_high))
    if not matches:
        return low, high
    return (
        min(item[0] for item in matches),
        max(item[1] for item in matches),
    )


def _apply_hollow_edge_moves(records, config):
    changed_any = False
    for _ in range(8):
        candidates = _hollow_edge_candidates(records, config)
        best = None
        locked = _locked_shared_lengths(
            records,
            {
                (first.room_id, second.room_id)
                for first_index, first in enumerate(records)
                for second in records[first_index + 1 :]
                if _shared_boundary_length(first, second)
                >= config.min_shared_length
            },
            config,
        )
        for candidate in candidates:
            first = candidate["first"]
            second = candidate["second"]
            mover, anchor = _choose_mover(
                first["record"],
                second["record"],
            )
            if mover is None or anchor is None:
                continue
            if mover.room_id == first["room_id"]:
                source = first
                target = second
            else:
                source = second
                target = first
            if source["orientation"] != target["orientation"]:
                continue
            old_polygon = mover.polygon
            old_shared = _shared_boundary_length(mover, anchor)
            old_overlap = _total_overlap_score(records)
            full_low, full_high = _maximal_collinear_span(
                old_polygon,
                source["orientation"],
                source["coord"],
                source["low"],
                source["high"],
                config,
            )
            spans = [(source["low"], source["high"], False)]
            if (
                full_low < source["low"] - 1e-7
                or full_high > source["high"] + 1e-7
            ):
                spans.append((full_low, full_high, True))
            for span_low, span_high, full_edge in spans:
                moved = _move_polygon_subedge(
                    old_polygon,
                    source["orientation"],
                    source["coord"],
                    span_low,
                    span_high,
                    target["coord"],
                    config,
                )
                if moved is None:
                    continue
                min_x, min_y, max_x, max_y = moved.bounds
                if min(max_x - min_x, max_y - min_y) <= 0.02:
                    continue
                mover.polygon = moved
                new_overlap = _non_auxiliary_overlap_score(records)
                new_shared = _shared_boundary_length(mover, anchor)
                locked_ok = _preserves_locked_shared(
                    records,
                    locked,
                    config,
                )
                overlap_ok = (
                    new_overlap
                    <= old_overlap + config.overlap_tolerance
                )
                if full_edge and not locked_ok:
                    overlap_ok = (
                        new_overlap
                        <= old_overlap
                        + config.min_overlap_action_area
                    )
                if (
                    not overlap_ok
                    or new_shared <= old_shared + 1e-7
                ):
                    mover.polygon = old_polygon
                    continue
                score = (
                    len(moved.exterior.coords) - 1,
                    0 if locked_ok else 1,
                    -new_shared,
                    moved.area,
                    abs(target["coord"] - source["coord"]),
                )
                if best is None or score < best[0]:
                    best = (score, mover, old_polygon, moved)
                mover.polygon = old_polygon
        if best is None:
            break
        _, mover, _, moved = best
        mover.polygon = moved
        changed_any = True
    return changed_any


def _parallel_full_edge_pairs(first, second, config):
    result = []
    for first_edge in _edge(first.polygon):
        orientation = _edge_orientation(*first_edge)
        if orientation is None:
            continue
        axis = _edge_axis(orientation)
        varying = 0 if axis == 1 else 1
        first_coord = float(
            0.5 * (first_edge[0][axis] + first_edge[1][axis])
        )
        first_low, first_high = sorted(
            [
                float(first_edge[0][varying]),
                float(first_edge[1][varying]),
            ]
        )
        for second_edge in _edge(second.polygon):
            if _edge_orientation(*second_edge) != orientation:
                continue
            second_coord = float(
                0.5 * (second_edge[0][axis] + second_edge[1][axis])
            )
            second_low, second_high = sorted(
                [
                    float(second_edge[0][varying]),
                    float(second_edge[1][varying]),
                ]
            )
            projection = min(first_high, second_high) - max(
                first_low,
                second_low,
            )
            if projection < config.min_shared_length:
                continue
            perpendicular = abs(first_coord - second_coord)
            if perpendicular > config.max_edge_move:
                continue
            result.append(
                {
                    "orientation": orientation,
                    "axis": axis,
                    "source_coord": first_coord,
                    "target_coord": second_coord,
                    "source_low": first_low,
                    "source_high": first_high,
                    "target_low": second_low,
                    "target_high": second_high,
                    "projection": projection,
                    "perpendicular": perpendicular,
                }
            )
    return result


def resolve_overlaps_by_edge_translation(records, config):
    changed_any = False
    for _ in range(6):
        locked = _locked_shared_lengths(
            records,
            {
                (first.room_id, second.room_id)
                for first_index, first in enumerate(records)
                for second in records[first_index + 1 :]
                if _shared_boundary_length(first, second)
                >= config.min_shared_length
            },
            config,
        )
        best = None
        for first_index, first in enumerate(records):
            for second in records[first_index + 1 :]:
                overlap = _overlap_area(first, second)
                if overlap < config.min_overlap_action_area:
                    continue
                mover, anchor = _choose_mover(first, second)
                if mover is None or anchor is None:
                    continue
                old_total = _total_overlap_score(records)
                old_polygon = mover.polygon
                for pair in _parallel_full_edge_pairs(
                    mover,
                    anchor,
                    config,
                ):
                    delta = np.zeros(2, dtype=float)
                    delta[pair["axis"]] = (
                        pair["target_coord"] - pair["source_coord"]
                    )
                    if abs(delta[pair["axis"]]) > config.max_edge_move:
                        continue
                    translated = _translated_polygon(
                        old_polygon,
                        delta,
                    )
                    if translated is None:
                        continue
                    mover.polygon = translated
                    new_total = _total_overlap_score(records)
                    if (
                        new_total >= old_total - 1e-7
                        or not _preserves_locked_shared(
                            records,
                            locked,
                            config,
                        )
                    ):
                        mover.polygon = old_polygon
                        continue
                    score = (
                        -pair["projection"],
                        pair["perpendicular"],
                        abs(delta[pair["axis"]]),
                        mover.polygon.area,
                    )
                    if best is None or score < best[0]:
                        best = (score, mover, old_polygon, translated)
                    mover.polygon = old_polygon
        if best is None:
            break
        _, mover, _, translated = best
        mover.polygon = translated
        changed_any = True
    return changed_any


def _hole_area(records):
    return sum(
        float(hole.area)
        for hole in _hole_polygons(records)
    )


def _record_polygon_snapshot(records):
    return [record.polygon for record in records]


def _restore_polygon_snapshot(records, snapshot):
    for record, polygon in zip(records, snapshot):
        record.polygon = polygon


def _run_hole_guarded(records, action, config):
    old_hole = _hole_area(records)
    snapshot = _record_polygon_snapshot(records)
    action()
    new_hole = _hole_area(records)
    if new_hole > max(old_hole, config.max_hole_area) + 1e-8:
        _restore_polygon_snapshot(records, snapshot)
        return False
    return True


def _short_edge_count(polygon, threshold=0.08):
    count = 0
    for start, end in _edge(polygon):
        if np.linalg.norm(end - start) < threshold - 1e-7:
            count += 1
    return count


def resolve_remaining_overlaps_by_edge_move(
    records,
    config,
    min_action=None,
):
    if min_action is None:
        min_action = config.overlap_tolerance
    changed_any = False
    for _ in range(6):
        best = None
        old_total = _total_overlap_score(records)
        old_hole = _hole_area(records)
        for first_index, first in enumerate(records):
            for second in records[first_index + 1 :]:
                if (
                    first.type_name in {"front_door", "door"}
                    or second.type_name in {"front_door", "door"}
                ):
                    continue
                pair_overlap = _overlap_area(first, second)
                if pair_overlap < min_action:
                    continue
                mover, anchor = _choose_mover(first, second)
                if mover is None or anchor is None or mover.fixed:
                    continue
                old_polygon = mover.polygon
                old_short = _short_edge_count(old_polygon)
                old_pair_overlap = pair_overlap
                for pair in _parallel_full_edge_pairs(
                    mover,
                    anchor,
                    config,
                ):
                    span_low, span_high = _maximal_collinear_span(
                        old_polygon,
                        pair["orientation"],
                        pair["source_coord"],
                        pair["source_low"],
                        pair["source_high"],
                        config,
                    )
                    moved = _move_polygon_subedge(
                        old_polygon,
                        pair["orientation"],
                        pair["source_coord"],
                        span_low,
                        span_high,
                        pair["target_coord"],
                        config,
                    )
                    if moved is None:
                        continue
                    mover.polygon = moved
                    new_pair_overlap = _overlap_area(mover, anchor)
                    new_total = _total_overlap_score(records)
                    new_hole = _hole_area(records)
                    compact = (
                        new_pair_overlap <= config.overlap_tolerance
                        and new_total <= old_total + 1e-8
                        and new_hole <= old_hole + 1e-8
                    )
                    if not compact:
                        mover.polygon = old_polygon
                        continue
                    new_short = _short_edge_count(moved)
                    score = (
                        new_short - old_short,
                        len(moved.exterior.coords) - 1,
                        new_hole,
                        abs(moved.area - old_polygon.area),
                        abs(
                            pair["target_coord"]
                            - pair["source_coord"]
                        ),
                        mover.polygon.area,
                    )
                    if best is None or score < best[0]:
                        best = (score, mover, old_polygon, moved)
                    mover.polygon = old_polygon
        if best is None:
            break
        _, mover, _, moved = best
        mover.polygon = moved
        changed_any = True
    return changed_any


def clip_small_fixed_head_overlaps(records, config):
    changed_any = False
    for _ in range(6):
        changed = False
        for first_index, first in enumerate(records):
            for second in records[first_index + 1 :]:
                if first.fixed == second.fixed:
                    continue
                fixed = first if first.fixed else second
                other = second if first.fixed else first
                if other.type_name in {"front_door", "door"}:
                    continue
                overlap = fixed.polygon.intersection(other.polygon)
                if (
                    overlap.is_empty
                    or overlap.area <= config.overlap_tolerance
                ):
                    continue
                min_x, min_y, max_x, max_y = overlap.bounds
                span_x = max_x - min_x
                span_y = max_y - min_y
                if (
                    min(span_x, span_y) > config.max_head_width
                    or max(span_x, span_y) > 0.20
                    or overlap.area > config.max_head_area
                ):
                    continue
                difference = _clean_polygon(
                    fixed.polygon.difference(other.polygon)
                )
                if difference is None:
                    continue
                before_points = len(fixed.polygon.exterior.coords) - 1
                after_points = len(difference.exterior.coords) - 1
                if after_points > before_points + 2:
                    continue
                old_polygon = fixed.polygon
                old_hole = _hole_area(records)
                old_total = _total_overlap_score(records)
                fixed.polygon = difference
                if (
                    _hole_area(records) > old_hole + 1e-8
                    or _total_overlap_score(records)
                    > old_total + 1e-8
                ):
                    fixed.polygon = old_polygon
                    continue
                changed = True
                changed_any = True
        if not changed:
            break
    return changed_any


def remove_small_rectangular_steps(records, config):
    changed_any = False
    for _ in range(6):
        changed = False
        for record in records:
            coords = np.asarray(
                record.polygon.exterior.coords[:-1],
                dtype=float,
            )
            if len(coords) <= 4:
                continue
            for index in range(len(coords)):
                points = [
                    coords[(index + offset) % len(coords)]
                    for offset in range(4)
                ]
                first = points[1] - points[0]
                second = points[2] - points[1]
                third = points[3] - points[2]
                first_length = float(np.linalg.norm(first))
                second_length = float(np.linalg.norm(second))
                third_length = float(np.linalg.norm(third))
                if (
                    first_length < config.min_shared_length
                    or third_length < config.min_shared_length
                ):
                    continue
                if (
                    second_length > config.max_head_width
                    or min(first_length, third_length) > 0.25
                ):
                    continue
                first_axis = (
                    0 if abs(first[1]) <= abs(first[0]) else 1
                )
                second_axis = 1 - first_axis
                third_axis = (
                    0 if abs(third[1]) <= abs(third[0]) else 1
                )
                if first_axis != third_axis:
                    continue
                if abs(second[first_axis]) > 1e-6:
                    continue
                offset = 0.5 * (
                    float(points[1][first_axis])
                    + float(points[2][first_axis])
                )
                base_a = float(points[0][first_axis])
                base_b = float(points[3][first_axis])
                if abs(base_a - base_b) <= 1e-8:
                    continue
                if (
                    min(first_length, third_length) * second_length
                    > config.max_head_area
                ):
                    continue
                target = (
                    base_a
                    if abs(offset - base_a) <= abs(offset - base_b)
                    else base_b
                )
                candidate = coords.copy()
                candidate[(index + 1) % len(coords), first_axis] = target
                candidate[(index + 2) % len(coords), first_axis] = target
                polygon = _polygon_from_points(candidate)
                if polygon is None:
                    continue
                if (
                    polygon.symmetric_difference(record.polygon).area
                    > config.max_head_area
                ):
                    continue
                old_polygon = record.polygon
                old_total = _total_overlap_score(records)
                old_hole = _hole_area(records)
                record.polygon = polygon
                if (
                    _total_overlap_score(records)
                    > old_total + config.overlap_tolerance
                    or _hole_area(records) > old_hole + 1e-8
                ):
                    record.polygon = old_polygon
                    continue
                changed = True
                changed_any = True
                break
            if changed:
                break
        if not changed:
            break
    return changed_any


def fill_long_short_long_self_holes(records, config):
    changed_any = False
    for _ in range(4):
        changed = False
        for record in records:
            if record.type_name in {"front_door", "door"}:
                continue
            coords = np.asarray(
                record.polygon.exterior.coords[:-1],
                dtype=float,
            )
            if len(coords) <= 4:
                continue
            for index in range(len(coords)):
                previous = coords[(index - 1) % len(coords)]
                first = coords[index]
                second = coords[(index + 1) % len(coords)]
                following = coords[(index + 2) % len(coords)]
                previous_edge = first - previous
                middle_edge = second - first
                following_edge = following - second
                middle_length = float(np.linalg.norm(middle_edge))
                previous_length = float(np.linalg.norm(previous_edge))
                following_length = float(np.linalg.norm(following_edge))
                if middle_length > 0.12:
                    continue
                if (
                    previous_length < 0.12
                    or following_length < 0.12
                ):
                    continue
                middle_orientation = _edge_orientation(first, second)
                if middle_orientation is None:
                    continue
                if _edge_orientation(previous, first) != (
                    "v" if middle_orientation == "h" else "h"
                ):
                    continue
                if _edge_orientation(second, following) != (
                    "v" if middle_orientation == "h" else "h"
                ):
                    continue
                if middle_orientation == "h":
                    normal_axis = 1
                    tangent_axis = 0
                else:
                    normal_axis = 0
                    tangent_axis = 1
                first_tangent = float(first[tangent_axis])
                second_tangent = float(second[tangent_axis])
                previous_low, previous_high = sorted(
                    [float(previous[normal_axis]), float(first[normal_axis])]
                )
                following_low, following_high = sorted(
                    [float(second[normal_axis]), float(following[normal_axis])]
                )
                low = min(previous_low, following_low)
                high = max(previous_high, following_high)
                if high - low > config.max_self_hole_span:
                    continue
                if middle_orientation == "h":
                    strip = box(
                        min(first_tangent, second_tangent),
                        low,
                        max(first_tangent, second_tangent),
                        high,
                    )
                else:
                    strip = box(
                        low,
                        min(first_tangent, second_tangent),
                        high,
                        max(first_tangent, second_tangent),
                    )
                gap = strip
                for component in _polygon_parts(gap):
                    if (
                        component.area < 1e-7
                        or component.area > config.max_self_hole_area
                    ):
                        continue
                    contact = component.boundary.intersection(
                        record.polygon.boundary
                    ).length
                    if contact < config.min_shared_length:
                        continue
                    merged = _clean_polygon(
                        unary_union([record.polygon, component])
                    )
                    if merged is None:
                        continue
                    old_polygon = record.polygon
                    old_total = _non_auxiliary_overlap_score(records)
                    old_hole = _hole_area(records)
                    record.polygon = merged
                    if (
                        _non_auxiliary_overlap_score(records)
                        > old_total + config.overlap_tolerance
                        or _hole_area(records) > old_hole + 1e-8
                    ):
                        record.polygon = old_polygon
                        continue
                    changed = True
                    changed_any = True
                    break
                if changed:
                    break
            if changed:
                break
        if not changed:
            break
    return changed_any


def fill_open_u_voids(records, config):
    changed_any = False
    original_bounds = {
        record.room_id: record.polygon.bounds
        for record in records
    }

    def changes_both_sides(room_id, polygon):
        original = original_bounds[room_id]
        current = polygon.bounds
        changed_x = (
            abs(current[0] - original[0]) > 1e-8
            and abs(current[2] - original[2]) > 1e-8
        )
        changed_y = (
            abs(current[1] - original[1]) > 1e-8
            and abs(current[3] - original[3]) > 1e-8
        )
        return changed_x or changed_y

    for _ in range(4):
        best = None
        union = unary_union([record.polygon for record in records])
        locked = _locked_shared_lengths(
            records,
            {
                (first.room_id, second.room_id)
                for first_index, first in enumerate(records)
                for second in records[first_index + 1 :]
                if _shared_boundary_length(first, second)
                >= config.min_shared_length
            },
            config,
        )
        for first_index, first in enumerate(records):
            if first.type_name in {"front_door", "door"}:
                continue
            for second in records[first_index + 1 :]:
                if second.type_name in {"front_door", "door"}:
                    continue
                for first_edge in _edge(first.polygon):
                    orientation = _edge_orientation(*first_edge)
                    if orientation is None:
                        continue
                    axis = _edge_axis(orientation)
                    varying = 0 if axis == 1 else 1
                    first_coord = float(
                        0.5 * (
                            first_edge[0][axis]
                            + first_edge[1][axis]
                        )
                    )
                    first_low, first_high = sorted(
                        [
                            float(first_edge[0][varying]),
                            float(first_edge[1][varying]),
                        ]
                    )
                    for second_edge in _edge(second.polygon):
                        if _edge_orientation(*second_edge) != orientation:
                            continue
                        second_coord = float(
                            0.5 * (
                                second_edge[0][axis]
                                + second_edge[1][axis]
                            )
                        )
                        perpendicular = abs(
                            first_coord - second_coord
                        )
                        if (
                            perpendicular <= 1e-8
                            or perpendicular > config.max_hollow_gap
                        ):
                            continue
                        second_low, second_high = sorted(
                            [
                                float(second_edge[0][varying]),
                                float(second_edge[1][varying]),
                            ]
                        )
                        projection = min(first_high, second_high) - max(
                            first_low,
                            second_low,
                        )
                        if projection < 0.02:
                            continue
                        if orientation == "h":
                            strip = box(
                                max(first_low, second_low),
                                min(first_coord, second_coord),
                                min(first_high, second_high),
                                max(first_coord, second_coord),
                            )
                        else:
                            strip = box(
                                min(first_coord, second_coord),
                                max(first_low, second_low),
                                max(first_coord, second_coord),
                                min(first_high, second_high),
                            )
                        void = strip.difference(union)
                        for component in _polygon_parts(void):
                            if (
                                component.area < 0.001
                                or component.area > config.max_hollow_area
                            ):
                                continue
                            contact = component.boundary.intersection(
                                union.boundary
                            ).length
                            if (
                                contact / max(component.boundary.length, 1e-9)
                                < 0.60
                            ):
                                continue
                            mover, anchor = _choose_mover(first, second)
                            if mover is None or anchor is None:
                                continue
                            if mover.room_id == first.room_id:
                                source_coord = first_coord
                                source_low = first_low
                                source_high = first_high
                                target_coord = second_coord
                            else:
                                source_coord = second_coord
                                source_low = second_low
                                source_high = second_high
                                target_coord = first_coord
                            merged = _clean_polygon(
                                unary_union([mover.polygon, component])
                            )
                            if merged is not None:
                                touches_balcony = any(
                                    other.type_name == "balcony"
                                    and (
                                        other.polygon.intersects(component)
                                        or other.polygon.distance(component)
                                        <= config.coordinate_tolerance
                                    )
                                    for other in records
                                    if other.room_id != mover.room_id
                                )
                                if (
                                    mover.type_name == "bathroom"
                                    and touches_balcony
                                ):
                                    merged = None
                            if merged is not None:
                                if changes_both_sides(
                                    mover.room_id,
                                    merged,
                                ):
                                    merged = None
                            if merged is not None:
                                old_polygon = mover.polygon
                                old_overlap = _non_auxiliary_overlap_score(
                                    records
                                )
                                old_hole = _hole_area(records)
                                mover.polygon = merged
                                new_overlap = _non_auxiliary_overlap_score(
                                    records
                                )
                                new_hole = _hole_area(records)
                                if (
                                    new_overlap
                                    <= old_overlap
                                    + config.overlap_tolerance
                                    and new_hole
                                    <= max(old_hole, config.max_hole_area)
                                    + 1e-8
                                    and _preserves_locked_shared(
                                        records,
                                        locked,
                                        config,
                                    )
                                ):
                                    score = (
                                        len(merged.exterior.coords) - 1,
                                        -component.area,
                                        abs(
                                            merged.area
                                            - old_polygon.area
                                        ),
                                        mover.polygon.area,
                                    )
                                    if best is None or score < best[0]:
                                        best = (
                                            score,
                                            mover,
                                            old_polygon,
                                            merged,
                                        )
                                mover.polygon = old_polygon
                            moved = _move_polygon_subedge(
                                mover.polygon,
                                orientation,
                                source_coord,
                                source_low,
                                source_high,
                                target_coord,
                                config,
                            )
                            if moved is None:
                                continue
                            if changes_both_sides(
                                mover.room_id,
                                moved,
                            ):
                                continue
                            old_polygon = mover.polygon
                            old_overlap = _non_auxiliary_overlap_score(records)
                            old_hole = _hole_area(records)
                            mover.polygon = moved
                            new_overlap = _non_auxiliary_overlap_score(records)
                            new_hole = _hole_area(records)
                            if (
                                new_overlap
                                > old_overlap + config.overlap_tolerance
                                or new_hole
                                > max(old_hole, config.max_hole_area) + 1e-8
                                or not _preserves_locked_shared(
                                    records,
                                    locked,
                                    config,
                                )
                            ):
                                mover.polygon = old_polygon
                                component_low = (
                                    component.bounds[1]
                                    if orientation == "h"
                                    else component.bounds[0]
                                )
                                component_high = (
                                    component.bounds[3]
                                    if orientation == "h"
                                    else component.bounds[2]
                                )
                                sub_low = max(source_low, component_low)
                                sub_high = min(source_high, component_high)
                                if sub_high - sub_low < 0.01:
                                    continue
                                moved = _move_polygon_subedge(
                                    old_polygon,
                                    orientation,
                                    source_coord,
                                    sub_low,
                                    sub_high,
                                    target_coord,
                                    config,
                                )
                                if moved is None:
                                    continue
                                mover.polygon = moved
                                new_overlap = _non_auxiliary_overlap_score(
                                    records
                                )
                                new_hole = _hole_area(records)
                                if (
                                    new_overlap
                                    > old_overlap
                                    + config.overlap_tolerance
                                    or new_hole
                                    > max(old_hole, config.max_hole_area)
                                    + 1e-8
                                    or not _preserves_locked_shared(
                                        records,
                                        locked,
                                        config,
                                    )
                                ):
                                    mover.polygon = old_polygon
                                    continue
                            score = (
                                len(moved.exterior.coords) - 1,
                                -component.area,
                                abs(moved.area - old_polygon.area),
                                mover.polygon.area,
                            )
                            if best is None or score < best[0]:
                                best = (
                                    score,
                                    mover,
                                    old_polygon,
                                    moved,
                                )
                            mover.polygon = old_polygon
        if best is None:
            break
        _, mover, _, moved = best
        mover.polygon = moved
        changed_any = True
    return changed_any


def _non_auxiliary_overlap_score(records):
    total = 0.0
    for first_index, first in enumerate(records):
        if first.type_name in {"front_door", "door"}:
            continue
        for second in records[first_index + 1 :]:
            if second.type_name in {"front_door", "door"}:
                continue
            total += _overlap_area(first, second)
    return total


def fill_medium_enclosed_holes(records, config):
    changed_any = False
    for _ in range(4):
        holes = [
            hole
            for hole in _hole_polygons(records)
            if 1e-7 < hole.area <= config.max_hollow_area
        ]
        if not holes:
            break
        changed = False
        for hole in holes:
            best = None
            for record in records:
                if record.type_name in {"front_door", "door"}:
                    continue
                touches_balcony = any(
                    other.type_name == "balcony"
                    and (
                        other.polygon.intersects(hole)
                        or other.polygon.distance(hole)
                        <= config.coordinate_tolerance
                    )
                    for other in records
                    if other.room_id != record.room_id
                )
                if record.type_name == "bathroom" and touches_balcony:
                    continue
                contact = hole.boundary.intersection(
                    record.polygon.boundary
                ).length
                if contact < config.min_shared_length:
                    continue
                merged = _clean_polygon(
                    unary_union([record.polygon, hole])
                )
                if merged is None:
                    continue
                old_polygon = record.polygon
                old_overlap = _non_auxiliary_overlap_score(records)
                old_hole = _hole_area(records)
                record.polygon = merged
                new_overlap = _non_auxiliary_overlap_score(records)
                new_hole = _hole_area(records)
                if (
                    new_overlap
                    > old_overlap + config.overlap_tolerance
                    or new_hole >= old_hole - 1e-8
                ):
                    record.polygon = old_polygon
                    continue
                score = (
                    len(merged.exterior.coords) - 1,
                    -contact,
                    abs(merged.area - old_polygon.area),
                    record.polygon.area,
                )
                if best is None or score < best[0]:
                    best = (score, record, old_polygon, merged)
                record.polygon = old_polygon
            if best is not None:
                _, record, _, merged = best
                record.polygon = merged
                changed = True
                changed_any = True
        if not changed:
            break
    return changed_any


def snap_parallel_edges_safely(records, config):
    old_overlap = _non_auxiliary_overlap_score(records)
    old_hole = _hole_area(records)
    old_areas = {
        record.room_id: record.polygon.area
        for record in records
    }
    snapshot = _record_polygon_snapshot(records)
    snap_all_parallel_edges(records, config)
    lost_area = any(
        old_areas[record.room_id] - record.polygon.area
        > config.max_head_area
        for record in records
        if record.type_name not in {"front_door", "door"}
    )
    if (
        lost_area
        or _non_auxiliary_overlap_score(records)
        > old_overlap + config.overlap_tolerance
        or _hole_area(records) > old_hole + 1e-8
    ):
        _restore_polygon_snapshot(records, snapshot)
        return False
    return True


def snap_global_seams(records, config):
    old_overlap = _non_auxiliary_overlap_score(records)
    old_hole = _hole_area(records)
    old_areas = {
        record.room_id: record.polygon.area
        for record in records
    }
    snapshot = _record_polygon_snapshot(records)
    x_values = []
    y_values = []
    for record in records:
        coords = np.asarray(
            record.polygon.exterior.coords[:-1],
            dtype=float,
        )
        x_values.extend(coords[:, 0].tolist())
        y_values.extend(coords[:, 1].tolist())
    x_centers = _cluster_1d(
        x_values,
        config.global_seam_tolerance,
    )
    y_centers = _cluster_1d(
        y_values,
        config.global_seam_tolerance,
    )
    for record in records:
        coords = np.asarray(
            record.polygon.exterior.coords[:-1],
            dtype=float,
        ).copy()
        for index in range(len(coords)):
            x_target = _nearest_cluster(coords[index, 0], x_centers)
            y_target = _nearest_cluster(coords[index, 1], y_centers)
            if abs(x_target - coords[index, 0]) <= (
                config.global_seam_tolerance
            ):
                coords[index, 0] = x_target
            if abs(y_target - coords[index, 1]) <= (
                config.global_seam_tolerance
            ):
                coords[index, 1] = y_target
        polygon = _polygon_from_points(coords)
        if polygon is not None:
            record.polygon = polygon
    lost_area = any(
        old_areas[record.room_id] - record.polygon.area
        > config.max_head_area
        for record in records
        if record.type_name not in {"front_door", "door"}
    )
    if (
        lost_area
        or _non_auxiliary_overlap_score(records)
        > old_overlap + config.overlap_tolerance
        or _hole_area(records) > old_hole + 1e-8
    ):
        _restore_polygon_snapshot(records, snapshot)
        return False
    return True


def orthogonalize_balconies(records, config):
    changed_any = False
    for _ in range(2):
        changed = False
        for record in records:
            if record.type_name != "balcony":
                continue
            source_polygon = (
                record.source_polygon
                if record.source_polygon is not None
                else record.polygon
            )
            if len(source_polygon.exterior.coords) - 1 != 4:
                continue
            coords = np.asarray(
                source_polygon.exterior.coords[:-1],
                dtype=float,
            )
            xs = coords[:, 0]
            ys = coords[:, 1]
            x_min = float(xs.min())
            x_max = float(xs.max())
            y_min = float(ys.min())
            y_max = float(ys.max())
            neighbor_x = []
            neighbor_y = []
            for other in records:
                if other.room_id == record.room_id:
                    continue
                if other.type_name in {"front_door", "door"}:
                    continue
                for start, end in _edge(other.polygon):
                    orientation = _edge_orientation(start, end)
                    if orientation == "v":
                        neighbor_x.append(
                            0.5 * (start[0] + end[0])
                        )
                    elif orientation == "h":
                        neighbor_y.append(
                            0.5 * (start[1] + end[1])
                        )
            if neighbor_x:
                nearest_min = min(
                    neighbor_x,
                    key=lambda value: abs(value - x_min),
                )
                nearest_max = min(
                    neighbor_x,
                    key=lambda value: abs(value - x_max),
                )
                if abs(nearest_min - x_min) <= config.balcony_snap_tolerance:
                    x_min = nearest_min
                if abs(nearest_max - x_max) <= config.balcony_snap_tolerance:
                    x_max = nearest_max
            if neighbor_y:
                nearest_min = min(
                    neighbor_y,
                    key=lambda value: abs(value - y_min),
                )
                nearest_max = min(
                    neighbor_y,
                    key=lambda value: abs(value - y_max),
                )
                if abs(nearest_min - y_min) <= config.balcony_snap_tolerance:
                    y_min = nearest_min
                if abs(nearest_max - y_max) <= config.balcony_snap_tolerance:
                    y_max = nearest_max
            rectangle = _clean_polygon(
                box(x_min, y_min, x_max, y_max)
            )
            if rectangle is None:
                continue
            if (
                rectangle.symmetric_difference(record.polygon).area
                > config.max_hollow_area
            ):
                continue
            old_polygon = record.polygon
            old_overlap = _non_auxiliary_overlap_score(records)
            old_hole = _hole_area(records)
            record.polygon = rectangle
            if (
                _non_auxiliary_overlap_score(records)
                > old_overlap + config.overlap_tolerance
                or _hole_area(records) > old_hole + 1e-8
            ):
                record.polygon = old_polygon
                continue
            changed = True
            changed_any = True
        if not changed:
            break
    return changed_any


def snap_adjacent_edges_final(records, adjacency, config):
    by_id = {record.room_id: record for record in records}
    changed_any = False
    for first_id, second_id in sorted(adjacency):
        first = by_id.get(int(first_id))
        second = by_id.get(int(second_id))
        if first is None or second is None:
            continue
        if first.type_name in {"front_door", "door"}:
            continue
        if second.type_name in {"front_door", "door"}:
            continue
        if first.fixed:
            mover, anchor = second, first
        elif second.fixed:
            mover, anchor = first, second
        elif first.type_name == "living":
            mover, anchor = second, first
        elif second.type_name == "living":
            mover, anchor = first, second
        else:
            continue
        old_shared = _shared_boundary_length(mover, anchor)
        old_overlap = _non_auxiliary_overlap_score(records)
        old_hole = _hole_area(records)
        for mover_edge in _edge(mover.polygon):
            orientation = _edge_orientation(*mover_edge)
            if orientation is None:
                continue
            axis = _edge_axis(orientation)
            varying = 0 if axis == 1 else 1
            mover_low, mover_high = sorted(
                [
                    float(mover_edge[0][varying]),
                    float(mover_edge[1][varying]),
                ]
            )
            mover_coord = float(
                0.5 * (mover_edge[0][axis] + mover_edge[1][axis])
            )
            for anchor_edge in _edge(anchor.polygon):
                if _edge_orientation(*anchor_edge) != orientation:
                    continue
                anchor_low, anchor_high = sorted(
                    [
                        float(anchor_edge[0][varying]),
                        float(anchor_edge[1][varying]),
                    ]
                )
                projection = min(mover_high, anchor_high) - max(
                    mover_low,
                    anchor_low,
                )
                if projection < config.min_shared_length:
                    continue
                anchor_coord = float(
                    0.5
                    * (anchor_edge[0][axis] + anchor_edge[1][axis])
                )
                if abs(mover_coord - anchor_coord) > 0.10:
                    continue
                moved = _move_polygon_subedge(
                    mover.polygon,
                    orientation,
                    mover_coord,
                    mover_low,
                    mover_high,
                    anchor_coord,
                    config,
                )
                if moved is None:
                    continue
                old_polygon = mover.polygon
                mover.polygon = moved
                new_shared = _shared_boundary_length(mover, anchor)
                if (
                    new_shared > old_shared + 1e-7
                    and _non_auxiliary_overlap_score(records)
                    <= old_overlap + config.overlap_tolerance
                    and _hole_area(records) <= old_hole + 1e-8
                ):
                    changed_any = True
                    break
                mover.polygon = old_polygon
            if changed_any:
                break
        if changed_any:
            continue
    return changed_any


def snap_balconies_to_neighbors(records, config):
    changed_any = False
    for record in records:
        if record.type_name != "balcony":
            continue
        source = (
            record.source_polygon
            if record.source_polygon is not None
            else record.polygon
        )
        if len(source.exterior.coords) - 1 != 4:
            continue
        bounds = source.bounds
        x_options = [bounds[0], bounds[2]]
        y_options = [bounds[1], bounds[3]]
        for other in records:
            if other.room_id == record.room_id:
                continue
            if other.type_name in {"front_door", "door"}:
                continue
            for start, end in _edge(other.polygon):
                orientation = _edge_orientation(start, end)
                if orientation == "v":
                    x_options.append(0.5 * (start[0] + end[0]))
                elif orientation == "h":
                    y_options.append(0.5 * (start[1] + end[1]))
        x_options = [
            value
            for value in sorted(set(round(v, 8) for v in x_options))
            if abs(value - bounds[0]) <= config.balcony_snap_tolerance
            or abs(value - bounds[2]) <= config.balcony_snap_tolerance
        ]
        y_options = [
            value
            for value in sorted(set(round(v, 8) for v in y_options))
            if abs(value - bounds[1]) <= config.balcony_snap_tolerance
            or abs(value - bounds[3]) <= config.balcony_snap_tolerance
        ]
        best = None
        old_polygon = record.polygon
        old_overlap = _non_auxiliary_overlap_score(records)
        old_hole = _hole_area(records)
        old_cost = old_overlap * 1000.0 + old_hole * 1000.0
        for x0 in x_options:
            if x0 >= bounds[2]:
                continue
            for x1 in x_options:
                if x1 <= x0:
                    continue
                for y0 in y_options:
                    if y0 >= bounds[3]:
                        continue
                    for y1 in y_options:
                        if y1 <= y0:
                            continue
                        if (
                            x1 - x0
                            < max(0.03, 0.35 * (bounds[2] - bounds[0]))
                            or y1 - y0
                            < max(0.03, 0.35 * (bounds[3] - bounds[1]))
                        ):
                            continue
                        rectangle = _clean_polygon(
                            box(x0, y0, x1, y1)
                        )
                        if rectangle is None:
                            continue
                        record.polygon = rectangle
                        new_overlap = _non_auxiliary_overlap_score(records)
                        new_hole = _hole_area(records)
                        if (
                            new_overlap
                            > old_overlap + config.overlap_tolerance
                            or new_hole > old_hole + 1e-8
                        ):
                            continue
                        alignment = (
                            abs(x0 - bounds[0])
                            + abs(x1 - bounds[2])
                            + abs(y0 - bounds[1])
                            + abs(y1 - bounds[3])
                        )
                        area_change = rectangle.symmetric_difference(
                            old_polygon
                        ).area
                        contact = 0.0
                        for other in records:
                            if other.room_id == record.room_id:
                                continue
                            if other.type_name in {"front_door", "door"}:
                                continue
                            contact += rectangle.boundary.intersection(
                                other.polygon.boundary
                            ).length
                        cost = (
                            new_overlap * 1000.0
                            + new_hole * 1000.0
                            + 0.2 * area_change
                            - 2.0 * contact
                            + alignment
                        )
                        if best is None or cost < best[0]:
                            best = (cost, rectangle)
        record.polygon = old_polygon
        if best is None:
            continue
        _, rectangle = best
        record.polygon = rectangle
        if (
            _non_auxiliary_overlap_score(records)
            <= old_overlap + config.overlap_tolerance
            and _hole_area(records) <= old_hole + 1e-8
        ):
            changed_any = True
        else:
            record.polygon = old_polygon
    return changed_any


def snap_non_adjacent_seams(records, config):
    changed_any = False
    for _ in range(3):
        changed = False
        for first_index, first in enumerate(records):
            if first.type_name in {"front_door", "door"}:
                continue
            for second in records[first_index + 1 :]:
                if second.type_name in {"front_door", "door"}:
                    continue
                distance = first.polygon.distance(second.polygon)
                if distance <= 1e-8 or distance > config.seam_snap_tolerance:
                    continue
                mover, anchor = _choose_mover(first, second)
                if mover is None or anchor is None:
                    continue
                old_shared = _shared_boundary_length(mover, anchor)
                old_overlap = _non_auxiliary_overlap_score(records)
                old_hole = _hole_area(records)
                for mover_edge in _edge(mover.polygon):
                    orientation = _edge_orientation(*mover_edge)
                    if orientation is None:
                        continue
                    axis = _edge_axis(orientation)
                    varying = 0 if axis == 1 else 1
                    mover_low, mover_high = sorted(
                        [
                            float(mover_edge[0][varying]),
                            float(mover_edge[1][varying]),
                        ]
                    )
                    mover_coord = float(
                        0.5 * (
                            mover_edge[0][axis]
                            + mover_edge[1][axis]
                        )
                    )
                    for anchor_edge in _edge(anchor.polygon):
                        if _edge_orientation(*anchor_edge) != orientation:
                            continue
                        anchor_low, anchor_high = sorted(
                            [
                                float(anchor_edge[0][varying]),
                                float(anchor_edge[1][varying]),
                            ]
                        )
                        projection = min(mover_high, anchor_high) - max(
                            mover_low,
                            anchor_low,
                        )
                        if projection < config.min_shared_length:
                            continue
                        anchor_coord = float(
                            0.5
                            * (
                                anchor_edge[0][axis]
                                + anchor_edge[1][axis]
                            )
                        )
                        if abs(mover_coord - anchor_coord) > (
                            config.seam_snap_tolerance
                        ):
                            continue
                        moved = _move_polygon_subedge(
                            mover.polygon,
                            orientation,
                            mover_coord,
                            mover_low,
                            mover_high,
                            anchor_coord,
                            config,
                        )
                        if moved is None:
                            continue
                        old_polygon = mover.polygon
                        mover.polygon = moved
                        new_shared = _shared_boundary_length(
                            mover,
                            anchor,
                        )
                        if (
                            new_shared > old_shared + 1e-7
                            and _non_auxiliary_overlap_score(records)
                            <= old_overlap + config.overlap_tolerance
                            and _hole_area(records) <= old_hole + 1e-8
                        ):
                            changed = True
                            changed_any = True
                            break
                        mover.polygon = old_polygon
                    if changed:
                        break
                if changed:
                    break
            if changed:
                break
        if not changed:
            break
    return changed_any


def orthogonalize_near_axis_edges(records, config):
    changed_any = False
    for record in records:
        if record.type_name in {"front_door", "door"}:
            continue
        coords = np.asarray(
            record.polygon.exterior.coords[:-1],
            dtype=float,
        ).copy()
        changed = False
        for index in range(len(coords)):
            following = (index + 1) % len(coords)
            delta = coords[following] - coords[index]
            dx = abs(float(delta[0]))
            dy = abs(float(delta[1]))
            if (
                1e-8 < dx <= config.coordinate_tolerance
                and dy > config.coordinate_tolerance
            ):
                target = 0.5 * (
                    coords[index, 0] + coords[following, 0]
                )
                coords[index, 0] = target
                coords[following, 0] = target
                changed = True
            elif (
                1e-8 < dy <= config.coordinate_tolerance
                and dx > config.coordinate_tolerance
            ):
                target = 0.5 * (
                    coords[index, 1] + coords[following, 1]
                )
                coords[index, 1] = target
                coords[following, 1] = target
                changed = True
        if not changed:
            continue
        polygon = _polygon_from_points(coords)
        if polygon is None:
            continue
        old_polygon = record.polygon
        old_overlap = _non_auxiliary_overlap_score(records)
        old_hole = _hole_area(records)
        record.polygon = polygon
        if (
            _non_auxiliary_overlap_score(records)
            > old_overlap + config.max_head_area
            or _hole_area(records)
            > max(old_hole, config.max_hollow_area) + 1e-8
        ):
            record.polygon = old_polygon
        else:
            changed_any = True
    return changed_any


def fill_small_envelope_notches(records, config):
    changed_any = False
    for record in records:
        if record.type_name in {"living", "front_door", "door"}:
            continue
        min_x, min_y, max_x, max_y = record.polygon.bounds
        rectangle = _clean_polygon(box(min_x, min_y, max_x, max_y))
        if rectangle is None:
            continue
        missing = rectangle.difference(record.polygon).area
        if missing <= 1e-8 or missing > config.max_hollow_area:
            continue
        old_polygon = record.polygon
        old_overlap = _non_auxiliary_overlap_score(records)
        old_hole = _hole_area(records)
        record.polygon = rectangle
        for other in records:
            if other.room_id == record.room_id:
                continue
            if other.type_name in {"front_door", "door"}:
                continue
            if not rectangle.intersects(other.polygon):
                continue
            clipped = _clean_polygon(
                other.polygon.difference(rectangle)
            )
            if clipped is not None:
                other.polygon = clipped
        if (
            _non_auxiliary_overlap_score(records)
            > old_overlap + config.max_hollow_area
            or _hole_area(records) > old_hole + 1e-8
        ):
            record.polygon = old_polygon
            continue
        changed_any = True
    return changed_any


def reconstruct_thin_rooms(records, config):
    changed_any = False
    for record in records:
        if record.type_name in {
            "living",
            "front_door",
            "door",
        }:
            continue
        if len(record.polygon.exterior.coords) - 1 != 4:
            continue
        min_x, min_y, max_x, max_y = record.polygon.bounds
        width = max_x - min_x
        height = max_y - min_y
        short = min(width, height)
        long = max(width, height)
        if (
            short <= 1e-8
            or short > config.thin_room_max_short_edge
            or long / short < config.thin_room_min_aspect_ratio
        ):
            continue
        if width >= height:
            bottom = LineString([(min_x, min_y), (max_x, min_y)])
            top = LineString([(min_x, max_y), (max_x, max_y)])
            bottom_shared = sum(
                bottom.intersection(other.polygon.boundary).length
                for other in records
                if other.room_id != record.room_id
            )
            top_shared = sum(
                top.intersection(other.polygon.boundary).length
                for other in records
                if other.room_id != record.room_id
            )
            expand = (config.thin_room_expand_factor - 1.0) * short
            if bottom_shared <= top_shared:
                candidate = box(
                    min_x,
                    min_y - expand,
                    max_x,
                    max_y,
                )
            else:
                candidate = box(
                    min_x,
                    min_y,
                    max_x,
                    max_y + expand,
                )
        else:
            left = LineString([(min_x, min_y), (min_x, max_y)])
            right = LineString([(max_x, min_y), (max_x, max_y)])
            left_shared = sum(
                left.intersection(other.polygon.boundary).length
                for other in records
                if other.room_id != record.room_id
            )
            right_shared = sum(
                right.intersection(other.polygon.boundary).length
                for other in records
                if other.room_id != record.room_id
            )
            expand = (config.thin_room_expand_factor - 1.0) * short
            if left_shared <= right_shared:
                candidate = box(
                    min_x - expand,
                    min_y,
                    max_x,
                    max_y,
                )
            else:
                candidate = box(
                    min_x,
                    min_y,
                    max_x + expand,
                    max_y,
                )
        candidate = _clean_polygon(candidate)
        if candidate is None:
            continue
        old_polygon = record.polygon
        old_overlap = _non_auxiliary_overlap_score(records)
        old_hole = _hole_area(records)
        record.polygon = candidate
        if (
            _non_auxiliary_overlap_score(records)
            > old_overlap + config.overlap_tolerance
            or _hole_area(records) > old_hole + 1e-8
        ):
            record.polygon = old_polygon
            continue
        changed_any = True
    return changed_any


def finalize_general_rules(records, adjacency, config):
    old_overlap = _non_auxiliary_overlap_score(records)
    old_hole = _hole_area(records)
    snapshot = _record_polygon_snapshot(records)
    reconstruct_thin_rooms(records, config)
    fill_open_u_voids(records, config)
    fill_medium_enclosed_holes(records, config)
    snap_adjacent_edges_final(records, adjacency, config)
    fill_open_u_voids(records, config)
    fill_medium_enclosed_holes(records, config)
    snap_balconies_to_neighbors(records, config)
    orthogonalize_near_axis_edges(records, config)
    fill_small_envelope_notches(records, config)
    snap_non_adjacent_seams(records, config)
    if (
        _non_auxiliary_overlap_score(records)
        > old_overlap + config.overlap_tolerance
        or _hole_area(records) > old_hole + 1e-8
    ):
        _restore_polygon_snapshot(records, snapshot)


def clip_overlaps_by_edge_guard(
    records,
    adjacency,
    config,
    min_action=None,
    ignore_auxiliary=False,
):
    locked = _locked_shared_lengths(records, adjacency, config)
    changed_any = False
    if min_action is None:
        min_action = config.min_overlap_action_area
    for _ in range(6):
        changed = False
        for first_index, first in enumerate(records):
            for second in records[first_index + 1 :]:
                if ignore_auxiliary and (
                    first.type_name in {"front_door", "door"}
                    or second.type_name in {"front_door", "door"}
                ):
                    continue
                overlap = _overlap_area(first, second)
                if overlap < min_action:
                    continue
                smaller, larger = (
                    (first, second)
                    if first.polygon.area <= second.polygon.area
                    else (second, first)
                )
                if (
                    overlap
                    >= 0.75 * max(smaller.polygon.area, 1e-12)
                    and smaller.polygon.difference(
                        larger.polygon
                    ).area
                    <= config.max_hollow_area
                ):
                    old_larger = larger.polygon
                    carved = _clean_polygon(
                        larger.polygon.difference(smaller.polygon)
                    )
                    if carved is not None:
                        larger.polygon = carved
                        changed = True
                        changed_any = True
                        continue
                    larger.polygon = old_larger
                intersection = first.polygon.intersection(
                    second.polygon
                )
                first_ratio, second_ratio = _overlap_boundary_owner_ratio(
                    first,
                    second,
                    intersection,
                )
                clipped_record = None
                if first_ratio >= 0.70 and first_ratio > second_ratio:
                    clipped_record = first
                elif second_ratio >= 0.70 and second_ratio > first_ratio:
                    clipped_record = second
                else:
                    old_first = first.polygon
                    old_second = second.polygon
                    if not _clip_overlap_with_edge_guard(
                        records,
                        first,
                        second,
                        config,
                    ):
                        continue
                    if _preserves_locked_shared(
                        records,
                        locked,
                        config,
                    ):
                        changed = True
                        changed_any = True
                    else:
                        first.polygon = old_first
                        second.polygon = old_second
                    continue
                old_polygon = clipped_record.polygon
                other = (
                    second
                    if clipped_record is first
                    else first
                )
                difference = _clean_polygon(
                    old_polygon.difference(other.polygon)
                )
                if difference is None:
                    continue
                clipped_record.polygon = difference
                if _preserves_locked_shared(
                    records,
                    locked,
                    config,
                ):
                    changed = True
                    changed_any = True
                else:
                    clipped_record.polygon = old_polygon
        if not changed:
            break
    return changed_any


def postprocess_records_general_rules(
    records,
    adjacency,
    config=None,
    ensure_doorable=True,
):
    config = config or PostProcessConfig()
    door_polygons = {
        record.room_id: record.polygon
        for record in records
        if record.type_name in {"front_door", "door"}
    }
    _run_hole_guarded(
        records,
        lambda: postprocess_records_canonical(
            records,
            adjacency,
            config,
        ),
        config,
    )
    _run_hole_guarded(
        records,
        lambda: snap_global_seams(records, config),
        config,
    )
    _run_hole_guarded(
        records,
        lambda: resolve_overlaps_by_edge_translation(
            records,
            config,
        ),
        config,
    )
    _run_hole_guarded(
        records,
        lambda: clip_overlaps_by_edge_guard(
            records,
            adjacency,
            config,
        ),
        config,
    )
    _run_hole_guarded(
        records,
        lambda: fill_medium_enclosed_holes(records, config),
        config,
    )
    snap_parallel_edges_safely(records, config)
    orthogonalize_balconies(records, config)
    _run_hole_guarded(
        records,
        lambda: _apply_hollow_edge_moves(records, config),
        config,
    )
    _run_hole_guarded(
        records,
        lambda: clip_small_fixed_head_overlaps(records, config),
        config,
    )
    _run_hole_guarded(
        records,
        lambda: resolve_remaining_overlaps_by_edge_move(
            records,
            config,
        ),
        config,
    )
    _run_hole_guarded(
        records,
        lambda: remove_small_rectangular_steps(records, config),
        config,
    )
    _run_hole_guarded(
        records,
        lambda: merge_near_vertices(
            records,
            tolerance=config.point_merge_tolerance,
        ),
        config,
    )
    _run_hole_guarded(
        records,
        lambda: fill_long_short_long_self_holes(records, config),
        config,
    )
    _run_hole_guarded(
        records,
        lambda: fill_open_u_voids(records, config),
        config,
    )
    _run_hole_guarded(
        records,
        lambda: fill_medium_enclosed_holes(records, config),
        config,
    )
    _run_hole_guarded(
        records,
        lambda: clip_overlaps_by_edge_guard(
            records,
            adjacency,
            config,
            min_action=config.overlap_tolerance,
            ignore_auxiliary=True,
        ),
        config,
    )
    finalize_general_rules(records, adjacency, config)
    orthogonalize_near_axis_edges(records, config)
    fill_small_envelope_notches(records, config)
    snap_non_adjacent_seams(records, config)
    snap_global_seams(records, config)
    fill_open_u_voids(records, config)
    fill_medium_enclosed_holes(records, config)
    clip_overlaps_by_edge_guard(
        records,
        adjacency,
        config,
        min_action=config.overlap_tolerance,
        ignore_auxiliary=True,
    )
    if ensure_doorable:
        ensure_doorable_connected_edges(
            records,
            adjacency,
            config,
        )
    for record in records:
        if (
            record.type_name in {"front_door", "door"}
            and record.room_id in door_polygons
        ):
            record.polygon = door_polygons[record.room_id]
    metrics = {
        "rooms": len(records),
        "connected_gaps": _connected_gap_stats(records, adjacency),
        "overlap_areas": _overlap_stats(
            records,
            ignore_auxiliary=True,
        ),
        "holes": len(_hole_polygons(records)),
        "reverted_heads": 0,
    }
    return records, metrics


def snap_all_parallel_edges(records, config):
    """Unify every pair of near-parallel overlapping room edges."""
    for _ in range(4):
        changed = False
        for first_index in range(len(records)):
            first = records[first_index]
            first_edges = _edge(first.polygon)
            for second_index in range(first_index + 1, len(records)):
                second = records[second_index]
                second_edges = _edge(second.polygon)
                for first_edge in first_edges:
                    orientation = _edge_orientation(*first_edge)
                    if orientation is None:
                        continue
                    if orientation == "h":
                        first_lo = min(first_edge[0][0], first_edge[1][0])
                        first_hi = max(first_edge[0][0], first_edge[1][0])
                        first_coord = 0.5 * (
                            first_edge[0][1] + first_edge[1][1]
                        )
                    else:
                        first_lo = min(first_edge[0][1], first_edge[1][1])
                        first_hi = max(first_edge[0][1], first_edge[1][1])
                        first_coord = 0.5 * (
                            first_edge[0][0] + first_edge[1][0]
                        )
                    for second_edge in second_edges:
                        if _edge_orientation(*second_edge) != orientation:
                            continue
                        if orientation == "h":
                            second_lo = min(
                                second_edge[0][0],
                                second_edge[1][0],
                            )
                            second_hi = max(
                                second_edge[0][0],
                                second_edge[1][0],
                            )
                            second_coord = 0.5 * (
                                second_edge[0][1] + second_edge[1][1]
                            )
                        else:
                            second_lo = min(
                                second_edge[0][1],
                                second_edge[1][1],
                            )
                            second_hi = max(
                                second_edge[0][1],
                                second_edge[1][1],
                            )
                            second_coord = 0.5 * (
                                second_edge[0][0] + second_edge[1][0]
                            )
                        if (
                            abs(first_coord - second_coord)
                            > config.coordinate_merge_tolerance
                        ):
                            continue
                        overlap = min(first_hi, second_hi) - max(
                            first_lo,
                            second_lo,
                        )
                        if overlap < config.min_shared_length:
                            continue
                        if first.fixed:
                            target = first_coord
                        elif second.fixed:
                            target = second_coord
                        elif _priority(first) >= _priority(second):
                            target = first_coord
                        else:
                            target = second_coord
                        moved_first = _set_edge_coordinate(
                            first,
                            first_edge,
                            orientation,
                            target,
                            config,
                        )
                        moved_second = _set_edge_coordinate(
                            second,
                            second_edge,
                            orientation,
                            target,
                            config,
                        )
                        changed = changed or moved_first or moved_second
        if not changed:
            break


def _priority(record):
    return (
        TYPE_PRIORITY.get(record.type_name, 0),
        record.polygon.area,
    )


def _choose_mover(first, second):
    if first.fixed and second.fixed:
        return None, None
    if first.fixed:
        return second, first
    if second.fixed:
        return first, second
    if _priority(first) > _priority(second):
        return second, first
    return first, second


def _edge_orientation(first, second, tolerance=1e-6):
    delta = second - first
    if abs(delta[1]) <= tolerance:
        return "h"
    if abs(delta[0]) <= tolerance:
        return "v"
    return None


def _parallel_edge_candidate(
    mover,
    anchor,
    gap_tolerance,
    min_shared_length,
):
    mover_edges = _edge(mover.polygon)
    anchor_edges = _edge(anchor.polygon)
    best = None
    for mover_edge in mover_edges:
        mover_orientation = _edge_orientation(*mover_edge)
        if mover_orientation is None:
            continue
        mover_low = min(mover_edge[0][0], mover_edge[1][0])
        mover_high = max(mover_edge[0][0], mover_edge[1][0])
        mover_low = min(mover_edge[0][1], mover_edge[1][1])
        mover_high = max(mover_edge[0][1], mover_edge[1][1])
        for anchor_edge in anchor_edges:
            if _edge_orientation(*anchor_edge) != mover_orientation:
                continue
            if mover_orientation == "h":
                perpendicular = abs(
                    mover_edge[0][1] - anchor_edge[0][1]
                )
                mover_low = min(mover_edge[0][0], mover_edge[1][0])
                mover_high = max(mover_edge[0][0], mover_edge[1][0])
                anchor_low = min(anchor_edge[0][0], anchor_edge[1][0])
                anchor_high = max(anchor_edge[0][0], anchor_edge[1][0])
            else:
                perpendicular = abs(
                    mover_edge[0][0] - anchor_edge[0][0]
                )
                mover_low = min(mover_edge[0][1], mover_edge[1][1])
                mover_high = max(mover_edge[0][1], mover_edge[1][1])
                anchor_low = min(anchor_edge[0][1], anchor_edge[1][1])
                anchor_high = max(anchor_edge[0][1], anchor_edge[1][1])
            overlap = min(mover_high, anchor_high) - max(
                mover_low,
                anchor_low,
            )
            if overlap >= min_shared_length:
                projection_gap = 0.0
            else:
                projection_gap = max(
                    anchor_low - mover_high,
                    mover_low - anchor_high,
                    0.0,
                )
            if perpendicular > gap_tolerance:
                continue
            if (
                overlap < min_shared_length
                and projection_gap > gap_tolerance
            ):
                continue
            score = (
                perpendicular
                + 1.5 * projection_gap
                - overlap
            )
            candidate = (
                score,
                mover_edge,
                anchor_edge,
                mover_orientation,
                perpendicular,
                projection_gap,
            )
            if best is None or candidate[0] < best[0]:
                best = candidate
    return best


def _translated_polygon(polygon, delta):
    return _clean_polygon(
        Polygon(np.asarray(polygon.exterior.coords) + delta)
    )


def resolve_overlaps(records, config):
    for _ in range(4):
        changed = False
        for first_index in range(len(records)):
            first = records[first_index]
            for second_index in range(first_index + 1, len(records)):
                second = records[second_index]
                if not first.polygon.intersects(second.polygon):
                    continue
                intersection = first.polygon.intersection(second.polygon)
                if intersection.area <= config.overlap_tolerance:
                    continue
                loser, winner = _choose_mover(first, second)
                if loser is None:
                    continue
                min_x, min_y, max_x, max_y = intersection.bounds
                overlap_width = max_x - min_x
                overlap_height = max_y - min_y
                loser_center = loser.polygon.centroid
                winner_center = winner.polygon.centroid
                if overlap_width <= overlap_height:
                    direction = (
                        1.0
                        if loser_center.x >= winner_center.x
                        else -1.0
                    )
                    delta = np.asarray(
                        [
                            direction
                            * min(
                                overlap_width + 1e-6,
                                config.max_edge_move * 2.0,
                            ),
                            0.0,
                        ]
                    )
                else:
                    direction = (
                        1.0
                        if loser_center.y >= winner_center.y
                        else -1.0
                    )
                    delta = np.asarray(
                        [
                            0.0,
                            direction
                            * min(
                                overlap_height + 1e-6,
                                config.max_edge_move * 2.0,
                            ),
                        ]
                    )
                moved = _translated_polygon(loser.polygon, delta)
                if moved is None:
                    continue
                old_overlap = loser.polygon.intersection(
                    winner.polygon
                ).area
                new_overlap = moved.intersection(winner.polygon).area
                if new_overlap > 0.5 * old_overlap:
                    continue
                old_total = 0.0
                new_total = 0.0
                for other in records:
                    if other is loser:
                        continue
                    old_total += loser.polygon.intersection(
                        other.polygon
                    ).area
                    new_total += moved.intersection(other.polygon).area
                if new_total > old_total + config.overlap_tolerance:
                    continue
                loser.polygon = moved
                changed = True
        if not changed:
            break

    for _ in range(3):
        changed = False
        for first_index in range(len(records)):
            first = records[first_index]
            for second_index in range(first_index + 1, len(records)):
                second = records[second_index]
                if not first.polygon.intersects(second.polygon):
                    continue
                intersection = first.polygon.intersection(second.polygon)
                if intersection.area <= config.overlap_tolerance:
                    continue
                loser, winner = _choose_mover(first, second)
                if loser is None:
                    continue
                difference = loser.polygon.difference(winner.polygon)
                difference = _clean_polygon(difference)
                if difference is None:
                    continue
                if difference.area < 0.2 * loser.polygon.area:
                    continue
                loser.polygon = difference
                changed = True
        if not changed:
            break


def _hole_polygons(records):
    union = unary_union([record.polygon for record in records])
    holes = []
    for polygon in _polygon_parts(union):
        for ring in polygon.interiors:
            hole = Polygon(ring)
            if hole.is_valid and hole.area > 0:
                holes.append(hole)
    return holes


def _line_parts(geometry):
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]
    if geometry.geom_type in {"MultiLineString", "GeometryCollection"}:
        return [
            part
            for part in geometry.geoms
            if part.geom_type == "LineString"
        ]
    return []


def _cluster_1d(values, tolerance, anchor_values=()):
    values = np.asarray(sorted(values), dtype=float)
    anchors = np.asarray(sorted(anchor_values), dtype=float)
    if values.size == 0:
        return np.empty(0, dtype=float)
    groups = [[float(values[0])]]
    for value in values[1:]:
        if value - groups[-1][-1] <= tolerance:
            groups[-1].append(float(value))
        else:
            groups.append([float(value)])
    centers = []
    for group in groups:
        lower = min(group)
        upper = max(group)
        fixed = anchors[
            (anchors >= lower - 1e-9)
            & (anchors <= upper + 1e-9)
        ]
        if fixed.size:
            centers.append(float(np.median(fixed)))
        else:
            centers.append(float(np.mean(group)))
    return np.asarray(centers, dtype=float)


def _nearest_cluster(value, coordinates):
    index = int(np.argmin(np.abs(coordinates - value)))
    return float(coordinates[index])


def _connected_gap_stats(records, adjacency):
    by_id = {record.room_id: record for record in records}
    gaps = []
    for first_id, second_id in adjacency:
        first = by_id.get(int(first_id))
        second = by_id.get(int(second_id))
        if first is None or second is None:
            continue
        gaps.append(first.polygon.distance(second.polygon))
    return gaps


def _overlap_stats(records, ignore_auxiliary=False):
    values = []
    for first_index in range(len(records)):
        for second_index in range(first_index + 1, len(records)):
            first = records[first_index]
            second = records[second_index]
            if ignore_auxiliary and (
                first.type_name in {"front_door", "door"}
                or second.type_name in {"front_door", "door"}
            ):
                continue
            if not first.polygon.intersects(second.polygon):
                continue
            area = first.polygon.intersection(second.polygon).area
            if area > 1e-8:
                values.append(area)
    return values


def _shared_boundary_length(first, second):
    intersection = first.polygon.boundary.intersection(
        second.polygon.boundary
    )
    return float(intersection.length)


def _shared_boundary_segment_lengths(first, second):
    intersection = first.polygon.boundary.intersection(
        second.polygon.boundary
    )
    return [
        float(part.length)
        for part in _line_parts(intersection)
        if part.length > 1e-8
    ]


def _max_shared_boundary_segment(first, second):
    lengths = _shared_boundary_segment_lengths(first, second)
    return max(lengths, default=0.0)


def _has_doorable_shared_segment(first, second, config):
    return (
        _max_shared_boundary_segment(first, second)
        >= config.min_door_shared_length - 1e-7
    )


def _overlap_area(first, second):
    if not first.polygon.intersects(second.polygon):
        return 0.0
    return float(first.polygon.intersection(second.polygon).area)


def _total_overlap_score(records, fixed_rooms=None):
    fixed_rooms = fixed_rooms or {}
    total = 0.0
    for first_index in range(len(records)):
        for second_index in range(first_index + 1, len(records)):
            first = records[first_index]
            second = records[second_index]
            area = _overlap_area(first, second)
            if area <= 0:
                continue
            if first.fixed or second.fixed:
                total += area * 4.0
            else:
                total += area
    return total


def records_to_dicts(records):
    output = []
    for record in records:
        coords = np.asarray(
            record.polygon.exterior.coords[:-1],
            dtype=float,
        )
        output.append(
            {
                "room_id": int(record.room_id),
                "name": record.name,
                "type": record.type_name,
                "fixed": bool(record.fixed),
                "area": float(record.polygon.area),
                "polygon": coords.tolist(),
            }
        )
    return output
