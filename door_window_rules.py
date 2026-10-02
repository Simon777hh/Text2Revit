"""Deterministic door/window placement from final room polygons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union


BALCONY_DOOR_SCALE = 2.0


def opening_lines_conflict(first, second, pixel_to_meter, common_room=False):
    """Keep 200 mm between openings; opposed doors need leaf-depth clearance."""
    distance = first.distance(second) * pixel_to_meter
    if distance < .2 - 1e-7:
        return True
    a, b = np.asarray(first.coords), np.asarray(second.coords)
    u, v = a[-1]-a[0], b[-1]-b[0]
    u /= np.linalg.norm(u); v /= np.linalg.norm(v)
    parallel = abs(float(np.dot(u, v))) > .999
    delta = b[0]-a[0]
    normal_separation = abs(float(u[0]*delta[1]-u[1]*delta[0])) * pixel_to_meter
    if parallel and common_room and normal_separation > 1e-4:
        first_interval = sorted(a @ u); second_interval = sorted(b @ u)
        overlap = min(first_interval[1],second_interval[1])-max(first_interval[0],second_interval[0])
        if overlap > 1e-6 and distance < max(first.length, second.length)*pixel_to_meter + .1:
            return True
    if not parallel and common_room and distance < .35:
        return True
    return False


def _place_clear_door(segments, preferred, length, thickness, gap, occupied, scale, rooms):
    for segment in segments:
        anchors = [preferred] + [segment[0] + f*(segment[1]-segment[0]) for f in np.linspace(0,1,51)]
        anchors.sort(key=lambda point: float(np.linalg.norm(point-preferred)))
        for anchor in anchors:
            candidate = _rectangle_on_edge(segment, length, thickness, gap, anchor)
            if candidate is None:
                continue
            line = LineString([candidate[2],candidate[3]])
            if line.length * scale < .6:
                continue
            if any(opening_lines_conflict(line, LineString([p.long_start,p.long_end]), scale,
                                          common_room=bool(({p.owner_room_id,p.target_room_id}-{None}) & set(rooms)))
                   for p in occupied):
                continue
            return candidate
    return None


@dataclass
class DoorWindowPiece:
    kind: str
    polygon: Polygon
    owner_room_id: int
    target_room_id: int | None = None
    anchor: tuple[float, float] | None = None
    long_start: tuple[float, float] | None = None
    long_end: tuple[float, float] | None = None
    thickness: float = 0.0
    source: str = "rule"
    outward_blocked: bool = False


def _polygon_edges(polygon):
    coords = list(polygon.exterior.coords)
    for start, end in zip(coords[:-1], coords[1:]):
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length <= 1e-9:
            continue
        unit = delta / length
        outward = np.asarray([unit[1], -unit[0]], dtype=float)
        if not polygon.exterior.is_ccw:
            outward = -outward
        midpoint = 0.5 * (start + end)
        yield start, end, unit, outward, midpoint, length


def _edge_clear_run(
    edge,
    room_polygons,
    clearance,
    samples=81,
):
    start, end, _, outward, _, length = edge
    blocked = np.zeros(samples, dtype=bool)
    for index, fraction in enumerate(np.linspace(0.0, 1.0, samples)):
        point = start + fraction * (end - start)
        probe = point + outward * clearance
        for polygon in room_polygons:
            if polygon.contains(Point(probe)):
                blocked[index] = True
                break
            if polygon.distance(Point(probe)) <= clearance * 0.10:
                blocked[index] = True
                break
    best_start = 0
    best_end = -1
    current_start = None
    for index, is_blocked in enumerate(blocked):
        if not is_blocked and current_start is None:
            current_start = index
        if is_blocked and current_start is not None:
            if index - current_start > best_end - best_start:
                best_start = current_start
                best_end = index
            current_start = None
    if current_start is not None and samples - current_start > best_end - best_start:
        best_start = current_start
        best_end = samples
    if best_end <= best_start:
        return None
    fraction_low = best_start / (samples - 1)
    fraction_high = (best_end - 1) / (samples - 1)
    clear_start = start + fraction_low * (end - start)
    clear_end = start + fraction_high * (end - start)
    return {
        "start": clear_start,
        "end": clear_end,
        "length": float(np.linalg.norm(clear_end - clear_start)),
        "center": 0.5 * (clear_start + clear_end),
    }


def _rectangle_on_edge(
    segment,
    length,
    thickness,
    gap,
    anchor=None,
):
    start, end = segment
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    delta = end - start
    segment_length = float(np.linalg.norm(delta))
    if segment_length <= 1e-9:
        return None
    unit = delta / segment_length
    normal = np.asarray([unit[1], -unit[0]], dtype=float)
    usable_start = gap
    usable_end = segment_length - gap
    usable_length = usable_end - usable_start
    if usable_length <= 0.0:
        return None
    actual_length = min(length, usable_length)
    if actual_length <= max(gap, 1e-6):
        return None
    if anchor is None:
        center_t = 0.5 * segment_length
    else:
        anchor = np.asarray(anchor, dtype=float)
        center_t = float(np.dot(anchor - start, unit))
    center_t = float(
        np.clip(
            center_t,
            usable_start + actual_length / 2.0,
            usable_end - actual_length / 2.0,
        )
    )
    center = start + unit * center_t
    half_along = unit * actual_length / 2.0
    half_across = normal * thickness / 2.0
    points = [
        center - half_along - half_across,
        center + half_along - half_across,
        center + half_along + half_across,
        center - half_along + half_across,
    ]
    polygon = Polygon(points)
    if not polygon.is_valid or polygon.area <= 1e-9:
        return None
    long_start = center - unit * actual_length / 2.0
    long_end = center + unit * actual_length / 2.0
    return (
        polygon,
        center,
        (float(long_start[0]), float(long_start[1])),
        (float(long_end[0]), float(long_end[1])),
    )


def _polygon_axes(polygon):
    rectangle = polygon.minimum_rotated_rectangle
    if rectangle is None or rectangle.is_empty:
        return None
    if rectangle.geom_type != "Polygon":
        return None
    coords = list(rectangle.exterior.coords[:-1])
    if len(coords) < 4:
        return None
    edges = []
    for start, end in zip(coords, coords[1:] + coords[:1]):
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        length = float(np.linalg.norm(end - start))
        if length <= 1e-9:
            continue
        edges.append((length, start, end))
    if len(edges) < 2:
        return None
    edges.sort(key=lambda item: item[0])
    short = edges[0]
    long = edges[-1]
    return {
        "long_start": long[1],
        "long_end": long[2],
        "thickness": float(short[0]),
    }


def i_shape_segments(piece):
    """Return stem and two cap segments for the Roman-I door symbol."""
    if piece.long_start is None or piece.long_end is None:
        return None
    start = np.asarray(piece.long_start, dtype=float)
    end = np.asarray(piece.long_end, dtype=float)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length <= 1e-9:
        return None
    unit = delta / length
    normal = np.asarray([-unit[1], unit[0]], dtype=float)
    cap_half = max(piece.thickness * 1.25, length * 0.16)
    caps = []
    for center in (start, end):
        caps.append(
            (
                center - normal * cap_half,
                center + normal * cap_half,
            )
        )
    return {
        "stem": (start, end),
        "caps": caps,
    }


def _shared_segments(first, second):
    intersection = first.polygon.boundary.intersection(
        second.polygon.boundary
    )
    if intersection.is_empty:
        return []
    parts = []
    if intersection.geom_type == "LineString":
        parts = [intersection]
    elif intersection.geom_type == "MultiLineString":
        parts = list(intersection.geoms)
    segments = []
    for part in parts:
        if part.length <= 1e-9:
            continue
        coords = list(part.coords)
        segments.append(
            (
                np.asarray(coords[0], dtype=float),
                np.asarray(coords[-1], dtype=float),
            )
        )
    return segments


def _room_scale(records):
    polygons = [
        record.polygon
        for record in records
        if record.type_name not in {"front_door", "door"}
        and record.polygon is not None
        and not record.polygon.is_empty
    ]
    if not polygons:
        return 1.0
    min_x = min(polygon.bounds[0] for polygon in polygons)
    min_y = min(polygon.bounds[1] for polygon in polygons)
    max_x = max(polygon.bounds[2] for polygon in polygons)
    max_y = max(polygon.bounds[3] for polygon in polygons)
    return max(max_x - min_x, max_y - min_y) / 2.0


def _farthest_boundary_point(polygon, point, samples=240):
    boundary = polygon.exterior
    point = np.asarray(point, dtype=float)
    best_point = None
    best_distance = -1.0
    for fraction in np.linspace(0.0, 1.0, samples):
        candidate = np.asarray(
            boundary.interpolate(
                fraction,
                normalized=True,
            ).coords[0],
            dtype=float,
        )
        distance = float(np.linalg.norm(candidate - point))
        if distance > best_distance:
            best_distance = distance
            best_point = candidate
    return best_point


def build_front_door(
    records,
    living_id,
    door_length_ratio,
    door_thickness_ratio,
    door_gap_ratio,
):
    by_id = {record.room_id: record for record in records}
    living = by_id.get(int(living_id))
    if living is None:
        return None
    room_polygons = [
        record.polygon
        for record in records
        if record.room_id != living_id
        and record.type_name not in {"front_door", "door"}
    ]
    scale = _room_scale(records)
    existing = next(
        (
            record
            for record in records
            if record.type_name == "front_door"
            and record.polygon is not None
            and not record.polygon.is_empty
        ),
        None,
    )
    edge_list = list(_polygon_edges(living.polygon))
    clearance = max(
        door_thickness_ratio * scale * 1.7,
        1.0,
    )
    clear_runs = [
        _edge_clear_run(edge, room_polygons, clearance)
        for edge in edge_list
    ]
    required_length = (
        door_length_ratio * scale
        + 2.0 * door_gap_ratio * scale
    )
    usable_indices = [
        index
        for index, clear_run in enumerate(clear_runs)
        if clear_run is not None
        and clear_run["length"] >= required_length
    ]
    if existing is not None:
        existing_center = np.asarray(
            existing.polygon.centroid.coords[0],
            dtype=float,
        )
        nearest_index = min(
            range(len(edge_list)),
            key=lambda index: np.linalg.norm(
                existing_center - edge_list[index][4]
            ),
        )
        order = [
            (nearest_index + offset) % len(edge_list)
            for offset in range(len(edge_list))
        ]
        selected_index = next(
            (
                index
                for index in order
                if index in usable_indices
            ),
            None,
        )
        outward_blocked = selected_index != nearest_index
        if selected_index is None:
            selected_index = max(
                range(len(edge_list)),
                key=lambda index: (
                    clear_runs[index]["length"]
                    if clear_runs[index] is not None
                    else -1.0
                ),
            )
            outward_blocked = True
        edge = edge_list[selected_index]
        clear_run = clear_runs[selected_index]
        anchor = existing_center
        source = (
            "flow"
            if selected_index == nearest_index
            else "patched"
        )
    else:
        candidates = [
            index
            for index in usable_indices
        ]
        if not candidates:
            candidates = list(range(len(edge_list)))
        selected_index = max(
            candidates,
            key=lambda index: (
                clear_runs[index]["length"]
                if clear_runs[index] is not None
                else edge_list[index][-1]
            ),
        )
        edge = edge_list[selected_index]
        clear_run = clear_runs[selected_index]
        anchor = edge[4]
        source = "rule"
        outward_blocked = True
    if clear_run is not None:
        edge = (clear_run["start"], clear_run["end"])
    placed = _rectangle_on_edge(
        (edge[0], edge[1]),
        door_length_ratio * scale,
        door_thickness_ratio * scale,
        door_gap_ratio * scale,
        anchor=anchor,
    )
    if placed is None:
        return None
    polygon, center, long_start, long_end = placed
    return DoorWindowPiece(
        kind="front_door",
        polygon=polygon,
        owner_room_id=int(living_id),
        anchor=(float(center[0]), float(center[1])),
        long_start=long_start,
        long_end=long_end,
        thickness=door_thickness_ratio * scale,
        source=source,
        outward_blocked=outward_blocked,
    )


def build_internal_doors(
    records,
    adjacency,
    living_id,
    front_door,
    door_length_ratio,
    door_thickness_ratio,
    door_gap_ratio,
    pixel_to_meter=None,
):
    by_id = {record.room_id: record for record in records}
    living = by_id.get(int(living_id))
    if living is None:
        return []
    scale = _room_scale(records)
    front_anchor = (
        np.asarray(front_door.anchor, dtype=float)
        if front_door is not None
        else np.asarray(living.polygon.centroid.coords[0], dtype=float)
    )
    living_cluster_anchor = _farthest_boundary_point(
        living.polygon,
        front_anchor,
    )
    pieces = []
    room_door_anchors = []
    for first_id, second_id in sorted(adjacency):
        first = by_id.get(int(first_id))
        second = by_id.get(int(second_id))
        if first is None or second is None:
            continue
        if first.type_name in {"front_door", "door"}:
            continue
        if second.type_name in {"front_door", "door"}:
            continue
        pair_types = {first.type_name, second.type_name}
        target_door_length_ratio = (
            door_length_ratio * BALCONY_DOOR_SCALE
            if "balcony" in pair_types
            else door_length_ratio
        )
        required_length = (
            target_door_length_ratio * scale
            + 2.0 * door_gap_ratio * scale
        )
        segments = _shared_segments(first, second)
        if not segments:
            continue
        valid_segments = [
            segment
            for segment in segments
            if np.linalg.norm(segment[1] - segment[0])
            >= required_length
        ]
        if not valid_segments:
            valid_segments = segments
        if "balcony" in pair_types:
            segment = max(
                valid_segments,
                key=lambda item: np.linalg.norm(
                    item[1] - item[0]
                ),
            )
            anchor = 0.5 * (segment[0] + segment[1])
        elif living_id in {first.room_id, second.room_id}:
            segment = min(
                valid_segments,
                key=lambda item: np.linalg.norm(
                    0.5 * (item[0] + item[1])
                    - living_cluster_anchor
                ),
            )
            anchor = living_cluster_anchor
        elif room_door_anchors:
            segment = min(
                valid_segments,
                key=lambda item: min(
                    np.linalg.norm(
                        0.5 * (item[0] + item[1]) - anchor
                    )
                    for anchor in room_door_anchors
                ),
            )
            segment_center = 0.5 * (segment[0] + segment[1])
            anchor = min(
                room_door_anchors,
                key=lambda candidate: np.linalg.norm(
                    segment_center - candidate
                ),
            )
        else:
            segment = max(
                valid_segments,
                key=lambda item: np.linalg.norm(
                    0.5 * (item[0] + item[1]) - front_anchor
                ),
            )
            start_distance = np.linalg.norm(
                segment[0] - front_anchor
            )
            end_distance = np.linalg.norm(
                segment[1] - front_anchor
            )
            anchor = (
                segment[0]
                if start_distance >= end_distance
                else segment[1]
            )
        placed = _rectangle_on_edge(
            segment,
            target_door_length_ratio * scale,
            door_thickness_ratio * scale,
            door_gap_ratio * scale,
            anchor=anchor,
        )
        if pixel_to_meter is not None:
            occupied = ([front_door] if front_door is not None else []) + pieces
            placed = _place_clear_door(
                [segment] + [item for item in valid_segments if item is not segment], anchor,
                target_door_length_ratio * scale, door_thickness_ratio * scale,
                max(door_gap_ratio * scale, .2 / pixel_to_meter), occupied,
                pixel_to_meter, {first.room_id, second.room_id})
        if placed is None:
            continue
        polygon, center, long_start, long_end = placed
        piece = DoorWindowPiece(
            kind="door",
            polygon=polygon,
            owner_room_id=int(first.room_id),
            target_room_id=int(second.room_id),
            anchor=(float(center[0]), float(center[1])),
            long_start=long_start,
            long_end=long_end,
            thickness=door_thickness_ratio * scale,
        )
        pieces.append(piece)
        room_door_anchors.append(center)
    return pieces


def _sample_living_window_count(rng):
    values = np.asarray([1, 2, 3], dtype=np.int16)
    weights = np.asarray([0.45, 0.35, 0.20], dtype=float)
    weights /= weights.sum()
    return int(rng.choice(values, p=weights))


def _window_length_ratio_for_area(
    area_ratio,
    minimum_ratio,
):
    return float(
        np.clip(
            0.14 + 0.55 * area_ratio,
            minimum_ratio,
            0.45,
        )
    )


def _front_door_wall_edges(room_edges, front_door):
    """Find the entrance wall, including consecutive collinear edge segments.

    A wall remains one face when the polygon contains intermediate vertices.
    Separate parallel walls are not excluded.
    """
    if front_door is None or not room_edges:
        return set()
    if front_door.anchor is None:
        return set()
    anchor = np.asarray(front_door.anchor, dtype=float)
    direction = None
    if front_door.long_start is not None and front_door.long_end is not None:
        delta = np.asarray(front_door.long_end) - np.asarray(front_door.long_start)
        length = float(np.linalg.norm(delta))
        if length > 1e-9:
            direction = delta / length

    def distance_to_edge(index):
        start, _, unit, _, _, length = room_edges[index]
        along = np.clip(np.dot(anchor - start, unit), 0.0, length)
        return float(np.linalg.norm(anchor - (start + along * unit)))

    eligible = [
        index for index, edge in enumerate(room_edges)
        if direction is None or abs(float(np.dot(direction, edge[2]))) >= 0.999
    ]
    if not eligible:
        eligible = list(range(len(room_edges)))
    entrance_index = min(eligible, key=distance_to_edge)
    blocked = {entrance_index}
    start, _, unit, _, _, _ = room_edges[entrance_index]
    tolerance = max(1e-7, max(edge[-1] for edge in room_edges) * 1e-6)

    def on_entrance_line(edge):
        if abs(float(np.dot(unit, edge[2]))) < 1.0 - 1e-8:
            return False
        normal = np.asarray([-unit[1], unit[0]])
        return all(abs(float(np.dot(point - start, normal))) <= tolerance
                   for point in edge[:2])

    for step in (-1, 1):
        index = (entrance_index + step) % len(room_edges)
        while index not in blocked and on_entrance_line(room_edges[index]):
            blocked.add(index)
            index = (index + step) % len(room_edges)
    return blocked


def build_windows(
    records,
    window_length_ratio,
    window_thickness_ratio,
    window_gap_ratio,
    rng,
    front_door=None,
    balcony_count_by_room=None,
    skip_room_ids=(),
    occupied_polygons=(),
    pixel_to_meter=None,
):
    scale = _room_scale(records)
    pieces = []
    skip_room_ids = set(skip_room_ids)
    balcony_count_by_room = balcony_count_by_room or {}
    functional_records = [
        record
        for record in records
        if record.type_name not in {"front_door", "door"}
    ]
    total_area = sum(
        max(record.polygon.area, 0.0)
        for record in functional_records
    )
    room_polygons = [
        record.polygon
        for record in functional_records
    ]
    facade_segments = []
    if front_door is not None and front_door.anchor is not None and room_polygons:
        envelope = unary_union(room_polygons)
        envelopes = list(envelope.geoms) if envelope.geom_type == "MultiPolygon" else [envelope]
        exterior = min(envelopes, key=lambda p: p.exterior.distance(Point(front_door.anchor)))
        exterior_edges = list(_polygon_edges(exterior))
        facade_segments = [LineString(exterior_edges[i][:2])
                           for i in _front_door_wall_edges(exterior_edges, front_door)]
    facade_tolerance = max(1e-7, scale * 1e-6)
    for record in functional_records:
        if record.type_name not in {
            "living",
            "kitchen",
            "bedroom",
            "bathroom",
            "storage",
        }:
            continue
        if record.room_id in skip_room_ids:
            continue
        if (
            record.type_name == "bedroom"
            and balcony_count_by_room.get(record.room_id, 0) > 0
        ):
            continue
        area_ratio = (
            max(record.polygon.area, 0.0) / total_area
            if total_area > 0.0
            else 0.0
        )
        normal_ratio = _window_length_ratio_for_area(
            area_ratio,
            window_length_ratio,
        )
        minimum_length = (
            window_length_ratio * scale
            + 2.0 * window_gap_ratio * scale
        )
        candidates = []
        others = [
            polygon
            for polygon in room_polygons
            if polygon is not record.polygon
        ]
        room_edges = list(_polygon_edges(record.polygon))
        for edge_index, edge in enumerate(room_edges):
            edge_line = LineString(edge[:2])
            if any(edge_line.intersection(segment.buffer(facade_tolerance)).length
                   > facade_tolerance * 4 for segment in facade_segments):
                continue
            clear_run = _edge_clear_run(
                edge,
                others,
                clearance=max(
                    window_thickness_ratio * scale * 1.7,
                    1.0,
                ),
            )
            if (
                clear_run is None
                or clear_run["length"] < minimum_length
            ):
                continue
            candidates.append(
                {
                    "edge_index": edge_index,
                    "clear_run": clear_run,
                }
            )
        if not candidates:
            continue
        candidates.sort(
            key=lambda item: item["clear_run"]["length"],
            reverse=True,
        )
        candidate_by_edge = {
            item["edge_index"]: item
            for item in candidates
        }
        if record.type_name == "living":
            target_count = max(
                0,
                _sample_living_window_count(rng)
                - int(balcony_count_by_room.get(record.room_id, 0)),
            )
        else:
            target_count = 1
        selected = candidates[:target_count]
        if not selected:
            continue
        front_anchor = (
            np.asarray(front_door.anchor, dtype=float)
            if front_door is not None
            else None
        )
        placed_polygons = [
            occupied
            for occupied in occupied_polygons
            if occupied is not None and not occupied.is_empty
        ]
        has_balcony = (
            int(balcony_count_by_room.get(record.room_id, 0)) > 0
        )
        farthest_index = None
        long_candidate = None
        if (
            record.type_name == "living"
            and front_anchor is not None
            and not has_balcony
        ):
            farthest_index = max(
                range(len(selected)),
                key=lambda index: np.linalg.norm(
                    selected[index]["clear_run"]["center"]
                    - front_anchor
                ),
            )
            longest = selected[farthest_index]
            required_double_length = (
                normal_ratio * 1.35 * scale
                + 2.0 * window_gap_ratio * scale
            )
            if (
                longest["clear_run"]["length"]
                >= required_double_length
            ):
                long_candidate = longest
            else:
                base_edge = longest["edge_index"]
                for offset in range(1, len(room_edges) + 1):
                    for adjacent in (
                        (base_edge - offset) % len(room_edges),
                        (base_edge + offset) % len(room_edges),
                    ):
                        candidate = candidate_by_edge.get(adjacent)
                        if candidate is None:
                            continue
                        if (
                            candidate["clear_run"]["length"]
                            >= required_double_length
                        ):
                            long_candidate = candidate
                            break
                    if long_candidate is not None:
                        break
            if long_candidate is not None:
                selected[farthest_index] = long_candidate
        selected = list(
            {
                candidate["edge_index"]: candidate
                for candidate in selected
            }.values()
        )
        for index, candidate in enumerate(selected):
            clear_run = candidate["clear_run"]
            ratio = normal_ratio
            if (
                long_candidate is not None
                and candidate is long_candidate
            ):
                ratio = normal_ratio * 1.35
            placed = _rectangle_on_edge(
                (clear_run["start"], clear_run["end"]),
                ratio * scale,
                window_thickness_ratio * scale,
                window_gap_ratio * scale,
                anchor=clear_run["center"],
            )
            if placed is None:
                continue
            polygon, center, long_start, long_end = placed
            if any(
                polygon.distance(occupied) < (.2 / pixel_to_meter if pixel_to_meter else 1e-9)
                for occupied in placed_polygons
            ):
                continue
            pieces.append(
                DoorWindowPiece(
                    kind="window",
                    polygon=polygon,
                    owner_room_id=int(record.room_id),
                    anchor=(float(center[0]), float(center[1])),
                    long_start=long_start,
                    long_end=long_end,
                    thickness=window_thickness_ratio * scale,
                )
            )
            placed_polygons.append(polygon)
    return pieces


def build_door_window_pieces(
    records,
    adjacency,
    living_id,
    rng=None,
    door_length_ratio=0.15,
    door_thickness_ratio=0.03,
    door_gap_ratio=0.03,
    window_length_ratio=0.15,
    window_thickness_ratio=0.03,
    window_gap_ratio=0.04,
    pixel_to_meter=None,
):
    rng = rng or np.random.default_rng()
    balcony_count_by_room = {}
    bedroom_with_balcony = set()
    for first_id, second_id in adjacency:
        first = next(
            (record for record in records if record.room_id == first_id),
            None,
        )
        second = next(
            (record for record in records if record.room_id == second_id),
            None,
        )
        if first is None or second is None:
            continue
        pair_types = {first.type_name, second.type_name}
        if "balcony" not in pair_types:
            continue
        if "living" in pair_types:
            living = first if first.type_name == "living" else second
            balcony_count_by_room[living.room_id] = (
                balcony_count_by_room.get(living.room_id, 0) + 1
            )
        if "bedroom" in pair_types:
            bedroom = (
                first if first.type_name == "bedroom" else second
            )
            bedroom_with_balcony.add(bedroom.room_id)
            balcony_count_by_room[bedroom.room_id] = (
                balcony_count_by_room.get(bedroom.room_id, 0) + 1
            )
    front_door = build_front_door(
        records,
        living_id,
        door_length_ratio,
        door_thickness_ratio,
        door_gap_ratio,
    )
    doors = build_internal_doors(
        records,
        adjacency,
        living_id,
        front_door,
        door_length_ratio,
        door_thickness_ratio,
        door_gap_ratio,
        pixel_to_meter=pixel_to_meter,
    )
    windows = build_windows(
        records,
        window_length_ratio,
        window_thickness_ratio,
        window_gap_ratio,
        rng,
        front_door=front_door,
        balcony_count_by_room=balcony_count_by_room,
        skip_room_ids=bedroom_with_balcony,
        occupied_polygons=[
            piece.polygon
            for piece in [front_door, *doors]
            if piece is not None
        ],
        pixel_to_meter=pixel_to_meter,
    )
    return {
        "front_door": front_door,
        "doors": doors,
        "windows": windows,
    }
