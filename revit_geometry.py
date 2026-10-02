"""Convert final polygons into unique Revit wall hosts and physical openings."""
from collections import defaultdict
import math
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union, linemerge
from door_window_rules import opening_lines_conflict

# Inclusive millimetre ranges; sampled values are multiples of 100 mm.
WINDOW_RANGES_MM = {
    "living": ((900, 1000), (1500, 1700)),
    "bedroom": ((900, 1000), (1500, 1700)),
    "kitchen": ((900, 1000), (1200, 1500)),
    "bathroom": ((1200, 1500), (600, 1200)),
}


def sample_window_dimensions(room_type, rng):
    sill, height = WINDOW_RANGES_MM.get(room_type, WINDOW_RANGES_MM["living"])
    return tuple(int(rng.integers(low // 100, high // 100 + 1)) / 10
                 for low, high in (sill, height))


def build_revit_geometry(rooms, openings, pixel_to_meter, seed):
    functional = [r for r in rooms if r["type"] not in {"door", "front_door"}]
    by_type = {r["room_id"]: r["type"] for r in functional}
    polygons = {r["room_id"]: Polygon(r["polygon"]) for r in functional}
    for room in functional:
        polygon = polygons[room["room_id"]]
        if not polygon.is_valid or polygon.is_empty:
            raise ValueError(f"invalid room polygon: {room['room_id']}")
        point = polygon.representative_point()
        room["interior_point"] = [point.x, point.y]
    ids = list(polygons)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if polygons[a].intersection(polygons[b]).area * pixel_to_meter ** 2 > 0.0025:
                raise ValueError(f"overlapping rooms: {a}, {b}")
    boundary = unary_union([p.boundary for p in polygons.values()])
    lines = list(boundary.geoms) if hasattr(boundary, "geoms") else [boundary]
    groups = defaultdict(list)
    for line in lines:
        for start, end in zip(list(line.coords), list(line.coords)[1:]):
            segment = LineString([start, end])
            if segment.length * pixel_to_meter < 0.005:
                continue
            owners = tuple(sorted(rid for rid, p in polygons.items()
                if p.boundary.intersection(segment).length >= segment.length - 1e-5))
            if not owners or len(owners) > 2:
                raise ValueError("invalid wall ownership")
            orientation = "h" if abs(start[1] - end[1]) < 1e-5 else "v" if abs(start[0] - end[0]) < 1e-5 else None
            if orientation is None:
                raise ValueError("non-orthogonal wall")
            coordinate = start[1] if orientation == "h" else start[0]
            groups[(owners, orientation, round(coordinate, 5))].append(segment)
    walls, railings = [], []
    for (owners, _, _), segments in sorted(groups.items()):
        merged = linemerge(segments) if len(segments) > 1 else segments[0]
        pieces = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
        for piece in pieces:
            line = [list(piece.coords[0]), list(piece.coords[-1])]
            if len(owners) == 1 and by_type[owners[0]] == "balcony":
                railings.append(dict(railing_id=f"railing_{len(railings):03d}",
                    reference_line=line, owner_room_id=owners[0], height_m=1.1))
                continue
            walls.append(dict(wall_id=f"wall_{len(walls):03d}", reference_line=line,
                host="exterior_wall" if len(owners) == 1 else "shared_wall",
                owner_room_ids=list(owners), thickness_m=0.4 if len(owners) == 1 else 0.2,
                height_m=3.3))
    rng = np.random.default_rng(int(seed) ^ 0x75379)
    for opening in openings:
        line = LineString(opening["reference_line"])
        owners = {opening["owner_room_id"]}
        if opening.get("target_room_id") is not None:
            owners.add(opening["target_room_id"])
        candidates = [w for w in walls if owners <= set(w["owner_room_ids"])
            and w["host"] == opening["host"]
            and LineString(w["reference_line"]).buffer(1e-4).covers(line)]
        if not candidates:
            raise ValueError(f"opening has no complete wall host: {opening['type']}, {sorted(owners)}")
        opening["wall_id"] = candidates[0]["wall_id"]
        opening["width_m"] = float(opening["width"]) * pixel_to_meter
        if not math.isfinite(opening["width_m"]) or opening["width_m"] <= 0.05:
            raise ValueError("invalid opening width")
        if opening["type"] == "window":
            kind = by_type.get(opening["owner_room_id"], "living")
            opening["sill_m"], opening["height_m"] = sample_window_dimensions(kind, rng)
        else:
            opening["sill_m"] = 0.0
            opening["height_m"] = 2.1
    validate_opening_clearances(openings, pixel_to_meter)
    return walls, railings


def validate_opening_clearances(openings, pixel_to_meter):
    for index, first in enumerate(openings):
        for second in openings[index + 1:]:
            owners_a = {first["owner_room_id"], first.get("target_room_id")} - {None}
            owners_b = {second["owner_room_id"], second.get("target_room_id")} - {None}
            both_doors = first["type"] != "window" and second["type"] != "window"
            if opening_lines_conflict(LineString(first["reference_line"]), LineString(second["reference_line"]),
                                      pixel_to_meter, common_room=both_doors and bool(owners_a & owners_b)):
                raise ValueError(f"Openings are too close: {sorted(owners_a)} and {sorted(owners_b)}")
