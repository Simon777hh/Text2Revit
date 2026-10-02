"""Physical minimum-area checks for generated rooms."""
import json
from pathlib import Path

from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

DEFAULT_MIN_AREAS_M2 = {
    "kitchen": 2.0, "bathroom": 2.0,
}


class PlanQualityError(ValueError):
    """A generated layout must be retried instead of imported into Revit."""


def load_min_areas():
    path = Path(__file__).resolve().parent / "data" / "geometry_policy.json"
    values = DEFAULT_MIN_AREAS_M2.copy()
    if path.is_file():
        values.update(json.loads(path.read_text(encoding="utf-8"))["minimum_room_areas_m2"])
    if any(not isinstance(v, (int, float)) or v <= 0 for v in values.values()):
        raise ValueError("Minimum room areas must be positive numbers")
    return values


def measure_room_areas(rooms, pixel_to_meter, walls=()):
    """Estimate clear floor area after subtracting wall footprints."""
    if pixel_to_meter <= 0:
        raise ValueError("Metre scale must be positive")
    footprints = [LineString(w["reference_line"]).buffer(
        w["thickness_m"] / (2.0 * pixel_to_meter), cap_style=2, join_style=2)
        for w in walls]
    wall_area = unary_union(footprints)
    areas = {}
    for room in rooms:
        if room["type"] in {"front_door", "door"}:
            continue
        polygon = Polygon(room["polygon"])
        if polygon.is_empty or not polygon.is_valid:
            raise PlanQualityError(f"Room {room['room_id']} has invalid geometry")
        areas[str(room["room_id"])] = float(polygon.difference(wall_area).area * pixel_to_meter ** 2)
    return areas


def validate_room_areas(rooms, pixel_to_meter, minimum_areas=None, walls=()):
    limits = load_min_areas() if minimum_areas is None else minimum_areas
    areas = measure_room_areas(rooms, pixel_to_meter, walls)
    for room in rooms:
        kind = room["type"]
        if kind in {"front_door", "door"}:
            continue
        area = areas[str(room["room_id"])]
        minimum = float(limits.get(kind, 0.0))
        if area + 1e-8 < minimum:
            raise PlanQualityError(
                f"{kind} {room['room_id']} is too small: {area:.2f} m²; minimum {minimum:.2f} m²")
    return areas
