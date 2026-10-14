"""Deployment physical scale and clear-floor-area checks for generated rooms."""
import json
import math
from pathlib import Path

from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

DEFAULT_MIN_AREAS_M2 = {
    "kitchen": 2.0, "bathroom": 2.0,
}


class PlanQualityError(ValueError):
    """A generated layout must be retried instead of imported into Revit."""


def load_physical_scale(size="medium", data_dir=None):
    """Apply deployment area calibration without altering measured dataset stats.

    Multipliers describe polygon area, so lengths change by their square root.
    Clear floor area must still subtract the unchanged physical wall widths.
    """
    if size not in {"small", "medium", "large"}:
        raise ValueError("Physical size must be small, medium or large")
    data_dir = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parent / "data"
    statistics_path = data_dir / "raw_recovery_scale_stats.json"
    reference = 0.05
    if statistics_path.is_file():
        statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
        reference = float(statistics["resplan_meter_scale"]["meters_per_raw_pixel"]["median"])
    policy_path = data_dir / "geometry_policy.json"
    multiplier = 1.0
    if policy_path.is_file():
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        multiplier = float(policy.get("output_area_multipliers", {}).get(size, 1.0))
    if not math.isfinite(reference) or reference <= 0:
        raise ValueError("Reference metre scale must be finite and positive")
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("Output area multiplier must be finite and positive")
    linear = math.sqrt(multiplier)
    return {
        "size": size,
        "reference_pixel_to_meter": reference,
        "output_area_multiplier": multiplier,
        "output_length_multiplier": linear,
        "pixel_to_meter": reference * linear,
    }


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
