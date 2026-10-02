from __future__ import annotations

import math
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon


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

CANVAS_SIZE = 255.0
CANVAS_AREA = CANVAS_SIZE * CANVAS_SIZE
DATA_DIR = Path(__file__).resolve().parent / "data"
ASPECT_RANGES_PATH = DATA_DIR / "room_aspect_ranges.json"
ASPECT_SAMPLES_PATH = DATA_DIR / "room_aspect_samples.npz"

# Deployment ranges selected from the raw canvas occupancy study.
OCCUPANCY_RANGES = {
    "small": (0.3314, 0.4020, 0.5040),
    "medium": (0.5185, 0.5489, 0.6554),
    "large": (0.6700, 0.6956, 0.8707),
}


def load_room_aspect_stats(
    ranges_path=ASPECT_RANGES_PATH,
    samples_path=ASPECT_SAMPLES_PATH,
):
    with open(ranges_path, "r", encoding="utf-8") as handle:
        ranges = json.load(handle)
    samples = {}
    if Path(samples_path).exists():
        with np.load(samples_path, allow_pickle=False) as payload:
            samples = {
                name: np.asarray(payload[name], dtype=float)
                for name in payload.files
            }
    stats = {}
    for type_name, values in ranges.items():
        if int(values.get("count", 0)) <= 0:
            continue
        if type_name in samples and samples[type_name].size:
            entries = samples[type_name]
        else:
            entries = np.asarray(
                [
                    math.log(float(values["q25"])),
                    math.log(float(values["median"])),
                    math.log(float(values["q85"])),
                ],
                dtype=float,
            )
        stats[type_name] = AspectBounds(
            lower=math.log(float(values["p10"])),
            upper=math.log(float(values["p90"])),
            median=math.log(float(values["median"])),
            sample_lower=math.log(float(values["q25"])),
            sample_upper=math.log(float(values["q85"])),
            samples=entries,
        )
    return stats


@dataclass(frozen=True)
class AspectBounds:
    lower: float
    upper: float
    median: float
    sample_lower: float
    sample_upper: float
    samples: np.ndarray = field(repr=False)


def _room_polygon(features, valid, room_ids, room_id):
    indices = np.flatnonzero(valid & (room_ids == room_id))
    if indices.size < 3:
        return None
    polygon = Polygon(features[indices, :2])
    if polygon.area <= 1e-8:
        return None
    return polygon


def build_room_aspect_stats(raw_features, node_type_ids):
    values = {
        name: []
        for name in ROOM_TYPE_NAMES.values()
        if name not in {"front_door", "door"}
    }
    for plan_index in range(len(raw_features)):
        features = raw_features[plan_index]
        valid = features[:, 4:24].sum(axis=1) > 0.5
        room_ids = features[:, 4:24].argmax(axis=1)
        for room_id in np.unique(room_ids[valid]):
            room_index = int(room_id)
            type_index = int(node_type_ids[plan_index, room_index])
            type_name = ROOM_TYPE_NAMES.get(type_index)
            if type_name not in values:
                continue
            polygon = _room_polygon(
                features,
                valid,
                room_ids,
                room_id,
            )
            if polygon is None:
                continue
            min_x, min_y, max_x, max_y = polygon.bounds
            width = max_x - min_x
            height = max_y - min_y
            if min(width, height) <= 1e-8:
                continue
            ratio = max(width, height) / min(width, height)
            values[type_name].append(math.log(ratio))

    stats = {}
    for type_name, entries in values.items():
        if not entries:
            continue
        entries = np.asarray(entries, dtype=float)
        stats[type_name] = AspectBounds(
            lower=float(np.quantile(entries, 0.10)),
            upper=float(np.quantile(entries, 0.90)),
            median=float(np.median(entries)),
            sample_lower=float(np.quantile(entries, 0.25)),
            sample_upper=float(np.quantile(entries, 0.85)),
            samples=entries,
        )
    return stats


def _record_room_type(record):
    if getattr(record, "type_name", None):
        return str(record.type_name)
    return ""


def room_aspect(record):
    min_x, min_y, max_x, max_y = record.polygon.bounds
    width = max_x - min_x
    height = max_y - min_y
    if min(width, height) <= 1e-8:
        return None
    return max(width, height) / min(width, height)


def _signed_log_ratio(record):
    min_x, min_y, max_x, max_y = record.polygon.bounds
    width = max_x - min_x
    height = max_y - min_y
    if min(width, height) <= 1e-8:
        return None
    return math.log(height / width)


def detect_global_aspect_scale(
    records,
    stats,
    max_abs_position=0.50,
    sample_target=True,
    rng=None,
):
    entries = []
    for record in records:
        type_name = _record_room_type(record)
        bounds = stats.get(type_name)
        signed = _signed_log_ratio(record)
        if bounds is None or signed is None:
            continue
        entries.append(
            {
                "record": record,
                "type": type_name,
                "signed": signed,
                "bounds": bounds,
            }
        )
    if not entries:
        return 0.0, {}

    def evaluate(position):
        violations = 0
        total_excess = 0.0
        max_excess = 0.0
        for entry in entries:
            magnitude = abs(
                entry["signed"] - 2.0 * position
            )
            bounds = entry["bounds"]
            low = min(bounds.lower, bounds.upper)
            high = max(bounds.lower, bounds.upper)
            if magnitude < low:
                excess = low - magnitude
            elif magnitude > high:
                excess = magnitude - high
            else:
                continue
            violations += 1
            total_excess += excess
            max_excess = max(max_excess, excess)
        return violations, total_excess, max_excess

    before = evaluate(0.0)
    anomalies = []
    for entry in entries:
        signed = entry["signed"]
        bounds = entry["bounds"]
        magnitude = abs(signed)
        low = min(bounds.lower, bounds.upper)
        high = max(bounds.lower, bounds.upper)
        if magnitude < low:
            excess = low - magnitude
        elif magnitude > high:
            excess = magnitude - high
        else:
            continue
        anomalies.append(
            {
                **entry,
                "magnitude": magnitude,
                "lower": low,
                "upper": high,
                "excess": excess,
            }
        )
    if not anomalies:
        return 0.0, {
            "triggered": False,
            "violations_before": 0,
            "violations_after": 0,
            "total_excess_before": 0.0,
            "total_excess_after": 0.0,
        }

    selected = max(
        anomalies,
        key=lambda entry: (
            entry["excess"],
            entry["magnitude"],
        ),
    )
    signed = selected["signed"]
    bounds = selected["bounds"]
    target_log = bounds.median
    sampled_target = False
    if sample_target:
        candidates = bounds.samples[
            (bounds.samples >= bounds.sample_lower)
            & (bounds.samples <= bounds.sample_upper)
        ]
        if not candidates.size:
            candidates = bounds.samples[
                (bounds.samples >= bounds.lower)
                & (bounds.samples <= bounds.upper)
            ]
        if candidates.size:
            generator = rng or np.random.default_rng()
            target_log = float(generator.choice(candidates))
            sampled_target = True
    direction = 1.0 if signed >= 0.0 else -1.0
    target_signed = direction * target_log
    selected_position = 0.5 * (signed - target_signed)

    bounded = max(1, int(max_abs_position / 0.01))
    candidates = {
        0.0,
        float(
            np.clip(
                selected_position,
                -max_abs_position,
                max_abs_position,
            )
        ),
    }
    candidates.update(
        float(value)
        for value in np.linspace(
            -max_abs_position,
            max_abs_position,
            2 * bounded + 1,
        )
    )
    for entry in entries:
        bounds = entry["bounds"]
        low = min(bounds.lower, bounds.upper)
        high = max(bounds.lower, bounds.upper)
        entry_direction = (
            1.0 if entry["signed"] >= 0.0 else -1.0
        )
        for target_magnitude in (low, high, bounds.median):
            candidate = 0.5 * (
                entry["signed"]
                - entry_direction * target_magnitude
            )
            candidates.add(
                float(
                    np.clip(
                        candidate,
                        -max_abs_position,
                        max_abs_position,
                    )
                )
            )

    def candidate_key(position):
        violations, total_excess, max_excess = evaluate(position)
        return (
            violations,
            total_excess,
            max_excess,
            abs(position),
        )

    best_position = min(candidates, key=candidate_key)
    if candidate_key(best_position)[:3] >= before:
        best_position = 0.0
    position = float(best_position)
    after = evaluate(position)
    violations_after = after[0]
    return position, {
        "triggered": abs(position) > 1e-12,
        "reason": (
            "global_violation_reduction"
            if abs(position) > 1e-12
            else "global_transform_not_beneficial"
        ),
        "violations_before": len(anomalies),
        "violations_after": violations_after,
        "total_excess_before": before[1],
        "total_excess_after": after[1],
        "max_excess_before": before[2],
        "max_excess_after": after[2],
        "selected_room_id": int(selected["record"].room_id),
        "selected_room_type": selected["type"],
        "selected_before_aspect": math.exp(selected["magnitude"]),
        "selected_target_aspect": math.exp(target_log),
        "selected_target_sampled": sampled_target,
        "selected_after_aspect": math.exp(
            abs(signed - 2.0 * position)
        ),
        "selected_excess_log": selected["excess"],
    }


def apply_global_aspect_scale(records, position):
    if abs(position) <= 1e-12:
        return records
    polygons = [
        record.polygon
        for record in records
        if record.polygon is not None and not record.polygon.is_empty
    ]
    if not polygons:
        return records
    min_x = min(polygon.bounds[0] for polygon in polygons)
    min_y = min(polygon.bounds[1] for polygon in polygons)
    max_x = max(polygon.bounds[2] for polygon in polygons)
    max_y = max(polygon.bounds[3] for polygon in polygons)
    center = (
        0.5 * (min_x + max_x),
        0.5 * (min_y + max_y),
    )
    scale_x = math.exp(position)
    scale_y = math.exp(-position)
    for record in records:
        record.polygon = affinity.scale(
            record.polygon,
            xfact=scale_x,
            yfact=scale_y,
            origin=center,
        )
        if getattr(record, "source_polygon", None) is not None:
            record.source_polygon = affinity.scale(
                record.source_polygon,
                xfact=scale_x,
                yfact=scale_y,
                origin=center,
            )
    return records


def restore_canvas_aspect(
    records,
    stats,
    sample_target=True,
    rng=None,
):
    position, info = detect_global_aspect_scale(
        records,
        stats,
        sample_target=sample_target,
        rng=rng,
    )
    apply_global_aspect_scale(records, position)
    return {
        **info,
        "position": position,
        "scale_x": math.exp(position),
        "scale_y": math.exp(-position),
    }


def functional_room_area(records) -> float:
    total = 0.0
    for record in records:
        if record.type_name in {"front_door", "door"}:
            continue
        if record.polygon is None or record.polygon.is_empty:
            continue
        total += float(record.polygon.area)
    return total


def sample_occupancy_target(
    group: str,
    rng: np.random.Generator,
) -> float:
    if group not in OCCUPANCY_RANGES:
        raise KeyError(f"unknown occupancy group: {group}")
    lower, peak, upper = OCCUPANCY_RANGES[group]
    return float(
        rng.triangular(lower, peak, upper)
    )


def _record_bounds(records):
    polygons = [
        record.polygon
        for record in records
        if record.polygon is not None and not record.polygon.is_empty
    ]
    if not polygons:
        return None
    min_x = min(polygon.bounds[0] for polygon in polygons)
    min_y = min(polygon.bounds[1] for polygon in polygons)
    max_x = max(polygon.bounds[2] for polygon in polygons)
    max_y = max(polygon.bounds[3] for polygon in polygons)
    return (
        0.5 * (min_x + max_x),
        0.5 * (min_y + max_y),
    )


def restore_canvas_area(
    records,
    group: str,
    rng: np.random.Generator,
):
    """Map normalized room areas back to raw 255x255 canvas occupancy."""
    current_area = functional_room_area(records)
    if current_area <= 1e-9:
        raise ValueError("cannot restore area for empty records")
    target_occupancy = sample_occupancy_target(group, rng)
    target_area = target_occupancy * CANVAS_AREA
    area_scale = target_area / current_area
    linear_scale = math.sqrt(area_scale)
    center = _record_bounds(records)
    if center is not None and abs(linear_scale - 1.0) > 1e-12:
        for record in records:
            record.polygon = affinity.scale(
                record.polygon,
                xfact=linear_scale,
                yfact=linear_scale,
                origin=center,
            )
            if getattr(record, "source_polygon", None) is not None:
                record.source_polygon = affinity.scale(
                    record.source_polygon,
                    xfact=linear_scale,
                    yfact=linear_scale,
                    origin=center,
                )
    return {
        "occupancy_group": group,
        "current_area": current_area,
        "target_occupancy": target_occupancy,
        "target_area": target_area,
        "area_scale": area_scale,
        "linear_scale": linear_scale,
        "current_occupancy_before": current_area / CANVAS_AREA,
        "current_occupancy_after": (
            functional_room_area(records) / CANVAS_AREA
        ),
    }


def place_records_on_canvas(
    records,
    canvas_size=CANVAS_SIZE,
):
    """Center raw-scale room polygons inside the fixed 255 x 255 canvas."""
    center = _record_bounds(records)
    if center is None:
        return {
            "canvas_size": float(canvas_size),
            "translation": (0.0, 0.0),
        }
    target_center = 0.5 * canvas_size
    delta_x = target_center - center[0]
    delta_y = target_center - center[1]
    for record in records:
        record.polygon = affinity.translate(
            record.polygon,
            xoff=delta_x,
            yoff=delta_y,
        )
        if getattr(record, "source_polygon", None) is not None:
            record.source_polygon = affinity.translate(
                record.source_polygon,
                xoff=delta_x,
                yoff=delta_y,
            )
    return {
        "canvas_size": float(canvas_size),
        "translation": (float(delta_x), float(delta_y)),
    }


def fit_records_to_canvas(
    records,
    canvas_size=CANVAS_SIZE,
    preserve_area=True,
):
    """Fit the final bbox inside the fixed canvas without clipping."""
    polygons = [
        record.polygon
        for record in records
        if record.polygon is not None and not record.polygon.is_empty
    ]
    if not polygons:
        return {
            "canvas_fit": "empty",
            "scale_x": 1.0,
            "scale_y": 1.0,
        }
    min_x = min(polygon.bounds[0] for polygon in polygons)
    min_y = min(polygon.bounds[1] for polygon in polygons)
    max_x = max(polygon.bounds[2] for polygon in polygons)
    max_y = max(polygon.bounds[3] for polygon in polygons)
    width = max_x - min_x
    height = max_y - min_y
    center = (0.5 * (min_x + max_x), 0.5 * (min_y + max_y))
    if width <= canvas_size and height <= canvas_size:
        return {
            "canvas_fit": "none",
            "scale_x": 1.0,
            "scale_y": 1.0,
            "bounds_before": (
                float(width),
                float(height),
            ),
            "bounds_after": (
                float(width),
                float(height),
            ),
        }

    candidates = []
    if preserve_area:
        if height > canvas_size:
            scale_y = canvas_size / height
            scale_x = 1.0 / scale_y
            candidates.append(
                (scale_x, scale_y, "height_preserve_area")
            )
        if width > canvas_size:
            scale_x = canvas_size / width
            scale_y = 1.0 / scale_x
            candidates.append(
                (scale_x, scale_y, "width_preserve_area")
            )
    valid = [
        candidate
        for candidate in candidates
        if width * candidate[0] <= canvas_size + 1e-9
        and height * candidate[1] <= canvas_size + 1e-9
    ]
    if valid:
        scale_x, scale_y, mode = min(
            valid,
            key=lambda candidate: max(
                candidate[0],
                candidate[1],
            ),
        )
    else:
        uniform = min(
            canvas_size / width,
            canvas_size / height,
        )
        scale_x = uniform
        scale_y = uniform
        mode = "uniform_fit"

    for record in records:
        record.polygon = affinity.scale(
            record.polygon,
            xfact=scale_x,
            yfact=scale_y,
            origin=center,
        )
        if getattr(record, "source_polygon", None) is not None:
            record.source_polygon = affinity.scale(
                record.source_polygon,
                xfact=scale_x,
                yfact=scale_y,
                origin=center,
            )
    return {
        "canvas_fit": mode,
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        "bounds_before": (
            float(width),
            float(height),
        ),
        "bounds_after": (
            float(width * scale_x),
            float(height * scale_y),
        ),
    }


def resolve_occupancy_group(
    room_count: int,
    requested: str = "auto",
) -> str:
    if requested != "auto":
        if requested not in OCCUPANCY_RANGES:
            raise ValueError(
                "size must be one of small, medium, large, auto"
            )
        return requested
    # Dataset prompt convention: small/large are explicit, while
    # the no-size headline group is medium.
    return "medium"


def restore_canvas_geometry(
    records,
    aspect_stats,
    room_count: int,
    rng: np.random.Generator,
    size: str = "auto",
):
    aspect_info = restore_canvas_aspect(
        records,
        aspect_stats,
        sample_target=True,
        rng=rng,
    )
    group = resolve_occupancy_group(room_count, size)
    area_info = restore_canvas_area(
        records,
        group,
        rng,
    )
    fit_info = fit_records_to_canvas(records)
    canvas_info = place_records_on_canvas(records)
    return {
        "aspect": aspect_info,
        "area": area_info,
        "fit": fit_info,
        "canvas": canvas_info,
    }
