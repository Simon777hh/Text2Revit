"""Fit ResPlan plan geometry into a 0-255 canvas using each plan's bbox.

ResPlan rooms are normalized around the inner boundary, so rooms and
balconies often fall outside the canvas (negative coordinates or values
above 255). This script computes the bbox over the plan's own geometry
(rooms, doors, windows, walls, inner) and uniformly scales/translates it
so every such coordinate lands inside ``[0, canvas]`` while keeping the
aspect ratio unchanged.

The neighbor plot is transformed with the same matrix when present but is
excluded from the bbox, because it describes a different house.
"""

import sys as _sys
from pathlib import Path as _Path
_RELEASE_ROOT = _Path(__file__).resolve().parents[2]
if str(_RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_RELEASE_ROOT))


import argparse
import math
import pickle
from pathlib import Path

import numpy as np
from shapely import affinity
from tqdm import tqdm


ROOT = _RELEASE_ROOT / "data" / "sources"
GEOMETRY_KEYS = [
    "living",
    "kitchen",
    "bedroom",
    "bathroom",
    "balcony",
    "storage",
    "stair",
    "front_door",
    "door",
    "window",
    "wall",
    "inner",
    "new_wall_lines",
]
PIXEL_AREA_KEYS = [
    "rooms_area",
    "net_area_rebuilt",
    "balcony_area",
    "living_area",
]


def collect_bounds(value, bounds):
    if value is None:
        return
    if hasattr(value, "is_empty") and value.is_empty:
        return
    if hasattr(value, "geom_type"):
        box = value.bounds
        if box and all(math.isfinite(v) for v in box):
            bounds["min_x"] = min(bounds["min_x"], box[0])
            bounds["min_y"] = min(bounds["min_y"], box[1])
            bounds["max_x"] = max(bounds["max_x"], box[2])
            bounds["max_y"] = max(bounds["max_y"], box[3])
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            collect_bounds(item, bounds)
    elif isinstance(value, dict):
        for item in value.values():
            collect_bounds(item, bounds)


def plan_bounds(plan):
    bounds = {
        "min_x": math.inf,
        "min_y": math.inf,
        "max_x": -math.inf,
        "max_y": -math.inf,
    }
    for key in GEOMETRY_KEYS:
        collect_bounds(plan.get(key), bounds)
    return bounds


def transform_value(value, matrix):
    if value is None:
        return None
    if hasattr(value, "geom_type"):
        try:
            return affinity.affine_transform(value, matrix)
        except Exception:
            return value
    if isinstance(value, list):
        return [transform_value(item, matrix) for item in value]
    if isinstance(value, tuple):
        return tuple(transform_value(item, matrix) for item in value)
    if isinstance(value, dict):
        return {
            key: transform_value(item, matrix)
            for key, item in value.items()
        }
    return value


def fit_plan(plan, canvas):
    bounds = plan_bounds(plan)
    if not math.isfinite(bounds["min_x"]):
        return dict(plan)

    width = bounds["max_x"] - bounds["min_x"]
    height = bounds["max_y"] - bounds["min_y"]
    span = max(width, height)
    if span <= 1e-9:
        return dict(plan)

    scale = canvas / span
    center_x = (bounds["min_x"] + bounds["max_x"]) / 2.0
    center_y = (bounds["min_y"] + bounds["max_y"]) / 2.0
    offset_x = canvas / 2.0 - scale * center_x
    offset_y = canvas / 2.0 - scale * center_y
    matrix = [scale, 0.0, 0.0, scale, offset_x, offset_y]

    new_plan = {}
    for key, value in plan.items():
        if key == "conn":
            conn = transform_value(value, matrix)
            if conn.get("node_areas") is not None:
                conn["node_areas"] = [
                    float(area) * scale * scale
                    for area in conn["node_areas"]
                ]
            new_plan[key] = conn
        elif key in PIXEL_AREA_KEYS and value is not None:
            new_plan[key] = float(value) * scale * scale
        elif key == "wall_depth" and value is not None:
            new_plan[key] = float(value) * scale
        else:
            new_plan[key] = transform_value(value, matrix)
    return new_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(ROOT / "ResPlan" / "ResPlan_filtered.pkl"),
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "ResPlan" / "ResPlan_filtered_canvas.pkl"),
    )
    parser.add_argument("--canvas", type=float, default=255.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    with open(args.input, "rb") as handle:
        plans = pickle.load(handle)
    if args.limit > 0:
        plans = plans[: args.limit]

    fitted = []
    scales = []
    for plan in tqdm(plans, desc="fitting ResPlan to canvas"):
        new_plan = fit_plan(plan, args.canvas)
        fitted.append(new_plan)
        bounds = plan_bounds(new_plan)
        span = max(
            bounds["max_x"] - bounds["min_x"],
            bounds["max_y"] - bounds["min_y"],
        )
        scales.append(args.canvas / span if span > 1e-9 else 1.0)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as handle:
        pickle.dump(fitted, handle)

    scales = np.asarray(scales, dtype=float)
    print(f"saved {args.out} plans={len(fitted)}")
    print(
        f"linear scale median={np.median(scales):.6f} "
        f"mean={scales.mean():.6f}"
    )

    bad = 0
    for plan in fitted:
        bounds = plan_bounds(plan)
        if (
            bounds["min_x"] < -1e-6
            or bounds["min_y"] < -1e-6
            or bounds["max_x"] > args.canvas + 1e-6
            or bounds["max_y"] > args.canvas + 1e-6
        ):
            bad += 1
    print(f"plans with geometry outside [0, {args.canvas}]: {bad}")


if __name__ == "__main__":
    main()
