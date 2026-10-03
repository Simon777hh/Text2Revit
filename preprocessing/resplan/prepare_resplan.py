"""Prepare ResPlan raw data as one filtered 0-255 canvas dataset."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter

from shapely import affinity
from shapely.errors import GEOSException
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from resplan_utils import get_geometries, plan_to_graph

ROOM_TYPES = {"living", "kitchen", "bedroom", "bathroom", "balcony", "storage", "stair", "front_door"}


def geometry_parts(value):
    for item in get_geometries(value):
        if hasattr(item, "geoms"):
            yield from geometry_parts(item)
        else:
            yield item


def clean_reason(plan, max_corners=256):
    kitchens = sum(g.geom_type == "Polygon" and g.is_valid and g.area > 1.0
                   for g in geometry_parts(plan.get("kitchen")))
    if kitchens >= 2:
        return "multiple_kitchens"
    corners = 0
    for key in ROOM_TYPES | {"door", "window"}:
        for geom in geometry_parts(plan.get(key)):
            if geom.is_empty or not geom.is_valid:
                continue
            if geom.geom_type == "Polygon" and geom.area > 1.0:
                corners += len(geom.exterior.coords) - 1
            elif geom.geom_type == "LineString":
                corners += len(geom.coords)
    return "corner_budget" if corners > max_corners else None


def refresh_topology(plan):
    graph = plan_to_graph(plan)
    names = sorted(n for n in graph.nodes if n.rsplit("_", 1)[0] in ROOM_TYPES)
    indices = {name: i for i, name in enumerate(names)}
    edges = sorted((min(indices[a], indices[b]), max(indices[a], indices[b]),
                    attributes.get("type", "adjacency"))
                   for a, b, attributes in graph.edges(data=True)
                   if a in indices and b in indices)
    plan["conn"] = {"node_names": names,
                    "room_types": [n.rsplit("_", 1)[0] for n in names],
                    "node_areas": [float(graph.nodes[n].get("area", 0.0)) for n in names],
                    "edge_list": [[a, b] for a, b, _ in edges],
                    "edge_type_names": [kind for _, _, kind in edges]}
    return plan


def filter_reason(plan):
    from preprocessing.features.extract_raw_data import extract_corner_features

    conn = plan.get("conn") or {}
    types = conn.get("room_types") or []
    if types.count("living") != 1 or types.count("kitchen") != 1:
        return "living_or_kitchen_count"
    if "bedroom" not in types or "front_door" not in types:
        return "missing_bedroom_or_entrance"
    if len(types) > 20:
        return "room_budget"
    for kind in set(types):
        for geom in geometry_parts(plan.get(kind)):
            if geom.is_empty or not geom.is_valid:
                return "invalid_geometry"
            if geom.geom_type == "Polygon" and len(geom.exterior.coords) - 1 > 28:
                return "room_corner_budget"
    try:
        features = extract_corner_features(plan)
    except ValueError:
        return "corner_budget"
    if features is None or features["num_corners"] > 128:
        return "corner_budget"
    return None



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
        return affinity.affine_transform(value, matrix)
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


def fit_plan(plan, canvas=255.0):
    """Fit owned geometry uniformly; keep physical m² metadata unchanged."""
    if not math.isfinite(canvas) or canvas <= 0:
        raise ValueError("Canvas must be finite and positive")
    bounds = plan_bounds(plan)
    if not math.isfinite(bounds["min_x"]):
        raise ValueError("Plan has no finite, nondegenerate canvas bounds")

    width = bounds["max_x"] - bounds["min_x"]
    height = bounds["max_y"] - bounds["min_y"]
    span = max(width, height)
    if span <= 1e-9:
        raise ValueError("Plan has no finite, nondegenerate canvas bounds")

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
    new_plan["canvas_transform"] = {
        "scale": scale, "offset_x": offset_x, "offset_y": offset_y,
        "canvas": canvas,
    }
    return new_plan



def source_coordinates(plan):
    """Recover reconstruction units for GT extraction without a second dataset.

    Historical files without transform metadata retain their existing units.
    Canvas geometry remains the default representation in the saved dataset.
    """
    transform = plan.get("canvas_transform")
    if transform is None:
        return plan
    scale = float(transform["scale"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid canvas transform scale")
    inverse = [1 / scale, 0, 0, 1 / scale,
               -transform["offset_x"] / scale, -transform["offset_y"] / scale]
    restored = {}
    for key, value in plan.items():
        if key == "canvas_transform":
            continue
        if key == "conn":
            value = transform_value(value, inverse)
            if value.get("node_areas") is not None:
                value["node_areas"] = [float(area) / scale**2
                                       for area in value["node_areas"]]
        elif key in PIXEL_AREA_KEYS and value is not None:
            value = float(value) / scale**2
        elif key == "wall_depth" and value is not None:
            value = float(value) / scale
        else:
            value = transform_value(value, inverse)
        restored[key] = value
    return restored


def prepare_plan(plan, max_corners=256, canvas=255.0):
    """Clean, reconstruct, refresh topology, filter and fit one raw plan."""
    from preprocessing.resplan.corner_features import process_plan
    from preprocessing.resplan.wall_reconstruction import reconstruct_plan

    stage = "clean"
    try:
        plan = dict(plan)
        reason = clean_reason(plan, max_corners)
        if reason:
            return None, {"stage": stage, "status": "removed", "reason": reason}
        stage = "reconstruct"
        features = process_plan(plan)
        if features is None:
            raise ValueError("No usable corner features")
        plan = reconstruct_plan(plan, features["features"], features["room_order"])
        refresh_topology(plan)
        stage = "filter"
        reason = filter_reason(plan)
        if reason:
            return None, {"stage": stage, "status": "removed", "reason": reason}
        stage = "canvas"
        plan = fit_plan(plan, canvas)
        return plan, {"stage": stage, "status": "kept", "reason": None}
    except (ValueError, RuntimeError, IndexError, TypeError, GEOSException) as error:
        return None, {"stage": stage, "status": "failed", "reason": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source_dir = ROOT / "data" / "sources" / "ResPlan"
    parser.add_argument("--input", type=Path, default=source_dir / "ResPlan.pkl")
    parser.add_argument("--output", type=Path,
                        default=source_dir / "ResPlan_filtered_canvas.pkl")
    parser.add_argument("--max-corners", type=int, default=256,
                        help="Raw-plan corner budget before reconstruction")
    parser.add_argument("--canvas", type=float, default=255.0)
    parser.add_argument("--limit", type=int, default=0, help="0 processes all remaining plans")
    parser.add_argument("--start", type=int, default=0, help="Starting raw source index")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must be different files")
    if args.limit < 0 or args.start < 0 or args.max_corners <= 0:
        parser.error("Invalid limit, start, or corner budget")
    if not math.isfinite(args.canvas) or args.canvas <= 0:
        parser.error("Canvas must be finite and positive")
    report_path = args.output.with_suffix(".report.json")
    if not args.overwrite and (args.output.exists() or report_path.exists()):
        parser.error("Output or report already exists; use --overwrite to replace it")
    with args.input.open("rb") as handle:
        plans = pickle.load(handle)
    if not isinstance(plans, (list, tuple)) or any(not isinstance(p, dict) for p in plans):
        parser.error("Input must contain a list of plan dictionaries")
    if args.start >= len(plans) and (plans or args.start):
        parser.error("Start index is outside the input dataset")
    stop = min(len(plans), args.start + args.limit) if args.limit else len(plans)
    kept, rows = [], []
    for index in range(args.start, stop):
        plan, row = prepare_plan(plans[index], args.max_corners, args.canvas)
        row.update(source_index=index, id=plans[index].get("id"))
        if plan is not None:
            kept.append(plan)
        rows.append(row)
        print(f"[{index + 1}/{stop}] {row['status']} ({row['stage']})", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial")
    with temporary.open("wb") as handle:
        pickle.dump(kept, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(args.output)
    report = {"input": str(args.input), "output": str(args.output),
              "canvas": args.canvas, "processed": len(rows),
              "counts": dict(Counter(row["status"] for row in rows)), "plans": rows}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Kept {len(kept)} of {len(rows)} plans: {args.output}")


if __name__ == "__main__":
    main()
