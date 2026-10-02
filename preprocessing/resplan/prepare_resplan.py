"""Clean, reconstruct, or filter ResPlan data without replacing the input."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from resplan_utils import get_geometries, plan_to_graph
from preprocessing.features.extract_raw_data import extract_corner_features

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["clean", "rebuild", "filter"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-corners", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--refresh-topology", action="store_true")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must be different files")
    if args.limit < 0 or args.start < 0 or args.max_corners <= 0:
        parser.error("Invalid limit, start, or corner budget")
    with args.input.open("rb") as handle:
        plans = pickle.load(handle)
    if args.stage == "rebuild":
        from preprocessing.resplan.extract_points import process_plan
        from preprocessing.resplan.reconstruct import reconstruct_plan
    kept, report = [], []
    stop = min(len(plans), args.start + args.limit) if args.limit else len(plans)
    for index in range(args.start, stop):
        plan = dict(plans[index])
        try:
            if args.stage == "clean":
                reason = clean_reason(plan, args.max_corners)
            elif args.stage == "rebuild":
                features = process_plan(plan)
                if features is None:
                    raise ValueError("No usable corner features")
                plan = reconstruct_plan(plan, features["features"], features["room_order"])
                if plan is None:
                    raise ValueError("Wall graph has no closed room faces")
                refresh_topology(plan)
                reason = None
            else:
                if args.refresh_topology or not plan.get("conn"):
                    refresh_topology(plan)
                reason = filter_reason(plan)
            if reason is None:
                kept.append(plan)
            report.append({"source_index": index, "id": plans[index].get("id"),
                           "status": "kept" if reason is None else "removed", "reason": reason})
        except (ValueError, RuntimeError, IndexError, TypeError) as error:
            report.append({"source_index": index, "id": plans[index].get("id"),
                           "status": "failed", "reason": str(error)})
        if (index + 1) % 100 == 0 or args.stage == "rebuild":
            print(f"[{index + 1}/{stop}] {report[-1]['status']}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial")
    with temporary.open("wb") as handle:
        pickle.dump(kept, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(args.output)
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Kept {len(kept)} of {stop - args.start} plans: {args.output}")


if __name__ == "__main__":
    main()
