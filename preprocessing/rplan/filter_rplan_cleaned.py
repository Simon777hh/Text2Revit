"""Save a cleaned RPLAN pickle with all detected problem plans removed."""

import sys as _sys
from pathlib import Path as _Path
_RELEASE_ROOT = _Path(__file__).resolve().parents[2]
if str(_RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_RELEASE_ROOT))


import pickle
from pathlib import Path

import numpy as np
from shapely.ops import unary_union


ROOT = _RELEASE_ROOT / "data" / "sources" / "RPLAN"
SOURCE_PKL = ROOT / "rplan_resplan_style.pkl"
CLEANED_PKL = ROOT / "rplan_resplan_style_cleaned.pkl"


def polygons_for_plan(plan):
    result = []
    for name, room_type in zip(
        plan["conn"]["node_names"],
        plan["conn"]["room_types"],
    ):
        if room_type == "front_door":
            continue
        geometry = plan.get(room_type)
        polygons = geometry if isinstance(geometry, list) else [geometry]
        index = int(name.rsplit("_", 1)[-1])
        if index < len(polygons):
            result.append((name, room_type, polygons[index]))
    return result


def has_holes(poly):
    if poly.geom_type == "Polygon":
        return len(poly.interiors) > 0
    return any(len(part.interiors) > 0 for part in poly.geoms)


def has_short_edge(poly, min_len=2.0):
    parts = poly.geoms if poly.geom_type == "MultiPolygon" else [poly]
    for part in parts:
        coords = np.asarray(part.exterior.coords[:-1])
        for i in range(len(coords)):
            p0 = coords[i]
            p1 = coords[(i + 1) % len(coords)]
            length = max(
                abs(p1[0] - p0[0]),
                abs(p1[1] - p0[1]),
            )
            if min_len > length > 0.01:
                return True
    return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SOURCE_PKL)
    parser.add_argument("--output", type=Path, default=CLEANED_PKL)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must be different files")
    with args.input.open("rb") as handle:
        plans = pickle.load(handle)
    problem_ids = set()
    for plan in plans:
        names = plan["conn"]["node_names"]
        types = plan["conn"]["room_types"]
        edges = plan["conn"]["edge_list"]
        edge_types = plan["conn"]["edge_type_names"]
        rooms = polygons_for_plan(plan)
        polys = [poly for _, _, poly in rooms]
        polygons_by_name = {name: poly for name, _, poly in rooms}
        plan_id = plan["id"]

        connected = set()
        for (a, b), edge_type in zip(edges, edge_types):
            if edge_type in ("via_door", "via_opening", "direct"):
                connected.add(a)
                connected.add(b)
        for index, room_type in enumerate(types):
            if room_type == "front_door":
                continue
            if index not in connected:
                problem_ids.add(plan_id)

        if sum(1 for t in types if t == "living") > 1:
            problem_ids.add(plan_id)

        if "front_door_0" in names:
            front_index = names.index("front_door_0")
            front_direct_living = False
            front_bad = False
            for (a, b), edge_type in zip(edges, edge_types):
                if front_index not in (a, b):
                    continue
                other = b if a == front_index else a
                if edge_type != "direct":
                    front_bad = True
                if edge_type == "direct" and names[other].startswith("living"):
                    front_direct_living = True
            if not front_direct_living or front_bad:
                problem_ids.add(plan_id)

        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                if polys[i].intersection(polys[j]).area > 0.5:
                    problem_ids.add(plan_id)
                    break
            if plan_id in problem_ids:
                break

        if any(has_holes(poly) for poly in polys):
            problem_ids.add(plan_id)

        if any(has_short_edge(poly) for poly in polys):
            problem_ids.add(plan_id)

        union = unary_union([poly.buffer(0) for poly in polys])
        inner = plan.get("inner")
        if inner is not None and not inner.is_empty:
            try:
                if inner.buffer(0).difference(union.buffer(0)).area > 0.5:
                    problem_ids.add(plan_id)
            except Exception:
                problem_ids.add(plan_id)

        for (a, b), edge_type in zip(edges, edge_types):
            if edge_type != "via_door":
                continue
            first = polygons_by_name.get(names[a])
            second = polygons_by_name.get(names[b])
            if first is None or second is None:
                continue
            shared = first.boundary.intersection(second.boundary).length
            if shared < 1.0:
                problem_ids.add(plan_id)

    cleaned = [plan for plan in plans if plan["id"] not in problem_ids]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(cleaned, handle)
    print("source", len(plans))
    print("deleted", len(problem_ids))
    print("cleaned", len(cleaned))
    print("saved", args.output)


if __name__ == "__main__":
    main()
