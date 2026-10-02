"""Verify opening policy on the JSON consumed by the C# add-in."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import unary_union


def verify(plan):
    rooms = {r["room_id"]: r for r in plan["rooms"]}
    windows = [o for o in plan["openings"] if o["type"] == "window"]
    entrance = next(o for o in plan["openings"]
                    if o["type"] != "window" and o["host"] == "exterior_wall")
    entrance_line = LineString(entrance["reference_line"])
    living_id = entrance["owner_room_id"]
    envelope = unary_union([Polygon(r["polygon"]) for r in rooms.values()
                            if r["type"] not in {"front_door", "door"}])
    polygons = list(envelope.geoms) if envelope.geom_type == "MultiPolygon" else [envelope]
    polygon = min(polygons, key=lambda p: p.exterior.distance(entrance_line.centroid))
    coords = list(polygon.exterior.coords)
    segments = [LineString([a, b]) for a, b in zip(coords[:-1], coords[1:])]
    axis = np.asarray(entrance["reference_line"][1]) - np.asarray(entrance["reference_line"][0])
    axis /= np.linalg.norm(axis)
    normal = np.array([-axis[1], axis[0]])
    anchor = np.array(entrance_line.centroid.coords[0])
    # Collect the connected collinear wall containing the entrance, independently
    # of the placement helper, so split polygon segments cannot hide violations.
    aligned = [s for s in segments if all(abs(np.dot(np.asarray(p) - anchor, normal)) < 1e-4
                                        for p in s.coords)]
    connected = min(aligned, key=lambda s: s.distance(entrance_line.centroid))
    remaining = [s for s in aligned if s is not connected]
    while True:
        touching = [s for s in remaining if s.distance(connected) < 1e-4]
        if not touching:
            break
        for segment in touching:
            connected = connected.union(segment)
            remaining.remove(segment)
    living_windows = [o for o in windows if o["owner_room_id"] == living_id]
    for opening in windows:
        center = LineString(opening["reference_line"]).centroid
        assert connected.distance(center) > 1e-4, "window is on the continuous entrance facade"
    bedrooms_with_balcony = set()
    for a, b in plan["debug"]["edge_lists"]:
        if a not in rooms or b not in rooms:
            continue
        if {rooms[a]["type"], rooms[b]["type"]} == {"bedroom", "balcony"}:
            bedrooms_with_balcony.add(a if rooms[a]["type"] == "bedroom" else b)
    assert not any(o["owner_room_id"] in bedrooms_with_balcony for o in windows), \
        "bedroom with a balcony has a window"
    from revit_geometry import validate_opening_clearances
    validate_opening_clearances(plan["openings"], plan["pixel_to_meter"])
    return {"seed": plan["seed"], "living_windows": len(living_windows),
            "bedrooms_with_balcony": sorted(bedrooms_with_balcony),
            "window_count": len(windows), "policy_passed": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = verify(json.loads(args.plan.read_text(encoding="utf-8")))
    if args.report:
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
