"""Scale RPLAN plans to ResPlan's coordinate-area size.

Scales every geometry about each plan's own room-bbox center so that
corner movement magnitudes become comparable across the two datasets.
Writes a new pickle and leaves the original untouched.
"""

import sys as _sys
from pathlib import Path as _Path
_RELEASE_ROOT = _Path(__file__).resolve().parents[2]
if str(_RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_RELEASE_ROOT))


import argparse
import pickle
from pathlib import Path

import numpy as np
from shapely import affinity
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from tqdm import tqdm


ROOT = _RELEASE_ROOT / "data" / "sources"
ROOM_TYPES = [
    "living",
    "bedroom",
    "bathroom",
    "kitchen",
    "balcony",
    "storage",
]
GEOMETRY_KEYS = [
    "living",
    "bedroom",
    "bathroom",
    "kitchen",
    "balcony",
    "storage",
    "stair",
    "front_door",
    "door",
    "window",
    "wall",
    "inner",
]


def polygon_list(geometry):
    if geometry is None:
        return []
    if isinstance(geometry, list):
        return geometry
    if hasattr(geometry, "geoms"):
        return list(geometry.geoms)
    return [geometry]


def plan_area(plan):
    total = 0.0
    for node_name, room_type in zip(
        plan["conn"]["node_names"],
        plan["conn"]["room_types"],
    ):
        if room_type not in ROOM_TYPES:
            continue
        polygons = polygon_list(plan.get(room_type))
        index = int(node_name.rsplit("_", 1)[-1])
        if index < len(polygons):
            total += float(polygons[index].area)
    return total


def plan_room_center(plan):
    bounds = []
    for room_type in ROOM_TYPES:
        for polygon in polygon_list(plan.get(room_type)):
            if polygon.is_empty:
                continue
            bounds.append(polygon.bounds)
    if not bounds:
        return 128.0, 128.0
    bounds = np.asarray(bounds, dtype=float)
    min_x = bounds[:, 0].min()
    min_y = bounds[:, 1].min()
    max_x = bounds[:, 2].max()
    max_y = bounds[:, 3].max()
    return (min_x + max_x) / 2.0, (min_y + max_y) / 2.0


def scale_geometry(geom, linear, center):
    if geom is None:
        return None
    return affinity.scale(
        geom,
        xfact=linear,
        yfact=linear,
        origin=center,
    )


def scale_geometry_field(value, linear, center):
    if value is None:
        return None
    if isinstance(value, list):
        return [
            scale_geometry(item, linear, center)
            for item in value
        ]
    return scale_geometry(value, linear, center)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rplan",
        default=str(ROOT / "RPLAN" / "rplan_resplan_style_cleaned.pkl"),
    )
    parser.add_argument(
        "--resplan",
        default=str(ROOT / "ResPlan" / "ResPlan_filtered.pkl"),
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "RPLAN" / "rplan_resplan_style_scaled.pkl"),
    )
    parser.add_argument(
        "--scale-area",
        type=float,
        default=None,
        help="Override the area scale factor. Defaults to median ratio.",
    )
    args = parser.parse_args()

    rplan_plans = pickle.load(open(args.rplan, "rb"))
    resplan_plans = pickle.load(open(args.resplan, "rb"))
    rplan_areas = np.asarray(
        [plan_area(plan) for plan in rplan_plans], dtype=float
    )
    resplan_areas = np.asarray(
        [plan_area(plan) for plan in resplan_plans], dtype=float
    )

    area_factor = args.scale_area
    if area_factor is None:
        area_factor = float(
            np.median(resplan_areas) / np.median(rplan_areas)
        )
    linear = float(np.sqrt(area_factor))
    print(
        f"area factor {area_factor:.4f}  "
        f"linear factor {linear:.4f}"
    )
    print(
        f"before: RPLAN median {np.median(rplan_areas):.1f}  "
        f"ResPlan median {np.median(resplan_areas):.1f}"
    )

    scaled = []
    for plan in tqdm(rplan_plans, desc="scaling RPLAN"):
        center = plan_room_center(plan)
        new_plan = dict(plan)
        for key in GEOMETRY_KEYS:
            if key in new_plan:
                new_plan[key] = scale_geometry_field(
                    new_plan[key], linear, center
                )

        conn = dict(new_plan.get("conn", {}))
        if conn.get("node_areas") is not None:
            conn["node_areas"] = [
                float(area) * area_factor
                for area in conn["node_areas"]
            ]
        new_plan["conn"] = conn

        for key in [
            "net_area",
            "area",
            "rooms_area",
            "net_area_rebuilt",
            "balcony_area",
            "living_area",
        ]:
            value = new_plan.get(key)
            if value is not None:
                new_plan[key] = float(value) * area_factor

        scaled.append(new_plan)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as handle:
        pickle.dump(scaled, handle)

    scaled_areas = np.asarray(
        [plan_area(plan) for plan in scaled], dtype=float
    )
    print(
        f"after: RPLAN median {np.median(scaled_areas):.1f}  "
        f"ResPlan median {np.median(resplan_areas):.1f}  "
        f"ratio {np.median(resplan_areas) / np.median(scaled_areas):.3f}"
    )
    print("saved", args.out)


if __name__ == "__main__":
    main()
