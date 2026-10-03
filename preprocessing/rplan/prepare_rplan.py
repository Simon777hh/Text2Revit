"""Convert and filter RPLAN into one dataset with recorded CLIP area scaling."""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from shapely.errors import GEOSException
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from resplan_utils import get_geometries

AREA_ROOM_TYPES = {"living", "bedroom", "bathroom", "kitchen", "balcony", "storage"}


def room_polygons(value):
    # RPLAN stores repeated room types as lists; ResPlan uses multi-geometries.
    return list(value) if isinstance(value, (list, tuple)) else get_geometries(value)


def polygons_for_plan(plan):
    rooms = []
    conn = plan["conn"]
    for name, kind in zip(conn["node_names"], conn["room_types"]):
        if kind == "front_door":
            continue
        polygons = room_polygons(plan.get(kind))
        index = int(name.rsplit("_", 1)[-1])
        if index >= len(polygons):
            raise ValueError(f"Missing polygon for {name}")
        rooms.append((name, kind, polygons[index]))
    return rooms


def has_holes(poly):
    parts = poly.geoms if hasattr(poly, "geoms") else [poly]
    return any(len(part.interiors) > 0 for part in parts)


def has_short_edge(poly, min_len=2.0):
    parts = poly.geoms if hasattr(poly, "geoms") else [poly]
    for part in parts:
        coords = np.asarray(part.exterior.coords[:-1])
        lengths = np.abs(coords - np.roll(coords, -1, axis=0)).max(axis=1)
        if np.any((lengths < min_len) & (lengths > 0.01)):
            return True
    return False


def filter_reasons(plan):
    """Retain the existing connectivity, entrance and geometry filter rules."""
    conn = plan["conn"]
    names, types = conn["node_names"], conn["room_types"]
    edges, edge_types = conn["edge_list"], conn["edge_type_names"]
    if len(names) != len(types) or len(edges) != len(edge_types):
        raise ValueError("Inconsistent topology fields")
    if any(len(edge) != 2 or min(edge) < 0 or max(edge) >= len(names) for edge in edges):
        raise ValueError("Topology edge index outside room range")
    rooms = polygons_for_plan(plan)
    polygons = [polygon for _, _, polygon in rooms]
    by_name = {name: polygon for name, _, polygon in rooms}
    if not polygons or any(p.is_empty or not p.is_valid or p.geom_type not in
                           ("Polygon", "MultiPolygon") for p in polygons):
        return ["invalid_room_geometry"]
    reasons = []
    connected = {node for edge, kind in zip(edges, edge_types)
                 if kind in ("via_door", "via_opening", "direct") for node in edge}
    if any(i not in connected for i, kind in enumerate(types) if kind != "front_door"):
        reasons.append("disconnected_room")
    if types.count("living") > 1:
        reasons.append("multiple_living_rooms")
    if "front_door_0" in names:
        front = names.index("front_door_0")
        entrance_edges = [(b if a == front else a, kind)
                          for (a, b), kind in zip(edges, edge_types) if front in (a, b)]
        if (any(kind != "direct" for _, kind in entrance_edges) or
                not any(kind == "direct" and names[other].startswith("living")
                        for other, kind in entrance_edges)):
            reasons.append("entrance_connection")
    if any(a.intersection(b).area > 0.5 for i, a in enumerate(polygons)
           for b in polygons[i + 1:]):
        reasons.append("overlapping_rooms")
    if any(has_holes(p) for p in polygons):
        reasons.append("room_holes")
    if any(has_short_edge(p) for p in polygons):
        reasons.append("short_room_edge")
    inner = plan.get("inner")
    if inner is not None and not inner.is_empty:
        union = unary_union([p.buffer(0) for p in polygons])
        if inner.buffer(0).difference(union.buffer(0)).area > 0.5:
            reasons.append("uncovered_inner_area")
    for (a, b), kind in zip(edges, edge_types):
        if kind != "via_door":
            continue
        first, second = by_name.get(names[a]), by_name.get(names[b])
        if first is not None and second is not None:
            if first.boundary.intersection(second.boundary).length < 1.0:
                reasons.append("door_without_shared_wall")
                break
    return reasons


def plan_area(plan):
    conn = plan["conn"]
    total = 0.0
    for name, kind in zip(conn["node_names"], conn["room_types"]):
        if kind in AREA_ROOM_TYPES:
            polygons = room_polygons(plan.get(kind))
            index = int(name.rsplit("_", 1)[-1])
            if index < len(polygons):
                total += float(polygons[index].area)
    return total


def area_scale(plans, references):
    """Match source coordinate-area medians, as in the former scaling stage."""
    from preprocessing.resplan.prepare_resplan import source_coordinates
    areas = np.asarray([plan_area(p) for p in plans])
    reference_areas = np.asarray([plan_area(source_coordinates(p)) for p in references])
    if (not len(areas) or not len(reference_areas) or
            not np.isfinite(areas).all() or not np.isfinite(reference_areas).all() or
            np.any(areas <= 0) or np.any(reference_areas <= 0)):
        raise ValueError("Area calibration requires nonempty, positive finite room areas")
    scale = float(np.median(reference_areas) / np.median(areas))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Calibrated area multiplier is not finite and positive")
    return scale


def clip_area_plan(plan):
    """Expose scaled node areas for CLIP labels without a second geometry file."""
    scale = float(plan.get("clip_area_scale", 1.0))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("CLIP area scale must be finite and positive")
    result = dict(plan)
    conn = dict(plan["conn"])
    conn["node_areas"] = [float(area) * scale for area in conn["node_areas"]]
    result["conn"] = conn
    return result


def _convert_item(item):
    index, path, id_offset = item
    from preprocessing.rplan.image_conversion import plan_from_image
    row = {"source_index": index, "filename": path.name, "stage": "convert"}
    try:
        plan = plan_from_image(path, id_offset=id_offset)
        if plan is None:
            row.update(status="removed", reasons=["no_usable_rooms"], id=None)
        else:
            row.update(id=plan.get("id"))
        return plan, row
    except (ValueError, RuntimeError, IndexError, TypeError, OSError, GEOSException) as error:
        row.update(status="failed", reasons=[str(error)], id=None)
        return None, row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source_dir = ROOT / "data" / "sources"
    parser.add_argument("--input", type=Path,
                        default=source_dir / "RPLAN" / "dataset" / "floorplan_dataset",
                        help="Raw PNG folder, or an existing converted source pickle")
    parser.add_argument("--output", type=Path,
                        default=source_dir / "RPLAN" / "RPLAN_filtered.pkl")
    parser.add_argument("--resplan", type=Path,
                        default=source_dir / "ResPlan" / "ResPlan_filtered_canvas.pkl")
    parser.add_argument("--scale-area", type=float, help="Override the CLIP area multiplier")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--id-offset", type=int, default=200000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.start < 0 or args.limit < 0 or args.workers < 1:
        parser.error("Start/limit must be nonnegative and workers must be positive")
    if args.scale_area is not None and (not math.isfinite(args.scale_area) or args.scale_area <= 0):
        parser.error("Area multiplier must be finite and positive")
    report_path = args.output.with_suffix(".report.json")
    if args.output.resolve() in (args.input.resolve(), args.resplan.resolve()):
        parser.error("Output must differ from the source and reference files")
    if not args.overwrite and (args.output.exists() or report_path.exists()):
        parser.error("Output or report exists; use --overwrite to replace it")
    if args.scale_area is None and not args.resplan.is_file():
        parser.error("Prepare ResPlan first, or provide an explicit --scale-area")
    if args.input.is_dir():
        sources = sorted(args.input.glob("*.png"))
        if not sources:
            parser.error("Input folder contains no PNG files")
        is_png = True
    elif args.input.is_file():
        with args.input.open("rb") as handle:
            sources = pickle.load(handle)
        if not isinstance(sources, (list, tuple)) or any(not isinstance(p, dict) for p in sources):
            parser.error("Pickle must contain plan dictionaries")
        is_png = False
    else:
        parser.error("Input does not exist")
    if args.start >= len(sources):
        parser.error("Start index is outside the input dataset")
    stop = min(len(sources), args.start + args.limit) if args.limit else len(sources)
    if is_png:
        items = [(i, sources[i], args.id_offset) for i in range(args.start, stop)]
        if args.workers > 1:
            from multiprocessing import Pool
            with Pool(args.workers) as pool:
                converted = list(pool.imap(_convert_item, items, chunksize=16))
        else:
            converted = []
            for item in items:
                converted.append(_convert_item(item))
                if len(converted) % 100 == 0:
                    print(f"Converted {len(converted)}/{len(items)}", flush=True)
    else:
        converted = [(dict(sources[i]), {"source_index": i, "id": sources[i].get("id")})
                     for i in range(args.start, stop)]
    kept, rows, bad_ids = [], [], set()
    for plan, row in converted:
        if plan is not None:
            row["stage"] = "filter"
            try:
                row["reasons"] = filter_reasons(plan)
                row["status"] = "removed" if row["reasons"] else "kept"
            except (ValueError, RuntimeError, IndexError, KeyError, TypeError, GEOSException) as error:
                row.update(status="failed", reasons=[str(error)])
            if row["status"] != "kept":
                bad_ids.add(plan.get("id"))
        rows.append(row)
    # The former filter removed all records belonging to a rejected source ID.
    for (plan, _), row in zip(converted, rows):
        if row["status"] == "kept":
            if plan.get("id") in bad_ids:
                row.update(status="removed", reasons=["rejected_duplicate_id"])
            else:
                kept.append(plan)
    report = {"input": str(args.input), "output": str(args.output),
              "processed": len(rows), "counts": dict(Counter(r["status"] for r in rows)),
              "clip_area_scale": None, "plans": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not kept:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise SystemExit("No plans survived; see the report. No dataset was written.")
    if args.scale_area is None:
        with args.resplan.open("rb") as handle:
            references = pickle.load(handle)
        scale = area_scale(kept, references)
    else:
        scale = args.scale_area
    for plan in kept:
        plan["clip_area_scale"] = scale
    report["clip_area_scale"] = scale
    report["reference"] = str(args.resplan) if args.scale_area is None else None
    temporary = args.output.with_suffix(".partial")
    with temporary.open("wb") as handle:
        pickle.dump(kept, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(args.output)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Kept {len(kept)} of {len(rows)}; CLIP area multiplier={scale:.6f}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
