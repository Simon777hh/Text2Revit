"""Reconstruct room polygons and snap shared boundaries."""
import numpy as np
try:
    from . import build_skeleton_plan0 as bs
except ImportError:
    import build_skeleton_plan0 as bs
SKIP_MERGE_PARALLEL = False
SKIP_MERGE_DUP = False
SKIP_SNAP_SHARED = False

def reconstruct_plan(plan, features, room_order):
    kitchen = bs.unary_union(bs.polys(plan, "kitchen"))
    mask, bounds = bs.raster_shell(plan)
    skel = bs.skeletonize(mask > 0)
    chains = bs.trace_chains(skel)
    lines = bs.pixel_to_world(chains, bounds)
    lines = [l for l in lines if l.length > 1.5]
    lines = bs.straighten_walls(lines)
    nodes, edges = bs.build_graph(lines)
    lines2, _, _, _ = bs.close_kitchen_no_door(plan, lines, nodes, edges, kitchen,
                                               features=features)
    pre_n, pre_e = bs.build_graph(lines2, tol=1.5)
    pre_e = bs.prune_dangling_edges(pre_n, pre_e)
    pre_e = bs.close_corner_gaps(pre_n, pre_e)
    pre_faces = bs.planar_faces(pre_n, pre_e)
    lines2 = bs.add_door_centerline_walls(plan, features, room_order, lines2, pre_faces)
    d_n, d_e = bs.build_graph(lines2, tol=1.5)
    d_e = bs.prune_dangling_edges(d_n, d_e)
    d_faces = bs.planar_faces(d_n, d_e)
    unmatched = bs.compute_unmatched(plan, features, room_order, d_faces)
    lines2 = bs.add_missing_room_edges(plan, features, room_order, lines2,
                                       only_rooms=unmatched)
    n2, e2 = bs.build_graph(lines2, tol=1.5)
    e2 = bs.prune_dangling_edges(n2, e2)
    e2 = bs.close_corner_gaps(n2, e2)
    if not SKIP_MERGE_PARALLEL:
        n2, e2 = bs.merge_parallel_walls(n2, e2, tol=2.0)
    e2 = bs.prune_dangling_edges(n2, e2)
    faces = [f for f in bs.planar_faces(n2, e2) if f[0] > 20.0]
    labeled = bs.label_faces(faces, plan)
    big = max(faces, key=lambda f: f[0])[1]
    kept = []
    for area, poly, rk, ratio in sorted(labeled, key=lambda x: -x[0]):
        if poly.equals(big):
            continue
        if rk and rk != "living" and ratio >= 0.7:
            clean = bs.simplify_face_poly(poly)
            if clean is None:
                clean = poly
            snapped = bs.snap_room_to_graph(clean, n2, e2, tol=0.65)
            if snapped is not None:
                bad = False
                for _, op, ork, _ in kept:
                    if op.intersection(snapped).area > 0.5:
                        bad = True
                        break
                if not bad:
                    clean = snapped
            kept.append((clean.area, clean, rk, ratio))
    candidates = [(a, p, rk) for a, p, rk, _ in labeled
                  if rk and not p.equals(big)]
    kept = bs.add_missing_rooms(plan, features, room_order, kept, lines, candidates,
                                snap_nodes=n2, snap_edges=e2)
    if not SKIP_MERGE_DUP:
        kept = bs.merge_duplicate_room_faces(plan, features, room_order, kept)
    occ = bs.unary_union([f[1] for f in kept if f[2] != "living"])
    rest = big.difference(occ)
    parts = list(rest.geoms) if rest.geom_type == "MultiPolygon" else [rest]
    living = []
    for p in parts:
        if p.geom_type == "Polygon" and p.area > 1.0:
            living.append(bs.clean_living_poly(p))
    if living:
        biggest = max(p.area for p in living)
        living = [p for p in living if p.area >= max(20.0, biggest * 0.02)]
    living_corner = None
    living_rings = []
    if features is not None and room_order is not None:
        corner_rooms = bs.room_rings_from_features(features, room_order, plan)
        living_corner = corner_rooms.get("living_0")
        living_rings = [p for name, p in corner_rooms.items()
                        if name.rsplit("_", 1)[0] == "living"]
    living_area = sum(p.area for p in living)
    ring_area = living_corner.area if living_corner is not None else 0.0
    if (len(living) == 0 or living_area < 0.3 * ring_area) and \
            living_corner is not None and len(kept) > 0:
        occupied = bs.unary_union([f[1] for f in kept if f[2] != "living"])
        wd = float(plan.get("wall_depth") or 4.0)
        snapped = bs.snap_ring_to_walls(living_corner, lines, wd)
        base = snapped if snapped is not None else living_corner
        rest = base.difference(occupied)
        parts = list(rest.geoms) if rest.geom_type == "MultiPolygon" else [rest]
        living = []
        for p in parts:
            if p.geom_type != "Polygon" or p.area <= 1.0:
                continue
            living.append(bs.clean_living_poly(p))
        if len(living) > 0:
            biggest = max(p.area for p in living)
            living = [p for p in living if p.area >= max(50.0, biggest * 0.02)]
    if living_rings and len(living) > 0:
        rings_union = bs.unary_union(living_rings)
        living = [p for p in living
                  if p.intersection(rings_union).area >= 0.3 * p.area]
    if living:
        room_polys = [f[1] for f in kept if f[2] != "living"]
        living = [bs.snap_living_to_rooms(p, room_polys,
                                          float(plan.get("wall_depth") or 4.0)) for p in living]
        # living is leftover space; cut any part that still covers a room
        occupied = bs.unary_union(room_polys)
        new_living = []
        for p in living:
            ov = p.intersection(occupied)
            if ov.is_empty or ov.area < 1.0:
                new_living.append(p)
                continue
            diff = p.difference(ov.buffer(0.01))
            if diff.geom_type == "MultiPolygon":
                diff = max(diff.geoms, key=lambda g: g.area)
            if diff.geom_type == "Polygon" and not diff.is_empty and diff.area > 10.0:
                new_living.append(diff)
        living = new_living
    if living and not SKIP_SNAP_SHARED:
        all_polys = [f[1] for f in kept if f[2] != "living"] + living
        snapped = bs.snap_shared_edges(all_polys, tol=3.0)
        snapped = bs.resolve_thin_overlaps(snapped, tol=1.0)
        n_rooms = len([f for f in kept if f[2] != "living"])
        room_polys = snapped[:n_rooms]
        living = [p for p in snapped[n_rooms:] if not p.is_empty and p.area > 1.0]
        occupied = bs.unary_union(room_polys)
        new_living = []
        for p in living:
            ov = p.intersection(occupied)
            if ov.is_empty or ov.area < 1.0 or ov.area > 500.0:
                new_living.append(p)
                continue
            diff = p.difference(ov.buffer(0.01))
            if diff.geom_type == "MultiPolygon":
                diff = max(diff.geoms, key=lambda g: g.area)
            if diff.geom_type == "Polygon" and not diff.is_empty and diff.area > 10.0:
                new_living.append(diff)
        living = new_living
        kept = [(p.area, p, rk, ratio) for (_, p, rk, ratio), p in
                zip([f for f in kept if f[2] != "living"], room_polys)]

    rebuilt = dict(plan)
    for key in bs.ROOM_KEYS:
        polygons = living if key == "living" else [p for _, p, kind, _ in kept if kind == key]
        if key == "balcony" and not polygons:
            polygons = bs.polys(plan, key)
        rebuilt[key] = bs.unary_union(polygons)
    rebuilt["new_wall_lines"] = bs.unary_union(lines2)
    # Keep the physical area metadata and connector geometry from the source.
    # Rebuild topology from the resulting polygons in the next extraction stage.
    rebuilt.pop("conn", None)
    return rebuilt
