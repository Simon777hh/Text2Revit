"""Wall centerline reconstruction, face labeling, and shared-edge snapping."""

import sys as _sys
from pathlib import Path as _Path
_RELEASE_ROOT = _Path(__file__).resolve().parents[2]
if str(_RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_RELEASE_ROOT))


import math
import logging
_LOG = logging.getLogger(__name__)
from collections import defaultdict

RASTER_SIZE = 1024


import cv2
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from skimage.morphology import skeletonize


from resplan_utils import get_geometries

ROOM_KEYS = ["living", "kitchen", "bedroom", "bathroom", "balcony", "storage", "stair"]
CONNECTOR_KEYS = ["door", "front_door", "window"]
ROOM_COLORS = {
    "living": "#d9d9d9",
    "kitchen": "#8da0cb",
    "bedroom": "#66c2a5",
    "bathroom": "#fc8d62",
    "balcony": "#b3b3b3",
    "storage": "#a37c52",
    "stair": "#9e9ac8",
}
CONNECTOR_COLORS = {"door": "#e78ac3", "front_door": "#a63603", "window": "#a6d854"}


def polys(plan, key):
    return [g for g in get_geometries(plan.get(key)) if isinstance(g, Polygon) and not g.is_empty]


def shell_polys(plan):
    return [g for k in ["wall"] + CONNECTOR_KEYS for g in polys(plan, k)]


def raster_shell(plan, size=RASTER_SIZE):
    """Rasterize every wall/connector polygon as a filled band."""
    parts = shell_polys(plan)
    xs, ys = [], []
    for g in parts:
        x1, y1, x2, y2 = g.bounds
        xs += [x1, x2]
        ys += [y1, y2]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    mask = np.zeros((size, size), np.uint8)
    for g in parts:
        coords = np.array(g.exterior.coords)
        px = ((coords[:, 0] - x1) / (x2 - x1) * (size - 1)).round().astype(np.int32)
        py = ((coords[:, 1] - y1) / (y2 - y1) * (size - 1)).round().astype(np.int32)
        cv2.fillPoly(mask, [np.stack([px, py], axis=1)], 1)
    return mask, (x1, y1, x2, y2)


def straighten_walls(lines):
    """Replace short skeleton zigzag diagonals with axis-aligned L corners.
    Only the diagonal itself is replaced; no neighbouring wall is extended,
    so room separation in the graph is not affected."""
    raw = [np.asarray(line.coords, dtype=float) for line in lines]
    segs = []
    for c in raw:
        for i in range(len(c) - 1):
            segs.append((c[i], c[i + 1]))
    out = []
    for a, b in segs:
        dx, dy = abs(b[0] - a[0]), abs(b[1] - a[1])
        if min(dx, dy) < 0.2 or max(dx, dy) >= 2.0:
            out.append(np.array([a, b], dtype=float))
            continue
        # snap the diagonal to an L corner: first endpoint keeps its axis
        # coordinate, second endpoint keeps its axis coordinate
        if dx >= dy:
            corner = (b[0], a[1])
            out.append(np.array([a, corner], dtype=float))
            out.append(np.array([corner, b], dtype=float))
        else:
            corner = (a[0], b[1])
            out.append(np.array([a, corner], dtype=float))
            out.append(np.array([corner, b], dtype=float))
    return [LineString(w) for w in out]


def has_living_kitchen_door(plan, features, living, kitchen):
    """
    Door between living and kitchen, judged by the cleaned v2 corner
    features: a door instance has 4 corners and spans both room boundaries.
    """
    inner = plan.get("inner")
    x1, y1, x2, y2 = inner.bounds

    def to_pixel(coords):
        out = np.empty_like(coords)
        out[:, 0] = (coords[:, 0] + 1.0) * 0.5 * (x2 - x1) + x1
        out[:, 1] = (coords[:, 1] + 1.0) * 0.5 * (y2 - y1) + y1
        return out

    groups = defaultdict(list)
    for row in features:
        if abs(row[0]) < 1e-6 and abs(row[1]) < 1e-6:
            continue
        if row[44:].sum() == 0:
            continue
        type_idx = int(np.argmax(row[2:12]))
        if type_idx not in (7, 8):  # door / front_door
            continue
        room_idx = int(np.argmax(row[44:]))
        groups[(room_idx, type_idx)].append((int(np.argmax(row[12:44])),
                                             np.array([row[0], row[1]])))

    tol = (plan.get("wall_depth") or 4.0) * 0.6 + living.distance(kitchen)
    for corners in groups.values():
        if len(corners) < 4:
            continue
        arr = to_pixel(np.array([c[1] for c in corners], dtype=float))
        near_a = sum(living.boundary.distance(Point(c)) < tol for c in arr)
        near_b = sum(kitchen.boundary.distance(Point(c)) < tol for c in arr)
        if near_a >= 2 and near_b >= 2:
            return True
    return False


def trace_chains(skel):
    """Trace 8-connected skeleton pixels into polylines (pixel coords)."""
    nbr = np.zeros_like(skel, dtype=np.int8)
    padded = np.pad(skel, 1)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nbr += padded[1 + dy: padded.shape[0] - 1 + dy, 1 + dx: padded.shape[1] - 1 + dx]
    endpoints = set(zip(*np.nonzero((nbr <= 1) & skel)))
    junctions = set(zip(*np.nonzero((nbr >= 3) & skel)))
    starts = endpoints | junctions

    visited = set()
    chains = []
    for s in starts:
        if s in visited:
            continue
        # each start may have several branches
        for n0 in _neighbors(skel, s):
            if (s, n0) in visited:
                continue
            chain = [s, n0]
            visited.add((s, n0))
            visited.add((n0, s))
            prev, cur = s, n0
            while cur not in starts:
                nxts = [n for n in _neighbors(skel, cur) if n != prev]
                if len(nxts) != 1:
                    break
                nxt = nxts[0]
                chain.append(nxt)
                visited.add((cur, nxt))
                visited.add((nxt, cur))
                prev, cur = cur, nxt
            chains.append(np.array(chain, dtype=float))
    return chains


def _neighbors(skel, p):
    y, x = p
    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            yy, xx = y + dy, x + dx
            if 0 <= yy < skel.shape[0] and 0 <= xx < skel.shape[1] and bool(skel[yy, xx]):
                out.append((yy, xx))
    return out


def pixel_to_world(chains, bounds):
    x1, y1, x2, y2 = bounds
    out = []
    for c in chains:
        wx = x1 + c[:, 1] / (RASTER_SIZE - 1) * (x2 - x1)
        wy = y1 + c[:, 0] / (RASTER_SIZE - 1) * (y2 - y1)
        line = LineString(np.stack([wx, wy], axis=1)).simplify(0.5, preserve_topology=False)
        out.append(line)
    return [l for l in out if l.length > 0.01]


def build_graph(lines, tol=1.0):
    """Snap endpoints, split at junctions, return node->edges and edge list."""
    snap_axis = 0.6
    raw_edges = []
    for line in lines:
        coords = np.array(line.coords)
        for i in range(len(coords) - 1):
            raw_edges.append((coords[i], coords[i + 1]))
    nodes = []

    def get_node(p, create=True):
        for nid, n in enumerate(nodes):
            if np.linalg.norm(n - p) <= tol:
                return nid
        if create:
            nodes.append(np.array(p, dtype=float))
            return len(nodes) - 1
        return None

    edge_pairs = set()
    for a, b in raw_edges:
        u, v = get_node(a), get_node(b)
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key in edge_pairs:
            continue
        edge_pairs.add(key)

    edges = sorted(edge_pairs)
    # split edges that pass through a node placed on their interior
    changed = True
    while changed:
        changed = False
        new_edges = []
        for u, v in edges:
            a, b = nodes[u], nodes[v]
            seg_len = np.linalg.norm(b - a)
            if seg_len < 1e-9:
                continue
            splits = []
            for nid, n in enumerate(nodes):
                if nid in (u, v):
                    continue
                t = np.clip(np.dot(n - a, b - a) / (seg_len * seg_len), 0.0, 1.0)
                proj = a + t * (b - a)
                if np.linalg.norm(n - proj) <= tol * 0.5:
                    splits.append((t, nid))
            if len(splits) > 0:
                changed = True
                splits.sort()
                prev = u
                for t, nid in splits:
                    if nid != prev:
                        new_edges.append((min(prev, nid), max(prev, nid)))
                    prev = nid
                if prev != v:
                    new_edges.append((min(prev, v), max(prev, v)))
            else:
                new_edges.append((u, v))
        edges = sorted(set(new_edges))

    # snap near-horizontal / near-vertical wall centerlines back to axis-aligned
    for _ in range(6):
        moved = False
        for u, v in edges:
            p0, p1 = nodes[u], nodes[v]
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                continue
            if abs(dx) > 1e-9 and abs(dy) <= abs(dx):
                y = (p0[1] + p1[1]) / 2.0
                if abs(p0[1] - y) > 1e-9:
                    p0[1] = y
                    moved = True
                if abs(p1[1] - y) > 1e-9:
                    p1[1] = y
                    moved = True
            elif abs(dy) > 1e-9 and abs(dx) <= abs(dy):
                x = (p0[0] + p1[0]) / 2.0
                if abs(p0[0] - x) > 1e-9:
                    p0[0] = x
                    moved = True
                if abs(p1[0] - x) > 1e-9:
                    p1[0] = x
                    moved = True
        if not moved:
            break

    # snap near-collinear wall runs sharing a T-junction to one line
    group_tol = 1.0
    for _ in range(6):
        moved = False
        horiz = []
        vert = []
        for u, v in edges:
            p0, p1 = nodes[u], nodes[v]
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            if abs(dx) > 1e-9 and abs(dy) <= abs(dx):
                horiz.append((u, v, (p0[1] + p1[1]) / 2.0,
                              min(p0[0], p1[0]), max(p0[0], p1[0])))
            elif abs(dy) > 1e-9 and abs(dx) <= abs(dy):
                vert.append((u, v, (p0[0] + p1[0]) / 2.0,
                             min(p0[1], p1[1]), max(p0[1], p1[1])))

        def _group(items):
            parent = list(range(len(items)))

            def find_root(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def merge(a, b):
                ra, rb = find_root(a), find_root(b)
                if ra != rb:
                    parent[rb] = ra

            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    a, b = items[i], items[j]
                    if abs(a[2] - b[2]) <= group_tol and \
                            a[4] >= b[3] - 0.8 and b[4] >= a[3] - 0.8:
                        merge(i, j)
            groups = {}
            for i, item in enumerate(items):
                groups.setdefault(find_root(i), []).append(item)
            return groups.values()

        for items in _group(horiz):
            if len(items) < 2:
                continue
            y = sum(it[2] for it in items) / len(items)
            for u, v, _, _, _ in items:
                if abs(nodes[u][1] - y) > 1e-9:
                    nodes[u][1] = y
                    moved = True
                if abs(nodes[v][1] - y) > 1e-9:
                    nodes[v][1] = y
                    moved = True
        for items in _group(vert):
            if len(items) < 2:
                continue
            x = sum(it[2] for it in items) / len(items)
            for u, v, _, _, _ in items:
                if abs(nodes[u][0] - x) > 1e-9:
                    nodes[u][0] = x
                    moved = True
                if abs(nodes[v][0] - x) > 1e-9:
                    nodes[v][0] = x
                    moved = True
        if not moved:
            break

    # re-cluster nodes that became identical after snapping
    remap = {}
    new_nodes = []
    for i, n in enumerate(nodes):
        hit = None
        for j, m in enumerate(new_nodes):
            if np.linalg.norm(m - n) <= tol:
                hit = j
                break
        if hit is None:
            hit = len(new_nodes)
            new_nodes.append(n.copy())
        remap[i] = hit
    nodes = new_nodes
    edges = sorted({(min(remap[u], remap[v]), max(remap[u], remap[v]))
                    for u, v in edges if remap[u] != remap[v]})
    return nodes, edges


def prune_dangling_edges(nodes, edges):
    """Remove dangling wall stubs that do not belong to any closed loop."""
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    changed = True
    while changed:
        changed = False
        keep = set()
        for u, v in edges:
            if len(adj[u]) > 1 and len(adj[v]) > 1:
                keep.add((u, v))
            else:
                changed = True
        if changed:
            edges = sorted(keep)
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
    return edges


def close_corner_gaps(nodes, edges, gap_tol=3.0):
    """Connect near-perpendicular walls whose endpoints almost meet."""
    extra = set(edges)
    for a, b in edges:
        d1 = nodes[b] - nodes[a]
        n1 = np.linalg.norm(d1)
        if n1 < 1e-9:
            continue
        for c, d in edges:
            if (a, b) == (c, d) or (a, b) == (d, c):
                continue
            d2 = nodes[d] - nodes[c]
            n2 = np.linalg.norm(d2)
            if n2 < 1e-9:
                continue
            if abs(np.dot(d1, d2)) > 0.3 * n1 * n2:
                continue
            for u in (a, b):
                for v in (c, d):
                    if u == v:
                        continue
                    key = (min(u, v), max(u, v))
                    if key in extra:
                        continue
                    if np.linalg.norm(nodes[u] - nodes[v]) <= gap_tol:
                        extra.add(key)
    return sorted(extra)


def merge_parallel_walls(nodes, edges, tol=1.0):
    """Merge nearly-parallel walls (offset < tol) into one shared coordinate
    so planar faces do not produce slanted edges or thin slivers."""
    nodes = [np.array(n, dtype=float) for n in nodes]
    edges = list(edges)

    def spans(nid):
        return nodes[nid]

    vwalls = defaultdict(list)   # x -> [(y_lo, y_hi, edge)]
    hwalls = defaultdict(list)   # y -> [(x_lo, x_hi, edge)]
    for eid, (u, v) in enumerate(edges):
        a, b = nodes[u], nodes[v]
        if abs(a[1] - b[1]) <= abs(a[0] - b[0]):
            hwalls[round((a[1] + b[1]) / 2.0, 3)].append(
                (min(a[0], b[0]), max(a[0], b[0]), eid))
        else:
            vwalls[round((a[0] + b[0]) / 2.0, 3)].append(
                (min(a[1], b[1]), max(a[1], b[1]), eid))

    def group_walls(items, axis):
        keys = sorted(items.keys())
        groups = []
        cur = []
        for k in keys:
            if cur and k - cur[-1] > tol:
                groups.append(cur)
                cur = []
            cur.append(k)
        if cur:
            groups.append(cur)
        out = []
        for g in groups:
            if len(g) == 1:
                continue
            spans_list = []
            for k in g:
                spans_list.extend(items[k])
            # only merge when the spans overlap or are near each other
            if axis == "V":
                spans_list.sort(key=lambda s: s[0])
                merged = [[spans_list[0][0], spans_list[0][1], spans_list[0][2]]]
                for lo, hi, eid in spans_list[1:]:
                    if lo <= merged[-1][1] + 2.0:
                        merged[-1][1] = max(merged[-1][1], hi)
                    else:
                        merged.append([lo, hi, None])
                out.append((sum(g) / len(g), merged))
            else:
                spans_list.sort(key=lambda s: s[0])
                merged = [[spans_list[0][0], spans_list[0][1], spans_list[0][2]]]
                for lo, hi, eid in spans_list[1:]:
                    if lo <= merged[-1][1] + 2.0:
                        merged[-1][1] = max(merged[-1][1], hi)
                    else:
                        merged.append([lo, hi, None])
                out.append((sum(g) / len(g), merged))
        return out

    vgroups = group_walls(vwalls, "V")
    hgroups = group_walls(hwalls, "H")

    def set_coord(nid, axis, value):
        nodes[nid][axis] = value

    for x, spans in vgroups:
        for lo, hi, _ in spans:
            for eid in range(len(edges)):
                u, v = edges[eid]
                a, b = nodes[u], nodes[v]
                if abs(a[1] - b[1]) <= abs(a[0] - b[0]):
                    continue
                ay = min(a[1], b[1])
                by = max(a[1], b[1])
                if ay <= hi + 0.5 and by >= lo - 0.5 and \
                        (abs(a[0] - x) <= tol or abs(b[0] - x) <= tol):
                    set_coord(u, 0, x)
                    set_coord(v, 0, x)
    for y, spans in hgroups:
        for lo, hi, _ in spans:
            for eid in range(len(edges)):
                u, v = edges[eid]
                a, b = nodes[u], nodes[v]
                if abs(a[1] - b[1]) > abs(a[0] - b[0]):
                    continue
                ax = min(a[0], b[0])
                bx = max(a[0], b[0])
                if ax <= hi + 0.5 and bx >= lo - 0.5 and \
                        (abs(a[1] - y) <= tol or abs(b[1] - y) <= tol):
                    set_coord(u, 1, y)
                    set_coord(v, 1, y)

    # re-cluster nodes and drop zero-length / tiny slanted edges
    remap = {}
    new_nodes = []
    for i, n in enumerate(nodes):
        hit = None
        for j, m in enumerate(new_nodes):
            if np.linalg.norm(m - n) <= 0.2:
                hit = j
                break
        if hit is None:
            hit = len(new_nodes)
            new_nodes.append(n.copy())
        remap[i] = hit
    new_edges = set()
    for u, v in edges:
        a, b = remap[u], remap[v]
        if a == b:
            continue
        p0, p1 = new_nodes[a], new_nodes[b]
        if np.linalg.norm(p1 - p0) < 0.2:
            continue
        new_edges.add((min(a, b), max(a, b)))
    return new_nodes, sorted(new_edges)


def planar_faces(nodes, edges):
    adj = defaultdict(list)
    for eid, (u, v) in enumerate(edges):
        adj[u].append((v, eid))
        adj[v].append((u, eid))
    for u in adj:
        adj[u].sort(key=lambda item: math_atan2(nodes[item[0]] - nodes[u]))
    seen = set()
    faces = []
    for eid, (u, v) in enumerate(edges):
        for start in ((u, v), (v, u)):
            if start in seen:
                continue
            face = []
            cur = start
            while cur not in seen and len(face) < len(edges) + 1:
                seen.add(cur)
                face.append(cur)
                a, b = cur
                nbrs = adj[b]
                # next half-edge at b: the neighbor after a in CCW order
                idx = None
                for i, (nb, _) in enumerate(nbrs):
                    if nb == a:
                        idx = i
                        break
                if idx is None:
                    break
                w = nbrs[(idx + 1) % len(nbrs)][0]
                cur = (b, w)
                if cur == start:
                    break
            if cur == start and len(face) > 0:
                poly_pts = [nodes[cur[0]] for cur in face] + [nodes[face[0][0]]]
                poly = Polygon(poly_pts)
                if abs(poly.area) > 1e-4:
                    if not poly.is_valid:
                        # dangling wall stubs make some faces self-touching
                        poly = poly.buffer(0)
                        if poly.geom_type == "MultiPolygon":
                            poly = max(poly.geoms, key=lambda g: g.area)
                        if poly.is_empty:
                            continue
                    faces.append((abs(poly.area), poly))
    return faces


def math_atan2(vec):
    return math.atan2(vec[1], vec[0])


def clean_room_poly(poly, grid=0.05, tol=0.15):
    """Snap near-axis edges flat and remove collinear T-junction vertices."""
    coords = np.array(poly.exterior.coords[:-1], dtype=float)
    for _ in range(4):
        for i in range(len(coords)):
            p0, p1 = coords[i], coords[(i + 1) % len(coords)]
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            if abs(dx) > 1e-9 and abs(dy) <= abs(dx):
                y = (p0[1] + p1[1]) / 2.0
                coords[i][1] = y
                coords[(i + 1) % len(coords)][1] = y
            elif abs(dy) > 1e-9 and abs(dx) <= abs(dy):
                x = (p0[0] + p1[0]) / 2.0
                coords[i][0] = x
                coords[(i + 1) % len(coords)][0] = x
        coords = np.round(coords / grid) * grid
        pts = []
        for p in coords:
            if len(pts) == 0 or np.linalg.norm(p - pts[-1]) > 1e-6:
                pts.append(p)
        if len(pts) > 1 and np.linalg.norm(pts[0] - pts[-1]) <= 1e-6:
            pts.pop()
        changed = True
        while changed and len(pts) > 3:
            changed = False
            for i in range(len(pts)):
                p, a, b = pts[i], pts[(i - 1) % len(pts)], pts[(i + 1) % len(pts)]
                v1 = p - a
                v2 = b - a
                length = np.linalg.norm(v2)
                if length < 1e-9:
                    continue
                dist = abs(v1[0] * v2[1] - v1[1] * v2[0]) / length
                if dist < tol:
                    pts.pop(i)
                    changed = True
                    break
        coords = np.array(pts)
        # delete short diagonal steps that sit between a vertical and a horizontal wall
        changed = True
        while changed and len(coords) > 3:
            changed = False
            n = len(coords)
            for i in range(n):
                a = coords[(i - 1) % n]
                b = coords[i]
                c = coords[(i + 1) % n]
                d = coords[(i + 2) % n]
                if np.linalg.norm(c - b) > 5.0:
                    continue
                if abs(b[0] - c[0]) < 0.3 or abs(b[1] - c[1]) < 0.3:
                    continue
                if abs(a[1] - b[1]) < 0.3 and abs(c[1] - d[1]) < 0.3:
                    coords[(i + 1) % n] = np.array([b[0], d[1]])
                    changed = True
                    break
                if abs(a[0] - b[0]) < 0.3 and abs(c[0] - d[0]) < 0.3:
                    coords[(i + 1) % n] = np.array([d[0], b[1]])
                    changed = True
                    break
                if np.linalg.norm(a - c) <= 5.0 and \
                        np.linalg.norm(b - a) > 0.3 and np.linalg.norm(b - c) > 0.3:
                    # narrow spike: drop the middle point
                    coords = np.delete(coords, i, axis=0)
                    changed = True
                    break
                if abs(a[0] - b[0]) < 0.3 and abs(c[1] - d[1]) < 0.3:
                    coords[i] = np.array([a[0], d[1]])
                    coords = np.delete(coords, (i + 1) % n, axis=0)
                    changed = True
                    break
                if abs(a[1] - b[1]) < 0.3 and abs(c[0] - d[0]) < 0.3:
                    coords[i] = np.array([d[0], a[1]])
                    coords = np.delete(coords, (i + 1) % n, axis=0)
                    changed = True
                    break
    if len(coords) < 3:
        return None
    p2 = Polygon(coords)
    if not p2.is_valid or p2.is_empty:
        return None
    return orient(p2, sign=1.0)


def simplify_face_poly(poly, grid=0.05):
    """Keep a graph face exactly on its wall coordinates; drop only points
    that are perfectly collinear with their neighbours."""
    coords = np.array(poly.exterior.coords[:-1], dtype=float)
    changed = True
    while changed and len(coords) > 3:
        changed = False
        n = len(coords)
        for i in range(n):
            a = coords[(i - 1) % n]
            b = coords[i]
            c = coords[(i + 1) % n]
            v1 = b - a
            v2 = c - a
            length = np.linalg.norm(v2)
            if length < 1e-9:
                continue
            dist = abs(v1[0] * v2[1] - v1[1] * v2[0]) / length
            if dist < 1e-6:
                coords = np.delete(coords, i, axis=0)
                changed = True
                break
    coords = np.round(coords / grid) * grid
    pts = []
    for p in coords:
        if len(pts) == 0 or np.linalg.norm(p - pts[-1]) > 1e-6:
            pts.append(p)
    if len(pts) > 1 and np.linalg.norm(pts[0] - pts[-1]) <= 1e-6:
        pts.pop()
    if len(pts) < 3:
        return None
    p2 = Polygon(pts)
    if not p2.is_valid or p2.is_empty:
        return None
    return orient(p2, sign=1.0)


def snap_room_to_graph(poly, nodes, edges, tol=0.65):
    """Snap a room's axis edges to the shared skeleton graph walls.
    Face rooms keep their steps (tight tol); corner-ring fallback rooms are
    pulled onto the nearest wall with a wider tol."""
    coords = np.array(poly.exterior.coords[:-1], dtype=float)
    vwalls = []
    hwalls = []
    for u, v in edges:
        a, b = nodes[u], nodes[v]
        dx, dy = b[0] - a[0], b[1] - a[1]
        if abs(dy) <= abs(dx):
            hwalls.append(((a[1] + b[1]) / 2.0, min(a[0], b[0]), max(a[0], b[0])))
        elif abs(dx) <= abs(dy):
            vwalls.append(((a[0] + b[0]) / 2.0, min(a[1], b[1]), max(a[1], b[1])))

    def nearest_h(y, lo, hi):
        best, best_d = None, tol
        for wy, wlo, whi in hwalls:
            if min(hi, whi) - max(lo, wlo) >= 0.15 * (hi - lo):
                d = abs(wy - y)
                if d < best_d:
                    best_d, best = d, wy
        return best

    def nearest_v(x, lo, hi):
        best, best_d = None, tol
        for wx, wlo, whi in vwalls:
            if min(hi, whi) - max(lo, wlo) >= 0.15 * (hi - lo):
                d = abs(wx - x)
                if d < best_d:
                    best_d, best = d, wx
        return best

    for _ in range(3):
        for i in range(len(coords)):
            p0, p1 = coords[i], coords[(i + 1) % len(coords)]
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                continue
            if abs(dy) <= abs(dx):
                y = nearest_h((p0[1] + p1[1]) / 2.0, min(p0[0], p1[0]), max(p0[0], p1[0]))
                if y is not None:
                    coords[i][1] = y
                    coords[(i + 1) % len(coords)][1] = y
            else:
                x = nearest_v((p0[0] + p1[0]) / 2.0, min(p0[1], p1[1]), max(p0[1], p1[1]))
                if x is not None:
                    coords[i][0] = x
                    coords[(i + 1) % len(coords)][0] = x
    if len(coords) < 3:
        return None
    p2 = Polygon(coords)
    if not p2.is_valid or p2.is_empty:
        return None
    return orient(p2, sign=1.0)


def snap_ring_to_rooms(poly, room_polys, tol=3.0):
    """Snap a corner-ring fallback room onto its neighbours' wall edges.
    Unlike snap_room_to_graph, the candidate edge only needs to be within
    tol and its span overlapping or adjacent to the ring edge, so a missing
    wall can inherit the coordinate from the room next to it."""
    coords = np.array(poly.exterior.coords[:-1], dtype=float)
    hwalls = []
    vwalls = []
    for rp in room_polys:
        rc = np.array(rp.exterior.coords[:-1], dtype=float)
        for i in range(len(rc)):
            a, b = rc[i], rc[(i + 1) % len(rc)]
            if abs(a[1] - b[1]) <= abs(a[0] - b[0]):
                hwalls.append(((a[1] + b[1]) / 2.0,
                               min(a[0], b[0]), max(a[0], b[0])))
            else:
                vwalls.append(((a[0] + b[0]) / 2.0,
                               min(a[1], b[1]), max(a[1], b[1])))

    def span_dist(lo, hi, wlo, whi):
        return max(0.0, max(lo, wlo) - min(hi, whi))

    def best_h(y, lo, hi):
        best, best_d = None, tol
        for wy, wlo, whi in hwalls:
            if span_dist(lo, hi, wlo, whi) <= tol and abs(wy - y) <= tol:
                if abs(wy - y) < best_d:
                    best_d, best = abs(wy - y), wy
        return best

    def best_v(x, lo, hi):
        best, best_d = None, tol
        for wx, wlo, whi in vwalls:
            if span_dist(lo, hi, wlo, whi) <= tol and abs(wx - x) <= tol:
                if abs(wx - x) < best_d:
                    best_d, best = abs(wx - x), wx
        return best

    for _ in range(2):
        for i in range(len(coords)):
            p0, p1 = coords[i], coords[(i + 1) % len(coords)]
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                continue
            if abs(dy) <= abs(dx):
                y = best_h((p0[1] + p1[1]) / 2.0,
                           min(p0[0], p1[0]), max(p0[0], p1[0]))
                if y is not None:
                    coords[i][1] = y
                    coords[(i + 1) % len(coords)][1] = y
            else:
                x = best_v((p0[0] + p1[0]) / 2.0,
                           min(p0[1], p1[1]), max(p0[1], p1[1]))
                if x is not None:
                    coords[i][0] = x
                    coords[(i + 1) % len(coords)][0] = x
    if len(coords) < 3:
        return poly
    p2 = Polygon(coords)
    if not p2.is_valid or p2.is_empty:
        return poly
    return orient(p2, sign=1.0)


def close_kitchen_no_door(plan, lines, nodes, edges, kitchen, features=None):
    """
    Kitchen without a door: find the room sharing its missing wall, extend
    that wall centerline, and extend open segments to intersect it.
    Returns (new lines, new nodes, new edges, info dict).
    """
    wd = float(plan.get("wall_depth") or 4.0)
    # 1. living <-> kitchen door, judged by 4 door corners from cleaned v2 features
    has_door = False
    if features is not None:
        living = unary_union(polys(plan, "living"))
        has_door = has_living_kitchen_door(plan, features, living, kitchen)
    info = {"has_door": has_door}

    # already closed by skeleton? (some plans have a wall there even without a door)
    pre_faces = planar_faces(nodes, edges)
    for area, poly in pre_faces:
        if area < 10.0:
            continue
        c = poly.representative_point()
        if kitchen.covers(c):
            overlap = kitchen.intersection(poly).area
            if overlap > 0.7 * min(poly.area, kitchen.area) and poly.area < 2.5 * kitchen.area:
                info["already_closed"] = True
                return lines, nodes, edges, info

    # 2. adjacent room with max shared boundary
    best_room, best_len = None, 0.0
    for rk in ROOM_KEYS:
        if rk == "kitchen":
            continue
        other = unary_union(polys(plan, rk)) if rk in plan else None
        if other is None or other.is_empty:
            continue
        shared = kitchen.boundary.intersection(other.boundary)
        length = shared.length if not shared.is_empty else 0.0
        if length > best_len:
            best_len, best_room = length, rk
    info["adjacent_room"] = best_room
    info["shared_length"] = round(best_len, 1)

    # 3. find kitchen edges without wall backing (the opening)
    shell = unary_union(shell_polys(plan))
    missing = None
    kc = np.array(kitchen.exterior.coords[:-1])
    for i in range(len(kc)):
        a, b = kc[i], kc[(i + 1) % len(kc)]
        mid = (a + b) / 2.0
        seg = LineString([a, b])
        probe = seg.buffer(0.6)
        backed = probe.intersection(shell).area > 0.1 * seg.length
        if not backed and seg.length > 4.0:
            missing = (a, b)
            info["missing_edge"] = [tuple(round(v, 2) for v in a), tuple(round(v, 2) for v in b)]
            break
    if missing is None:
        info["note"] = "no missing edge found"
        return lines, nodes, edges, info

    a, b = missing
    vertical = abs(a[0] - b[0]) < 1e-6

    # 4. existing wall-centerline stub parallel to the missing edge:
    #    the skeleton already has a short wall segment at the same line,
    #    so extend that one instead of inventing a new coordinate
    kb = kitchen.bounds  # (minx, miny, maxx, maxy)
    axis_coord = None
    for line in lines:
        coords = np.array(line.coords)
        for i in range(len(coords) - 1):
            p0, p1 = coords[i], coords[i + 1]
            if vertical and abs(p0[0] - p1[0]) < 0.6 and abs(p0[1] - p1[1]) > 2.0:
                y_lo, y_hi = min(p0[1], p1[1]), max(p0[1], p1[1])
                if abs(p0[0] - a[0]) <= wd and y_hi >= kb[1] - wd and y_lo <= kb[3] + wd:
                    axis_coord = (p0[0] + p1[0]) / 2.0
                    break
            elif not vertical and abs(p0[1] - p1[1]) < 0.6 and abs(p0[0] - p1[0]) > 2.0:
                x_lo, x_hi = min(p0[0], p1[0]), max(p0[0], p1[0])
                if abs(p0[1] - a[1]) <= wd and x_hi >= kb[0] - wd and x_lo <= kb[2] + wd:
                    axis_coord = (p0[1] + p1[1]) / 2.0
                    break
        if axis_coord is not None:
            break

    # fallback: average the collinear wall band edges
    if axis_coord is None:
        edge_coords = []
        for w in polys(plan, "wall"):
            wc = np.array(w.exterior.coords[:-1])
            for i in range(len(wc)):
                p0, p1 = wc[i], wc[(i + 1) % len(wc)]
                if np.linalg.norm(p1 - p0) < 3.0:
                    continue
                if vertical and abs(p0[0] - p1[0]) < 1e-6 and abs(p0[0] - a[0]) < wd * 0.8:
                    edge_coords.append(p0[0])
                elif not vertical and abs(p0[1] - p1[1]) < 1e-6 and abs(p0[1] - a[1]) < wd * 0.8:
                    edge_coords.append(p0[1])
        if len(edge_coords) > 0:
            axis_coord = (min(edge_coords) + max(edge_coords)) / 2.0
    if axis_coord is None:
        info["note"] = "no collinear wall band"
        return lines, nodes, edges, info

    # snap the closing line to the nearest existing collinear skeleton wall
    best_axis, best_d = None, wd * 2.0
    miss_lo, miss_hi = (min(a[1], b[1]), max(a[1], b[1])) if vertical \
        else (min(a[0], b[0]), max(a[0], b[0]))
    for line in lines:
        c = np.array(line.coords)
        for i in range(len(c) - 1):
            p0, p1 = c[i], c[i + 1]
            if vertical:
                if (abs(p0[1] - p1[1]) > 1e-9
                        and abs(p0[0] - p1[0]) <= abs(p0[1] - p1[1])):
                    x = (p0[0] + p1[0]) / 2.0
                    w0, w1 = min(p0[1], p1[1]), max(p0[1], p1[1])
                    ov = min(miss_hi, w1) - max(miss_lo, w0)
                    if ov >= 0.1 * (miss_hi - miss_lo) and abs(x - axis_coord) <= wd * 2.0:
                        d = abs(x - axis_coord)
                        if d < best_d:
                            best_d, best_axis = d, x
            else:
                if (abs(p0[0] - p1[0]) > 1e-9
                        and abs(p0[1] - p1[1]) <= abs(p0[0] - p1[0])):
                    y = (p0[1] + p1[1]) / 2.0
                    w0, w1 = min(p0[0], p1[0]), max(p0[0], p1[0])
                    ov = min(miss_hi, w1) - max(miss_lo, w0)
                    if ov >= 0.1 * (miss_hi - miss_lo) and abs(y - axis_coord) <= wd * 2.0:
                        d = abs(y - axis_coord)
                        if d < best_d:
                            best_d, best_axis = d, y
    if best_axis is not None:
        axis_coord = best_axis

    # 5. span of the closing wall: horizontal/vertical skeleton segments
    #    belonging to the kitchen loop and near its y/x range
    lo, hi = None, None
    for line in lines:
        coords = np.array(line.coords)
        for i in range(len(coords) - 1):
            p0, p1 = coords[i], coords[i + 1]
            if vertical and abs(p0[1] - p1[1]) < 0.6 and abs(p0[0] - p1[0]) > 0.5:
                y_line = p0[1]
                x_lo, x_hi = min(p0[0], p1[0]), max(p0[0], p1[0])
                if not (kb[1] - 6.0 <= y_line <= kb[3] + 6.0):
                    continue
                if not (x_hi >= kb[0] - 2.0 and x_lo <= kb[2] + 2.0):
                    continue
                gap_x = max(0.0, axis_coord - x_hi, x_lo - axis_coord)
                if gap_x > 2.0 * wd:
                    continue
                if lo is None or y_line < lo:
                    lo = y_line
                if hi is None or y_line > hi:
                    hi = y_line
            elif not vertical and abs(p0[0] - p1[0]) < 0.6 and abs(p0[1] - p1[1]) > 0.5:
                x_line = p0[0]
                y_lo, y_hi = min(p0[1], p1[1]), max(p0[1], p1[1])
                if not (kb[0] - 6.0 <= x_line <= kb[2] + 6.0):
                    continue
                if not (y_hi >= kb[1] - 2.0 and y_lo <= kb[3] + 2.0):
                    continue
                gap_y = max(0.0, axis_coord - y_hi, y_lo - axis_coord)
                if gap_y > 2.0 * wd:
                    continue
                if lo is None or x_line < lo:
                    lo = x_line
                if hi is None or x_line > hi:
                    hi = x_line
    if lo is None or hi is None:
        info["note"] = "no parallel skeleton segments"
        return lines, nodes, edges, info

    # trim the closing span so it stops at the first wall instead of crossing rooms
    for rk in ("bathroom", "bedroom", "balcony", "storage", "stair", "living"):
        if rk not in plan:
            continue
        for room in polys(plan, rk):
            if vertical:
                seg = LineString([(axis_coord, lo), (axis_coord, hi)])
                inter = seg.intersection(room)
                if inter.is_empty:
                    continue
                if inter.geom_type == "LineString":
                    ys = [p[1] for p in inter.coords]
                elif inter.geom_type == "MultiLineString":
                    ys = [p[1] for g in inter.geoms for p in g.coords]
                else:
                    ys = [inter.bounds[1], inter.bounds[3]]
                ymin, ymax = min(ys), max(ys)
                miss_lo, miss_hi = min(a[1], b[1]), max(a[1], b[1])
                if ymin > miss_hi + 1e-6:
                    hi = min(hi, ymin - 0.01)
                elif ymax < miss_lo - 1e-6:
                    lo = max(lo, ymax + 0.01)
            else:
                seg = LineString([(lo, axis_coord), (hi, axis_coord)])
                inter = seg.intersection(room)
                if inter.is_empty:
                    continue
                if inter.geom_type == "LineString":
                    xs = [p[0] for p in inter.coords]
                elif inter.geom_type == "MultiLineString":
                    xs = [p[0] for g in inter.geoms for p in g.coords]
                else:
                    xs = [inter.bounds[0], inter.bounds[2]]
                xmin, xmax = min(xs), max(xs)
                miss_lo, miss_hi = min(a[0], b[0]), max(a[0], b[0])
                if xmin > miss_hi + 1e-6:
                    hi = min(hi, xmin - 0.01)
                elif xmax < miss_lo - 1e-6:
                    lo = max(lo, xmax + 0.01)

    # stop at existing skeleton walls collinear with the closing line
    if vertical:
        miss_lo, miss_hi = min(a[1], b[1]), max(a[1], b[1])
    else:
        miss_lo, miss_hi = min(a[0], b[0]), max(a[0], b[0])
    for line in lines:
        c = np.array(line.coords)
        for i in range(len(c) - 1):
            p0, p1 = c[i], c[i + 1]
            if vertical:
                if (abs(p0[1] - p1[1]) > 1e-9
                        and abs(p0[0] - p1[0]) <= abs(p0[1] - p1[1])
                        and abs(p0[0] - axis_coord) <= 1.0):
                    w0, w1 = min(p0[1], p1[1]), max(p0[1], p1[1])
                    if w1 < miss_lo - 1e-6:
                        lo = max(lo, w1 + 0.01)
                    if w0 > miss_hi + 1e-6:
                        hi = min(hi, w0 - 0.01)
            else:
                if (abs(p0[0] - p1[0]) > 1e-9
                        and abs(p0[1] - p1[1]) <= abs(p0[0] - p1[0])
                        and abs(p0[1] - axis_coord) <= 1.0):
                    w0, w1 = min(p0[0], p1[0]), max(p0[0], p1[0])
                    if w1 < miss_lo - 1e-6:
                        lo = max(lo, w1 + 0.01)
                    if w0 > miss_hi + 1e-6:
                        hi = min(hi, w0 - 0.01)
    info["extended_span"] = [round(lo, 2), round(hi, 2)]

    # 6. build the closing centerline and splice horizontal segments
    new_lines = []
    if vertical:
        close_line = LineString([(axis_coord, lo), (axis_coord, hi)])
        cut = axis_coord
        for line in lines:
            coords = np.array(line.coords)
            for i in range(len(coords) - 1):
                p0, p1 = coords[i], coords[i + 1]
                if abs(p0[1] - p1[1]) > abs(p0[0] - p1[0]):
                    new_lines.append(LineString([p0, p1]))
                    continue
                y_line = (p0[1] + p1[1]) / 2.0
                if not (lo - 2.0 <= y_line <= hi + 2.0):
                    new_lines.append(LineString([p0, p1]))
                    continue
                x_lo, x_hi = min(p0[0], p1[0]), max(p0[0], p1[0])
                if not (x_hi >= kb[0] - 2.0 and x_lo <= kb[2] + 2.0):
                    new_lines.append(LineString([p0, p1]))
                    continue
                if x_lo < cut < x_hi:
                    new_lines.append(LineString([(x_lo, y_line), (cut, y_line)]))
                    new_lines.append(LineString([(cut, y_line), (x_hi, y_line)]))
                elif x_hi < cut and cut - x_hi <= 2.0 * wd:
                    new_lines.append(LineString([(x_lo, y_line), (cut, y_line)]))
                elif x_lo > cut and x_lo - cut <= 2.0 * wd:
                    new_lines.append(LineString([(cut, y_line), (x_hi, y_line)]))
                else:
                    new_lines.append(LineString([p0, p1]))
    else:
        close_line = LineString([(lo, axis_coord), (hi, axis_coord)])
        cut = axis_coord
        for line in lines:
            coords = np.array(line.coords)
            for i in range(len(coords) - 1):
                p0, p1 = coords[i], coords[i + 1]
                if abs(p0[0] - p1[0]) > abs(p0[1] - p1[1]):
                    new_lines.append(LineString([p0, p1]))
                    continue
                x_line = (p0[0] + p1[0]) / 2.0
                if not (lo - 2.0 <= x_line <= hi + 2.0):
                    new_lines.append(LineString([p0, p1]))
                    continue
                y_lo, y_hi = min(p0[1], p1[1]), max(p0[1], p1[1])
                if not (y_hi >= kb[1] - 2.0 and y_lo <= kb[3] + 2.0):
                    new_lines.append(LineString([p0, p1]))
                    continue
                if y_lo < cut < y_hi:
                    new_lines.append(LineString([(x_line, y_lo), (x_line, cut)]))
                    new_lines.append(LineString([(x_line, cut), (x_line, y_hi)]))
                elif y_hi < cut and cut - y_hi <= 2.0 * wd:
                    new_lines.append(LineString([(x_line, y_lo), (x_line, cut)]))
                elif y_lo > cut and y_lo - cut <= 2.0 * wd:
                    new_lines.append(LineString([(x_line, cut), (x_line, y_hi)]))
                else:
                    new_lines.append(LineString([p0, p1]))

    new_lines.append(close_line)
    info["closing_line"] = list(close_line.coords)

    # report the four skeleton corner points of the closed kitchen ring
    if vertical:
        east_x = None
        for line in lines:
            coords = np.array(line.coords)
            for i in range(len(coords) - 1):
                p0, p1 = coords[i], coords[i + 1]
                if abs(p0[0] - p1[0]) < 0.6 and abs(p0[1] - p1[1]) > 2.0:
                    y_lo, y_hi = min(p0[1], p1[1]), max(p0[1], p1[1])
                    if (abs(p0[0] - kb[2]) <= wd and y_lo <= hi and y_hi >= lo):
                        east_x = (p0[0] + p1[0]) / 2.0
                        break
            if east_x is not None:
                break
        if east_x is None:
            east_x = kb[2] + wd * 0.5
        info["kitchen_ring_corners"] = [
            [round(east_x, 3), round(lo, 3)],
            [round(axis_coord, 3), round(lo, 3)],
            [round(axis_coord, 3), round(hi, 3)],
            [round(east_x, 3), round(hi, 3)],
        ]
    else:
        south_y = None
        for line in lines:
            coords = np.array(line.coords)
            for i in range(len(coords) - 1):
                p0, p1 = coords[i], coords[i + 1]
                if abs(p0[1] - p1[1]) < 0.6 and abs(p0[0] - p1[0]) > 2.0:
                    x_lo, x_hi = min(p0[0], p1[0]), max(p0[0], p1[0])
                    if (abs(p0[1] - kb[1]) <= wd and x_lo <= hi and x_hi >= lo):
                        south_y = (p0[1] + p1[1]) / 2.0
                        break
            if south_y is not None:
                break
        if south_y is None:
            south_y = kb[1] - wd * 0.5
        info["kitchen_ring_corners"] = [
            [round(lo, 3), round(south_y, 3)],
            [round(lo, 3), round(axis_coord, 3)],
            [round(hi, 3), round(axis_coord, 3)],
            [round(hi, 3), round(south_y, 3)],
        ]
    return new_lines, nodes, edges, info


def label_faces(faces, plan):
    room_unions = {rk: unary_union(polys(plan, rk)) for rk in ROOM_KEYS if rk in plan}
    labeled = []
    for area, poly in faces:
        c = poly.representative_point()
        best, best_r = None, 0.0
        for rk, ru in room_unions.items():
            if ru.is_empty or not ru.intersects(poly):
                continue
            inter = ru.intersection(poly).area
            ratio = inter / poly.area
            if ratio > best_r:
                best_r, best = ratio, rk
        labeled.append((area, poly, best, best_r))
    return labeled


def room_rings_from_features(features, room_order, plan):
    """Rebuild each room ring from cleaned v2 corner tokens."""
    inner = plan.get("inner")
    if inner is None:
        return {}
    x1, y1, x2, y2 = inner.bounds

    def denorm(c):
        out = np.empty_like(c)
        out[:, 0] = (c[:, 0] + 1.0) * 0.5 * (x2 - x1) + x1
        out[:, 1] = (c[:, 1] + 1.0) * 0.5 * (y2 - y1) + y1
        return out

    rings = {}
    for ridx, name in enumerate(room_order):
        pts = []
        for i in range(features.shape[0]):
            row = features[i]
            if abs(row[0]) < 1e-6 and abs(row[1]) < 1e-6:
                continue
            if row[44:].sum() == 0:
                continue
            if int(np.argmax(row[44:])) != ridx:
                continue
            ci = int(np.argmax(row[12:44]))
            pts.append((ci, float(row[0]), float(row[1])))
        if len(pts) < 3:
            continue
        pts.sort(key=lambda p: p[0])
        arr = denorm(np.array([[p[1], p[2]] for p in pts], dtype=float))
        poly = Polygon(arr)
        if poly.is_valid and not poly.is_empty:
            rings[name] = orient(poly, sign=1.0)
    return rings


def close_room_with_walls(plan, corner_poly, lines):
    """Close a missing room using nearby skeleton walls; corner only locates it."""
    wd = float(plan.get("wall_depth") or 4.0)
    minx, miny, maxx, maxy = corner_poly.bounds

    def nearest_axis(orientation, value, lo, hi):
        best = None
        best_d = wd * 2.0
        for line in lines:
            c = np.array(line.coords)
            for i in range(len(c) - 1):
                p0, p1 = c[i], c[i + 1]
                if orientation == "H":
                    if (abs(p0[0] - p1[0]) > 1e-9
                            and abs(p0[1] - p1[1]) <= abs(p0[0] - p1[0])):
                        y = (p0[1] + p1[1]) / 2.0
                        if abs(y - value) <= wd * 2.0:
                            w0, w1 = min(p0[0], p1[0]), max(p0[0], p1[0])
                            overlap = min(hi, w1) - max(lo, w0)
                            if overlap >= 0.1 * (hi - lo):
                                d = abs(y - value)
                                if d < best_d:
                                    best_d, best = d, y
                else:
                    if (abs(p0[1] - p1[1]) > 1e-9
                            and abs(p0[0] - p1[0]) <= abs(p0[1] - p1[1])):
                        x = (p0[0] + p1[0]) / 2.0
                        if abs(x - value) <= wd * 2.0:
                            w0, w1 = min(p0[1], p1[1]), max(p0[1], p1[1])
                            overlap = min(hi, w1) - max(lo, w0)
                            if overlap >= 0.1 * (hi - lo):
                                d = abs(x - value)
                                if d < best_d:
                                    best_d, best = d, x
        return best

    top = nearest_axis("H", maxy, minx, maxx)
    bottom = nearest_axis("H", miny, minx, maxx)
    left = nearest_axis("V", minx, miny, maxy)
    right = nearest_axis("V", maxx, miny, maxy)
    wall_based = sum(v is not None for v in (top, bottom, left, right)) >= 2
    top = maxy if top is None else top
    bottom = miny if bottom is None else bottom
    left = minx if left is None else left
    right = maxx if right is None else right
    poly = Polygon([(left, bottom), (right, bottom), (right, top), (left, top)])
    if not poly.is_valid or poly.is_empty or poly.area < 1.0:
        return None, False
    return orient(poly, sign=1.0), wall_based


def snap_ring_to_walls(poly, lines, wd):
    """Snap a corner-ring's near-axis edges to the closest skeleton walls."""
    coords = np.array(poly.exterior.coords[:-1], dtype=float)

    def nearest_axis(orientation, value, lo, hi):
        best = None
        best_d = wd * 2.0
        for line in lines:
            c = np.array(line.coords)
            for i in range(len(c) - 1):
                p0, p1 = c[i], c[i + 1]
                if orientation == "H":
                    if (abs(p0[0] - p1[0]) > 1e-9
                            and abs(p0[1] - p1[1]) <= abs(p0[0] - p1[0])):
                        y = (p0[1] + p1[1]) / 2.0
                        if abs(y - value) <= wd * 2.0:
                            w0, w1 = min(p0[0], p1[0]), max(p0[0], p1[0])
                            if min(hi, w1) - max(lo, w0) >= 0.1 * (hi - lo):
                                d = abs(y - value)
                                if d < best_d:
                                    best_d, best = d, y
                else:
                    if (abs(p0[1] - p1[1]) > 1e-9
                            and abs(p0[0] - p1[0]) <= abs(p0[1] - p1[1])):
                        x = (p0[0] + p1[0]) / 2.0
                        if abs(x - value) <= wd * 2.0:
                            w0, w1 = min(p0[1], p1[1]), max(p0[1], p1[1])
                            if min(hi, w1) - max(lo, w0) >= 0.1 * (hi - lo):
                                d = abs(x - value)
                                if d < best_d:
                                    best_d, best = d, x
        return best

    for _ in range(3):
        for i in range(len(coords)):
            p0, p1 = coords[i], coords[(i + 1) % len(coords)]
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                continue
            lo, hi = (min(p0[0], p1[0]), max(p0[0], p1[0])) if abs(dx) >= abs(dy) \
                else (min(p0[1], p1[1]), max(p0[1], p1[1]))
            if abs(dx) > 1e-9 and abs(dy) <= abs(dx):
                y = nearest_axis("H", (p0[1] + p1[1]) / 2.0, lo, hi)
                if y is not None:
                    coords[i][1] = y
                    coords[(i + 1) % len(coords)][1] = y
            elif abs(dy) > 1e-9 and abs(dx) <= abs(dy):
                x = nearest_axis("V", (p0[0] + p1[0]) / 2.0, lo, hi)
                if x is not None:
                    coords[i][0] = x
                    coords[(i + 1) % len(coords)][0] = x
        coords = np.round(coords / 0.05) * 0.05
        pts = []
        for p in coords:
            if len(pts) == 0 or np.linalg.norm(p - pts[-1]) > 1e-6:
                pts.append(p)
        if len(pts) > 1 and np.linalg.norm(pts[0] - pts[-1]) <= 1e-6:
            pts.pop()
        coords = np.array(pts)

    if len(coords) < 3:
        return None
    p2 = Polygon(coords)
    if not p2.is_valid or p2.is_empty:
        return None
    return orient(p2, sign=1.0)


def clean_living_poly(p):
    # keep graph coordinates; flatten near-axis slanted edges (two parallel
    # wall stubs offset by <0.5) to their midpoint, never re-average steps
    coords = np.array(p.exterior.coords[:-1], dtype=float)
    # drop tiny edges (<0.3) that appear where the graph has two nearly
    # identical wall stubs; they read as thin slots in the drawn living area
    changed = True
    while changed and len(coords) > 3:
        changed = False
        n = len(coords)
        for i in range(n):
            a = coords[(i - 1) % n]
            b = coords[i]
            c = coords[(i + 1) % n]
            if np.linalg.norm(b - a) < 0.3 or np.linalg.norm(c - b) < 0.3:
                cand = np.delete(coords, i, axis=0)
                pp = Polygon(cand)
                if pp.is_valid and not pp.is_empty and abs(pp.area - p.area) < 0.01 * p.area:
                    coords = cand
                    changed = True
                    break
    for _ in range(3):
        changed = False
        n = len(coords)
        for i in range(n):
            p0 = coords[i]
            p1 = coords[(i + 1) % n]
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            if 0.02 < min(abs(dx), abs(dy)) <= 0.5 and max(abs(dx), abs(dy)) > 2.0:
                if abs(dx) <= 0.5:
                    x = (p0[0] + p1[0]) / 2.0
                    coords[i][0] = x
                    coords[(i + 1) % n][0] = x
                else:
                    y = (p0[1] + p1[1]) / 2.0
                    coords[i][1] = y
                    coords[(i + 1) % n][1] = y
                changed = True
                break
        if not changed:
            break
    if len(coords) < 3:
        return p
    p2 = Polygon(coords)
    if not p2.is_valid or p2.is_empty:
        return p
    return orient(p2, sign=1.0)


def snap_living_to_rooms(poly, room_polys, wd):
    """Snap living edges to the closest room boundaries so no sliver remains."""
    coords = np.array(poly.exterior.coords[:-1], dtype=float)
    snap_tol = 3.0

    def nearest(orientation, value, lo, hi):
        best = None
        best_d = snap_tol
        for room in room_polys:
            rc = np.array(room.exterior.coords[:-1])
            for i in range(len(rc)):
                a, b = rc[i], rc[(i + 1) % len(rc)]
                if orientation == "H":
                    if (abs(a[1] - b[1]) <= abs(a[0] - b[0])
                            and abs((a[1] + b[1]) / 2.0 - value) <= snap_tol):
                        w0, w1 = min(a[0], b[0]), max(a[0], b[0])
                        if min(hi, w1) - max(lo, w0) >= 0.05 * (hi - lo):
                            d = abs((a[1] + b[1]) / 2.0 - value)
                            if d < best_d:
                                best_d, best = d, (a[1] + b[1]) / 2.0
                else:
                    if (abs(a[0] - b[0]) <= abs(a[1] - b[1])
                            and abs((a[0] + b[0]) / 2.0 - value) <= snap_tol):
                        w0, w1 = min(a[1], b[1]), max(a[1], b[1])
                        if min(hi, w1) - max(lo, w0) >= 0.05 * (hi - lo):
                            d = abs((a[0] + b[0]) / 2.0 - value)
                            if d < best_d:
                                best_d, best = d, (a[0] + b[0]) / 2.0
        return best

    for _ in range(3):
        for i in range(len(coords)):
            p0, p1 = coords[i], coords[(i + 1) % len(coords)]
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                continue
            if abs(dy) <= abs(dx):
                y = nearest("H", (p0[1] + p1[1]) / 2.0,
                            min(p0[0], p1[0]), max(p0[0], p1[0]))
                if y is not None:
                    coords[i][1] = y
                    coords[(i + 1) % len(coords)][1] = y
            else:
                x = nearest("V", (p0[0] + p1[0]) / 2.0,
                            min(p0[1], p1[1]), max(p0[1], p1[1]))
                if x is not None:
                    coords[i][0] = x
                    coords[(i + 1) % len(coords)][0] = x
    if len(coords) < 3:
        return poly
    p2 = Polygon(coords)
    if not p2.is_valid or p2.is_empty:
        return poly
    return orient(p2, sign=1.0)


def snap_shared_edges(polys, tol=3.0):
    """Snap the shared edge of two neighbouring polygons to one coordinate.
    When room A's edge is at x1 and room B's parallel edge is at x2 with
    |x1-x2|<tol and overlapping spans, both edges are moved to (x1+x2)/2 so
    thin slots and small overlaps disappear. polys is a list of Shapely
    polygons; returns the updated list."""
    out = [np.array(p.exterior.coords[:-1], dtype=float) for p in polys]
    n = len(out)
    for _ in range(3):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                ci, cj = out[i], out[j]
                # horizontal edges
                for a, b in zip(ci, np.roll(ci, -1, axis=0)):
                    if abs(a[1] - b[1]) > abs(a[0] - b[0]):
                        continue
                    y = (a[1] + b[1]) / 2.0
                    lo, hi = min(a[0], b[0]), max(a[0], b[0])
                    for c, d in zip(cj, np.roll(cj, -1, axis=0)):
                        if abs(c[1] - d[1]) > abs(c[0] - d[0]):
                            continue
                        y2 = (c[1] + d[1]) / 2.0
                        if abs(y2 - y) > tol or abs(y2 - y) < 1e-6:
                            continue
                        lo2, hi2 = min(c[0], d[0]), max(c[0], d[0])
                        if min(hi, hi2) - max(lo, lo2) < 1.0:
                            continue
                        ny = (y + y2) / 2.0
                        ci[np.where(np.isclose(ci[:, 1], y, atol=1e-9))[0], 1] = ny
                        cj[np.where(np.isclose(cj[:, 1], y2, atol=1e-9))[0], 1] = ny
                        moved = True
                # vertical edges
                for a, b in zip(ci, np.roll(ci, -1, axis=0)):
                    if abs(a[0] - b[0]) > abs(a[1] - b[1]):
                        continue
                    x = (a[0] + b[0]) / 2.0
                    lo, hi = min(a[1], b[1]), max(a[1], b[1])
                    for c, d in zip(cj, np.roll(cj, -1, axis=0)):
                        if abs(c[0] - d[0]) > abs(c[1] - d[1]):
                            continue
                        x2 = (c[0] + d[0]) / 2.0
                        if abs(x2 - x) > tol or abs(x2 - x) < 1e-6:
                            continue
                        lo2, hi2 = min(c[1], d[1]), max(c[1], d[1])
                        if min(hi, hi2) - max(lo, lo2) < 1.0:
                            continue
                        nx = (x + x2) / 2.0
                        ci[np.where(np.isclose(ci[:, 0], x, atol=1e-9))[0], 0] = nx
                        cj[np.where(np.isclose(cj[:, 0], x2, atol=1e-9))[0], 0] = nx
                        moved = True
        if not moved:
            break
    res = []
    for c in out:
        p = Polygon(c)
        if p.is_valid and not p.is_empty:
            res.append(orient(p, sign=1.0))
        else:
            res.append(p)
    return res


def resolve_thin_overlaps(polys, tol=1.0, max_area=500.0):
    """Move facing edges apart so thin overlap strips disappear.

    A thin overlap is a strip whose width is below tol and whose other
    dimension is at least 1.0. Each polygon loses its own half of the
    strip, so both facing edges end at the strip midpoint: no overlap and
    no new gap on that span."""
    out = [np.array(p.exterior.coords[:-1], dtype=float) for p in polys]
    n = len(out)

    def largest(g):
        if g.is_empty:
            return None
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda p: p.area)
        if g.geom_type != "Polygon" or g.area < 1e-6:
            return None
        return g

    for _ in range(3):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                pa, pb = Polygon(out[i]), Polygon(out[j])
                if not pa.is_valid or not pb.is_valid:
                    continue
                inter = pa.intersection(pb)
                if inter.is_empty or inter.area < 0.5 or inter.area > max_area:
                    continue
                x0, y0, x1, y1 = inter.bounds
                w, h = x1 - x0, y1 - y0
                if h < tol and w >= 1.0:
                    mid = (y0 + y1) / 2.0
                    strip_a = Polygon([(x0, y0), (x1, y0), (x1, mid), (x0, mid)])
                    strip_b = Polygon([(x0, mid), (x1, mid), (x1, y1), (x0, y1)])
                elif w < tol and h >= 1.0:
                    mid = (x0 + x1) / 2.0
                    strip_a = Polygon([(x0, y0), (mid, y0), (mid, y1), (x0, y1)])
                    strip_b = Polygon([(mid, y0), (x1, y0), (x1, y1), (mid, y1)])
                else:
                    continue
                new_a = largest(pa.difference(strip_a))
                new_b = largest(pb.difference(strip_b))
                if new_a is None or new_b is None:
                    continue
                if new_a.area < 0.7 * pa.area or new_b.area < 0.7 * pb.area:
                    continue
                window = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
                wa = new_a.intersection(window)
                wb = new_b.intersection(window)
                if wa.is_empty or wb.is_empty or wa.distance(wb) > 0.05:
                    continue
                out[i] = np.array(new_a.exterior.coords[:-1], dtype=float)
                out[j] = np.array(new_b.exterior.coords[:-1], dtype=float)
                moved = True
        if not moved:
            break
    res = []
    for orig, c in zip(polys, out):
        p = Polygon(c)
        if p.is_valid and not p.is_empty:
            res.append(orient(p, sign=1.0))
        else:
            res.append(orig)
    return res


def add_missing_rooms(plan, features, room_order, kept, lines, candidates,
                      snap_nodes=None, snap_edges=None):
    """Match corner rooms; keep skeleton faces that cover them, else close with walls."""
    corner_rooms = room_rings_from_features(features, room_order, plan)
    if not corner_rooms:
        return kept
    g_nodes = snap_nodes if snap_nodes is not None else build_graph(lines)[0]
    g_edges = snap_edges if snap_edges is not None else build_graph(lines)[1]

    max_room_area = max(cp.area for cp in corner_rooms.values())
    candidates = [c for c in candidates if c[0] < 3.0 * max_room_area]
    if len(candidates) > 1:
        candidates = [c for c in candidates if c[0] < max(c[0] for c in candidates)]

    matched = set()
    for area, poly, rk, ratio in sorted(kept, key=lambda f: -f[0]):
        if rk not in ROOM_KEYS:
            continue
        for name, cp in corner_rooms.items():
            base = name.rsplit("_", 1)[0]
            if base != rk or name in matched or not cp.intersects(poly):
                continue
            ov = cp.intersection(poly).area / cp.area
            if ov > 0.5:
                matched.add(name)

    added = []
    used_faces = set()
    for name, cp in sorted(corner_rooms.items(), key=lambda kv: -kv[1].area):
        base = name.rsplit("_", 1)[0]
        if (name in matched or base in ("front_door", "living", "balcony")
                or base not in ROOM_KEYS):
            continue

        best, best_ov = None, 0.0
        for i, (area, poly, rk) in enumerate(candidates):
            if i in used_faces or not poly.intersects(cp):
                continue
            ov = poly.intersection(cp).area / cp.area
            if ov > best_ov:
                best_ov, best = ov, (i, poly)

        if best is not None and best_ov > 0.3:
            i, poly = best
            dup = False
            for _, op, ork, _ in kept + added:
                if ork == base:
                    continue
                inter = poly.intersection(op).area
                if inter > 0.9 * min(poly.area, op.area):
                    dup = True
                    break
            if dup:
                _LOG.debug(f"  face for {name} already used by another room, skip")
            else:
                clean = clean_room_poly(poly)
                if clean is None:
                    clean = orient(poly.simplify(0.0, preserve_topology=False), sign=1.0)
                added.append((clean.area, clean, base, 0.5))
                used_faces.add(i)
                matched.add(name)
                _LOG.debug(f"  kept skeleton face for {name} (overlap {best_ov:.2f}, area {clean.area:.1f})")
                continue

        # ring sits inside an already-built room face: snap it to the
        # neighbouring walls and carve it out of the parent face
        all_faces = kept + added
        parents = [f for f in all_faces
                   if f[2] != base and f[1].intersection(cp).area / cp.area > 0.5]
        if parents:
            fb = cp
            fb_clean = clean_room_poly(cp)
            if fb_clean is not None:
                fb = fb_clean
            snapped = snap_room_to_graph(fb, g_nodes, g_edges, tol=3.0)
            if snapped is not None:
                fb = snapped
            room_polys = [f[1] for f in kept if f[2] != base]
            snapped = snap_ring_to_rooms(fb, room_polys, tol=3.0)
            if snapped is not None:
                fb = snapped
            ok = fb.area > 20.0
            others = unary_union([f[1] for f in all_faces if f not in parents])
            if ok and not others.is_empty and \
                    fb.intersection(others).area > 0.05 * fb.area:
                ok = False
                _LOG.debug(f"  carve {name} failed: overlaps other rooms "
                      f"{fb.intersection(others).area:.1f}")
            new_kept = []
            new_added = []
            if ok:
                for lst, out in ((kept, new_kept), (added, new_added)):
                    for f in lst:
                        if f not in parents:
                            out.append(f)
                            continue
                        rem = f[1].difference(fb)
                        if rem.geom_type == "MultiPolygon":
                            pieces = [g for g in rem.geoms
                                      if g.geom_type == "Polygon" and g.area > 0.5]
                            if len(pieces) == 1:
                                rem = pieces[0]
                            elif len(pieces) > 1:
                                pieces.sort(key=lambda g: -g.area)
                                if sum(g.area for g in pieces[1:]) < 0.05 * pieces[0].area:
                                    rem = pieces[0]
                        if rem.geom_type != "Polygon" or rem.is_empty or \
                                rem.area < 20.0 or len(rem.interiors) > 0 or \
                                rem.area < 0.3 * f[1].area:
                            ok = False
                            _LOG.debug(f"  carve {name} failed: parent remainder "
                                  f"type={rem.geom_type} area={rem.area:.1f} "
                                  f"holes={len(rem.interiors) if rem.geom_type == 'Polygon' else -1}")
                            break
                        out.append((rem.area, rem, f[2], f[3]))
                    if not ok:
                        break
            if ok:
                kept[:] = new_kept
                added[:] = new_added
                added.append((fb.area, fb, base, -1.0))
                matched.add(name)
                _LOG.debug(f"  carved {name} out of {[f[2] for f in parents]} "
                      f"-> area {fb.area:.1f}")
                continue

        consumed = [op for _, op, ork, _ in kept + added if ork != base]
        covered = any(
            i not in used_faces and
            not any(op.intersection(poly).area > 0.9 * min(op.area, poly.area)
                    for op in consumed) and
            poly.intersection(cp).area / cp.area > 0.3
            for i, (_, poly, _) in enumerate(candidates))
        if covered:
            matched.add(name)
            continue

        # prefer the corner ring itself: it is the actual room shape and does
        # not grow into a neighbouring room like an axis-aligned box does
        fb = cp
        wall_based = False
        fb_clean = clean_room_poly(cp)
        if fb_clean is not None:
            fb = fb_clean
        bad = False
        for _, op, ork, _ in kept + added:
            if ork == base:
                continue
            ov = op.intersection(fb).area
            if ov > 0.05 * op.area and ov > 0.05 * fb.area:
                bad = True
                break
        if bad:
            # fall back to the wall-box only when it also does not collide
            fb2, wall_based = close_room_with_walls(plan, cp, lines)
            if fb2 is not None:
                fb_clean = clean_room_poly(fb2)
                if fb_clean is not None:
                    fb2 = fb_clean
                bad2 = False
                for _, op, ork, _ in kept + added:
                    if ork == base:
                        continue
                    ov = op.intersection(fb2).area
                    if ov > 0.05 * op.area and ov > 0.05 * fb2.area:
                        bad2 = True
                        break
                if not bad2:
                    fb = fb2
                    bad = False
        if not bad:
            snapped = snap_room_to_graph(fb, g_nodes, g_edges, tol=2.0)
            if snapped is not None:
                bad2 = False
                occ_others = unary_union([op for _, op, ork, _ in kept + added if ork != base])
                overlap_piece = snapped.intersection(occ_others)
                if not overlap_piece.is_empty and overlap_piece.area > 1.0:
                    trimmed = snapped.difference(overlap_piece.buffer(0.01))
                    if trimmed.geom_type == "MultiPolygon":
                        trimmed = max(trimmed.geoms, key=lambda g: g.area)
                    if trimmed.geom_type == "Polygon" and not trimmed.is_empty and \
                            trimmed.area > 0.9 * snapped.area:
                        snapped = trimmed
                for _, op, ork, _ in kept + added:
                    if ork == base:
                        continue
                    ov = op.intersection(snapped).area
                    if ov > 0.05 * op.area and ov > 0.05 * snapped.area:
                        bad2 = True
                        break
                if not bad2:
                    fb = snapped
            added.append((fb.area, fb, base, 0.5 if wall_based else -1.0))
            _LOG.debug(f"  closed room with walls: {name} "
                  f"(wall-based={wall_based}, area {fb.area:.1f})")
        else:
            _LOG.debug(f"  skipped overlapping fallback for {name} (area {fb.area:.1f})")
    return kept + added


def merge_duplicate_room_faces(plan, features, room_order, kept):
    """When one corner ring is covered by several kept faces of the same
    type, a skeleton wall that does not exist in the plan split one room in
    two. Merge those faces back into a single room."""
    corner_rooms = room_rings_from_features(features, room_order, plan)
    if not corner_rooms:
        return kept
    merged = []
    drop_ids = set()
    for name, cp in corner_rooms.items():
        base = name.rsplit("_", 1)[0]
        if base not in ROOM_KEYS:
            continue
        hits = [f for f in kept
                if f[2] == base and f[1].intersection(cp).area / cp.area > 0.3]
        if len(hits) < 2:
            continue
        union = unary_union([f[1] for f in hits])
        if union.geom_type == "MultiPolygon":
            union = max(union.geoms, key=lambda g: g.area)
        if union.geom_type != "Polygon" or union.is_empty or union.area < 10.0:
            continue
        clean = clean_room_poly(union)
        if clean is None:
            clean = orient(union, sign=1.0)
        _LOG.debug(f"  merged {len(hits)} {base} faces for {name} -> area {clean.area:.1f}")
        merged.append((clean.area, clean, base, 0.9))
        for h in hits:
            drop_ids.add(id(h))
    if not merged:
        return kept
    keep = [f for f in kept if id(f) not in drop_ids]
    return keep + merged


def add_door_centerline_walls(plan, features, room_order, lines, _faces=None):
    """
    When one skeleton face merges two door-connected corner rooms, add a wall
    along the door's centerline so they split.
    """
    corner_rooms = room_rings_from_features(features, room_order, plan)
    if not corner_rooms:
        return lines
    doors = [g for k in ("door", "front_door") for g in polys(plan, k)]
    max_area = max(cp.area for cp in corner_rooms.values())
    extra = []
    biggest_face = max((f[0] for f in _faces or []), default=0.0)
    for door in doors:
        coords = np.array(door.exterior.coords[:-1])
        edges = []
        for i in range(len(coords)):
            a, b = coords[i], coords[(i + 1) % len(coords)]
            edges.append((np.linalg.norm(b - a), a, b))
        edges.sort(key=lambda e: -e[0])
        if len(edges) < 2:
            continue
        mid1 = (edges[0][1] + edges[0][2]) / 2.0
        mid2 = (edges[1][1] + edges[1][2]) / 2.0
        center = (mid1 + mid2) / 2.0
        direction = edges[0][2] - edges[0][1]

        touched = [
            name for name, cp in corner_rooms.items()
            if name.rsplit("_", 1)[0] not in ("living", "balcony", "front_door")
            and door.distance(cp) < 1.0
        ]
        if len(touched) < 2:
            continue

        for area, face in _faces or []:
            if area >= biggest_face or area > 3.0 * max_area:
                continue
            if not all(face.intersection(corner_rooms[t]).area / corner_rooms[t].area > 0.3
                       for t in touched):
                continue
            if abs(direction[0]) >= abs(direction[1]):
                lo = min(min(edges[0][1][0], edges[0][2][0]), min(edges[1][1][0], edges[1][2][0]))
                hi = max(max(edges[0][1][0], edges[0][2][0]), max(edges[1][1][0], edges[1][2][0]))
                p0 = np.array([lo, center[1]])
                p1 = np.array([hi, center[1]])
            else:
                lo = min(min(edges[0][1][1], edges[0][2][1]), min(edges[1][1][1], edges[1][2][1]))
                hi = max(max(edges[0][1][1], edges[0][2][1]), max(edges[1][1][1], edges[1][2][1]))
                p0 = np.array([center[0], lo])
                p1 = np.array([center[0], hi])

            if abs(p1[0] - p0[0]) >= abs(p1[1] - p0[1]):
                p0[0], p1[0] = face.bounds[0], face.bounds[2]
            else:
                p0[1], p1[1] = face.bounds[1], face.bounds[3]
            wall = LineString([p0, p1])
            if wall.length > 1:
                extra.append(wall)
            _LOG.debug(f"  door centerline wall: {touched[0]} / {touched[1]}")
            break
    return lines + extra


def add_missing_room_edges(plan, features, room_order, lines, only_rooms=None):
    """
    Fill a missing room wall only when corner points AND skeleton/doors agree:
    either the edge's two ends meet existing skeleton walls, or a door sits on
    the edge between two rooms. New walls must not cross another room.
    """
    corner_rooms = room_rings_from_features(features, room_order, plan)
    if not corner_rooms:
        return lines
    wd = float(plan.get("wall_depth") or 4.0)
    doors = [g for k in ("door", "front_door") for g in polys(plan, k)]
    segs = []
    for line in lines:
        c = np.array(line.coords)
        for i in range(len(c) - 1):
            segs.append((c[i], c[i + 1]))

    room_keys = [n for n, cp in corner_rooms.items()
                 if n.rsplit("_", 1)[0] not in ("living", "balcony", "front_door")]
    extra = []
    for name in room_keys:
        if only_rooms is not None and name not in only_rooms:
            continue
        cp = corner_rooms[name]
        base = name.rsplit("_", 1)[0]
        ring = np.array(cp.exterior.coords[:-1])
        gap_tol = max(3.0, wd * 1.2)
        for i in range(len(ring)):
            p0, p1 = ring[i], ring[(i + 1) % len(ring)]
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                continue
            horiz = abs(dy) <= abs(dx)

            covered = False
            for a, b in segs:
                if horiz and abs(a[1] - b[1]) <= abs(a[0] - b[0]):
                    y = (a[1] + b[1]) / 2.0
                    if abs(y - (p0[1] + p1[1]) / 2.0) <= 0.8:
                        w0, w1 = min(a[0], b[0]), max(a[0], b[0])
                        ov = min(max(p0[0], p1[0]), w1) - max(min(p0[0], p1[0]), w0)
                        if ov >= 0.3 * abs(dx):
                            covered = True
                            break
                elif not horiz and abs(a[0] - b[0]) <= abs(a[1] - b[1]):
                    x = (a[0] + b[0]) / 2.0
                    if abs(x - (p0[0] + p1[0]) / 2.0) <= 0.8:
                        w0, w1 = min(a[1], b[1]), max(a[1], b[1])
                        ov = min(max(p0[1], p1[1]), w1) - max(min(p0[1], p1[1]), w0)
                        if ov >= 0.3 * abs(dy):
                            covered = True
                            break
            if covered:
                continue

            supports = []
            for ep in (p0, p1):
                sup = None
                for a, b in segs:
                    seg = LineString([a, b])
                    if seg.distance(Point(ep)) > gap_tol:
                        continue
                    if horiz and abs(a[0] - b[0]) <= abs(a[1] - b[1]):
                        sup = (a[0] + b[0]) / 2.0
                        break
                    if not horiz and abs(a[1] - b[1]) <= abs(a[0] - b[0]):
                        sup = (a[1] + b[1]) / 2.0
                        break
                supports.append(sup)

            door_support = any(
                dg.distance(cp) < 1.0 and any(
                    dg.distance(corner_rooms[n2]) < 1.0 for n2 in room_keys if n2 != name)
                for dg in doors)

            if supports[0] is not None and supports[1] is not None:
                if horiz:
                    wall = LineString([(supports[0], (p0[1] + p1[1]) / 2.0),
                                       (supports[1], (p0[1] + p1[1]) / 2.0)])
                else:
                    wall = LineString([((p0[0] + p1[0]) / 2.0, supports[0]),
                                       ((p0[0] + p1[0]) / 2.0, supports[1])])
            elif door_support:
                wall = LineString([p0, p1])
            else:
                continue

            bad = False
            for n2 in room_keys:
                if n2 == name:
                    continue
                other = corner_rooms[n2]
                inter = wall.intersection(other)
                if inter.is_empty:
                    continue
                inter_len = inter.length if hasattr(inter, "length") else 0.0
                if wall.intersection(other.buffer(-0.5)).length > 0.5:
                    bad = True
                    break
            if not bad and wall.length > 1:
                extra.append(wall)

        # repair broken collinear ring edges (same axis, separated by a gap)
        vsegs = []
        hsegs = []
        for i in range(len(ring)):
            a, b = ring[i], ring[(i + 1) % len(ring)]
            if abs(a[0] - b[0]) <= 0.5 and abs(a[1] - b[1]) > 0.5:
                vsegs.append((a, b))
            elif abs(a[1] - b[1]) <= 0.5 and abs(a[0] - b[0]) > 0.5:
                hsegs.append((a, b))

        def repair_vertical():
            groups = {}
            for a, b in vsegs:
                x = round((a[0] + b[0]) / 2.0, 1)
                groups.setdefault(x, []).append((a, b))
            hwalls = []
            for a, b in segs:
                if abs(a[1] - b[1]) <= abs(a[0] - b[0]):
                    hwalls.append(((a[1] + b[1]) / 2.0, min(a[0], b[0]), max(a[0], b[0])))
            for x, grp in groups.items():
                if len(grp) < 2:
                    continue
                spans = sorted([(min(a[1], b[1]), max(a[1], b[1])) for a, b in grp])
                for k in range(len(spans) - 1):
                    y0, y1 = spans[k]
                    y2, y3 = spans[k + 1]
                    if y2 - y1 < 1.0:
                        continue
                    best = None
                    for by, bx0, bx1 in hwalls:
                        if by > y1:
                            continue
                        for ay, ax0, ax1 in hwalls:
                            if ay < y2:
                                continue
                            ox0, ox1 = max(bx0, ax0), min(bx1, ax1)
                            if ox1 - ox0 < 1.0:
                                continue
                            score = abs((ox0 + ox1) / 2.0 - x) + abs(y1 - by) + abs(ay - y2)
                            if best is None or score < best[0]:
                                best = (score, by, ay, ox0, ox1)
                    if best is not None:
                        by, ay, ox0, ox1 = best[1:]
                        wx = min(max(x, ox0), ox1)
                        wall = LineString([(wx, by), (wx, ay)])
                        bad = False
                        for n2 in room_keys:
                            if n2 == name:
                                continue
                            other = corner_rooms[n2]
                            inter = wall.intersection(other)
                            if inter.is_empty:
                                continue
                            inter_len = inter.length if hasattr(inter, "length") else 0.0
                            if wall.intersection(other.buffer(-0.5)).length > 0.5:
                                bad = True
                                break
                        if not bad and wall.length > 1:
                            extra.append(wall)

        def repair_horizontal():
            groups = {}
            for a, b in hsegs:
                y = round((a[1] + b[1]) / 2.0, 1)
                groups.setdefault(y, []).append((a, b))
            vwalls = []
            for a, b in segs:
                if abs(a[0] - b[0]) <= abs(a[1] - b[1]):
                    vwalls.append(((a[0] + b[0]) / 2.0, min(a[1], b[1]), max(a[1], b[1])))
            for y, grp in groups.items():
                if len(grp) < 2:
                    continue
                spans = sorted([(min(a[0], b[0]), max(a[0], b[0])) for a, b in grp])
                for k in range(len(spans) - 1):
                    x0, x1 = spans[k]
                    x2, x3 = spans[k + 1]
                    if x2 - x1 < 1.0:
                        continue
                    best = None
                    for lx, ly0, ly1 in vwalls:
                        if lx > x1:
                            continue
                        for rx, ry0, ry1 in vwalls:
                            if rx < x2:
                                continue
                            oy0, oy1 = max(ly0, ry0), min(ly1, ry1)
                            if oy1 - oy0 < 1.0:
                                continue
                            score = abs((oy0 + oy1) / 2.0 - y) + abs(x1 - lx) + abs(rx - x2)
                            if best is None or score < best[0]:
                                best = (score, lx, rx, oy0, oy1)
                    if best is not None:
                        lx, rx, oy0, oy1 = best[1:]
                        wy = min(max(y, oy0), oy1)
                        wall = LineString([(lx, wy), (rx, wy)])
                        bad = False
                        for n2 in room_keys:
                            if n2 == name:
                                continue
                            other = corner_rooms[n2]
                            inter = wall.intersection(other)
                            if inter.is_empty:
                                continue
                            inter_len = inter.length if hasattr(inter, "length") else 0.0
                            if wall.intersection(other.buffer(-0.5)).length > 0.5:
                                bad = True
                                break
                        if not bad and wall.length > 1:
                            extra.append(wall)

        repair_vertical()
        repair_horizontal()

    return lines + extra


def compute_unmatched(plan, features, room_order, faces):
    corner_rooms = room_rings_from_features(features, room_order, plan)
    biggest = max((f[0] for f in faces), default=0.0)
    labeled = label_faces(faces, plan)
    unmatched = set()
    for name, cp in corner_rooms.items():
        base = name.rsplit("_", 1)[0]
        if base in ("living", "balcony", "front_door"):
            continue
        covered = False
        for area, poly, rk, ratio in labeled:
            if area >= biggest or rk != base:
                continue
            if poly.intersection(cp).area / cp.area > 0.5:
                covered = True
                break
        if not covered:
            unmatched.add(name)
    return unmatched
