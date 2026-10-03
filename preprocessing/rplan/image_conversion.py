"""Convert RPLAN 4-channel PNGs into ResPlan-style pickle plans.

Output plans use the shared ResPlan geometry and topology fields:
conn (node_names/room_types/node_areas/edge_list/edge_type_names),
per-type shapely geometries, inner boundary, front_door, and areas.
"""

import sys as _sys
from pathlib import Path as _Path
_RELEASE_ROOT = _Path(__file__).resolve().parents[2]
if str(_RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_RELEASE_ROOT))


import math
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import stats
from scipy import ndimage
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from skimage import draw, feature, measure
from skimage.morphology import skeletonize
try:
    from skimage.segmentation import watershed

    HAS_WATERSHED = True
except Exception:
    HAS_WATERSHED = False
from tqdm import tqdm


MAIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MAIN_DIR))
try:
    from preprocessing.resplan import wall_reconstruction as skel

    HAS_SKELETON = True
except Exception:
    HAS_SKELETON = False


import os
TOOLBOX_PATH = os.environ.get("RPLAN_TOOLBOX_PATH", "")
if TOOLBOX_PATH:
    sys.path.insert(0, TOOLBOX_PATH)
try:
    from rplan.floorplan import Floorplan

    HAS_TOOLBOX = True
except Exception:
    HAS_TOOLBOX = False

try:
    from rplan.align_python import align_fp_gt

    HAS_ALIGN = True
except Exception:
    HAS_ALIGN = False


ROOM_LABEL_TO_TYPE = {
    0: "living",    # LivingRoom
    1: "bedroom",   # MasterRoom
    2: "kitchen",   # Kitchen
    3: "bathroom",  # Bathroom
    4: "living",    # DiningRoom -> living
    5: "bedroom",   # ChildRoom
    6: "bedroom",   # StudyRoom
    7: "bedroom",   # SecondRoom
    8: "bedroom",   # GuestRoom
    9: "balcony",   # Balcony
    10: "balcony",  # Entrance -> balcony
    11: "storage",  # Storage
    12: "storage",  # Wall-in -> storage
}
ROOM_CATEGORIES = set(ROOM_LABEL_TO_TYPE.keys())
FRONT_DOOR_CATEGORY = 15
DOOR_CATEGORY = 17
ADJACENCY_MIN_LENGTH = 1.0


def contour_polygon(mask, epsilon=0.0):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    contour = cv2.approxPolyDP(contour, epsilon, True)
    points = contour[:, 0, :].astype(float)
    if len(points) < 3:
        return None
    polygon = Polygon(points)
    if polygon.is_valid and polygon.area > 1.0 and not polygon.is_empty:
        return polygon
    return None


def clean_room_poly(poly, grid=0.05, tol=0.15):
    """Legacy clean: snap axis edges flat, remove diagonal chamfer steps."""
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
                        np.linalg.norm(b - a) > 0.3 and \
                        np.linalg.norm(b - c) > 0.3:
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


def orthogonalize_poly(poly, max_step=0.5):
    """Legacy: split short slanted steps into L corners."""
    coords = np.array(poly.exterior.coords[:-1], dtype=float)
    changed = True
    while changed and len(coords) > 3:
        changed = False
        n = len(coords)
        for i in range(n):
            p0 = coords[(i - 1) % n]
            p1 = coords[i]
            p2 = coords[(i + 1) % n]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                coords = np.delete(coords, i, axis=0)
                changed = True
                break
            if min(abs(dx), abs(dy)) > 0.02 and \
                    min(abs(dx), abs(dy)) <= max_step and \
                    max(abs(dx), abs(dy)) <= 4.0:
                e1x, e1y = p1[0] - p0[0], p1[1] - p0[1]
                e2x, e2y = p2[0] - p1[0], p2[1] - p1[1]
                if abs(e1y) <= abs(e1x) and abs(e2x) <= abs(e2y):
                    corner = (p1[0], p2[1])
                elif abs(e1x) <= abs(e1y) and abs(e2y) <= abs(e2x):
                    corner = (p2[0], p1[1])
                else:
                    continue
                cand_a = np.insert(coords, i + 1, corner, axis=0)
                cand_b = list(coords)
                if abs(e1y) <= abs(e1x) and abs(e2x) <= abs(e2y):
                    other = (p2[0], p1[1])
                else:
                    other = (p1[0], p2[1])
                cand_b = np.insert(np.asarray(cand_b), i + 1, other, axis=0)
                if Polygon(cand_a).area > Polygon(cand_b).area:
                    corner = other
                coords = np.insert(coords, i + 1, corner, axis=0)
                changed = True
                break
    if len(coords) < 3:
        return None
    p2 = Polygon(coords)
    if not p2.is_valid or p2.is_empty:
        return None
    return p2


def shared_boundary_length(a, b):
    inter = a.boundary.intersection(b.boundary)
    return 0.0 if inter.is_empty else float(inter.length)


def room_components(category, instance, room_mask):
    labels = np.unique(instance[room_mask])
    components = []
    for label in labels:
        if label <= 0:
            continue
        mask = (instance == label) & room_mask
        if mask.sum() < 50:
            continue
        mask = ndimage.binary_dilation(mask, iterations=1)
        room_category = int(stats.mode(category[mask])[0])
        if room_category not in ROOM_CATEGORIES:
            continue
        polygon = contour_polygon(mask)
        if polygon is None:
            continue
        polygon = orthogonalize_poly(polygon)
        if polygon is not None:
            polygon = clean_room_poly(polygon)
        if polygon is None:
            continue
        components.append((label, ROOM_LABEL_TO_TYPE[room_category], polygon))
    return components


def grow_room_masks(instance, inside, boundary):
    """Grow rooms into interior-wall pixels so polygons share edges."""
    room_mask = (inside > 0) & (instance > 0)
    labels_img = np.where(room_mask, instance, 0)
    distance, indices = ndimage.distance_transform_edt(
        labels_img == 0,
        return_indices=True,
    )
    nearest_label = instance[indices[0], indices[1]]
    interior_wall = (labels_img == 0) & (
        (inside > 0) | (boundary == 127)
    )
    grown = labels_img.copy()
    grown[interior_wall] = nearest_label[interior_wall]
    return grown


def aligned_toolbox_components(fp):
    """Use RPLAN-Toolbox aligned room boundaries as exact shared-wall polygons."""
    if not HAS_ALIGN:
        return None
    data = fp.to_dict()
    _, _, room_boundaries = align_fp_gt(
        data["boundary"],
        data["boxes"],
        data["types"],
        data["edges"],
    )
    if len(room_boundaries) != len(data["types"]):
        return None
    components = []
    for label, ring in enumerate(room_boundaries, start=1):
        coords = np.asarray(ring, dtype=float)
        if len(coords) < 4:
            return None
        poly = Polygon(coords)
        if not poly.is_valid or poly.is_empty:
            poly = poly.buffer(0)
        if poly.is_empty:
            return None
        if poly.geom_type == "MultiPolygon":
            pieces = [p for p in poly.geoms if not p.is_empty and p.area > 1.0]
            if not pieces:
                return None
            poly = max(pieces, key=lambda p: p.area)
        if poly.area < 1.0:
            return None
        room_type = ROOM_LABEL_TO_TYPE.get(int(data["types"][label - 1]))
        if room_type is None:
            return None
        components.append((label, room_type, orient(poly, sign=1.0)))
    return components


def resolve_overlaps(components):
    """Subtract smaller/contained rooms from larger ones so polygons are disjoint."""
    items = sorted(
        [
            (label, room_type, polygon, polygon.area)
            for label, room_type, polygon in components
        ],
        key=lambda item: item[3],
    )
    kept = []
    for label, room_type, polygon, _ in items:
        for _, _, other in kept:
            if not polygon.intersects(other):
                continue
            inter = polygon.intersection(other)
            if inter.area <= 0.5:
                continue
            polygon = polygon.difference(other)
            if polygon.is_empty:
                break
            if polygon.geom_type == "MultiPolygon":
                parts = [part for part in polygon.geoms if part.area > 1.0]
                if not parts:
                    polygon = None
                    break
                polygon = (
                    parts[0]
                    if len(parts) == 1
                    else MultiPolygon(parts)
                )
            if polygon.area < 1.0:
                polygon = None
                break
        if polygon is not None and not polygon.is_empty:
            kept.append((label, room_type, orient(polygon, sign=1.0)))
    return kept


def snap_polygon(poly, grid=1.0):
    """Round polygon coordinates back to the pixel grid."""
    coords = np.round(np.asarray(poly.exterior.coords) / grid) * grid
    pts = []
    for p in coords:
        if not pts or np.linalg.norm(p - pts[-1]) > 1e-6:
            pts.append(p)
    if len(pts) > 1 and np.linalg.norm(pts[0] - pts[-1]) <= 1e-6:
        pts.pop()
    if len(pts) < 3:
        return None
    holes = []
    for ring in poly.interiors:
        coords = np.round(np.asarray(ring.coords) / grid) * grid
        hpts = []
        for p in coords:
            if not hpts or np.linalg.norm(p - hpts[-1]) > 1e-6:
                hpts.append(p)
        if len(hpts) > 1 and np.linalg.norm(hpts[0] - hpts[-1]) <= 1e-6:
            hpts.pop()
        if len(hpts) >= 3:
            holes.append(hpts)
    snapped = Polygon(pts, holes)
    if not snapped.is_valid:
        snapped = snapped.buffer(0)
    if snapped.is_empty:
        return None
    return orient(snapped, sign=1.0)


def fill_inner_gaps(components, inner):
    """Assign uncovered interior area to the room sharing its longest boundary."""
    if inner is None or inner.is_empty:
        return components
    result = list(components)
    union = unary_union([poly.buffer(0) for _, _, poly in result])
    uncovered = inner.buffer(0).difference(union)
    if uncovered.is_empty:
        return result
    pieces = (
        uncovered.geoms
        if uncovered.geom_type == "MultiPolygon"
        else [uncovered]
    )
    for piece in pieces:
        if piece.area <= 0.5:
            continue
        best_index = None
        best_len = 0.0
        best_dist = float("inf")
        for index, (_, _, poly) in enumerate(result):
            shared = piece.boundary.intersection(poly.boundary).length
            dist = piece.distance(poly)
            if shared > best_len or (
                shared == best_len and dist < best_dist
            ):
                best_len = shared
                best_dist = dist
                best_index = index
        if best_index is None:
            continue
        label, room_type, poly = result[best_index]
        merged = unary_union([poly, piece])
        result[best_index] = (label, room_type, merged)
    return result


def pixel_to_world_sized(chains, bounds, size):
    x1, y1, x2, y2 = bounds
    out = []
    for chain in chains:
        wx = x1 + chain[:, 1] / (size - 1) * (x2 - x1)
        wy = y1 + chain[:, 0] / (size - 1) * (y2 - y1)
        line = LineString(
            np.stack([wx, wy], axis=1)
        ).simplify(0.5, preserve_topology=False)
        out.append(line)
    return [line for line in out if line.length > 0.01]


def rooms_from_walls(category, inside, instance):
    """Wall mask -> skeleton -> graph faces -> room polygons (legacy path)."""
    wall = np.isin(category, [14, 15, 16, 17]).astype(np.uint8)
    skeleton = skeletonize(wall)
    chains = skel.trace_chains(skeleton)
    lines = [LineString(chain) for chain in chains]
    lines = [
        LineString(line)
        for line in skel.split_and_merge_walls(lines)
    ]
    lines = skel.straighten_walls(lines)
    nodes, edges = skel.build_graph(lines)
    edges = skel.close_corner_gaps(nodes, edges)
    edges = skel.prune_dangling_edges(nodes, edges)
    faces = skel.planar_faces(nodes, edges)

    results = []
    for area, polygon in faces:
        if area < 50:
            continue
        coords = np.asarray(polygon.exterior.coords)
        rows, cols = draw.polygon(
            coords[:, 1], coords[:, 0], shape=instance.shape
        )
        valid = inside[rows, cols] > 0
        if valid.sum() < 20:
            continue
        labels_in_face = instance[rows, cols][valid]
        counts = np.bincount(labels_in_face)
        best_label = int(np.argmax(counts))
        best_count = int(counts[best_label])
        if best_label <= 0 or best_count / valid.sum() < 0.7:
            continue
        matching = valid & (instance[rows, cols] == best_label)
        room_categories = category[rows, cols][matching]
        room_categories = room_categories[
            np.isin(room_categories, list(ROOM_CATEGORIES))
        ]
        if len(room_categories) == 0:
            continue
        room_category = int(stats.mode(room_categories)[0])
        results.append(
            (area, polygon, best_label, room_category)
        )
    return results


def wall_graph(category):
    """Wall mask -> skeleton -> planar graph nodes/edges (legacy path)."""
    wall = np.isin(category, [14, 15, 16, 17]).astype(np.uint8)
    skeleton = skeletonize(wall)
    chains = skel.trace_chains(skeleton)
    lines = [LineString(chain) for chain in chains]
    lines = [
        LineString(line)
        for line in skel.split_and_merge_walls(lines)
    ]
    lines = skel.straighten_walls(lines)
    nodes, edges = skel.build_graph(lines)
    edges = skel.close_corner_gaps(nodes, edges)
    return nodes, edges


def door_components(category):
    mask = (category == DOOR_CATEGORY).astype(np.uint8)
    labels = measure.label(mask, connectivity=2)
    doors = []
    for region in measure.regionprops(labels):
        y0, x0, y1, x1 = region.bbox
        if region.area < 10:
            continue
        doors.append((y0, x0, y1, x1, labels))
    return doors


def door_archs(category):
    """Split interior-door pixels into individual arch segments."""
    mask = (category == DOOR_CATEGORY).astype(np.uint8)
    if not HAS_WATERSHED:
        labels = measure.label(mask, connectivity=2)
    else:
        distance = ndimage.distance_transform_cdt(mask)
        local_maxi = (distance > 1).astype(np.uint8)
        local_maxi[feature.corner_harris(local_maxi) > 0] = 0
        markers = measure.label(local_maxi)
        labels = watershed(
            -distance,
            markers,
            mask=mask,
            connectivity=8,
        )
    archs = []
    for region in measure.regionprops(labels):
        if region.area < 10:
            continue
        archs.append(tuple(region.bbox))
    return archs


def door_room_pairs(instance, room_labels, archs):
    """Map each door arch to all room pairs it bridges."""
    room_labels = set(room_labels)
    pairs = set()

    def strip_label(region):
        counts = Counter(
            int(label)
            for label in region.flat
            if int(label) in room_labels
        )
        if not counts:
            return None
        return counts.most_common(1)[0][0]

    for y0, x0, y1, x1 in archs:
        horizontal = (x1 - x0) >= (y1 - y0)
        if horizontal:
            for x in range(max(x0, 0), min(x1, instance.shape[1])):
                top = strip_label(
                    instance[
                        max(y0 - 2, 0):y0,
                        max(x - 1, 0):min(x + 2, instance.shape[1]),
                    ]
                )
                bottom = strip_label(
                    instance[
                        y1:min(y1 + 2, instance.shape[0]),
                        max(x - 1, 0):min(x + 2, instance.shape[1]),
                    ]
                )
                if top is not None and bottom is not None and top != bottom:
                    pairs.add(tuple(sorted((top, bottom))))
        else:
            for y in range(max(y0, 0), min(y1, instance.shape[0])):
                left = strip_label(
                    instance[
                        max(y - 1, 0):min(y + 2, instance.shape[0]),
                        max(x0 - 2, 0):x0,
                    ]
                )
                right = strip_label(
                    instance[
                        max(y - 1, 0):min(y + 2, instance.shape[0]),
                        x1:min(x1 + 2, instance.shape[1]),
                    ]
                )
                if left is not None and right is not None and left != right:
                    pairs.add(tuple(sorted((left, right))))
    return sorted(pairs)


def door_room_pair(instance, room_labels, y0, x0, y1, x1):
    pairs = set()
    for row in range(y0, y1):
        for col in (max(x0 - 2, 0), min(x1 + 1, instance.shape[1] - 1)):
            label = int(instance[row, col])
            if label in room_labels:
                pairs.add(label)
    for col in range(x0, x1):
        for row in (max(y0 - 2, 0), min(y1 + 1, instance.shape[0] - 1)):
            label = int(instance[row, col])
            if label in room_labels:
                pairs.add(label)
    if len(pairs) == 2:
        return tuple(sorted(pairs))
    return None


def plan_from_image(path, id_offset=200000):
    with Image.open(path) as source_image:
        image = np.asarray(source_image)
    boundary = image[..., 0]
    category = image[..., 1]
    instance = image[..., 2]
    inside = image[..., 3]
    room_mask = inside > 0

    grown = grow_room_masks(instance, inside, boundary)
    components = room_components(category, grown, grown > 0)
    fp = None
    toolbox_used = False
    if HAS_TOOLBOX and len(components) >= 2:
        try:
            fp = Floorplan(str(path))
            aligned = aligned_toolbox_components(fp)
            if aligned is not None:
                resolved = resolve_overlaps(aligned)
                if resolved:
                    components = resolved
                toolbox_used = True
        except Exception:
            fp = None
    if not toolbox_used and len(components) >= 2:
        resolved = resolve_overlaps(components)
        if resolved:
            components = resolved
    if HAS_SKELETON and len(components) >= 2 and not toolbox_used:
        try:
            nodes, edges = wall_graph(category)
            snapped_components = []
            for label, room_type, polygon in components:
                snapped = skel.snap_room_to_graph(
                    polygon, nodes, edges, tol=1.5
                )
                if snapped is not None:
                    polygon = snapped
                snapped_components.append((label, room_type, polygon))
            components = snapped_components
        except Exception:
            pass
    if not components:
        return None

    front_polygon = None
    interior = None
    toolbox_edges = None
    if fp is not None:
        try:
            if len(fp.exterior_boundary) > 0:
                exterior = np.asarray(fp.exterior_boundary)
                interior = Polygon(exterior[:, [1, 0]][:, :2])
            if fp.front_door is not None:
                y0, x0, y1, x1 = fp.front_door
                front_polygon = Polygon(
                    [
                        [x0, y0],
                        [x1, y0],
                        [x1, y1],
                        [x0, y1],
                    ]
                )
        except Exception:
            interior = None
            front_polygon = None
        try:
            fp._get_archs()
            fp._get_graph()
            toolbox_edges = fp.graph
        except Exception:
            toolbox_edges = None

    if interior is None:
        interior = contour_polygon(room_mask.astype(np.uint8), epsilon=0.8)
    if front_polygon is None:
        front_mask = category == FRONT_DOOR_CATEGORY
        front_polygon = contour_polygon(
            front_mask.astype(np.uint8), epsilon=0.5
        )
    if interior is None:
        return None

    repaired = []
    for label, room_type, polygon in components:
        snapped = snap_polygon(polygon)
        if snapped is not None:
            repaired.append((label, room_type, snapped))
    if repaired:
        components = repaired
    components = resolve_overlaps(components) or components
    components = fill_inner_gaps(components, interior)
    components = resolve_overlaps(components) or components
    snapped_again = []
    for label, room_type, polygon in components:
        snapped = snap_polygon(polygon)
        if snapped is not None:
            snapped_again.append((label, room_type, snapped))
    if snapped_again:
        components = snapped_again
    components = fill_inner_gaps(components, interior)
    components = resolve_overlaps(components) or components
    components = fill_inner_gaps(components, interior)
    components = resolve_overlaps(components) or components
    if not components:
        return None

    rooms = []
    for label, room_type, polygon in components:
        rooms.append(
            {
                "label": label,
                "type": room_type,
                "polygon": polygon,
                "area": float(polygon.area),
            }
        )

    # Order node names like ResPlan: grouped by type, alphabetical type order.
    room_order = sorted(
        enumerate(rooms),
        key=lambda item: (rooms[item[0]]["type"], item[0]),
    )
    sorted_rooms = [rooms[i] for i, _ in room_order]
    room_labels = [r["label"] for r in sorted_rooms]

    type_counter = Counter()
    node_names = []
    room_types = []
    node_areas = []
    geometry_by_type = {}
    for room in sorted_rooms:
        room_type = room["type"]
        index = type_counter[room_type]
        type_counter[room_type] += 1
        node_names.append(f"{room_type}_{index}")
        room_types.append(room_type)
        node_areas.append(room["area"])
        geometry_by_type.setdefault(room_type, []).append(room["polygon"])

    # Front door as a node, like ResPlan.
    if front_polygon is not None:
        node_names.append("front_door_0")
        room_types.append("front_door")
        node_areas.append(float(front_polygon.area))
        geometry_by_type["front_door"] = [front_polygon]
    else:
        front_polygon = None

    edges = []
    edge_types = []

    label_to_sorted = {
        room["label"]: index for index, room in enumerate(sorted_rooms)
    }
    try:
        arch_pairs = door_room_pairs(
            instance,
            room_labels,
            door_archs(category),
        )
    except Exception:
        arch_pairs = []
    for pair in arch_pairs:
        if pair[0] not in label_to_sorted or pair[1] not in label_to_sorted:
            continue
        i = label_to_sorted[pair[0]]
        j = label_to_sorted[pair[1]]
        if (i, j) in edges or (j, i) in edges:
            continue
        edges.append([i, j])
        edge_types.append("via_door")
    if toolbox_edges is not None:
        for row in toolbox_edges:
            u, v, _, edge_type, _ = row
            label_u = int(u) + 1
            label_v = int(v) + 1
            if label_u not in label_to_sorted or label_v not in label_to_sorted:
                continue
            i = label_to_sorted[label_u]
            j = label_to_sorted[label_v]
            if int(edge_type) != 1:
                continue
            if (i, j) in edges or (j, i) in edges:
                continue
            edges.append([i, j])
            edge_types.append("via_door")
    def has_connection(index):
        return any(
            (edge[0] == index or edge[1] == index)
            and edge_type in ("via_door", "via_opening", "direct")
            for edge, edge_type in zip(edges, edge_types)
        )

    for index, room_type in enumerate(room_types):
        if room_type == "front_door":
            continue
        if has_connection(index):
            continue
        candidates = []
        for j in range(len(sorted_rooms)):
            if j == index:
                continue
            shared = shared_boundary_length(
                sorted_rooms[index]["polygon"],
                sorted_rooms[j]["polygon"],
            )
            if shared >= ADJACENCY_MIN_LENGTH:
                candidates.append((shared, j))
        if not candidates:
            nearest = min(
                range(len(sorted_rooms)),
                key=lambda j: (
                    sorted_rooms[index]["polygon"].distance(
                        sorted_rooms[j]["polygon"]
                    )
                    if j != index
                    else float("inf")
                ),
            )
            if nearest == index:
                continue
            best_index = nearest
        else:
            living_candidates = [
                (shared, j)
                for shared, j in candidates
                if room_types[j] == "living"
            ]
            connected_candidates = [
                (shared, j)
                for shared, j in candidates
                if has_connection(j)
            ]
            if living_candidates:
                best_index = max(
                    living_candidates, key=lambda item: item[0]
                )[1]
            elif connected_candidates:
                best_index = max(
                    connected_candidates, key=lambda item: item[0]
                )[1]
            else:
                best_index = max(candidates, key=lambda item: item[0])[1]
        existing = next(
            (
                edge_index
                for edge_index, edge in enumerate(edges)
                if set(edge) == {index, best_index}
            ),
            None,
        )
        if existing is not None:
            edge_types[existing] = "via_opening"
        else:
            edges.append([index, best_index])
            edge_types.append("via_opening")
    for i in range(len(sorted_rooms)):
        for j in range(i + 1, len(sorted_rooms)):
            if any(set([i, j]) == set(edge) for edge in edges):
                continue
            shared = shared_boundary_length(
                sorted_rooms[i]["polygon"],
                sorted_rooms[j]["polygon"],
            )
            if shared >= ADJACENCY_MIN_LENGTH:
                edges.append([i, j])
                edge_types.append("adjacency")

    living_indices = [
        i for i, room_type in enumerate(room_types) if room_type == "living"
    ]
    if front_polygon is not None and living_indices:
        front_idx = node_names.index("front_door_0")
        living_idx = min(
            living_indices,
            key=lambda i: (
                front_polygon.distance(sorted_rooms[i]["polygon"]),
                -sorted_rooms[i]["polygon"].area,
            ),
        )
        if (living_idx, front_idx) not in edges and (
            front_idx,
            living_idx,
        ) not in edges:
            edges.append([living_idx, front_idx])
            edge_types.append("direct")

    living_indices = [
        i for i, room_type in enumerate(room_types) if room_type == "living"
    ]
    if len(living_indices) > 1:
        keep = living_indices[0]
        dropped = living_indices[1:]
        living_polys = [
            sorted_rooms[i]["polygon"]
            for i in living_indices
        ]
        merged = unary_union(living_polys)
        geometry_by_type["living"] = [merged]
        node_areas[keep] = float(merged.area)
        index_map = {index: keep for index in dropped}
        new_edges = []
        new_edge_types = []
        for (a, b), edge_type in zip(edges, edge_types):
            a = index_map.get(a, a)
            b = index_map.get(b, b)
            if a == b:
                continue
            if (a, b) in new_edges or (b, a) in new_edges:
                continue
            new_edges.append([a, b])
            new_edge_types.append(edge_type)
        edges = new_edges
        edge_types = new_edge_types
        node_names = [
            name
            for index, name in enumerate(node_names)
            if index not in dropped
        ]
        room_types = [
            room_type
            for index, room_type in enumerate(room_types)
            if index not in dropped
        ]
        node_areas = [
            area
            for index, area in enumerate(node_areas)
            if index not in dropped
        ]

    plan = {
        "id": int(Path(path).stem) + id_offset,
        "inner": interior,
        "front_door": front_polygon,
        "conn": {
            "node_names": node_names,
            "room_types": room_types,
            "node_areas": node_areas,
            "edge_list": edges,
            "edge_type_names": edge_types,
            "num_rooms": len(node_names),
        },
        "net_area": sum(room["area"] for room in sorted_rooms),
        "area": sum(room["area"] for room in sorted_rooms),
        "rooms_area": sum(room["area"] for room in sorted_rooms),
        "net_area_rebuilt": sum(room["area"] for room in sorted_rooms),
        "balcony_area": sum(
            room["area"]
            for room in sorted_rooms
            if room["type"] == "balcony"
        ),
        "living_area": sum(
            room["area"]
            for room in sorted_rooms
            if room["type"] == "living"
        ),
        "missing_types": [],
        "reconstruct_failed": False,
    }
    for room_type, polygons in geometry_by_type.items():
        plan[room_type] = polygons[0] if len(polygons) == 1 else polygons
    plan["door"] = []
    plan["window"] = []
    return plan

