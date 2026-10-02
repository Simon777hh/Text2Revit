"""
extract_points.py
Extract corner-level features from ResPlan for DPFM-style training.

room_idx alignment:
    0 .. N-1: functional rooms + front_door (from topology order)
    N .. N+M-1: door
"""

import sys as _sys
from pathlib import Path as _Path
_RELEASE_ROOT = _Path(__file__).resolve().parents[2]
if str(_RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_RELEASE_ROOT))


import pickle
import numpy as np
from collections import defaultdict
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString, Point
from resplan_utils import get_geometries, CATEGORY_COLORS
from tqdm import tqdm


# Configuration

ROOM_TYPES = [
    'living', 'kitchen', 'bedroom', 'bathroom', 'balcony', 'storage', 'stair'
]

ROOM_TYPE_TO_IDX = {
    'living': 0,
    'kitchen': 1,
    'bedroom': 2,
    'bathroom': 3,
    'balcony': 4,
    'storage': 5,
    'stair': 6,
}

DOOR_TYPE_IDX = 7
FRONT_DOOR_TYPE_IDX = 8
WINDOW_TYPE_IDX = 9

NUM_TYPES = 10

MAX_CORNERS = 32
MAX_TOTAL_CORNERS = 256
MAX_ROOMS = 64

FEAT_DIM = 2 + NUM_TYPES + MAX_CORNERS + MAX_ROOMS

TYPE_NAMES = {
    0: 'living', 1: 'kitchen', 2: 'bedroom', 3: 'bathroom',
    4: 'balcony', 5: 'storage', 6: 'stair',
    7: 'door', 8: 'front_door', 9: 'window'
}

TYPE_COLORS = {
    'living': CATEGORY_COLORS.get('living', '#d9d9d9'),
    'kitchen': CATEGORY_COLORS.get('kitchen', '#8da0cb'),
    'bedroom': CATEGORY_COLORS.get('bedroom', '#66c2a5'),
    'bathroom': CATEGORY_COLORS.get('bathroom', '#fc8d62'),
    'balcony': CATEGORY_COLORS.get('balcony', '#b3b3b3'),
    'storage': CATEGORY_COLORS.get('storage', '#a37c52'),
    'stair': CATEGORY_COLORS.get('stair', '#9e9ac8'),
    'door': CATEGORY_COLORS.get('door', '#e78ac3'),
    'front_door': CATEGORY_COLORS.get('front_door', '#a63603'),
    'window': CATEGORY_COLORS.get('window', '#a6d854'),
}


# Helpers

def safe_get_geometries(geom):
    """Safely extract geometries from any shapely object."""
    if geom is None:
        return []
    if hasattr(geom, 'is_empty') and geom.is_empty:
        return []
    result = get_geometries(geom)
    if result is None:
        return []
    if isinstance(result, tuple):
        return list(result)
    if isinstance(result, list):
        return result
    if hasattr(result, 'geom_type'):
        return [result]
    return []


def simplify_polygon_coords(coords, tolerance=0.5):
    """Remove collinear points and duplicates."""
    if len(coords) <= 4:
        return coords
    unique_coords = []
    for i, (x, y) in enumerate(coords):
        if i == 0 or (abs(x - coords[i - 1][0]) > 1e-6 or abs(y - coords[i - 1][1]) > 1e-6):
            unique_coords.append((x, y))
    if len(unique_coords) <= 4:
        return unique_coords
    try:
        poly = Polygon(unique_coords)
        if poly.is_valid:
            simplified = poly.simplify(tolerance, preserve_topology=True)
            if simplified.geom_type == 'Polygon':
                return list(simplified.exterior.coords[:-1])
    except Exception:
        pass
    return unique_coords


def normalize_coords(coords, plan):
    """Normalize coordinates to [-1, 1] based on inner boundary."""
    if not coords:
        return coords
    inner = plan.get('inner')
    if inner is not None and not inner.is_empty:
        min_x, min_y, max_x, max_y = inner.bounds
    else:
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
    range_x, range_y = max_x - min_x, max_y - min_y
    if range_x == 0:
        range_x = 1.0
    if range_y == 0:
        range_y = 1.0
    return [
        (2.0 * (x - min_x) / range_x - 1.0, 2.0 * (y - min_y) / range_y - 1.0)
        for x, y in coords
    ]


def get_topology_order(plan):
    """
    Get room order from topology graph using plan_to_graph.

    Returns: list of node names, e.g. ['balcony_0', ..., 'front_door_0', 'kitchen_0', 'living_0']
    """
    from resplan_utils import plan_to_graph

    G = plan_to_graph(plan)

    nodes = []
    for node in G.nodes():
        if node.startswith('front_door'):
            node_type = 'front_door'
        else:
            node_type = node.split('_')[0]

        if node_type in ROOM_TYPES or node_type == 'front_door':
            nodes.append(node)

    # Keep the room order stable across geometry and topology extraction.
    nodes.sort(key=lambda x: x)

    return nodes


# Core extraction (aligned with topology)

def extract_corners_from_plan(plan, room_order, max_corners=MAX_CORNERS):
    """
    Extract corners with room_idx aligned to topology node order.

    room_idx assignment:
        0 .. len(room_order)-1: functional rooms + front_door (from topology)
        len(room_order) .. : door, window (appended at the end)

    Args:
        plan: ResPlan dict
        room_order: list of node names from topology graph
        max_corners: maximum corners per room

    Returns:
        list of corner dicts with room_idx aligned
    """
    all_corners = []

    # Part 1: Collect all polygons per room type from plan

    type_polygons = {}
    for room_type in ROOM_TYPES:
        geom = plan.get(room_type)
        if geom is None:
            continue
        polys = []
        for sub in get_geometries(geom):
            if sub.geom_type == 'Polygon' and sub.is_valid and not sub.is_empty and sub.area > 1.0:
                polys.append(sub)
        type_polygons[room_type] = polys

    # front_door polygons
    fd_polys = []
    fd_geom = plan.get('front_door')
    if fd_geom is not None and not fd_geom.is_empty:
        for sub in get_geometries(fd_geom):
            if sub.geom_type == 'Polygon' and sub.is_valid and not sub.is_empty:
                fd_polys.append(sub)
            elif sub.geom_type == 'LineString' and not sub.is_empty:
                # Convert LineString to thin polygon for corner extraction
                fd_polys.append(sub.buffer(0.5))
    type_polygons['front_door'] = fd_polys

    functional_room_count = len(room_order)

    # Part 2: Extract corners for each node in topology order

    for room_idx, node_name in enumerate(room_order):
        node_type = node_name.split('_')[0]
        if node_type == 'front':
            node_type = 'front_door'

        # Find which polygon of this type corresponds to this node
        type_nodes = [n for n in room_order if n.startswith(node_type + '_') or
                     (node_type == 'front_door' and n.startswith('front_door_'))]
        type_pos = type_nodes.index(node_name) if node_name in type_nodes else 0

        polys = type_polygons.get(node_type, [])
        if type_pos >= len(polys):
            continue

        poly = polys[type_pos]

        # Get coordinates (handle both Polygon and LineString)
        if poly.geom_type == 'Polygon':
            coords = list(poly.exterior.coords[:-1])
        elif poly.geom_type == 'LineString':
            coords = list(poly.coords)
        else:
            continue

        coords = simplify_polygon_coords(coords)

        # Get type index
        if node_type in ROOM_TYPE_TO_IDX:
            type_idx_val = ROOM_TYPE_TO_IDX[node_type]
        elif node_type == 'front_door':
            type_idx_val = FRONT_DOOR_TYPE_IDX
        else:
            continue

        for corner_pos, (x, y) in enumerate(coords):
            if corner_pos >= max_corners:
                break
            all_corners.append({
                'x': x,
                'y': y,
                'type': type_idx_val,
                'corner_idx': corner_pos,
                'room_idx': room_idx,
            })

    # Part 3: door and window (appended after functional rooms)

    current_room_idx = functional_room_count

    for key, type_idx_val in [('door', DOOR_TYPE_IDX)]:
        geom = plan.get(key)
        if geom is None or (hasattr(geom, 'is_empty') and geom.is_empty):
            continue

        for sub in get_geometries(geom):
            # Handle Polygon
            if sub.geom_type == 'Polygon':
                if sub.is_valid and not sub.is_empty and sub.area > 1.0:
                    coords = list(sub.exterior.coords[:-1])
                    coords = simplify_polygon_coords(coords)
                    for corner_pos, (x, y) in enumerate(coords):
                        if corner_pos >= max_corners:
                            break
                        all_corners.append({
                            'x': x,
                            'y': y,
                            'type': type_idx_val,
                            'corner_idx': corner_pos,
                            'room_idx': current_room_idx,
                        })
                    current_room_idx += 1

            # Handle LineString
            elif sub.geom_type == 'LineString':
                coords = list(sub.coords)
                for corner_pos, (x, y) in enumerate(coords):
                    if corner_pos >= max_corners:
                        break
                    all_corners.append({
                        'x': x,
                        'y': y,
                        'type': type_idx_val,
                        'corner_idx': corner_pos,
                        'room_idx': current_room_idx,
                    })
                current_room_idx += 1

            # Handle MultiLineString
            elif sub.geom_type == 'MultiLineString':
                for line in sub.geoms:
                    coords = list(line.coords)
                    for corner_pos, (x, y) in enumerate(coords):
                        if corner_pos >= max_corners:
                            break
                        all_corners.append({
                            'x': x,
                            'y': y,
                            'type': type_idx_val,
                            'corner_idx': corner_pos,
                            'room_idx': current_room_idx,
                        })
                    current_room_idx += 1

            # Handle MultiPolygon
            elif sub.geom_type == 'MultiPolygon':
                for poly in sub.geoms:
                    if poly.is_valid and not poly.is_empty and poly.area > 1.0:
                        coords = list(poly.exterior.coords[:-1])
                        coords = simplify_polygon_coords(coords)
                        for corner_pos, (x, y) in enumerate(coords):
                            if corner_pos >= max_corners:
                                break
                            all_corners.append({
                                'x': x,
                                'y': y,
                                'type': type_idx_val,
                                'corner_idx': corner_pos,
                                'room_idx': current_room_idx,
                            })
                        current_room_idx += 1

    return all_corners


def build_corner_token(corner, plan):
    """
    Build a single corner token.
    Token: [x, y, type_onehot(10), corner_idx_onehot(32), room_idx_onehot(64)]
    """
    x, y = corner['x'], corner['y']
    norm_coords = normalize_coords([(x, y)], plan)
    if norm_coords:
        x, y = norm_coords[0]

    type_onehot = [0.0] * NUM_TYPES
    type_onehot[corner['type']] = 1.0

    corner_onehot = [0.0] * MAX_CORNERS
    if corner['corner_idx'] < MAX_CORNERS:
        corner_onehot[corner['corner_idx']] = 1.0

    room_onehot = [0.0] * MAX_ROOMS
    if corner['room_idx'] < MAX_ROOMS:
        room_onehot[corner['room_idx']] = 1.0

    return [x, y] + type_onehot + corner_onehot + room_onehot


def pad_corners(corners, max_total=MAX_TOTAL_CORNERS, feat_dim=FEAT_DIM):
    """Pad corners to max_total with zeros."""
    if len(corners) > max_total:
        corners = corners[:max_total]
    padded = []
    for i in range(max_total):
        if i < len(corners):
            padded.append(corners[i])
        else:
            padded.append([0.0] * feat_dim)
    return np.array(padded, dtype=np.float32)


def process_plan(plan, room_order=None):
    """Process a single plan into corner features."""
    if room_order is None:
        room_order = get_topology_order(plan)

    corners = extract_corners_from_plan(plan, room_order)
    if not corners:
        return None

    tokens = [build_corner_token(c, plan) for c in corners]
    return {
        'features': pad_corners(tokens),
        'num_corners': len(tokens),
        'room_order': room_order,
    }


# Visualization
