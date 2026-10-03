"""Extract Flow Matching tensors directly from the source plan pickles."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_RELEASE_ROOT = _Path(__file__).resolve().parents[2]
if str(_RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_RELEASE_ROOT))


import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
)
from tqdm import tqdm

from data_utils import (
    DATA_DIR,
    PROJECT_ROOT,
    canonicalize_corner_rings,
)


DEFAULT_RPLAN_PKL = (
    PROJECT_ROOT / "RPLAN" / "RPLAN_filtered.pkl"
)
DEFAULT_RESPLAN_PKL = PROJECT_ROOT / "ResPlan" / "ResPlan_filtered_canvas.pkl"

from preprocessing.resplan.prepare_resplan import source_coordinates

NUM_ROOMS = 20
NUM_TYPES = 9
MAX_CORNERS_PER_ROOM = 28
MAX_TOTAL_CORNERS = 128
RW_PE_DIM = 16
LAP_PE_DIM = 16

TYPE_NAME_TO_IDX = {
    "living": 0,
    "kitchen": 1,
    "bedroom": 2,
    "bathroom": 3,
    "balcony": 4,
    "storage": 5,
    "stair": 6,
    "front_door": 7,
    "door": 8,
}

CONDITION_CONNECTION_TYPES = {
    "via_door",
    "via_opening",
    "via_window",
    "direct",
}

LOSS_CONNECTION_TYPES = {
    *CONDITION_CONNECTION_TYPES,
    "adjacency",
    "fallback",
}


def safe_geometries(geom: Any) -> list[Any]:
    if geom is None:
        return []
    if hasattr(geom, "is_empty") and geom.is_empty:
        return []
    if isinstance(geom, (Polygon, LineString, Point)):
        return [geom]
    if isinstance(geom, (MultiPolygon, MultiLineString, GeometryCollection)):
        result = []
        for part in geom.geoms:
            result.extend(safe_geometries(part))
        return result
    if isinstance(geom, (list, tuple)):
        result = []
        for part in geom:
            result.extend(safe_geometries(part))
        return result
    return []


def simplify_polygon_coords(
    coords: list[tuple[float, float]],
    tolerance: float = 0.5,
) -> list[tuple[float, float]]:
    if len(coords) <= 4:
        return coords
    unique_coords = []
    for index, (x, y) in enumerate(coords):
        if (
            index == 0
            or abs(x - coords[index - 1][0]) > 1e-6
            or abs(y - coords[index - 1][1]) > 1e-6
        ):
            unique_coords.append((x, y))
    if len(unique_coords) <= 4:
        return unique_coords
    try:
        polygon = Polygon(unique_coords)
        if polygon.is_valid:
            simplified = polygon.simplify(
                tolerance,
                preserve_topology=True,
            )
            if simplified.geom_type == "Polygon":
                return list(simplified.exterior.coords[:-1])
    except Exception:
        pass
    return unique_coords


def get_node_geometry(
    plan: dict[str, Any],
    node_name: str,
    room_order: list[str],
) -> Any:
    key = node_name.split("_")[0]
    if key == "front":
        key = "front_door"
    type_nodes = [
        name for name in room_order if name.startswith(key + "_")
    ]
    if node_name not in type_nodes:
        return None
    type_position = type_nodes.index(node_name)
    geometries = [
        geom
        for geom in safe_geometries(plan.get(key))
        if key == "front_door"
        or geom.geom_type != "Polygon"
        or geom.area > 1.0
    ]
    if type_position >= len(geometries):
        return None
    return geometries[type_position]


def signed_area(coords: list[tuple[float, float]]) -> float:
    points = np.asarray(coords, dtype=float)
    if points.shape[0] < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * (
        (x * np.roll(y, -1) - np.roll(x, -1) * y).sum()
    )


def renumber_ccw_min_y(
    coords: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    coords = [tuple(point) for point in coords]
    if len(coords) < 3:
        return coords
    if signed_area(coords) < 0:
        coords = coords[::-1]
    start = min(
        range(len(coords)),
        key=lambda index: (coords[index][1], coords[index][0]),
    )
    return coords[start:] + coords[:start]


def build_corner_tokens(
    coords: list[tuple[float, float]],
    room_index: int,
) -> tuple[list[np.ndarray], int]:
    coords = renumber_ccw_min_y(coords)
    coords = coords[:MAX_CORNERS_PER_ROOM]
    center = (
        np.mean(coords, axis=0)
        if coords
        else np.zeros(2, dtype=np.float32)
    )
    tokens = []
    for corner_index, (x, y) in enumerate(coords):
        room_onehot = np.zeros(NUM_ROOMS, dtype=np.float32)
        room_onehot[room_index] = 1.0
        corner_onehot = np.zeros(MAX_CORNERS_PER_ROOM, dtype=np.float32)
        corner_onehot[corner_index] = 1.0
        tokens.append(
            np.concatenate(
                [
                    [float(x), float(y)],
                    center.astype(np.float32),
                    room_onehot,
                    corner_onehot,
                ]
            )
        )
    return tokens, len(coords)


def pad_corner_tokens(tokens: list[np.ndarray]) -> np.ndarray:
    tokens = tokens[:MAX_TOTAL_CORNERS]
    padded = np.zeros(
        (MAX_TOTAL_CORNERS, 52),
        dtype=np.float32,
    )
    if tokens:
        padded[: len(tokens)] = np.asarray(tokens, dtype=np.float32)
    return padded


def extract_corner_features(plan: dict[str, Any]) -> dict[str, Any] | None:
    conn = plan.get("conn") or {}
    room_order = conn.get("node_names") or []
    room_types = conn.get("room_types") or []
    if not room_order or len(room_order) > NUM_ROOMS:
        return None

    tokens = []
    for room_index, node_name in enumerate(room_order):
        geometry = get_node_geometry(plan, node_name, room_order)
        if geometry is None:
            continue
        if geometry.geom_type == "Polygon":
            keep_polygon = (
                node_name.startswith("front_door")
                or geometry.area > 1.0
            )
            if (
                not geometry.is_valid
                or geometry.is_empty
                or not keep_polygon
            ):
                continue
            coords = list(geometry.exterior.coords[:-1])
        elif geometry.geom_type == "LineString":
            coords = list(geometry.coords)
        else:
            continue
        coords = simplify_polygon_coords(coords)
        room_tokens, _ = build_corner_tokens(coords, room_index)
        tokens.extend(room_tokens)

    if not tokens:
        return None
    return {
        "features": pad_corner_tokens(tokens),
        "num_corners": len(tokens),
        "room_order": list(room_order),
        "room_types": list(room_types),
    }


def build_adjacency(
    conn: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    condition_adjacency = np.zeros(
        (NUM_ROOMS, NUM_ROOMS),
        dtype=np.float32,
    )
    loss_adjacency = np.zeros(
        (NUM_ROOMS, NUM_ROOMS),
        dtype=np.float32,
    )
    edges = conn.get("edge_list") or []
    edge_types = conn.get("edge_type_names") or []
    for (room_a, room_b), edge_type in zip(edges, edge_types):
        if edge_type not in LOSS_CONNECTION_TYPES:
            continue
        if room_a >= NUM_ROOMS or room_b >= NUM_ROOMS:
            continue
        loss_adjacency[room_a, room_b] = 1.0
        loss_adjacency[room_b, room_a] = 1.0
        if edge_type in CONDITION_CONNECTION_TYPES:
            condition_adjacency[room_a, room_b] = 1.0
            condition_adjacency[room_b, room_a] = 1.0
    return condition_adjacency, loss_adjacency


def edge_list_from_adjacency(
    adjacency: np.ndarray,
) -> list[tuple[int, int]]:
    rows, cols = np.nonzero(np.triu(adjacency, k=1))
    return list(zip(rows.tolist(), cols.tolist()))


def random_walk_pe(
    adjacency: np.ndarray,
    dim: int = RW_PE_DIM,
) -> np.ndarray:
    size = adjacency.shape[0]
    augmented = adjacency + np.eye(size, dtype=adjacency.dtype)
    degree = augmented.sum(axis=1)
    inverse_degree = np.zeros_like(degree)
    inverse_degree[degree > 0] = 1.0 / degree[degree > 0]
    transition = augmented * inverse_degree[:, None]
    pe = np.zeros((size, dim), dtype=np.float32)
    power = np.eye(size, dtype=np.float64)
    for step in range(1, dim + 1):
        power = power @ transition
        pe[:, step - 1] = np.diag(power)
    return pe


def laplacian_pe(
    adjacency: np.ndarray,
    dim: int = LAP_PE_DIM,
) -> np.ndarray:
    size = adjacency.shape[0]
    degree = adjacency.sum(axis=1)
    inverse_sqrt = np.zeros(size, dtype=np.float64)
    inverse_sqrt[degree > 0] = 1.0 / np.sqrt(degree[degree > 0])
    inverse_sqrt_matrix = np.diag(inverse_sqrt)
    laplacian = (
        np.eye(size, dtype=np.float64)
        - inverse_sqrt_matrix @ adjacency @ inverse_sqrt_matrix
    )
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    pe = np.zeros((size, dim), dtype=np.float32)
    take = min(dim, size)
    pe[:, :take] = (
        eigenvectors[:, :take] * eigenvalues[:take][None, :]
    ).astype(np.float32)
    return pe


def build_topology_features(
    conn: dict[str, Any],
    adjacency: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    room_order = conn.get("node_names") or []
    room_types = conn.get("room_types") or []
    num_rooms = len(room_order)
    pe = np.concatenate(
        [random_walk_pe(adjacency), laplacian_pe(adjacency)],
        axis=1,
    )
    features = np.zeros((NUM_ROOMS, 61), dtype=np.float32)
    for room_index in range(num_rooms):
        type_name = room_types[room_index]
        type_index = TYPE_NAME_TO_IDX.get(type_name, -1)
        room_onehot = np.zeros(NUM_ROOMS, dtype=np.float32)
        room_onehot[room_index] = 1.0
        type_onehot = np.zeros(NUM_TYPES, dtype=np.float32)
        if type_index >= 0:
            type_onehot[type_index] = 1.0
        row = features[room_index]
        row[0:NUM_ROOMS] = room_onehot
        row[NUM_ROOMS:NUM_ROOMS + NUM_TYPES] = type_onehot
        row[NUM_ROOMS + NUM_TYPES:] = pe[room_index]

    valid_mask = np.zeros(NUM_ROOMS, dtype=bool)
    valid_mask[:num_rooms] = True
    return features, valid_mask, num_rooms


def extract_source(
    source_name: str,
    plans: list[dict[str, Any]],
    plan_id_offset: int,
) -> dict[str, Any]:
    corner_features = []
    topology_features = []
    valid_masks = []
    adjacency_rows = []
    loss_adjacency_rows = []
    edge_rows = []
    loss_edge_rows = []
    num_edges = []
    loss_num_edges = []
    num_rooms = []
    source_ids = []
    source_indices = []
    skipped = []

    for source_index, plan in enumerate(
        tqdm(plans, desc=f"extracting {source_name}")
    ):
        if source_name == "resplan":
            plan = source_coordinates(plan)
        conn = plan.get("conn") or {}
        result = extract_corner_features(plan)
        if result is None:
            skipped.append(source_index)
            continue

        adjacency, loss_adjacency = build_adjacency(conn)
        topo, valid_mask, room_count = build_topology_features(
            conn,
            adjacency,
        )
        edges = edge_list_from_adjacency(adjacency)
        loss_edges = edge_list_from_adjacency(loss_adjacency)

        corner_features.append(result["features"])
        topology_features.append(topo)
        valid_masks.append(valid_mask)
        adjacency_rows.append(adjacency)
        loss_adjacency_rows.append(loss_adjacency)
        edge_rows.append(edges)
        loss_edge_rows.append(loss_edges)
        num_edges.append(len(edges))
        loss_num_edges.append(len(loss_edges))
        num_rooms.append(room_count)
        source_ids.append(int(plan.get("id", -1)))
        source_indices.append(source_index)

    count = len(corner_features)
    plan_ids = np.arange(
        plan_id_offset,
        plan_id_offset + count,
        dtype=np.int64,
    )
    return {
        "source": source_name,
        "count": count,
        "skipped": skipped,
        "plan_ids": plan_ids,
        "source_ids": np.asarray(source_ids, dtype=np.int64),
        "source_indices": np.asarray(source_indices, dtype=np.int64),
        "corner_features": np.asarray(
            corner_features,
            dtype=np.float32,
        ),
        "topology_features": np.asarray(
            topology_features,
            dtype=np.float32,
        ),
        "valid_masks": np.asarray(valid_masks, dtype=bool),
        "adjacency": np.asarray(adjacency_rows, dtype=np.float32),
        "loss_adjacency": np.asarray(
            loss_adjacency_rows,
            dtype=np.float32,
        ),
        "edge_lists": np.asarray(edge_rows, dtype=object),
        "loss_edge_lists": np.asarray(
            loss_edge_rows,
            dtype=object,
        ),
        "num_edges": np.asarray(num_edges, dtype=np.int64),
        "loss_num_edges": np.asarray(
            loss_num_edges,
            dtype=np.int64,
        ),
        "num_rooms": np.asarray(num_rooms, dtype=np.int64),
    }


def concatenate_sources(
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    result = {}
    array_keys = [
        "plan_ids",
        "source_ids",
        "source_indices",
        "corner_features",
        "topology_features",
        "valid_masks",
        "adjacency",
        "loss_adjacency",
        "edge_lists",
        "loss_edge_lists",
        "num_edges",
        "loss_num_edges",
        "num_rooms",
    ]
    for key in array_keys:
        result[key] = np.concatenate(
            [source[key] for source in sources],
            axis=0,
        )
    result["source"] = np.concatenate(
        [
            np.full(source["count"], source["source"], dtype="<U16")
            for source in sources
        ]
    )
    result["source_summaries"] = [
        {
            "source": source["source"],
            "count": source["count"],
            "skipped": source["skipped"],
        }
        for source in sources
    ]
    return result


def build_manifest(data: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = []
    for index in range(len(data["plan_ids"])):
        manifest.append(
            {
                "id": int(data["plan_ids"][index]),
                "source": str(data["source"][index]),
                "source_id": int(data["source_ids"][index]),
                "source_idx": int(data["source_indices"][index]),
            }
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rplan", default=str(DEFAULT_RPLAN_PKL))
    parser.add_argument("--resplan", default=str(DEFAULT_RESPLAN_PKL))
    parser.add_argument("--out-dir", default=str(DATA_DIR))
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["rplan", "resplan"],
        default=["rplan", "resplan"],
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_results = []
    next_id = 0
    if "rplan" in args.sources:
        with open(args.rplan, "rb") as handle:
            rplan_plans = pickle.load(handle)
        rplan_result = extract_source(
            "rplan",
            rplan_plans,
            next_id,
        )
        source_results.append(rplan_result)
        next_id += rplan_result["count"]
        del rplan_plans
    if "resplan" in args.sources:
        with open(args.resplan, "rb") as handle:
            resplan_plans = pickle.load(handle)
        resplan_result = extract_source(
            "resplan",
            resplan_plans,
            next_id,
        )
        source_results.append(resplan_result)
        del resplan_plans

    data = concatenate_sources(source_results)
    data["corner_features"] = canonicalize_corner_rings(
        data["corner_features"]
    )
    plan_ids = data["plan_ids"]
    np.savez_compressed(
        out_dir / "raw_gt_tokens.npz",
        features=data["corner_features"],
        plan_ids=plan_ids,
        source=data["source"],
        source_id=data["source_ids"],
    )
    np.savez_compressed(
        out_dir / "raw_topology.npz",
        features=data["topology_features"],
        valid_mask=data["valid_masks"],
        plan_ids=plan_ids,
        num_rooms=data["num_rooms"],
    )
    np.savez_compressed(
        out_dir / "raw_edges.npz",
        edge_lists=data["edge_lists"],
        adjacency=data["adjacency"],
        loss_edge_lists=data["loss_edge_lists"],
        loss_adjacency=data["loss_adjacency"],
        num_edges=data["num_edges"],
        loss_num_edges=data["loss_num_edges"],
        plan_ids=plan_ids,
    )
    manifest = build_manifest(data)
    manifest_path = out_dir / "raw_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)

    summary = {
        "count": int(len(plan_ids)),
        "sources": data["source_summaries"],
        "outputs": {
            "gt": str((out_dir / "raw_gt_tokens.npz").resolve()),
            "topology": str(
                (out_dir / "raw_topology.npz").resolve()
            ),
            "edges": str((out_dir / "raw_edges.npz").resolve()),
            "manifest": str(manifest_path.resolve()),
        },
    }
    with open(
        out_dir / "raw_extraction_summary.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    print(f"saved {out_dir / 'raw_gt_tokens.npz'}")
    print(f"saved {out_dir / 'raw_topology.npz'}")
    print(f"saved {out_dir / 'raw_edges.npz'}")
    print(f"saved {manifest_path}")
    for source in data["source_summaries"]:
        print(
            f"{source['source']}: kept={source['count']} "
            f"skipped={len(source['skipped'])}"
        )


if __name__ == "__main__":
    main()
