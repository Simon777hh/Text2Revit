"""Assemble and validate the Flow Matching topology GT dataset."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_RELEASE_ROOT = _Path(__file__).resolve().parents[2]
if str(_RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_RELEASE_ROOT))


import argparse
import json
from pathlib import Path

import numpy as np

from data_utils import (
    DATA_DIR,
    DEFAULT_RAW_EDGES,
    DEFAULT_RAW_MANIFEST,
    DEFAULT_RAW_TOPOLOGY,
    load_plan_order,
)


from preprocessing.features.extract_raw_data import laplacian_pe, random_walk_pe


def remove_bedroom_connections(data):
    """Filter conditioning edges and refresh positional encodings in memory."""
    adjacency = data["adjacency"]
    removed = np.zeros(len(adjacency), dtype=np.int64)
    edge_lists = np.empty(len(adjacency), dtype=object)
    for index, current in enumerate(adjacency):
        bedrooms = data["valid_mask"][index] & (data["node_type_ids"][index] == 2)
        forbidden = bedrooms[:, None] & bedrooms[None, :]
        removed[index] = np.triu(current & forbidden, k=1).sum()
        current[forbidden] = 0
        rows, cols = np.nonzero(np.triu(current, k=1))
        edge_lists[index] = list(zip(rows.tolist(), cols.tolist()))
        data["num_edges"][index] = len(rows)
        if removed[index]:
            data["features"][index, :, 29:] = np.concatenate(
                [random_walk_pe(current.astype(np.float32)),
                 laplacian_pe(current.astype(np.float32))], axis=1)
    data["edge_lists"] = edge_lists
    return removed


ROOM_TYPE_NAMES = [
    "living",
    "kitchen",
    "bedroom",
    "bathroom",
    "balcony",
    "storage",
    "stair",
    "front_door",
    "door",
]


def quantiles(values: np.ndarray) -> dict[str, float]:
    points = np.quantile(
        values,
        [0.0, 0.25, 0.5, 0.75, 0.95, 1.0],
    )
    return {
        name: float(value)
        for name, value in zip(
            ["min", "q25", "median", "q75", "p95", "max"],
            points,
        )
    }


def build_topology_gt(
    topology: np.lib.npyio.NpzFile,
    edges: np.lib.npyio.NpzFile,
) -> dict[str, np.ndarray]:
    features = topology["features"].astype(np.float32)
    valid_mask = topology["valid_mask"].astype(bool)
    plan_ids = topology["plan_ids"].astype(np.int64)
    num_rooms = topology["num_rooms"].astype(np.int64)
    adjacency = edges["adjacency"].astype(bool)
    edge_lists = edges["edge_lists"]
    num_edges = edges["num_edges"].astype(np.int64)

    if not np.array_equal(plan_ids, edges["plan_ids"]):
        raise ValueError("topology and edge plan ids are not aligned")
    if features.shape[1:] != (20, 61):
        raise ValueError(f"unexpected topology shape: {features.shape}")
    if adjacency.shape[1:] != (20, 20):
        raise ValueError(f"unexpected adjacency shape: {adjacency.shape}")
    if not np.array_equal(valid_mask.sum(axis=1), num_rooms):
        raise ValueError("valid_mask count does not match num_rooms")

    node_type_ids = np.full(
        (len(features), 20),
        -1,
        dtype=np.int8,
    )
    type_onehot = features[:, :, 20:29]
    node_type_ids[valid_mask] = type_onehot[
        valid_mask
    ].argmax(axis=-1).astype(np.int8)

    invalid_edges = 0
    for plan_index in range(len(features)):
        valid = valid_mask[plan_index]
        current = adjacency[plan_index]
        if np.any(np.diag(current)):
            raise ValueError(
                f"plan {plan_index} has a topology self-loop"
            )
        if not np.array_equal(current, current.T):
            raise ValueError(
                f"plan {plan_index} topology is not symmetric"
            )
        if np.any(current[~valid]):
            invalid_edges += 1
        rows, cols = np.nonzero(np.triu(current, k=1))
        expected = list(zip(rows.tolist(), cols.tolist()))
        stored = [tuple(edge) for edge in edge_lists[plan_index]]
        if expected != stored:
            raise ValueError(
                f"plan {plan_index} edge list mismatch"
            )
        if len(stored) != int(num_edges[plan_index]):
            raise ValueError(
                f"plan {plan_index} num_edges mismatch"
            )

    if invalid_edges:
        raise ValueError(
            f"{invalid_edges} plans contain edges to padding nodes"
        )

    data = {
        "features": features,
        "valid_mask": valid_mask,
        "node_type_ids": node_type_ids,
        "adjacency": adjacency.astype(np.uint8),
        "edge_lists": edge_lists,
        "num_rooms": num_rooms,
        "num_edges": num_edges,
        "plan_ids": plan_ids,
    }
    remove_bedroom_connections(data)
    return data


def summarize_by_source(
    data: dict[str, np.ndarray],
    sources: np.ndarray,
) -> dict[str, dict]:
    summary = {}
    for source_name in sorted(np.unique(sources).tolist()):
        mask = sources == source_name
        type_ids = data["node_type_ids"][mask]
        valid = data["valid_mask"][mask]
        type_counts = {}
        for type_index, type_name in enumerate(ROOM_TYPE_NAMES):
            type_counts[type_name] = int(
                np.sum(valid & (type_ids == type_index))
            )
        summary[source_name] = {
            "plan_count": int(mask.sum()),
            "num_rooms": quantiles(data["num_rooms"][mask]),
            "num_edges": quantiles(data["num_edges"][mask]),
            "valid_nodes": int(valid.sum()),
            "valid_edges": int(data["adjacency"][mask].sum() // 2),
            "room_type_counts": type_counts,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topology",
        default=str(DEFAULT_RAW_TOPOLOGY),
    )
    parser.add_argument("--edges", default=str(DEFAULT_RAW_EDGES))
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_RAW_MANIFEST),
    )
    parser.add_argument(
        "--out",
        default=str(DATA_DIR / "topology_gt.npz"),
    )
    parser.add_argument(
        "--summary",
        default=str(DATA_DIR / "topology_gt_summary.json"),
    )
    args = parser.parse_args()

    topology = np.load(args.topology, allow_pickle=True)
    edges = np.load(args.edges, allow_pickle=True)
    plan_order = load_plan_order(args.manifest)
    if len(plan_order) != len(topology["plan_ids"]):
        raise ValueError("manifest and topology counts do not match")

    data = build_topology_gt(topology, edges)
    removed_by_plan = edges["num_edges"] - data["num_edges"]
    sources = np.asarray(
        [row["source"] for row in plan_order],
        dtype="<U16",
    )

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **data)

    summary = {
        "plan_count": int(len(data["plan_ids"])),
        "feature_dim": int(data["features"].shape[-1]),
        "max_nodes": int(data["features"].shape[1]),
        "room_type_order": ROOM_TYPE_NAMES,
        "sources": summarize_by_source(data, sources),
        "removed_bedroom_edges": int(removed_by_plan.sum()),
        "plans_with_removed_edges": int(np.count_nonzero(removed_by_plan)),
        "outputs": {
            "topology_gt": str(output_path.resolve()),
        },
    }
    summary_path = Path(args.summary)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    print(
        f"saved {output_path} "
        f"features={data['features'].shape} "
        f"adjacency={data['adjacency'].shape}"
    )
    print(f"saved {summary_path}")
    for source_name, values in summary["sources"].items():
        print(
            f"{source_name}: plans={values['plan_count']} "
            f"rooms_median={values['num_rooms']['median']:.1f} "
            f"edges_median={values['num_edges']['median']:.1f}"
        )


if __name__ == "__main__":
    main()
