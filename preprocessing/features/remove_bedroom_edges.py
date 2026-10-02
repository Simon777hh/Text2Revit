"""Remove bedroom-bedroom edges and rebuild topology graph encodings."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_RELEASE_ROOT = _Path(__file__).resolve().parents[2]
if str(_RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_RELEASE_ROOT))


import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


FLOW_ROOT = Path(__file__).resolve().parents[2]
if str(FLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(FLOW_ROOT))
if str(FLOW_ROOT / "preprocessing" / "features") not in sys.path:
    sys.path.insert(0, str(FLOW_ROOT / "preprocessing" / "features"))

from data_utils import (  # noqa: E402
    DATA_DIR,
    DEFAULT_RAW_MANIFEST,
    NUM_ROOMS,
    load_plan_order,
)
from preprocessing.features.extract_raw_data import (  # noqa: E402
    NUM_TYPES,
    laplacian_pe,
    random_walk_pe,
)
from preprocessing.features.prepare_topology_gt import (  # noqa: E402
    ROOM_TYPE_NAMES,
    summarize_by_source,
)


DEFAULT_TOPOLOGY = DATA_DIR / "topology_gt.npz"
DEFAULT_OUT = DATA_DIR / "topology_gt.npz"
DEFAULT_SUMMARY = DATA_DIR / "topology_gt_summary.json"


def edge_list_from_adjacency(
    adjacency: np.ndarray,
) -> list[tuple[int, int]]:
    rows, cols = np.nonzero(np.triu(adjacency, k=1))
    return list(zip(rows.tolist(), cols.tolist()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", default=str(DEFAULT_TOPOLOGY))
    parser.add_argument("--manifest", default=str(DEFAULT_RAW_MANIFEST))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    args = parser.parse_args()

    source = np.load(args.topology, allow_pickle=True)
    features = source["features"].astype(np.float32, copy=True)
    valid_mask = source["valid_mask"].astype(bool, copy=True)
    node_type_ids = source["node_type_ids"].astype(np.int8, copy=True)
    plan_ids = source["plan_ids"].astype(np.int64, copy=True)
    num_rooms = source["num_rooms"].astype(np.int64, copy=True)
    edge_lists = source["edge_lists"]

    if features.shape[1:] != (NUM_ROOMS, 61):
        raise ValueError(f"unexpected feature shape: {features.shape}")
    if len(edge_lists) != len(plan_ids):
        raise ValueError("edge list count does not match plans")

    adjacency = np.zeros(
        (len(plan_ids), NUM_ROOMS, NUM_ROOMS),
        dtype=np.uint8,
    )
    rebuilt_edges = np.empty(len(plan_ids), dtype=object)
    num_edges = np.zeros(len(plan_ids), dtype=np.int64)
    removed_by_plan = np.zeros(len(plan_ids), dtype=np.int64)
    pe_offset = NUM_ROOMS + NUM_TYPES

    for index in tqdm(
        range(len(plan_ids)),
        desc="removing bedroom edges",
    ):
        valid = valid_mask[index]
        bedrooms = valid & (node_type_ids[index] == 2)
        kept_edges = []
        removed = 0
        for room_a_value, room_b_value in edge_lists[index]:
            room_a = int(room_a_value)
            room_b = int(room_b_value)
            if bedrooms[room_a] and bedrooms[room_b]:
                removed += 1
                continue
            kept_edges.append((room_a, room_b))
            adjacency[index, room_a, room_b] = 1
            adjacency[index, room_b, room_a] = 1

        expected_edges = edge_list_from_adjacency(adjacency[index])
        if kept_edges != expected_edges:
            raise ValueError(
                f"plan {index} edge list does not match adjacency"
            )
        if np.any(adjacency[index, ~valid]):
            raise ValueError(f"plan {index} has edges to padded rooms")

        rebuilt_edges[index] = kept_edges
        num_edges[index] = len(kept_edges)
        removed_by_plan[index] = removed
        if removed > 0:
            pe = np.concatenate(
                [
                    random_walk_pe(
                        adjacency[index].astype(np.float32)
                    ),
                    laplacian_pe(
                        adjacency[index].astype(np.float32)
                    ),
                ],
                axis=1,
            )
            features[index, :, pe_offset:] = pe

    data = {
        "features": features,
        "valid_mask": valid_mask,
        "node_type_ids": node_type_ids,
        "adjacency": adjacency,
        "edge_lists": rebuilt_edges,
        "num_rooms": num_rooms,
        "num_edges": num_edges,
        "plan_ids": plan_ids,
    }

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **data)

    manifest = load_plan_order(args.manifest)
    if len(manifest) != len(plan_ids):
        raise ValueError("manifest and topology counts do not match")
    sources = np.asarray(
        [row["source"] for row in manifest],
        dtype="<U16",
    )
    summary = {
        "plan_count": int(len(plan_ids)),
        "feature_dim": int(features.shape[-1]),
        "max_nodes": int(features.shape[1]),
        "room_type_order": ROOM_TYPE_NAMES,
        "sources": summarize_by_source(data, sources),
        "outputs": {
            "topology_gt": str(output_path.resolve()),
        },
        "removed_bedroom_edges": int(removed_by_plan.sum()),
        "plans_with_removed_edges": int(
            np.count_nonzero(removed_by_plan)
        ),
    }
    summary_path = Path(args.summary)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    print(
        f"saved {output_path} features={features.shape} "
        f"adjacency={adjacency.shape}"
    )
    print(
        f"removed bedroom-bedroom edges="
        f"{int(removed_by_plan.sum())} "
        f"plans={int(np.count_nonzero(removed_by_plan))}"
    )
    print(f"saved {summary_path}")


if __name__ == "__main__":
    main()
