"""Build attention masks and next-corner mapping in one GT pass."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_RELEASE_ROOT = _Path(__file__).resolve().parents[2]
if str(_RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_RELEASE_ROOT))


import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


FLOW_ROOT = Path(__file__).resolve().parents[2]
if str(FLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(FLOW_ROOT))

from data_utils import (
    CORNER_START,
    CORNER_END,
    DATA_DIR,
    DEFAULT_OUT_GT,
    NUM_CORNERS,
    NUM_ROOMS,
    ROOM_END,
    ROOM_START,
)


DEFAULT_TOPOLOGY = DATA_DIR / "topology_gt.npz"
DEFAULT_OUT = DATA_DIR / "masks.npz"
DEFAULT_SAMPLE = DATA_DIR / "masks_sample.txt"


def build_corner_masks(
    features: np.ndarray,
    adjacency: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    room_onehot = features[:, ROOM_START:ROOM_END]
    valid = room_onehot.sum(axis=-1) > 0.5
    room_ids = room_onehot.argmax(axis=-1)
    valid_pair = valid[:, None] & valid[None, :]
    same_room = room_ids[:, None] == room_ids[None, :]
    connected = adjacency[
        room_ids[:, None],
        room_ids[None, :],
    ] > 0.5
    corner_global = valid_pair
    corner_room = valid_pair & same_room
    corner_conn = valid_pair & (same_room | connected)
    return corner_room, corner_conn, corner_global


def build_node_masks(
    valid_mask: np.ndarray,
    adjacency: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    node_global = valid_mask[:, None] & valid_mask[None, :]
    node_conn = node_global & (
        (adjacency > 0.5) | np.eye(NUM_ROOMS, dtype=bool)
    )
    return node_conn, node_global


def ascii_mask(mask: np.ndarray) -> str:
    return "\n".join(
        "".join("1" if value else "." for value in row)
        for row in mask
    )


def write_sample(
    path: Path,
    index: int,
    plan_id: int,
    corner_room: np.ndarray,
    corner_conn: np.ndarray,
    corner_global: np.ndarray,
    node_conn: np.ndarray,
    node_global: np.ndarray,
) -> None:
    lines = [
        f"index={index} plan_id={plan_id}",
        f"valid corners={int(corner_global.any(axis=-1).sum())}",
        f"valid rooms={int(node_global.any(axis=-1).sum())}",
        "",
        "corner_room:",
        ascii_mask(corner_room),
        "",
        "corner_conn:",
        ascii_mask(corner_conn),
        "",
        "corner_global:",
        ascii_mask(corner_global),
        "",
        "node_conn:",
        ascii_mask(node_conn),
        "",
        "node_global:",
        ascii_mask(node_global),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


MAX_CORNERS_PER_ROOM = 28


def build_mapping(
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    room_onehot = features[:, ROOM_START:ROOM_END]
    corner_onehot = features[:, CORNER_START:CORNER_END]
    valid = room_onehot.sum(axis=-1) > 0.5
    room_ids = room_onehot.argmax(axis=-1)
    corner_ids = corner_onehot.argmax(axis=-1)
    room_corner_counts = room_onehot.sum(axis=0)
    token_room_corner_count = room_corner_counts[room_ids]

    index_map = np.full(
        (NUM_ROOMS, MAX_CORNERS_PER_ROOM),
        -1,
        dtype=np.int16,
    )
    token_indices = np.arange(NUM_CORNERS)
    index_map[room_ids[valid], corner_ids[valid]] = token_indices[valid]

    next_corner_ids = np.where(
        corner_ids + 1 < token_room_corner_count,
        corner_ids + 1,
        0,
    )
    safe_next_corner_ids = np.where(valid, next_corner_ids, 0)
    next_index = index_map[room_ids, safe_next_corner_ids]
    next_index = np.where(next_index >= 0, next_index, 0)
    edge_valid = valid & (token_room_corner_count >= 2)
    return edge_valid.astype(bool), next_index.astype(np.int16)



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", default=str(DEFAULT_OUT_GT))
    parser.add_argument("--topology", default=str(DEFAULT_TOPOLOGY))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--mapping-out", default=str(DATA_DIR / "edge_mapping.npz"))
    parser.add_argument("--sample", default=str(DEFAULT_SAMPLE))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("Limit must be nonnegative")
    outputs = [Path(args.out).resolve(), Path(args.mapping_out).resolve(), Path(args.sample).resolve()]
    if len(set(outputs)) != len(outputs) or any(
            p in (Path(args.gt).resolve(), Path(args.topology).resolve()) for p in outputs):
        parser.error("Mask/mapping outputs must be distinct and cannot replace inputs")

    gt = np.load(args.gt, allow_pickle=True)
    topology = np.load(args.topology, allow_pickle=True)
    features = gt["features"]
    plan_ids = gt["plan_ids"].astype(np.int64)
    valid_masks = topology["valid_mask"].astype(bool)
    adjacency = topology["adjacency"].astype(bool)

    if not np.array_equal(
        plan_ids,
        topology["plan_ids"].astype(np.int64),
    ):
        raise ValueError("GT and topology plan ids are not aligned")
    expected_edges = np.triu(adjacency, k=1).sum(axis=(1, 2))
    if not np.array_equal(
        expected_edges,
        topology["num_edges"].astype(np.int64),
    ):
        raise ValueError(
            "topology adjacency does not match num_edges"
        )
    if np.any(adjacency & ~valid_masks[:, :, None]):
        raise ValueError("topology contains invalid-node edges")

    total = len(features)
    if args.limit > 0:
        total = min(total, args.limit)

    if not total:
        raise ValueError("Cannot build attention data from an empty GT dataset")
    next_index = np.zeros((total, NUM_CORNERS), dtype=np.int16)
    edge_valid = np.zeros((total, NUM_CORNERS), dtype=bool)
    corner_room = np.zeros(
        (total, NUM_CORNERS, NUM_CORNERS),
        dtype=bool,
    )
    corner_conn = np.zeros_like(corner_room)
    corner_global = np.zeros_like(corner_room)
    node_conn = np.zeros((total, NUM_ROOMS, NUM_ROOMS), dtype=bool)
    node_global = np.zeros_like(node_conn)

    for index in tqdm(
        range(total),
        desc="Building topology masks",
    ):
        (
            corner_room[index],
            corner_conn[index],
            corner_global[index],
        ) = build_corner_masks(features[index], adjacency[index])
        edge_valid[index], next_index[index] = build_mapping(features[index])
        node_conn[index], node_global[index] = build_node_masks(
            valid_masks[index],
            adjacency[index],
        )

    self_mask = ~corner_room
    door_mask = ~corner_conn
    gen_mask = ~corner_global
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        plan_ids=plan_ids[:total],
        corner_room=corner_room,
        corner_conn=corner_conn,
        corner_global=corner_global,
        self_mask=self_mask,
        door_mask=door_mask,
        gen_mask=gen_mask,
        node_conn=node_conn,
        node_global=node_global,
    )

    mapping_path = Path(args.mapping_out)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(mapping_path, plan_ids=plan_ids[:total],
                        corner_next_index=next_index, corner_edge_valid=edge_valid)
    print(f"saved {mapping_path} valid_edges={int(edge_valid.sum())}")
    sample_index = min(max(args.index, 0), total - 1)
    sample_path = Path(args.sample)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    write_sample(
        sample_path,
        sample_index,
        int(plan_ids[sample_index]),
        corner_room[sample_index],
        corner_conn[sample_index],
        corner_global[sample_index],
        node_conn[sample_index],
        node_global[sample_index],
    )
    print(
        f"saved {output_path} corner={corner_room.shape} "
        f"node={node_conn.shape}"
    )
    print(f"saved {sample_path}")


if __name__ == "__main__":
    main()
