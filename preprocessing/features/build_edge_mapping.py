"""Precompute the static corner-to-next-corner mapping for AU expansion."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_RELEASE_ROOT = _Path(__file__).resolve().parents[2]
if str(_RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_RELEASE_ROOT))


import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data_utils import (
    CORNER_END,
    CORNER_START,
    DATA_DIR,
    DEFAULT_OUT_GT,
    NUM_CORNERS,
    NUM_ROOMS,
    ROOM_END,
    ROOM_START,
)


DEFAULT_OUT = DATA_DIR / "edge_mapping.npz"
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
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    gt = np.load(args.gt, allow_pickle=True)
    features = gt["features"]
    plan_ids = gt["plan_ids"].astype(np.int64)
    if args.limit > 0:
        features = features[: args.limit]
        plan_ids = plan_ids[: args.limit]

    next_index = np.zeros(
        (len(features), NUM_CORNERS),
        dtype=np.int16,
    )
    edge_valid = np.zeros(
        (len(features), NUM_CORNERS),
        dtype=bool,
    )
    for index in tqdm(range(len(features)), desc="building edge mapping"):
        edge_valid[index], next_index[index] = build_mapping(
            features[index]
        )

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        plan_ids=plan_ids,
        corner_next_index=next_index,
        corner_edge_valid=edge_valid,
    )
    print(
        f"saved {output_path} valid_edges={int(edge_valid.sum())} "
        f"next_index={next_index.shape}"
    )


if __name__ == "__main__":
    main()
