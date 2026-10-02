"""Dense RAM cache for the precomputed attention masks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from data_utils import DATA_DIR


DEFAULT_MASKS = DATA_DIR / "masks.npz"
_CACHE = None


def load_mask_cache(path: str | Path = DEFAULT_MASKS):
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    with np.load(path, allow_pickle=True) as masks:
        _CACHE = {
            "plan_ids": masks["plan_ids"].astype(np.int64),
            "corner_room": torch.from_numpy(
                np.asarray(masks["corner_room"], dtype=np.bool_)
            ),
            "corner_conn": torch.from_numpy(
                np.asarray(masks["corner_conn"], dtype=np.bool_)
            ),
            "corner_global": torch.from_numpy(
                np.asarray(masks["corner_global"], dtype=np.bool_)
            ),
            "node_conn": torch.from_numpy(
                np.asarray(masks["node_conn"], dtype=np.bool_)
            ),
            "node_global": torch.from_numpy(
                np.asarray(masks["node_global"], dtype=np.bool_)
            ),
            "corner_mask_shape": (128, 128),
        }
    return _CACHE


def get_corner_mask(name, index):
    if name not in {
        "corner_room",
        "corner_conn",
        "corner_global",
    }:
        raise KeyError(f"unknown corner mask: {name}")
    cache = load_mask_cache()
    return cache[name][index]
