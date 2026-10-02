"""Central data-generation helpers for the floorplan model.

This module is the single entry point used by production inference for:

- prompt -> topology and corner-count sampling
- room/node features and positional encodings
- corner/node attention masks
- embedding layouts and corner edge mappings
- packing the model input sample

The original dataset extraction scripts can keep their historical imports;
new inference code should import from this module.
"""

from __future__ import annotations

import numpy as np
import torch

from data_utils import (
    CORNER_START,
    NUM_CORNERS,
    NUM_ROOMS,
    ROOM_START,
    build_corner_layout,
)
from edge_utils import build_corner_edge_mapping
from topology_rules import (
    TYPE_NAMES,
    CornerCountRules,
    RuleTopology,
    get_default_corner_count_rules,
    parse_prompt_room_counts,
    sample_rule_topology,
)


RW_PE_DIM = 16
LAP_PE_DIM = 16
NUM_TYPES = 9


def random_walk_pe(
    adjacency: np.ndarray,
    dim: int = RW_PE_DIM,
) -> np.ndarray:
    """Return diagonal random-walk positional encodings."""
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
    """Return Laplacian-eigenvector positional encodings."""
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
    adjacency: np.ndarray,
    node_type_ids: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Build the 61-d node feature tensor used by the transformer."""
    adjacency = np.asarray(adjacency, dtype=np.float32)
    node_type_ids = np.asarray(node_type_ids)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    pe = np.concatenate(
        [
            random_walk_pe(adjacency, RW_PE_DIM),
            laplacian_pe(adjacency, LAP_PE_DIM),
        ],
        axis=1,
    )
    node_features = np.zeros((NUM_ROOMS, 61), dtype=np.float32)
    for room_id in np.flatnonzero(valid_mask):
        room_id = int(room_id)
        type_id = int(node_type_ids[room_id])
        node_features[room_id, room_id] = 1.0
        if 0 <= type_id < NUM_TYPES:
            node_features[room_id, 20 + type_id] = 1.0
        node_features[room_id, 29:] = pe[room_id]
    return node_features


def build_corner_masks(
    features: np.ndarray,
    adjacency: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build corner-room, corner-connection, and corner-global masks."""
    room_onehot = features[:, ROOM_START:ROOM_START + NUM_ROOMS]
    valid = room_onehot.sum(axis=-1) > 0.5
    room_ids = room_onehot.argmax(axis=-1)
    valid_pair = valid[:, None] & valid[None, :]
    same_room = room_ids[:, None] == room_ids[None, :]
    connected_room = (
        adjacency[room_ids[:, None], room_ids[None, :]] > 0.5
    )
    corner_global = valid_pair
    corner_room = valid_pair & same_room
    corner_conn = valid_pair & (same_room | connected_room)
    return corner_room, corner_conn, corner_global


def build_node_masks(
    valid_mask: np.ndarray,
    adjacency: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build node-global and node-connection masks."""
    valid_mask = np.asarray(valid_mask, dtype=bool)
    adjacency = np.asarray(adjacency)
    node_global = valid_mask[:, None] & valid_mask[None, :]
    node_conn = node_global & (
        (adjacency > 0.5)
        | np.eye(NUM_ROOMS, dtype=bool)
    )
    return node_global, node_conn


def build_room_corner_features(
    topology: RuleTopology,
    room_counts: dict[int, int],
) -> np.ndarray:
    """Create the 128 x 52 room/corner token feature tensor."""
    features = np.zeros((NUM_CORNERS, 52), dtype=np.float32)
    token = 0
    for room_id in np.flatnonzero(topology.valid_mask):
        room_id = int(room_id)
        corner_count = int(room_counts[room_id])
        features[
            token : token + corner_count,
            ROOM_START + room_id,
        ] = 1.0
        features[
            token + np.arange(corner_count),
            CORNER_START + np.arange(corner_count),
        ] = 1.0
        token += corner_count
    return features


def sample_room_corner_counts(
    topology: RuleTopology,
    corner_rules: CornerCountRules,
    rng: np.random.Generator,
) -> tuple[dict[int, int], str]:
    """Sample corner counts from the dataset distributions."""
    source = corner_rules.sample_source(rng)
    room_counts: dict[int, int] = {}
    for room_id in np.flatnonzero(topology.valid_mask):
        room_id = int(room_id)
        room_type = str(
            TYPE_NAMES[int(topology.node_type_ids[room_id])]
        )
        room_counts[room_id] = corner_rules.sample(
            room_type,
            rng,
            source=source,
        )
    while sum(room_counts.values()) > NUM_CORNERS:
        room_id = max(
            room_counts,
            key=lambda key: room_counts[key],
        )
        if room_counts[room_id] <= 4:
            break
        room_counts[room_id] -= 1
    return room_counts, source


def build_apartment(
    topology: RuleTopology,
    corner_rules: CornerCountRules,
    rng: np.random.Generator,
) -> dict:
    """Build all tensors needed by the model from one rule topology."""
    room_counts, source = sample_room_corner_counts(
        topology,
        corner_rules,
        rng,
    )
    features = build_room_corner_features(topology, room_counts)
    node_features = build_topology_features(
        topology.adjacency,
        topology.node_type_ids,
        topology.valid_mask,
    )
    corner_room, corner_conn, corner_global = build_corner_masks(
        features,
        topology.adjacency,
    )
    node_global, node_conn = build_node_masks(
        topology.valid_mask,
        topology.adjacency,
    )
    layout_batch = build_corner_layout(features[None])
    layout = {
        key: value[0]
        for key, value in layout_batch.items()
    }
    edge_valid, next_index = build_corner_edge_mapping(
        torch.from_numpy(features[None])
    )
    return {
        "features": features,
        "node_features": node_features,
        "node_type_ids": topology.node_type_ids,
        "adjacency": topology.adjacency,
        "corner_room": corner_room,
        "corner_conn": corner_conn,
        "corner_global": corner_global,
        "node_conn": node_conn,
        "node_global": node_global,
        "corner_next_index": next_index[0].numpy(),
        "corner_edge_valid": edge_valid[0].numpy(),
        "corner_layout": layout,
        "room_counts": room_counts,
        "source": source,
        "text": topology.text,
    }


def to_sample(apartment: dict, clip_embedding) -> tuple:
    """Pack one apartment dictionary into the model sample tuple."""
    return (
        torch.from_numpy(apartment["features"]),
        torch.from_numpy(
            np.asarray(clip_embedding, dtype=np.float32)
        ),
        torch.from_numpy(apartment["corner_room"]),
        torch.from_numpy(apartment["corner_conn"]),
        torch.from_numpy(apartment["corner_global"]),
        torch.from_numpy(apartment["node_conn"]),
        torch.from_numpy(apartment["node_global"]),
        torch.from_numpy(apartment["node_features"]),
        torch.from_numpy(apartment["corner_next_index"]),
        torch.from_numpy(apartment["corner_edge_valid"]),
        {
            key: torch.from_numpy(value)
            for key, value in apartment["corner_layout"].items()
        },
        -1,
    )


def generate_prompt_data(
    prompt: str,
    seed: int,
    corner_rules: CornerCountRules | None = None,
) -> tuple[RuleTopology, dict, np.random.Generator]:
    """Generate topology and model input data from one prompt."""
    rng = np.random.default_rng(seed)
    topology = sample_rule_topology(prompt, rng)
    corner_rules = (
        corner_rules
        if corner_rules is not None
        else get_default_corner_count_rules()
    )
    apartment = build_apartment(topology, corner_rules, rng)
    return topology, apartment, rng


__all__ = [
    "CornerCountRules",
    "RuleTopology",
    "build_apartment",
    "build_corner_masks",
    "build_node_masks",
    "build_room_corner_features",
    "build_topology_features",
    "generate_prompt_data",
    "get_default_corner_count_rules",
    "laplacian_pe",
    "parse_prompt_room_counts",
    "random_walk_pe",
    "sample_room_corner_counts",
    "sample_rule_topology",
    "to_sample",
]
