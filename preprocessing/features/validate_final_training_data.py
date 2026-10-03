"""Validate GT, topology, masks, next-corner mapping and pooled CLIP alignment."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import zipfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from data_utils import DATA_DIR, ROOM_START, ROOM_END
from mmap_cache import load_cached
from preprocessing.features.prepare_attention import build_mapping


def check(name, actual, expected):
    if np.shape(actual) != np.shape(expected) or not np.array_equal(actual, expected):
        raise AssertionError(name + " does not match")
    print(name + ": OK", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    with np.load(DATA_DIR / "plan_norm_gt.npz", allow_pickle=True) as gt:
        ids = gt["plan_ids"]
    with np.load(DATA_DIR / "topology_gt.npz", allow_pickle=True) as topo:
        check("topology.plan_ids", topo["plan_ids"], ids)
        valid, types, adjacency = topo["valid_mask"].astype(bool), topo["node_type_ids"], topo["adjacency"].astype(bool)
        check("num_rooms", topo["num_rooms"], valid.sum(axis=1))
        check("symmetric_adjacency", adjacency, adjacency.transpose(0, 2, 1))
        if np.any(adjacency & ~(valid[:, :, None] & valid[:, None, :])):
            raise AssertionError("Topology references padded rooms")
        bed = valid & (types == 2)
        if np.any(adjacency & bed[:, :, None] & bed[:, None, :]):
            raise AssertionError("Bedroom-bedroom conditioning edges remain")
        expected_counts = np.triu(adjacency, k=1).sum(axis=(1, 2))
        check("num_edges", topo["num_edges"], expected_counts)
        for index, edges in enumerate(topo["edge_lists"]):
            a, b = np.nonzero(np.triu(adjacency[index], k=1))
            if list(map(tuple, edges)) != list(zip(a.tolist(), b.tolist())):
                raise AssertionError(f"Edge list mismatch on plan {index}")
    print("Topology invariants: OK", flush=True)
    with np.load(DATA_DIR / "prompt_areas.npz") as areas:
        check("prompt_areas.plan_ids", areas["plan_ids"], ids)
    clip = np.load(DATA_DIR / "clip" / "pooled_embeddings.npy", mmap_mode="r")
    check("clip.plan_ids", np.load(DATA_DIR / "clip" / "plan_ids.npy"), ids)
    if clip.shape != (len(ids), 768) or not np.isfinite(clip).all():
        raise AssertionError("Invalid pooled CLIP embeddings")
    print("Pooled CLIP [N,768]: OK", flush=True)
    features = load_cached("gt_features")
    with np.load(DATA_DIR / "edge_mapping.npz") as mapping:
        check("edge_mapping.plan_ids", mapping["plan_ids"], ids)
        next_indices, edge_valid = mapping["corner_next_index"], mapping["corner_edge_valid"]
        for index, plan in enumerate(features):
            expected_valid, expected_next = build_mapping(plan)
            if not np.array_equal(next_indices[index], expected_next) or not np.array_equal(edge_valid[index], expected_valid):
                raise AssertionError(f"Next-corner mapping mismatch on plan {index}")
    print("Next-corner mapping: OK", flush=True)
    room_onehot = features[:, :, ROOM_START:ROOM_END]
    corner_valid = room_onehot.sum(axis=-1) > .5
    room_ids = room_onehot.argmax(axis=-1)
    with np.load(DATA_DIR / "masks.npz") as masks:
        check("masks.plan_ids", masks["plan_ids"], ids)
        node_global = valid[:, :, None] & valid[:, None, :]
        check("node_global", masks["node_global"], node_global)
        check("node_conn", masks["node_conn"], node_global & (adjacency | np.eye(20, dtype=bool)))
        for name in ["corner_room", "corner_conn", "corner_global", "self_mask", "door_mask", "gen_mask"]:
            with zipfile.ZipFile(DATA_DIR / "masks.npz") as archive:
                stream = archive.open(name + ".npy")
                version = np.lib.format.read_magic(stream)
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, fortran, dtype = reader(stream)
                if shape != (len(ids), 128, 128) or fortran or dtype != np.dtype(bool):
                    raise AssertionError(f"Invalid mask layout: {name}")
                _validate_mask_stream(name, stream, corner_valid, room_ids, adjacency, len(ids))
            print(name + ": OK", flush=True)
    print(f"All prepared training tensors verified: {len(ids)} plans", flush=True)


def _validate_mask_stream(name, stream, corner_valid, room_ids, adjacency, count):
    for start in range(0, count, 256):
        stop = min(start + 256, count)
        raw = stream.read((stop - start) * 128 * 128)
        actual = np.frombuffer(raw, dtype=bool).reshape(stop - start, 128, 128)
        cv, ri = corner_valid[start:stop], room_ids[start:stop]
        expected = cv[:, :, None] & cv[:, None, :]
        if name in {"corner_room", "self_mask", "corner_conn", "door_mask"}:
            same = ri[:, :, None] == ri[:, None, :]
            if name in {"corner_conn", "door_mask"}:
                connection = adjacency[np.arange(start, stop)[:, None, None], ri[:, :, None], ri[:, None, :]]
                same |= connection
            expected &= same
        if name in {"self_mask", "door_mask", "gen_mask"}:
            expected = ~expected
        if not np.array_equal(actual, expected):
            raise AssertionError(f"{name} mismatch in plans {start}:{stop}")
    stream.close()

if __name__ == "__main__":
    main()
