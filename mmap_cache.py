"""Create and load uncompressed .npy caches for the large training arrays."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile

import numpy as np

from data_utils import DATA_DIR


CACHE_DIR = DATA_DIR / "mmap_cache"

CACHE_SPECS = {
    "gt_features": (
        DATA_DIR / "plan_norm_gt.npz",
        "features",
        np.float32,
    ),
}


def cache_path(name):
    if name not in CACHE_SPECS:
        raise KeyError(f"unknown mmap cache: {name}")
    return CACHE_DIR / f"{name}.npy"


def _metadata_path(name):
    return cache_path(name).with_suffix(".meta.json")


def _source_signature(name):
    source_path, key, _ = CACHE_SPECS[name]
    stat = source_path.stat()
    with np.load(source_path, allow_pickle=True) as source:
        plan_ids = np.ascontiguousarray(
            source["plan_ids"].astype(np.int64)
        )
    return {
        "source": source_path.name,
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "key": key,
        "plan_ids_sha256": hashlib.sha256(
            plan_ids.tobytes()
        ).hexdigest(),
    }


def _cache_is_current(name):
    output_path = cache_path(name)
    metadata_path = _metadata_path(name)
    if not output_path.exists() or not metadata_path.exists():
        return False
    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    if metadata.get("signature") != _source_signature(name):
        return False
    try:
        cached = np.load(
            output_path,
            mmap_mode="r",
            allow_pickle=False,
        )
    except (OSError, ValueError):
        return False
    return (
        list(cached.shape) == metadata.get("shape")
        and str(cached.dtype) == metadata.get("dtype")
    )


def build_cache(name, force=False):
    if name not in CACHE_SPECS:
        raise KeyError(f"unknown mmap cache: {name}")
    output_path = cache_path(name)
    if not force and _cache_is_current(name):
        return output_path

    source_path, key, dtype = CACHE_SPECS[name]
    signature = _source_signature(name)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.stem}.{os.getpid()}.tmp.npy")
    with zipfile.ZipFile(source_path) as archive:
        with archive.open(key + ".npy") as source, temporary.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        array = np.load(temporary, mmap_mode="r", allow_pickle=False)
        if array.dtype != np.dtype(dtype):
            raise ValueError(f"Unexpected cache dtype for {name}: {array.dtype}")
        metadata = {
            "signature": signature,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
        del array
        os.replace(temporary, output_path)
    temporary_metadata = _metadata_path(name).with_name(
        f"{_metadata_path(name).stem}.{os.getpid()}.tmp.json"
    )
    with open(temporary_metadata, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=True)
    os.replace(temporary_metadata, _metadata_path(name))
    print(f"built mmap cache: {output_path}")
    return output_path


def ensure_cache(force=False):
    paths = {
        name: build_cache(name, force=force)
        for name in CACHE_SPECS
    }
    return paths


def load_cached(name):
    path = cache_path(name)
    if not _cache_is_current(name):
        build_cache(name)
    return np.load(path, mmap_mode="r", allow_pickle=False)


if __name__ == "__main__":
    for cache_name, cache_file in ensure_cache().items():
        print(f"{cache_name}: {cache_file}")
