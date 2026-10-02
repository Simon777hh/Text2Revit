"""Build plan-level normalized GT data for the Flow Matching workspace."""

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
    build_corner_layout,
    canonicalize_corner_rings,
    DATA_DIR,
    DEFAULT_OUT_GT,
    DEFAULT_OUT_META,
    DEFAULT_RAW_GT,
    DEFAULT_RAW_MANIFEST,
    denormalize_coords,
    load_plan_order,
    normalize_plans_stretched,
    source_labels,
    valid_corner_mask,
)


def quantiles(values: np.ndarray) -> dict[str, float]:
    points = np.quantile(
        values,
        [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0],
    )
    names = [
        "min",
        "p01",
        "p05",
        "q25",
        "median",
        "q75",
        "p95",
        "p99",
        "max",
    ]
    return {
        name: float(value)
        for name, value in zip(names, points)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", default=str(DEFAULT_RAW_GT))
    parser.add_argument(
        "--plan-order",
        default=str(DEFAULT_RAW_MANIFEST),
    )
    parser.add_argument("--out-gt", default=str(DEFAULT_OUT_GT))
    parser.add_argument("--out-meta", default=str(DEFAULT_OUT_META))
    parser.add_argument(
        "--summary",
        default=str(DATA_DIR / "plan_norm_summary.json"),
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    gt_data = np.load(args.gt, allow_pickle=True)
    raw_features = canonicalize_corner_rings(
        gt_data["features"].astype(np.float32)
    )
    plan_ids = gt_data["plan_ids"].astype(np.int64)
    gt_data.close()

    plan_order = load_plan_order(args.plan_order)
    if len(plan_order) != len(raw_features):
        raise ValueError(
            f"manifest has {len(plan_order)} plans but GT has "
            f"{len(raw_features)}"
        )

    if args.limit > 0:
        raw_features = raw_features[: args.limit]
        plan_ids = plan_ids[: args.limit]
        plan_order = plan_order[: args.limit]

    normalized, metadata = normalize_plans_stretched(
        raw_features,
        plan_ids,
    )
    normalized = canonicalize_corner_rings(normalized)
    corner_layout = build_corner_layout(normalized)
    sources, source_ids = source_labels(plan_order)
    metadata["source"] = sources
    metadata["source_id"] = source_ids

    valid = valid_corner_mask(normalized)
    normalized_xy = normalized[:, :, :2]
    finite = normalized_xy[valid]
    per_plan_extent = (
        normalized_xy.max(axis=1) - normalized_xy.min(axis=1)
    )
    if finite.min() < -1.00001 or finite.max() > 1.00001:
        raise AssertionError(
            f"normalized coordinates escaped [-1,1]: "
            f"{finite.min():.6f}, {finite.max():.6f}"
        )
    if not np.allclose(per_plan_extent, 2.0, atol=1e-4):
        raise AssertionError(
            "normalized x/y extents were not both mapped to [-1,1]"
        )

    reconstruction_error = 0.0
    sample_count = min(len(normalized), 256)
    sample_indices = np.linspace(
        0,
        len(normalized) - 1,
        sample_count,
        dtype=np.int64,
    )
    for plan_index in sample_indices:
        restored = denormalize_coords(
            normalized[plan_index, :, :2],
            int(plan_index),
            metadata,
        )
        plan_valid = valid[plan_index]
        reconstruction_error = max(
            reconstruction_error,
            float(
                np.abs(
                    restored[plan_valid]
                    - raw_features[plan_index, plan_valid, :2]
                ).max()
            ),
        )
    if reconstruction_error > 1e-4:
        raise AssertionError(
            f"inverse normalization error is {reconstruction_error:.6f}"
        )

    out_gt = Path(args.out_gt)
    out_meta = Path(args.out_meta)
    out_gt.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_gt,
        features=normalized.astype(np.float32),
        plan_ids=plan_ids,
        **corner_layout,
    )
    np.savez_compressed(out_meta, **metadata)

    source_summary = {}
    for source in sorted(np.unique(sources).tolist()):
        mask = sources == source
        source_summary[source] = {
            "count": int(mask.sum()),
            "aspect_ratio": quantiles(metadata["aspect_ratio"][mask]),
            "absolute_footprint_long_side": quantiles(
                metadata["bbox_size"][mask].max(axis=1)
            ),
            "wide_count": int(metadata["orientation_wide"][mask].sum()),
            "tall_count": int((~metadata["orientation_wide"][mask]).sum()),
        }

    summary = {
        "plan_count": int(len(normalized)),
        "coordinate_rule": (
            "per-plan bbox center; independent x/y scales; both bbox axes "
            "mapped to [-1,1]; aspect ratio removed"
        ),
        "aspect_ratio_removed": True,
        "normalized_coord_range": [
            float(finite.min()),
            float(finite.max()),
        ],
        "max_inverse_error": reconstruction_error,
        "sources": source_summary,
        "outputs": {
            "gt": str(out_gt.resolve()),
            "meta": str(out_meta.resolve()),
        },
    }
    summary_path = (
        Path(args.summary)
        if args.summary
        else out_gt.parent / "plan_norm_summary.json"
    )
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    print(
        f"saved {out_gt} features={normalized.shape} "
        f"coord_range=[{finite.min():.4f}, {finite.max():.4f}]"
    )
    print(f"saved {out_meta}")
    print(f"saved {summary_path}")
    print(
        f"max reconstruction error={reconstruction_error:.8f}; "
        f"plans={len(normalized)}"
    )
    for source, values in source_summary.items():
        aspect = values["aspect_ratio"]
        print(
            f"{source}: n={values['count']} "
            f"aspect median={aspect['median']:.4f} "
            f"q25={aspect['q25']:.4f} q75={aspect['q75']:.4f}"
        )


if __name__ == "__main__":
    main()
