# Data files

[中文](README.zh.md)

The six committed data files total approximately **1.13 MiB**. They contain
statistical priors and settings used by the code, rather than complete floor
plans, training tensors or model weights.

| File | Purpose | Used by |
| --- | --- | --- |
| `combined_area_stats_natural.json` | Canvas room/total-area quantiles for small, medium and large training prompt labels | `preprocessing/features/prepare_prompts.py` |
| `corner_count_distributions.json` | Room corner-count distributions for sampled topology conditions | `topology_rules.py` |
| `geometry_policy.json` | Minimum kitchen and bathroom areas in m² | `plan_quality.py` |
| `raw_recovery_scale_stats.json` | Coordinate-to-metre recovery and layout geometry statistics | `backend_cli.py`, `pipeline_generate.py` |
| `room_aspect_ranges.json` | Room aspect-ratio ranges used during geometry recovery | `restore_canvas_aspect.py` |
| `room_aspect_samples.npz` | Room aspect-ratio samples used during geometry recovery | `restore_canvas_aspect.py` |

The installer bundles the five inference files; the area quantile file is used
for training prompt preparation. All six have current callers and are retained.
Changing these files can change generated layouts or training labels.

Local preparation uses the same directory:

- `sources/ResPlan/ResPlan.pkl`: separately obtained raw source.
- `sources/ResPlan/ResPlan_filtered_canvas.pkl`: the single prepared ResPlan dataset.
- `sources/RPLAN/dataset/floorplan_dataset/`: separately obtained RPLAN PNGs.
- `sources/RPLAN/RPLAN_filtered.pkl`: the single prepared RPLAN dataset; its
  `clip_area_scale` records the CLIP label calibration without a second file.
- Generated `.npz` tensors, `prompts.json`, `raw_manifest.json`, `clip/` and
  `mmap_cache/`: local training data and caches, excluded from Git.

See the [data pipeline](../docs/data_pipeline.md) for preparation and validation.
Weights belong in [`checkpoints/`](../checkpoints/README.md).
