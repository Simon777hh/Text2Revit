# Current data pipeline

## Source preparation

Obtain ResPlan and RPLAN separately under their own terms. Place local inputs in
`data/sources/ResPlan/` and `data/sources/RPLAN/`. Dataset files are ignored by Git.
Only load pickle/NumPy object data from trusted sources.

One ResPlan command runs cleaning, wall reconstruction and snapping, topology
refresh, filtering and uniform canvas fitting in order:

```powershell
python -m preprocessing.resplan.prepare_resplan --input data/sources/ResPlan/ResPlan.pkl --output data/sources/ResPlan/ResPlan_filtered_canvas.pkl
```

Only `ResPlan_filtered_canvas.pkl` is written as the prepared dataset; no
cleaned, rebuilt or pre-canvas pickle versions are emitted. The separate
canvas-fitting CLI has been removed. `corner_features.py` and
`wall_reconstruction.py` are internal algorithms used by this single entry.
Owned geometry fits inside `[0, 255]` while preserving aspect ratio. Neighbor
geometry follows the same transform but is excluded from the fitting bounds.
Physical area metadata remains in m²; coordinate areas, topology node areas and
wall depth follow the canvas scale.

`--limit` and `--start` support small checks. Input files are never overwritten;
existing outputs require `--overwrite`. A companion `.report.json` records raw
source indices, IDs, processing stages and kept/removed/failed counts.
The saved `canvas_transform` allows GT extraction and source-area calculations
to recover reconstruction units in memory, preserving source-unit tolerances
without keeping another dataset file. GT extraction and prompt generation both
default to this one ResPlan file. CLIP size labels use its canvas geometry.
Historical canvas files without this metadata cannot recover their pre-canvas
units; regenerate them with this command for the unified training workflow.

RPLAN also has one entry for PNG conversion, connectivity/geometry filtering
and CLIP area calibration. Prepare ResPlan first, then run:

```powershell
python -m preprocessing.rplan.prepare_rplan
```

The default input is `data/sources/RPLAN/dataset/floorplan_dataset/`; the only
prepared dataset is `data/sources/RPLAN/RPLAN_filtered.pkl`, accompanied by a
`.report.json`. `image_conversion.py` is an internal conversion module, shared
wall algorithms live under `resplan/`. `--input` can also name a previously
converted source pickle for migration. `--start`, `--limit`, `--workers` and
`--overwrite` control batch processing. Raw sources and reference files cannot
be overwritten. A batch with no surviving plans writes a report and fails
without writing a dataset or a NaN area multiplier.

RPLAN geometry stays in its original pixel units for GT. `clip_area_scale`
records the former scaling stage's ResPlan/RPLAN median room-area ratio. Prompt
preparation applies that multiplier to node areas in memory, giving the same
CLIP size labels without storing separate cleaned/scaled/scaled-canvas files.
RPLAN `area` fields are coordinate areas, not measured m². Use `--scale-area`
to supply an explicit multiplier; calibration from a small `--limit` batch is
for verification and is not a substitute for the full dataset's calibration.
Historical separate files remain valid explicit feature-builder inputs, but
the new defaults require the unified file with its recorded multiplier.

The converter supports an optional external RPLAN toolbox through
`RPLAN_TOOLBOX_PATH`; it is not redistributed. Alignment behavior depends on
its availability. Per-plan conversion failures and filter reasons are reported.

The historical dataset was built by earlier scripts, including a batch driver
that was not available as a complete standalone source. The retained CLI exposes
the reconstruction core; exact regeneration of every historical sample from
raw data has not been established. Existing prepared inputs were retained
locally. This limitation does not invalidate verification of their training
contracts below.

## Training features

The committed [`data/`](../data/README.md) contains small statistical priors and
geometry settings. Raw datasets, GT, topology, masks and CLIP embeddings are
generated locally and are excluded from Git.

Run `python -m preprocessing.features.prepare_all_data --dry-run` to list the
exact order, or omit `--dry-run` to execute it:

| Stage | Outputs / purpose |
| --- | --- |
| `extract_raw_data` | Raw corner tokens, topology, edges and ID manifest |
| `prepare_plan_data` | Canonical normalized GT and recovery metadata |
| `prepare_topology_gt` | Padded room types, adjacency, bedroom-edge removal and updated graph encodings |
| `prepare_prompts` | Room-count/size prompts and aligned area metadata |
| `prepare_attention` | Attention masks, next-corner indices and valid polygon edges in one GT pass |
| `encode_clip` | Pooled CLIP text embeddings and explicit plan IDs |
| `validate_final_training_data` | Alignment and structural validation |

`--start-at STAGE` resumes from a stage. The current feature builders use the
prepared dataset paths configured in `extract_raw_data.py`; place files at those
paths before execution. No optional prior/loss-target scripts are included.
The seven-stage driver replaces the former nine-stage flow. Topology is written
once after applying the bedroom policy; `prepare_attention` writes both
`masks.npz` and `edge_mapping.npz` with unchanged field names and aligned IDs.

Corner GT is `[N,128,52]`: XY, room-centroid XY, 20 room identity channels
and 28 corner identity channels. Topology allows at most 20 rooms. CLIP is
`[N,768]`, with `plan_ids.npy` explicitly matching the same row order. Identical
prompts are encoded once, then expanded back to that order.

The validator checks plan IDs across GT/topology/prompts/CLIP/masks/edge mapping,
topology symmetry, padded nodes, bedroom edge policy, edge lists, next-corner
mapping and each mask against its definition. Large arrays use memory maps or
streamed compressed-array reads. Run it directly after any feature change:

```powershell
python -m preprocessing.features.validate_final_training_data
```

## Inference

Prompt counts create topology conditioning. The trained transformer integrates
Gaussian noise with the Heun solver, restores canvas aspect/physical scale,
repairs room boundaries, places openings and builds physical wall/railing data.
After physical scale recovery, `geometry_cleanup.py` transfers small narrow
appendages to neighbors and fills small gaps. Required doors search alternative
positions and all openings pass a physical spacing check across host walls.
`backend_cli.py` validates door connectivity and minimum clear kitchen/bathroom
areas, retries invalid layouts, writes JSON atomically and generates a labelled
PNG. C# reads this JSON and constructs native Revit elements.

The area check applies only to kitchen and bathroom, each at least 2 m². Preview
areas subtract generated wall footprints; native plan labels use Revit room
areas. Both totals include balconies. No dataset-derived minimum-area thresholds
or other-room minimum areas are used.
