# Current data pipeline

## Source preparation

Obtain ResPlan and RPLAN separately under their own terms. Place local inputs in
`data/sources/ResPlan/` and `data/sources/RPLAN/`. Dataset files are ignored by Git.
Only load pickle/NumPy object data from trusted sources.

The ResPlan CLI performs three separate stages without overwriting its input:

```powershell
python -m preprocessing.resplan.prepare_resplan clean --input data/sources/ResPlan/ResPlan.pkl --output data/sources/ResPlan/ResPlan_cleaned.pkl
python -m preprocessing.resplan.prepare_resplan rebuild --input data/sources/ResPlan/ResPlan_cleaned.pkl --output data/sources/ResPlan/ResPlan_new.pkl
python -m preprocessing.resplan.prepare_resplan filter --input data/sources/ResPlan/ResPlan_new.pkl --output data/sources/ResPlan/ResPlan_filtered.pkl
python -m preprocessing.rplan.fit_resplan_to_canvas --input data/sources/ResPlan/ResPlan_filtered.pkl --out data/sources/ResPlan/ResPlan_filtered_canvas.pkl
```

Reconstruction uses the retained wall-skeleton closure algorithm, reconstructs
room polygons, snaps walls and refreshes connectivity. `--limit` and `--start`
support small checks. A companion JSON report records kept, removed and failed
plans. RPLAN conversion and clean filtering have separate command-line entries;
use their `--help` for input/output arguments. The converter supports an optional
external RPLAN toolbox through `RPLAN_TOOLBOX_PATH`; it is not redistributed.

The historical dataset was built by earlier scripts, including a batch driver
that was not available as a complete standalone source. The retained CLI exposes
the reconstruction core; exact regeneration of every historical sample from
raw data has not been established. Existing prepared inputs were retained
locally. This limitation does not invalidate verification of their training
contracts below.

## Training features

Run `python -m preprocessing.features.prepare_all_data --dry-run` to list the
exact order, or omit `--dry-run` to execute it:

| Stage | Outputs / purpose |
| --- | --- |
| `extract_raw_data` | Raw corner tokens, topology, edges and ID manifest |
| `prepare_plan_data` | Canonical normalized GT and recovery metadata |
| `prepare_topology_gt` | Padded room types, adjacency and edge lists |
| `remove_bedroom_edges` | Remove bedroom-bedroom conditioning edges |
| `prepare_prompts` | Room-count/size prompts and aligned area metadata |
| `build_masks` | Validity, same-room and connected-room attention masks |
| `build_edge_mapping` | Next-corner indices and valid polygon edges |
| `encode_clip` | Pooled CLIP text embeddings and explicit plan IDs |
| `validate_final_training_data` | Alignment and structural validation |

`--start-at STAGE` resumes from a stage. The current feature builders use the
prepared dataset paths configured in `extract_raw_data.py`; place files at those
paths before execution. No optional prior/loss-target scripts are included.

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
