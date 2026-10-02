# Publication audit

Audit date: 2026-10-02. The publication directory is `final`; the parent workspace
contains local assets and preserved historical work and is not the upload root.

## Fixed

* All room windows exclude the entrance's continuous exterior facade, including
  collinear split segments. Separate parallel facades remain eligible. Bedrooms
  connected to balconies keep their access door and receive no extra window.
* Door placement now searches clear alternative positions. Openings keep at
  least 200 mm between reference segments. Perpendicular doors sharing a room
  keep 350 mm; opposed parallel doors with overlapping projections require the
  larger door width plus 100 mm. These checks span different host walls, rather
  than checking only overlap on one wall. Unplaceable required doors reject the
  layout instead of being silently omitted.
* A physical cleanup step removes narrow room appendages below 600 mm by
  transferring small patches to a neighboring room. Small narrow gaps can be
  filled. A whole narrow room, disconnected body, excessive repair or residual
  narrow appendage rejects the layout. Occupied floor area is preserved. The
  reported sample's 370 mm bedroom strip was removed before door placement.
* Only kitchen and bathroom have area thresholds, each 2 m². Preview estimates
  subtract wall footprints. Revit plans label each native room area and their
  sum, including balconies. Invalid geometry, missing doors and small areas
  retry up to ten attempts, including requests with an initial explicit seed.
* Window solid height/sill validation and the corrected cached window family
  origin are retained. Balcony outside edges use railings and separation lines.
* RPLAN filtering matches polygons by room name rather than assuming topology
  indices survive removal of the front-door placeholder.
* CLIP now stores explicit aligned plan IDs. Current pooled vectors were
  regenerated from 269 unique prompts and expanded to 75,379 ordered rows.
* Compressed GT caching and mask validation stream large arrays to avoid the
  validator's previous memory allocation failure. Training requires CLIP IDs.
* Personal model paths, obsolete file references, decorative comments and unused
  demo entry points were removed. Thirty-five unreachable postprocessing
  definitions (1,843 lines) were removed after checking the production call
  graph. Historical source remains archived locally.
* The installer manifest was restored and the installer successfully rebuilt.
  The release now includes the current backend, physical policies and notices.

## Verification

| Check | Result |
| --- | --- |
| Geometry/opening/area regressions | 22 tests passed |
| Prepared GT, topology, masks, mapping, CLIP | All 75,379 rows verified |
| ResPlan clean → rebuild → filter | Two real source plans retained through all stages |
| Real packaged-runtime GPU inference | Passed with the corrected sample seed |
| Native Revit 2022 construction and exports | Nine rooms, real doors/windows, open railings and room/total area notes |
| Entrance facade / balcony bedroom / spacing checks | Accepted JSON passed |
| Revit API targets | 2020–2026 DLLs compiled |
| Offline installer | Payload hash and packaged source bytes verified |
| Git publication | Fresh local source repository; weights excluded, inference weights bundled in installer |

`sample/floorplan.png`, `sample/model3d.png`, `sample/plan.json` and
`sample/model-metadata.json` describe one matching accepted layout. Detailed
logs and generated RVT/build products are local assets, excluded from Git.

The final installer is version `1.0.0-20261002211905`, 3,901,737,855 bytes.
The corrected native sample total is 106.2 m², with 28 wall segments. Prepared
training data, downloaded CLIP and build caches were physically moved to the
parent workspace's `_local_assets/` directory. The active sample RVT and loaded
DLLs remain in ignored local output directories so this Revit session remains
usable.

## Repository cleanup

The public feature pipeline contains only current raw/normalized GT, topology,
prompt, mask, edge mapping, pooled CLIP and their validator. Optional priors,
alternate loss-target generators, exploratory analysis and old visualizers were
moved out. Unique historical experiments, dataset backups and earlier source
were archived instead of deleted. Raw/prepared datasets, downloaded CLIP,
SDK/runtime caches, build outputs and installer binaries are local assets.

Only confirmed derived junk is deleted: Python bytecode caches, an accidental
file containing an `rg` error, and the redundant payload ZIP after its complete
bytes were verified inside the installer. Active installed runtime files remain
at their original location so the Revit registration continues to work. No
parent Git history, original supplied checkpoint or unique dataset was deleted.

## Remaining limits

* Native runtime verification covers Revit 2022. Other Revit versions compiled,
  but still need actual runtime testing. A clean external-PC install remains
  unverified. Binaries are unsigned.
* NuGet's online vulnerability feed was unavailable during compilation
  (`NU1900` warnings). Successful builds do not establish a clean dependency
  vulnerability audit.
* Exact raw-to-historical-dataset reproduction is not established; the earlier
  complete batch driver was unavailable. The current retained reconstruction
  core was checked on two real samples and current training tensors were fully
  verified. See [data_pipeline.md](data_pipeline.md).
* Minimum areas and conservative spacing checks improve output quality but do
  not establish architectural or accessibility code compliance. Door-spacing
  guards do not simulate every native hinge/swing configuration.
* The license reserves the author's rights and prohibits unauthorized reuse and
  commercial use; it cannot physically prevent copying. Third-party MIT rights
  and separately obtained dataset terms remain applicable. An independent legal
  or complete dependency-provenance review was not performed.

The local Revit registration points to the updated area-label assembly for its
next startup. The current session's old ribbon assembly stays loaded until
Revit restarts; sample validation used the freshly compiled production builders.
