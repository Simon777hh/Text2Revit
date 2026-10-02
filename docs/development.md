# Development and release packaging

## C# files

`revit/Text2Revit.Addin/App.cs` registers the ribbon button. `Command.cs` opens
the prompt dialog and creates the model in one undo group. `UI/PromptDialog.cs`
handles language, examples, inference progress and cancellation.

`Services/PythonBackendRunner.cs` launches the bundled backend and consumes its
JSON. `JsonPlanReader.cs` validates the schema. `WallBuilder`, `BalconyBuilder`,
`FamilyLibrary`, `OpeningBuilder`, `RoomBuilder` and `ModelViewBuilder` create
native 3D elements, hosted door/window families, area notes and the model view.
`OpeningGeometryValidator` checks the actual window solid against sill/height
dimensions, including family insertion-origin offsets.

The installer copies the runtime/models/add-ins into the current user's local
application data, writes version-specific `.addin` registrations and adds an
uninstaller. End users do not place C# source or DLLs manually.

## Build

From the repository root:

```powershell
python -m pip install -r requirements.txt
python scripts/setup_models.py
python revit/build_release.py --checkpoint checkpoints/flow_matching_best.pth
```

Or use `revit/Build-Release.cmd`. `TEXT2REVIT_BUILD_PYTHON` selects the prepared
Python environment to bundle. Packaging needs a Windows Conda environment and
Conda tooling; Conda is only a developer dependency. First builds download the
.NET SDK, API reference packages and packing tools into ignored `revit/build_tools`.
`REVIT2022_API_DIR` (and equivalent years) selects locally installed API files.

The builder compiles Revit 2020–2026 plugins, exports the author's checkpoint
without optimizer state, copies the CLIP text encoder, packs Python/PyTorch,
assembles `revit/dist/Text2Revit-Setup.exe` and verifies its appended payload hash.
Runtime/build caches are reusable. It also writes `SHA256.txt`.

| Revit | Target framework |
| --- | --- |
| 2020 | .NET Framework 4.7 |
| 2021–2024 | .NET Framework 4.8 |
| 2025–2026 | .NET 8 Windows |

All seven targets were compiled locally. Native model construction was tested
in Revit 2022 only. A clean external-PC installation and other Revit runtime
versions remain release validation tasks. The current binaries are unsigned.

## Distribution

The source repository excludes model weights, generated datasets, downloaded
CLIP files, build caches and installer outputs through `.gitignore`. Source
inference and release builds require a separately supplied compatible checkpoint.

The full training checkpoint is kept locally. The builder removes optimizer
state and bundles only inference weights and architecture arguments, together
with the CLIP encoder, in the installer. End users do not download weights
separately. Git LFS is not required for the source repository.

The current Windows installer is approximately 3.63 GiB. Distribute it through
a download service that accepts its size, together with the generated
`SHA256.txt`; link that download from the release page. Confirm the host's
current per-file limits before publishing. See
[GitHub's release documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).

## Checks

```powershell
python -m unittest discover -s tests -v
python -m preprocessing.features.validate_final_training_data
python tests/verify_opening_placement.py outputs/verification/plan.json
revit/dist/Text2Revit-Setup.exe --verify-only
```

`tests/verify_opening_placement.py` independently checks a real generated JSON:
all windows avoid the continuous entrance facade, and bedrooms connected to
balconies have no window. Unit tests cover collinear split walls, different
parallel walls, balcony access, minimum areas and wall-footprint subtraction.
No automated test is claimed to establish architectural code compliance.
