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

By default, the original runtime and model files are compressed together into
a solid LZMA2 archive with a 128 MiB dictionary. The installer bundles the
public-domain LZMA SDK extractor, validates archive paths against a file
inventory and checks each extracted file's size and SHA-256 before initializing
Python. No user-installed archive tool is required. The intermediate runtime
ZIP is a build cache and is expanded before LZMA2 compression. Use
`--compression zip` for the previous ZIP-based payload format.

### GitHub online installer

Build the small online installer and data attachments for a specific tag:

```powershell
python revit/build_release.py --checkpoint checkpoints/flow_matching_best.pth --online-release-url https://github.com/Simon777hh/Text2Revit/releases/download/v1.0.0
```

Upload `Text2Revit-Setup.exe`, all `environment.7z.*` parts and `SHA256.txt` from
`revit/dist/` to that release. Keep asset names and the configured tag unchanged:
their URLs are embedded in the installer. Each data part is at most 1.5 GiB.
Publish all assets together so the installer can retrieve its complete payload.

The installer uses HTTPS and the Windows system proxy, reuses validated cached
parts, requests byte ranges for interrupted downloads and checks per-part and
combined SHA-256 values. Successful installation removes its download cache.
Inference works offline after installation. Isolated `--test-root` checks also
permit loopback HTTP for local fixtures; normal installation requires HTTPS.
The extractor and .NET file operations support long installation paths. The
installer regression script covers extraction, traversal rejection, integrity,
long paths, resumed downloads and cache reuse.

The first release targets Windows x64 only. No macOS or Linux installer is
included.

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

The complete LZMA2 offline installer is approximately 2.44 GiB, compared with
3.63 GiB for the earlier ZIP build. The online setup executable is approximately
3.4 MiB and downloads two data attachments of 1.50 GiB and 0.93 GiB. These can
be hosted together in a GitHub release. Distribute the complete offline installer
through a download service that accepts its size. Confirm the host's current
per-file limits before publishing. See
[GitHub's release documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).

## Checks

```powershell
python -m unittest discover -s tests -v
python -m preprocessing.features.validate_final_training_data
python tests/verify_opening_placement.py outputs/verification/plan.json
python tests/verify_installer_compression.py --stub revit/Installer/bin/Release/net47/Text2Revit.Setup.exe --extractor revit/build_tools/lzma/sdk/bin/x64/7zr.exe --work-dir outputs/installer-compression-tests
revit/dist/Text2Revit-Setup.exe --verify-only
```

`tests/verify_opening_placement.py` independently checks a real generated JSON:
all windows avoid the continuous entrance facade, and bedrooms connected to
balconies have no window. Unit tests cover collinear split walls, different
parallel walls, balcony access, minimum areas and wall-footprint subtraction.
No automated test is claimed to establish architectural code compliance.
