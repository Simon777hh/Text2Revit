# Text2Revit user and developer guide

[中文版本](guide.zh.md)

## Install and use

Download `Text2Revit-Setup.exe` from the Windows release, close Revit, double-click the EXE and click **Install**. The online installer automatically downloads, verifies and installs the runtime and models and registers the tool. Do not download or extract the data attachments manually. Allow at least 12 GB of free disk space. If a download is interrupted, run the installer again; validated parts are reused and partial downloads can resume. A complete offline installer performs the same installation without downloads. No separate archive utility is required.

Start Revit, open a project's floor plan view and select **Text2Revit → Generate Model**. Use one of the English examples:

```text
A small 2-bedroom apartment with 1 bathroom and 1 balcony.
A 3-bedroom apartment with 2 bathrooms and 1 balcony.
A large 4-bedroom apartment with 2 bathrooms and 2 balconies.
```

Change only the size (`small / large`; omit the size for medium), bedroom count, bathroom count and balcony count. One living room and one kitchen are included. At least one bedroom and one bathroom are required; total nodes, including the entry, must not exceed 20.

Click **Generate**. The tool runs inference and creates 3D walls, real hosted doors/windows, open balcony railings, rooms and room tags on the current level. It opens a shaded 3D view automatically. The same button works again from that generated 3D view. When walls already exist, the new layout is placed to their right. The complete generation can be undone in one operation. **Cancel** stops inference.

English is the default interface. Choose **中文** in the installer or prompt dialog for Chinese. The selection is remembered; ribbon labels use that preference on the next Revit startup. Prompts remain in the supported English format in both interface languages.

Users do not need to install Python, Conda or PyTorch separately. The online release includes these dependencies and both model weights as data attachments that the installer retrieves automatically. Internet is required during online installation. Inference then works offline and uses a compatible NVIDIA GPU automatically, with CPU fallback when CUDA is unavailable or fails.

Installation applies to the current Windows user. Other Windows accounts on the same PC install separately. Uninstall through Windows **Installed apps**; generated jobs and family caches are preserved. The current build is unsigned, so Revit can ask whether to load it on first startup. Choose **Always Load** for this tool if you intend to use it.

## Generated dimensions

| Element | Rule |
| --- | --- |
| Exterior / interior wall thickness | 400 / 200 mm |
| Wall height | 3.3 m |
| Balcony exterior boundary | Open Revit railing, 1100 mm high; interior access walls and doors remain |
| Door height | 2.1 m |
| Door / window width | Final Python opening-line length multiplied by the metre scale |
| Living room / bedroom window | Height 1.5–1.7 m; sill 0.9–1.0 m |
| Kitchen window | Height 1.2–1.5 m; sill 0.9–1.0 m |
| Bathroom window | Height 0.6–1.2 m; sill 1.2–1.5 m |

Window heights and sills are sampled in 100 mm increments, including both endpoints of each range. These dimensions are automatic; the first version exposes only the four prompt conditions. Balcony separation lines preserve room areas without enclosing the balcony with full-height walls. The project template must include a railing type, which is duplicated for the balcony.

Doors contain 3D leaves/jambs and plan swing arcs. Windows contain frames, glass and mullions. Dimension-specific families are generated and cached automatically using the corresponding Revit core door/window templates. A normal Revit core-content installation is required; users do not need to locate individual RFA files.

Window geometry and wall openings use the template's actual horizontal insertion-origin plane. The add-in verifies the solid window bottom and top against the requested sill/height after placement. Corrected windows use a new family-cache version to avoid reusing older geometry.

The original ribbon icon combines a blue floor plan with a gold sparkle, at 16 and 32 pixels. Icon selection research included [Lucide house](https://lucide.dev/icons/house).

## Where the code belongs

Everything is already under `revit` in this project. Keep the layout as supplied; no manual file moves are required.

| File or directory | Purpose |
| --- | --- |
| `Build-Release.cmd` | Double-click developer entry point for building the installer |
| `build_release.py` | Compile add-ins, export inference models, pack Python and create one EXE |
| `Text2Revit.Addin/App.cs` | Create the Revit ribbon tab, panel and icon |
| `Text2Revit.Addin/Command.cs` | Coordinate the dialog, inference, family loading and Revit transactions |
| `Text2Revit.Addin/UI/PromptDialog.cs` | Prompt, examples, language selection, progress and cancellation |
| `Text2Revit.Addin/Services/PythonBackendRunner.cs` | Start the bundled Python process and read its JSON |
| `Text2Revit.Addin/Services/WallBuilder.cs` | Build unique exterior/interior wall segments |
| `Text2Revit.Addin/Services/BalconyBuilder.cs` | Create open balcony railings and room separation lines |
| `Text2Revit.Addin/Services/ModelViewBuilder.cs` | Create and frame the generated 3D view, linked to its source floor plan |
| `Text2Revit.Addin/Services/FamilyLibrary.cs` | Generate, load and cache real door/window families |
| `Text2Revit.Addin/Services/OpeningBuilder.cs` | Place hosted family instances using explicit wall IDs |
| `Text2Revit.Addin/Services/RoomBuilder.cs` | Create rooms, tags and room/total area notes |
| `Text2Revit.Addin/Models/PlanData.cs` | Define the JSON fields C# reads |
| `Shared/UiLanguage.cs` | English/Chinese interface preference |
| `Shared/ProcessEnvironment.cs` | Configure an isolated Python child environment and remove duplicate-case variables |
| `Installer/` | Automatic extraction, runtime initialization, registration and uninstall |
| `build_tools/` | Developer-only SDK, API references and conda-pack |
| `build/` | Intermediate exported models and packed runtime |
| `dist/` | Final installer and SHA256 checksum |

Python source is in the repository root. The build copies only inference modules and small statistics files. `inference_runtime.py` does not load training datasets; `revit_geometry.py` defines unique walls, host IDs and physical dimensions; `backend_cli.py` writes atomic JSON/status files and retries invalid layouts up to ten attempts.

## Rebuild the installer

On the current developer PC, double-click `Build-Release.cmd`. The first build needs internet access for the .NET SDK, yearly API references and packaging tooling. Users do not need these tools.

To use a different trained checkpoint:

```powershell
python build_release.py --checkpoint "PATH_TO_CHECKPOINT"
```

The build caches the SDK, unchanged exported models and packed Conda runtime. Conda is used only on the developer PC to prepare that runtime; users run the packaged Python directly.

| Revit year | Target framework |
| --- | --- |
| 2020 | .NET Framework 4.7 |
| 2021–2024 | .NET Framework 4.8 |
| 2025–2026 | .NET 8 Windows |

Each year has its own API references and DLL, sharing one backend and model set. Compilation checks API compatibility; actual runtime compatibility requires testing in each Revit version. Revit 2022 English ribbon generation, balcony railings, 3D view creation and corrected solid window heights were verified. Other Revit years remain untested at runtime. See [the audit](../docs/audit.md).

## Automatically managed locations

- Product: `%LocalAppData%/Text2Revit/releases/VERSION/`
- Registration: `%AppData%/Autodesk/Revit/Addins/YEAR/Text2Revit.addin`
- Jobs and logs: `%LocalAppData%/Text2Revit/jobs/`
- Family cache: `%LocalAppData%/Text2Revit/families/YEAR/`
- Language preference: `%LocalAppData%/Text2Revit/preferences.language`

The installer and add-in manage these paths. Users do not copy files into them.

All room windows avoid the continuous entrance facade. Separate parallel facades remain eligible. Bedrooms connected to balconies have no extra window. Kitchen/bathroom clear areas must be at least 2 m²; invalid layouts retry. Room and total areas are labelled in the preview and native plan. Totals include balconies.
