@echo off
cd /d "%~dp0"
if defined TEXT2REVIT_BUILD_PYTHON (
  "%TEXT2REVIT_BUILD_PYTHON%" build_release.py --python "%TEXT2REVIT_BUILD_PYTHON%"
) else (
  python build_release.py
)
if errorlevel 1 echo Build failed. See the error above and ..\docs\development.md.
pause
