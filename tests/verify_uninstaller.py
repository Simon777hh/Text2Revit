"""Verify the standalone uninstall UI and cleanup in isolated product roots."""
import argparse
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET


def fixture(root):
    preserved = {}
    for relative in ("jobs/example/plan.json", "families/2022/door.rfa", "preferences.language"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"keep user data")
        preserved[path] = path.read_bytes()
    for relative in ("releases/probe/runtime/python.exe", "releases/probe/models/weights.pth",
                     "downloads/probe/environment.7z.001.partial"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"remove product data")
    for year in range(2020, 2027):
        path = root / f"test-addins/{year}/Text2Revit.addin"
        path.parent.mkdir(parents=True, exist_ok=True)
        document = ET.Element("RevitAddIns")
        addin = ET.SubElement(document, "AddIn")
        assembly = root / "releases/probe/addin/plugin.dll" if year != 2026 else root.parent / "other-product/plugin.dll"
        ET.SubElement(addin, "Assembly").text = str(assembly)
        ET.ElementTree(document).write(path, encoding="utf-8")
        if year == 2026:
            preserved[path] = path.read_bytes()
        other = path.with_name("OtherProduct.addin")
        other.write_bytes(b"keep other plugin")
        preserved[other] = other.read_bytes()
    return preserved


def check(root, preserved):
    assert not list((root / "releases").iterdir())
    assert not list((root / "downloads").iterdir())
    for year in range(2020, 2026):
        assert not (root / f"test-addins/{year}/Text2Revit.addin").exists()
    for path, data in preserved.items():
        assert path.read_bytes() == data, path


def run(tool, work):
    if work.exists():
        raise ValueError("Use a fresh work directory")
    work.mkdir(parents=True)
    probe = work / "RenamedTool.exe"
    subprocess.run(["C:/Windows/Microsoft.NET/Framework64/v4.0.30319/csc.exe",
                    "/nologo", "/target:exe", "/out:"+str(probe),
                    "/r:System.Windows.Forms.dll", "/r:System.Core.dll",
                    str(Path(__file__).with_name("UninstallerProbe.cs"))], check=True)
    root = work / "ui-root"
    preserved = fixture(root)
    # No command-line switch or appended installation payload is supplied.
    subprocess.run([str(probe), str(tool), str(root)], check=True, timeout=15)
    check(root, preserved)
    print("PASS: standalone UI defaults to uninstall, cleans product files, preserves user data and unrelated add-ins")
    for _ in range(2):
        subprocess.run([str(tool), "--test-uninstall", str(root)], check=True, timeout=15)
        check(root, preserved)
    print("PASS: repeated uninstall is safe")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.tool.resolve(), args.work_dir.resolve())
