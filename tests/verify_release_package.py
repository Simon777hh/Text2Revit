"""Check the self-extracting installer against the repository's current source."""
import argparse
import hashlib
import importlib.util
import json
import struct
import zipfile
import re
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
release_spec = importlib.util.spec_from_file_location(
    "text2revit_build_release", ROOT / "revit" / "build_release.py")
if release_spec is None or release_spec.loader is None:
    raise ImportError("Cannot load revit/build_release.py")
release_builder = importlib.util.module_from_spec(release_spec)
release_spec.loader.exec_module(release_builder)
BACKEND_FILES = release_builder.BACKEND_FILES
DATA_FILES = release_builder.DATA_FILES
VERSIONS = release_builder.VERSIONS


def verify(path):
    with path.open("rb") as stream:
        stream.seek(-56, 2)
        magic, offset, length, expected = struct.unpack("<8sQQ32s", stream.read(56))
        assert magic == b"T2RPKG01"
        assert offset + length + 56 == path.stat().st_size
        stream.seek(offset)
        digest = hashlib.sha256()
        remaining = length
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            assert chunk
            digest.update(chunk)
            remaining -= len(chunk)
        assert digest.digest() == expected, "Payload hash differs"
    with zipfile.ZipFile(path) as package:
        manifest = json.loads(package.read("release.json"))
        assert manifest["revit_versions"] == list(VERSIONS)
        environment_format = manifest.get("environment_format", "zip")
        assert environment_format in ("zip", "7z")
        remote = manifest.get("remote_environment")
        if environment_format == "7z":
            if remote:
                assert "environment.7z" not in package.namelist()
                combined = hashlib.sha256()
                total = 0
                for part in remote["parts"]:
                    assert re.fullmatch(r"environment\.7z\.[0-9]{3}", part["name"])
                    assert part["url"].startswith("https://") and part["url"].endswith("/" + part["name"])
                    assert 0 < part["size"] < 2**31
                    part_digest = hashlib.sha256()
                    source = path.parent / part["name"]
                    assert source.stat().st_size == part["size"]
                    with source.open("rb") as stream:
                        while chunk := stream.read(4 * 1024**2):
                            combined.update(chunk);part_digest.update(chunk)
                    assert part_digest.hexdigest() == part["sha256"]
                    total += part["size"]
                assert total == remote["size"] and combined.hexdigest() == remote["sha256"]
            else: assert package.getinfo("environment.7z").compress_type == zipfile.ZIP_STORED
            assert package.read("tools/7zr.exe").startswith(b"MZ")
            inventory = json.loads(package.read("environment-files.json"))["files"]
            paths = set()
            for record in inventory:
                name = record["path"]
                path_parts = PurePosixPath(name)
                assert not path_parts.is_absolute() and ".." not in path_parts.parts and ":" not in name
                assert name.startswith(("runtime/", "models/"))
                assert name.lower() not in paths
                paths.add(name.lower())
                assert record["size"] >= 0 and re.fullmatch("[a-f0-9]{64}", record["sha256"])
            assert {"runtime/python.exe", "models/flow_matching_best.pth", "models/clip/model.safetensors"} <= paths
            assert "runtime.zip" not in package.namelist()
            assert b"public domain" in package.read("third_party/LZMA-SDK.txt").lower()
        for name in BACKEND_FILES:
            assert package.read("backend/" + name) == (ROOT / name).read_bytes(), name
        for name in DATA_FILES:
            source = ROOT / "data" / name
            if not source.exists() and name == "room_aspect_samples.npz":
                continue
            actual = package.read("backend/data/" + name)
            if name == "raw_recovery_scale_stats.json":
                expected = json.loads(source.read_text(encoding="utf-8"))
                expected.pop("source", None)
                assert json.loads(actual) == expected
            else:
                assert actual == source.read_bytes(), name
        for year in VERSIONS:
            assert package.read(f"addin/{year}/Text2Revit.Addin.dll") == (
                ROOT / f"revit/Text2Revit.Addin/bin/Release/{year}/Text2Revit.Addin.dll").read_bytes()
        for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
            assert package.read(name) == (ROOT / name).read_bytes()
        assert json.loads(package.read("backend/data/geometry_policy.json"))["minimum_room_areas_m2"] == {
            "kitchen": 2.0, "bathroom": 2.0}
    return {"version": manifest["version"], "size_bytes": path.stat().st_size,
            "payload_hash_verified": True, "source_bytes_verified": True,
            "environment_format": environment_format,
            "online": remote is not None,
            "revit_targets": manifest["revit_versions"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("installer", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = verify(args.installer)
    text = json.dumps(result, indent=2) + "\n"
    if args.report:
        args.report.write_text(text, encoding="utf-8")
    print(text)
