"""Check the self-extracting installer against the repository's current source."""
import argparse
import hashlib
import json
import struct
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "revit"))
from build_release import BACKEND_FILES, DATA_FILES, VERSIONS


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
