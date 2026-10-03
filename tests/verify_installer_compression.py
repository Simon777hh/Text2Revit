"""Exercise the compiled installer's extraction checks in an isolated directory."""
import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def assemble(stub, tool, archive, inventory, output, version, archive_format="7z", remote=None):
    payload = output.with_suffix(".zip")
    release = {"version": version, "revit_versions": [2022], "required_files": [],
               "required_free_bytes": 0, "environment_format": archive_format}
    if remote: release["remote_environment"] = remote
    with zipfile.ZipFile(payload, "w") as z:
        z.write(tool, "tools/7zr.exe")
        if not remote: z.write(archive, "environment.7z")
        z.writestr("environment-files.json", json.dumps({"files": inventory}))
        z.writestr("release.json", json.dumps(release))
    data = payload.read_bytes()
    with output.open("wb") as stream:
        stream.write(stub.read_bytes())
        offset = stream.tell()
        stream.write(data)
        stream.write(struct.pack("<8sQQ32s", b"T2RPKG01", offset, len(data), hashlib.sha256(data).digest()))


def run(stub, tool, destination):
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("Use an empty test directory; existing files will not be overwritten")
    source = destination / "source"
    inventory = []
    for name, content in [("runtime/a.txt", b"runtime probe"), ("models/a.txt", b"model probe"), ("runtime/long/"+"x"*170+".txt", b"long path probe")]:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        inventory.append({"path": name, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()})
    archive = destination / "valid.7z"
    subprocess.run([str(tool), "a", str(archive), *[r["path"] for r in inventory], "-bso0", "-bsp0"], cwd=source, check=True)
    escaped = destination / "unexpected.7z"
    shutil.copy2(archive, escaped)
    subprocess.run([str(tool), "rn", str(escaped), "models/a.txt", "../escaped.txt", "-bso0", "-bsp0"], check=True)
    changed = [dict(record) for record in inventory]
    changed[0]["sha256"] = "0" * 64
    duplicate = inventory + [dict(inventory[0])]
    outside = [dict(record) for record in inventory]
    outside[0]["path"] = "runtime/../../escaped.txt"
    cases = [
        ("valid", archive, inventory, "The runtime relocation script is missing.", "7z", "--test-root"),
        ("archive-traversal", escaped, inventory, "outside the installation directory", "7z", "--test-root"),
        ("checksum", archive, changed, "Extracted file checksum differs", "7z", "--test-root"),
        ("duplicate", archive, duplicate, "Invalid environment file record", "7z", "--verify-only"),
        ("inventory-traversal", archive, outside, "outside the installation directory", "7z", "--verify-only"),
        ("format", archive, inventory, "Invalid release manifest", "unknown", "--verify-only"),
    ]
    for name, packed, records, expected_error, archive_format, command in cases:
        installer = destination / (name + ".exe")
        assemble(stub, tool, packed, records, installer, name, archive_format)
        test_root = destination / (name + "-install")
        args = [str(installer), command]
        if command == "--test-root": args.append(str(test_root))
        result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        assert result.returncode != 0, (name, result.stdout, result.stderr)
        assert expected_error in result.stderr, (name, result.stdout, result.stderr)
        assert not (test_root / "releases" / name).exists(), name
        assert not (test_root / "escaped.txt").exists(), name
        assert not (destination / "escaped.txt").exists(), name
        print(f"PASS: {name}", flush=True)
    data = archive.read_bytes()
    pieces = {"environment.7z.001": data[:len(data)//2], "environment.7z.002": data[len(data)//2:]}
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass

        def do_GET(self):
            name = self.path.rsplit("/", 1)[-1]
            body = pieces.get(name)
            if body is None:
                self.send_error(404)
                return
            requested_range = self.headers.get("Range")
            requests.append((name, requested_range))
            start = int(requested_range.removeprefix("bytes=").removesuffix("-")) if requested_range else 0
            self.send_response(206 if requested_range else 200)
            if requested_range: self.send_header("Content-Range", f"bytes {start}-{len(body)-1}/{len(body)}")
            self.send_header("Content-Length", str(len(body)-start))
            self.end_headers()
            self.wfile.write(body[start:])

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}/"
    remote = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "parts": [
        {"name": name, "url": base+name, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        for name, content in pieces.items()]}
    try:
        for name in ("online-resume", "online-cache", "online-part-checksum", "online-archive-checksum"):
            requests.clear()
            description = json.loads(json.dumps(remote))
            test_root = destination / (name+"-install")
            cache = test_root / "downloads" / name
            cache.mkdir(parents=True)
            first_name, first_bytes = next(iter(pieces.items()))
            if name == "online-resume": (cache/(first_name+".partial")).write_bytes(first_bytes[:10])
            if name == "online-cache": (cache/first_name).write_bytes(first_bytes)
            if name == "online-part-checksum": description["parts"][0]["sha256"] = "0"*64
            if name == "online-archive-checksum": description["sha256"] = "0"*64
            expected = "Downloaded file checksum differs" if name == "online-part-checksum" else "Downloaded archive checksum differs" if name == "online-archive-checksum" else "The runtime relocation script is missing."
            installer = destination / (name+".exe")
            assemble(stub, tool, archive, inventory, installer, name, remote=description)
            result = subprocess.run([str(installer), "--test-root", str(test_root)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
            assert result.returncode != 0 and expected in result.stderr, (name, result.stdout, result.stderr)
            assert not (test_root/"releases"/name).exists(), name
            if name == "online-resume": assert requests[0] == (first_name, "bytes=10-")
            if name == "online-cache": assert all(item[0] != first_name for item in requests)
            if name != "online-part-checksum":
                for filename, content in pieces.items(): assert (cache/filename).read_bytes() == content
            print(f"PASS: {name}", flush=True)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    return {"checks": len(cases)+4, "passed": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stub", type=Path, required=True)
    parser.add_argument("--extractor", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.stub.resolve(), args.extractor.resolve(), args.work_dir), indent=2))
