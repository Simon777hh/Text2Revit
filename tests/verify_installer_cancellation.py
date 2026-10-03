"""Drive the actual Windows setup form while a loopback download stalls."""
import argparse
import hashlib
import json
import struct
import subprocess
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def run(stub, work):
    work.mkdir(parents=True, exist_ok=True)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "1000000")
            self.end_headers()
            self.wfile.write(b"x" * 65536)
            self.wfile.flush()
            stop.wait(15)

    stop = threading.Event()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    probe = work / "probe.exe"
    csc = Path("C:/Windows/Microsoft.NET/Framework64/v4.0.30319/csc.exe")
    subprocess.run([str(csc), "/nologo", "/target:exe", "/out:"+str(probe),
                    "/r:System.Windows.Forms.dll", "/r:System.Core.dll",
                    str(Path(__file__).with_name("InstallerCancellationProbe.cs"))], check=True)
    payload = work / "probe.zip"
    release = {"version": "cancel-probe", "revit_versions": [2022], "required_files": [],
               "required_free_bytes": 0, "environment_format": "7z",
               "remote_environment": {"size": 1000000, "sha256": "0"*64,
                   "parts": [{"name": "environment.7z.001", "size": 1000000,
                              "sha256": "0"*64, "url": f"http://127.0.0.1:{server.server_port}/part"}]}}
    with zipfile.ZipFile(payload, "w") as z:
        z.writestr("release.json", json.dumps(release))
    data = payload.read_bytes()
    with probe.open("ab") as f:
        offset = f.tell()
        f.write(data)
        f.write(struct.pack("<8sQQ32s", b"T2RPKG01", offset, len(data), hashlib.sha256(data).digest()))
    try:
        for mode in ("button", "close", "extractor", "python", "copy", "hash"):
            root = work / mode
            if root.exists():
                raise ValueError("Use a fresh work directory")
            result = subprocess.run([str(probe), str(stub), str(root), mode],
                                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20)
            print(mode, result.stdout, result.stderr, flush=True)
            assert result.returncode == 0, mode
            if mode in ("button", "close"):
                assert (root / "downloads/cancel-probe/environment.7z.001.partial").exists()
    finally:
        stop.set()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stub", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.stub.resolve(), args.work_dir.resolve())
