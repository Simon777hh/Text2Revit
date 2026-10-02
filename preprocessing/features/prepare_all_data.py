"""Run the complete independent Flow Matching data pipeline."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_RELEASE_ROOT = _Path(__file__).resolve().parents[2]
if str(_RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_RELEASE_ROOT))


import subprocess
import argparse
import sys
from pathlib import Path


FLOW_DIR = Path(__file__).resolve().parent
SCRIPTS = [
    "extract_raw_data.py",
    "prepare_plan_data.py",
    "prepare_topology_gt.py",
    "remove_bedroom_edges.py",
    "prepare_prompts.py",
    "build_masks.py",
    "build_edge_mapping.py",
    "encode_clip.py",
    "validate_final_training_data.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-at", choices=[s[:-3] for s in SCRIPTS], default=SCRIPTS[0][:-3])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    stages = SCRIPTS[SCRIPTS.index(args.start_at + ".py"):]
    for script in stages:
        command = [sys.executable, "-m", "preprocessing.features." + script[:-3]]
        print("Running: " + " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=_RELEASE_ROOT, check=True)


if __name__ == "__main__":
    main()
