"""Check the inference dependencies and model files without generating a plan."""
from pathlib import Path
import argparse
import json
import os
import sys

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    os.environ["TEXT2REVIT_BACKEND"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    import transformers
    import shapely
    import pipeline_generate
    backend = Path(__file__).resolve().parent
    clip = backend / "models" / "clip"
    if not clip.is_dir():
        clip = backend.parent / "models" / "clip"
    checkpoint = torch.load(pipeline_generate.find_flow_best_checkpoint(), map_location="cpu", weights_only=True, mmap=True)
    if "model" not in checkpoint or "args" not in checkpoint:
        raise ValueError("Invalid inference checkpoint")
    from transformers import CLIPTokenizer
    CLIPTokenizer.from_pretrained(clip, local_files_only=True)
    if not (clip / "model.safetensors").is_file():
        raise FileNotFoundError("Missing CLIP text weights; run scripts/setup_models.py")
    for filename in ("corner_count_distributions.json", "room_aspect_ranges.json", "raw_recovery_scale_stats.json", "geometry_policy.json"):
        json.loads((backend / "data" / filename).read_text(encoding="utf-8"))
    print(json.dumps({"ok":True,"python":sys.version.split()[0],"cuda":torch.cuda.is_available(),"torch":torch.__version__}))

if __name__ == "__main__":
    main()
