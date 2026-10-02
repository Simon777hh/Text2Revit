"""Encode prompt heads as pooled CLIP vectors, preserving the GT plan order."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from data_utils import DATA_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA_DIR / "prompts.json")
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR / "clip")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "clip")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.batch < 1 or args.limit < 0:
        parser.error("Invalid batch size or limit")
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        parser.error("No prompt rows")
    texts = [row["clip"] for row in rows]
    ids = np.array([row["id"] for row in rows], dtype=np.int64)
    unique = list(dict.fromkeys(texts))
    indices = {text: index for index, text in enumerate(unique)}
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import CLIPTokenizer, CLIPTextModel
    tokenizer = CLIPTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = CLIPTextModel.from_pretrained(args.model_dir, local_files_only=True).to(args.device).eval()
    vectors = np.empty((len(unique), model.config.hidden_size), dtype=np.float16)
    for start in range(0, len(unique), args.batch):
        batch = tokenizer(unique[start:start + args.batch], padding="max_length", truncation=True,
                          max_length=77, return_tensors="pt").to(args.device)
        with torch.no_grad():
            vectors[start:start + len(batch.input_ids)] = model(**batch).pooler_output.float().cpu().numpy()
        print(f"Encoded {min(start + args.batch, len(unique))}/{len(unique)} unique prompts", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pooled_tmp = args.out_dir / "pooled_embeddings.partial.npy"
    pooled = np.lib.format.open_memmap(pooled_tmp, mode="w+", dtype=np.float16,
                                      shape=(len(rows), model.config.hidden_size))
    for start in range(0, len(rows), 1024):
        pooled[start:start + 1024] = vectors[[indices[text] for text in texts[start:start + 1024]]]
    pooled.flush()
    del pooled
    ids_tmp = args.out_dir / "plan_ids.partial.npy"
    np.save(ids_tmp, ids)
    pooled_tmp.replace(args.out_dir / "pooled_embeddings.npy")
    ids_tmp.replace(args.out_dir / "plan_ids.npy")
    summary = {"encoder":"openai/clip-vit-large-patch14", "plan_count":len(rows),
               "unique_prompt_count":len(unique), "shape":[len(rows), model.config.hidden_size],
               "prompt_sha256":hashlib.sha256("\n".join(texts).encode()).hexdigest(),
               "definition":"clip field only; the featuring clause is excluded"}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
