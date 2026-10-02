"""Download the public CLIP text encoder for subsequent offline inference."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "models" / "clip")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "0"
    from transformers import CLIPTextModel, CLIPTokenizer
    name = "openai/clip-vit-large-patch14"
    args.output.mkdir(parents=True, exist_ok=True)
    CLIPTextModel.from_pretrained(name).save_pretrained(args.output, safe_serialization=True)
    CLIPTokenizer.from_pretrained(name).save_pretrained(args.output)
    print(f"CLIP text encoder saved to {args.output}")


if __name__ == "__main__":
    main()
