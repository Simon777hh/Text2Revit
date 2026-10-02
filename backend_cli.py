"""Stable JSON backend used by the Revit C# add-in."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


os.environ["TEXT2REVIT_BACKEND"] = "1"

FLOW_ROOT = Path(__file__).resolve().parent
if str(FLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(FLOW_ROOT))

import pipeline_generate as pipeline  # noqa: E402
from pipeline_generate import generate_plan_data  # noqa: E402
from revit_geometry import build_revit_geometry
from plan_quality import validate_room_areas

MAX_GENERATION_ATTEMPTS = 10


def default_pixel_to_meter():
    path = FLOW_ROOT / "data" / "raw_recovery_scale_stats.json"
    try:
        stats = json.loads(path.read_text(encoding="utf-8"))
        return float(
            stats["resplan_meter_scale"]["meters_per_raw_pixel"][
                "median"
            ]
        )
    except Exception:
        return 0.05


def flatten_openings(door_window_rules):
    pieces = []
    front_door = door_window_rules.get("front_door")
    if front_door is not None:
        pieces.append(front_door)
    pieces.extend(door_window_rules.get("doors", []))
    pieces.extend(door_window_rules.get("windows", []))
    return [
        piece["opening"]
        for piece in pieces
        if piece is not None and "opening" in piece
    ]


def validate_response(response):
    if not response["rooms"]:
        raise ValueError("backend generated no rooms")
    for room in response["rooms"]:
        polygon = room.get("polygon") or []
        if len(polygon) < 3:
            raise ValueError(
                f"room {room.get('room_id')} has invalid polygon"
            )
    for index, opening in enumerate(response["openings"], start=1):
        line = opening.get("reference_line") or []
        if len(line) != 2:
            raise ValueError(
                f"opening #{index} has invalid reference_line"
            )
        if float(opening.get("width", 0.0)) <= 0.0:
            raise ValueError(
                f"opening #{index} has non-positive width"
            )


def validate_door_connections(rooms, openings, edges):
    functional = {r["room_id"] for r in rooms if r["type"] not in {"door", "front_door"}}
    required = {frozenset((a, b)) for a, b in edges if a in functional and b in functional}
    actual = {frozenset((o["owner_room_id"], o["target_room_id"])) for o in openings
              if o["type"] != "window" and o.get("target_room_id") is not None}
    missing = required - actual
    if missing:
        raise ValueError(f"connected rooms have no generated doorway: {sorted(tuple(sorted(pair)) for pair in missing)}")
    if not any(o["type"] != "window" and o["host"] == "exterior_wall" for o in openings):
        raise ValueError("the plan has no entrance door")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--status")
    args = parser.parse_args()

    request = json.loads(
        Path(args.request).read_text(encoding="utf-8")
    )
    prompt = request["prompt"]
    seed = request.get("seed")
    def message(english, chinese):
        return chinese if request.get("language", "en") == "zh" else english
    def status(message):
        if args.status:
            path = Path(args.status)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"message": message}, ensure_ascii=False), encoding="utf-8")
            temporary.replace(path)
        print(message, flush=True)

    device = "NVIDIA GPU" if pipeline.train.DEVICE.type == "cuda" else "CPU"
    status(message(f"Loading models and generating the floor plan ({device})…",f"正在加载模型并生成户型（{device}）…"))
    last_error = None
    for attempt in range(MAX_GENERATION_ATTEMPTS):
        try:
            processed, overlays, metadata = generate_plan_data(prompt, seed)
        except ValueError as error:
            last_error = error
            seed = None
            if attempt + 1 < MAX_GENERATION_ATTEMPTS:
                status(message(f"Geometry check failed: {error}. Generating another layout (attempt {attempt + 2}/{MAX_GENERATION_ATTEMPTS})…", f"几何检查未通过：{error}。正在重新生成（尝试 {attempt + 2}/{MAX_GENERATION_ATTEMPTS}）…"))
            continue
        except RuntimeError as error:
            # CUDA availability does not guarantee enough VRAM or a supported GPU.
            error_text = str(error).lower()
            if pipeline.train.DEVICE.type != "cuda" or not any(term in error_text for term in ("cuda", "cudnn", "cublas", "out of memory", "nvidia")):
                raise
            pipeline.train.DEVICE = pipeline.torch.device("cpu")
            status(message("GPU inference is unavailable. Switching to CPU…","显卡推理不可用，正在自动改用 CPU…"))
            processed, overlays, metadata = generate_plan_data(prompt, seed)
        openings = flatten_openings(overlays)
        try:
            # Repair diagnostics flag target shared-wall lengths, not missing doors.
            # The original placement code can fit a shorter door on such a wall.
            validate_door_connections(metadata["rooms"], openings, metadata["edge_lists"])
            walls, railings = build_revit_geometry(metadata["rooms"], openings,
                default_pixel_to_meter(), metadata["seed"])
            room_areas_m2 = validate_room_areas(metadata["rooms"], default_pixel_to_meter(), walls=walls)
            break
        except ValueError as error:
            last_error = error
            seed = None
            if attempt + 1 < MAX_GENERATION_ATTEMPTS:
                status(message(f"Layout check failed: {error}. Generating another layout (attempt {attempt + 2}/{MAX_GENERATION_ATTEMPTS})…",f"户型检查未通过：{error}。正在重新生成（尝试 {attempt + 2}/{MAX_GENERATION_ATTEMPTS}）…"))
    else:
        raise ValueError(f"Unable to generate a valid Revit plan after {MAX_GENERATION_ATTEMPTS} attempts: {last_error}")
    status(message("Inference finished. Preparing Revit data…","推理完成，正在准备 Revit 数据…"))

    response = {
        "schema_version": "1.0",
        "coordinate_system": "raw_canvas_px",
        "pixel_to_meter": default_pixel_to_meter(),
        "seed": metadata["seed"],
        "prompt": prompt,
        "clip_prompt": metadata["clip_prompt"],
        "room_areas_m2": room_areas_m2,
        "total_area_m2": sum(room_areas_m2.values()),
        "area_definition": "Estimated clear floor area including balconies; wall footprints excluded",
        "rooms": metadata["rooms"],
        "walls": walls,
        "railings": railings,
        "openings": openings,
        "debug": {
            "edge_lists": metadata["edge_lists"],
            "node_type_ids": metadata["node_type_ids"],
            "restore": metadata["restore"],
        },
    }
    validate_response(response)
    output_path = Path(args.response)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = pipeline.plt.subplots(figsize=(9, 9))
    pipeline.draw_records(axis, processed, metadata["clip_prompt"], overlays=overlays,
                          room_areas_m2=room_areas_m2)
    figure.tight_layout()
    preview_path = output_path.with_suffix(".png")
    preview_temporary = preview_path.with_suffix(".tmp")
    figure.savefig(preview_temporary, format="png", dpi=180, facecolor="white")
    pipeline.plt.close(figure)
    preview_temporary.replace(preview_path)
    response["preview_image"] = str(preview_path)
    temporary = output_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(response, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output_path)
    status(message("Generation complete","生成完成"))


if __name__ == "__main__":
    main()
