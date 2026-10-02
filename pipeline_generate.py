"""Production prompt-to-floorplan inference pipeline.

The pipeline emits room polygons, deterministic door/window placements,
post-processing metrics, aspect recovery, and raw-area recovery.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import torch


FLOW_ROOT = Path(__file__).resolve().parent
if str(FLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(FLOW_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import plan_postprocess as postprocess
import inference_runtime as train

if (
    os.name == "nt"
    and os.environ.get("TEXT2REVIT_BACKEND") != "1"
):
    matplotlib.use("TkAgg", force=True)

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolygonPatch

from data_generation import (
    build_apartment,
    get_default_corner_count_rules,
    sample_rule_topology,
    to_sample,
)
from door_window_rules import (
    build_door_window_pieces,
)
from model import FlowMatchingPlanTransformer
from restore_canvas_aspect import (
    fit_records_to_canvas,
    load_room_aspect_stats,
    place_records_on_canvas,
    restore_canvas_geometry,
)


CHECKPOINT_NAME = "flow_matching_best.pth"
CHECKPOINT_CANDIDATES = (
    FLOW_ROOT / "checkpoints" / CHECKPOINT_NAME,
    FLOW_ROOT / "models" / CHECKPOINT_NAME,
    FLOW_ROOT.parent / "models" / CHECKPOINT_NAME,
)
TYPE_COLORS = {
    "living": "#d9d9d9",
    "kitchen": "#8da0cb",
    "bedroom": "#66c2a5",
    "bathroom": "#fc8d62",
    "balcony": "#b3b3b3",
    "storage": "#a37c52",
    "stair": "#e5c494",
    "front_door": "#a63603",
}

# IDE SETTINGS: edit these values and run this file directly.
PROMPT = (
    "A large 4-bedroom apartment with 2 bathrooms and 2 balconies."
)

# None saves to flow_matching/viz/pipeline_generate/.
OUTPUT_DIR = None

# None generates a new plan/noise on every run.
# Set an integer only when a reproducible result is needed.
SEED = None

# Final door-length repair after all scaling, in normalized plan units.
INFERENCE_STEPS = 200
SIZE = "auto"
USE_DISCRETE = True
USE_POSTPROCESS = True
MIN_DOOR_SHARED_RATIO = 0.20
DOOR_LENGTH_RATIO = 0.15
WINDOW_LENGTH_RATIO = 0.15
MAX_DOOR_AREA_CHANGE_RATIO = 0.15


def find_flow_best_checkpoint():
    override = os.environ.get("TEXT2REVIT_CHECKPOINT")
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        return path
    for path in CHECKPOINT_CANDIDATES:
        if path.exists():
            return path
    candidates = "\n".join(str(path) for path in CHECKPOINT_CANDIDATES)
    raise FileNotFoundError(
        "flow_matching_best.pth was not found. Checked:\n"
        f"{candidates}"
    )


def encode_texts(texts, device):
    from transformers import CLIPTextModel, CLIPTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

    bundled_clip = Path(os.environ.get("TEXT2REVIT_CLIP_DIR", str(FLOW_ROOT / "models" / "clip")))
    if not bundled_clip.is_dir():
        bundled_clip = FLOW_ROOT.parent / "models" / "clip"
    model_name = str(bundled_clip) if bundled_clip.is_dir() else "openai/clip-vit-large-patch14"
    tokenizer = CLIPTokenizer.from_pretrained(
        model_name,
        local_files_only=True,
    )
    model = CLIPTextModel.from_pretrained(
        model_name,
        local_files_only=True,
    ).to(device)
    model.eval()
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    with torch.no_grad():
        output = model(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
        )
    pooled = (
        output.pooler_output
        if output.pooler_output is not None
        else output.last_hidden_state[:, 0]
    )
    return pooled.float().cpu().numpy().astype(np.float32)


def load_checkpoint(path):
    return torch.load(
        path,
        map_location=train.DEVICE,
        weights_only=False,
    )


def load_flow_model(path):
    checkpoint = load_checkpoint(path)
    saved_args = checkpoint.get("args", {})
    model = FlowMatchingPlanTransformer(
        hidden_dimension=saved_args.get("hidden", 256),
        num_heads=saved_args.get("heads", 8),
        head_dimension=saved_args.get("head_dim", 64),
        num_node_layers=saved_args.get("node_layers", 6),
        num_corner_layers=saved_args.get("corner_layers", 8),
        detach_bit_from_velocity=not saved_args.get(
            "joint_bit_finetune",
            True,
        ),
        base_logits_weight=saved_args.get(
            "base_logits_weight",
            0.0,
        ),
    ).to(train.DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def infer(
    model,
    sample,
    seed,
    steps,
    use_discrete,
):
    torch.manual_seed(seed)
    if train.DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    args = train.build_parser().parse_args(
        ["--infer-steps", str(steps)]
    )
    args.infer_discrete_steps = (
        int(getattr(args, "infer_discrete_steps", 12))
        if use_discrete
        else 0
    )
    args.infer_low_sigma_discrete = bool(use_discrete)
    args.infer_fixed_low_sigma = bool(use_discrete)
    return train.run_inference(
        model,
        sample,
        args,
        return_discrete=use_discrete,
    )


def build_raw_door_config(
    records,
    base_config,
    minimum_ratio=MIN_DOOR_SHARED_RATIO,
):
    polygons = [
        record.polygon
        for record in records
        if record.polygon is not None and not record.polygon.is_empty
    ]
    if not polygons:
        return copy.deepcopy(base_config), 1.0
    min_x = min(polygon.bounds[0] for polygon in polygons)
    min_y = min(polygon.bounds[1] for polygon in polygons)
    max_x = max(polygon.bounds[2] for polygon in polygons)
    max_y = max(polygon.bounds[3] for polygon in polygons)
    width = max_x - min_x
    height = max_y - min_y
    scale = max(width, height) / 2.0
    area_scale = max(scale * scale, 1e-9)
    config = copy.deepcopy(base_config)
    for name in (
        "coordinate_tolerance",
        "max_edge_move",
        "min_shared_length",
        "max_head_width",
        "wall_segment_tolerance",
        "grid_fill_distance",
        "thin_room_max_short_edge",
        "balcony_snap_tolerance",
        "seam_snap_tolerance",
        "global_seam_tolerance",
        "grid",
        "point_merge_tolerance",
        "coordinate_merge_tolerance",
        "endpoint_snap_tolerance",
        "max_door_edge_move",
    ):
        setattr(config, name, float(getattr(config, name)) * scale)
    config.min_door_shared_length = minimum_ratio * scale
    for name in (
        "max_head_area",
        "max_hole_area",
        "max_self_hole_area",
        "max_hollow_area",
        "overlap_tolerance",
        "min_overlap_action_area",
    ):
        setattr(
            config,
            name,
            float(getattr(config, name)) * area_scale,
        )
    config.max_door_edge_area_change = (
        MAX_DOOR_AREA_CHANGE_RATIO * area_scale
    )
    return config, scale


def serialize_door_window_piece(piece):
    if piece is None:
        return None
    start = np.asarray(piece.long_start, dtype=float)
    end = np.asarray(piece.long_end, dtype=float)
    delta = end - start
    width = float(np.linalg.norm(delta))
    direction = (
        delta / width
        if width > 1e-9
        else np.asarray([1.0, 0.0], dtype=float)
    )
    normal = np.asarray(
        [-direction[1], direction[0]],
        dtype=float,
    )
    center = 0.5 * (start + end)
    return {
        "kind": piece.kind,
        "source": piece.source,
        "outward_blocked": bool(piece.outward_blocked),
        "owner_room_id": int(piece.owner_room_id),
        "target_room_id": (
            int(piece.target_room_id)
            if piece.target_room_id is not None
            else None
        ),
        "opening": {
            "coordinate_system": "raw_canvas_px",
            "type": piece.kind,
            "host": (
                "shared_wall"
                if piece.target_room_id is not None
                else "exterior_wall"
            ),
            "reference_line": [
                [float(start[0]), float(start[1])],
                [float(end[0]), float(end[1])],
            ],
            "center": [
                float(center[0]),
                float(center[1]),
            ],
            "direction": [
                float(direction[0]),
                float(direction[1]),
            ],
            "normal": [
                float(normal[0]),
                float(normal[1]),
            ],
            "width": width,
            "owner_room_id": int(piece.owner_room_id),
            "target_room_id": (
                int(piece.target_room_id)
                if piece.target_room_id is not None
                else None
            ),
        },
        "polygon": [
            [float(x), float(y)]
            for x, y in piece.polygon.exterior.coords
        ],
    }


def draw_records(
    axis,
    records,
    title,
    canvas_size=255.0,
    overlays=None,
    room_areas_m2=None,
):
    room_records = records
    if overlays is not None:
        room_records = [
            record
            for record in records
            if record.type_name not in {"front_door", "door"}
        ]
    ordered = [
        record
        for record in room_records
        if record.type_name not in {"front_door", "door"}
    ] + [
        record
        for record in room_records
        if record.type_name in {"front_door", "door"}
    ]
    for record in ordered:
        axis.add_patch(
            PolygonPatch(
                np.asarray(record.polygon.exterior.coords),
                closed=True,
                facecolor=TYPE_COLORS.get(
                    record.type_name,
                    "#cccccc",
                ),
                edgecolor="black",
                linewidth=0.8,
                alpha=1.0,
            )
        )
        if record.type_name not in {"front_door", "door"} and room_areas_m2 is not None:
            point = record.polygon.representative_point()
            area = room_areas_m2[str(record.room_id)]
            axis.text(point.x, point.y,
                      f"{record.type_name.replace('_', ' ').title()}\n{area:.1f} m²",
                      ha="center", va="center", fontsize=8,
                      bbox={"facecolor": "white", "alpha": .75, "edgecolor": "none", "pad": 1})
    if overlays is not None:
        overlay_specs = (
            (overlays["windows"], "#1f77b4", 1.8),
            (overlays["doors"], "#e31a1c", 1.8),
            (
                [overlays["front_door"]]
                if overlays.get("front_door") is not None
                else [],
                "#6a3d9a",
                2.0,
            ),
        )
        for pieces, color, width in overlay_specs:
            for piece in pieces:
                axis.add_patch(
                    PolygonPatch(
                        np.asarray(
                            piece["polygon"],
                            dtype=float,
                        ),
                        closed=True,
                        facecolor=color,
                        edgecolor=color,
                        linewidth=width,
                        alpha=0.95,
                    )
                )
    axis.set_xlim(0.0, canvas_size)
    axis.set_ylim(0.0, canvas_size)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_title(title, fontsize=10)
    if room_areas_m2 is not None:
        axis.text(.5, -.035,
                  f"Total: {sum(room_areas_m2.values()):.1f} m² (including balconies; walls excluded)",
                  transform=axis.transAxes, ha="center", fontsize=9)


def resolve_output_path(output_dir):
    base = (
        Path(output_dir)
        if output_dir is not None
        else FLOW_ROOT / "outputs" / "inference"
    )
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base / f"floorplan_{stamp}.png"


def generate_plan_data(prompt, seed_override=None):
    checkpoint = find_flow_best_checkpoint()
    steps = INFERENCE_STEPS
    seed = (
        int(np.random.SeedSequence().entropy) & 0xFFFFFFFF
        if seed_override is None and SEED is None
        else int(
            SEED if seed_override is None else seed_override
        )
    )
    rng = np.random.default_rng(seed)
    topology = sample_rule_topology(prompt, rng)
    corner_rules = get_default_corner_count_rules()
    apartment = build_apartment(
        topology,
        corner_rules,
        rng,
    )
    clip_embedding = encode_texts(
        [apartment["text"]],
        train.DEVICE,
    )[0]
    sample = to_sample(apartment, clip_embedding)
    model = load_flow_model(checkpoint)
    result = infer(
        model,
        sample,
        seed,
        steps,
        use_discrete=USE_DISCRETE,
    )
    _, source_xy, output_xy, node_features, _, layout = result

    raw_records = postprocess.records_from_xy(
        output_xy[0],
        layout,
        node_features[0],
    )
    adjacency_pairs = postprocess.adjacency_pairs(
        apartment["adjacency"]
    )
    processed = copy.deepcopy(raw_records)
    postprocess_metrics = {}
    if USE_POSTPROCESS:
        processed, postprocess_metrics = (
            postprocess.postprocess_records_general_rules(
                processed,
                adjacency_pairs,
                postprocess.PostProcessConfig(),
                ensure_doorable=False,
            )
        )

    requested_size = SIZE
    size_source = "cli"
    if requested_size == "auto":
        if topology.counts.size is not None:
            requested_size = topology.counts.size
            size_source = "prompt"
        else:
            requested_size = "medium"
            size_source = "default_medium"
    restore_info = restore_canvas_geometry(
        processed,
        load_room_aspect_stats(),
        room_count=int(topology.valid_mask.sum()),
        rng=rng,
        size=requested_size,
    )
    base_door_config = postprocess.PostProcessConfig()
    raw_door_config, door_scale = build_raw_door_config(
        processed,
        base_door_config,
    )
    door_repaired = postprocess.ensure_doorable_connected_edges(
        processed,
        adjacency_pairs,
        raw_door_config,
    )
    if door_repaired:
        restore_info["fit_before_door_repair"] = restore_info["fit"]
        restore_info["fit"] = fit_records_to_canvas(processed)
        restore_info["canvas"] = place_records_on_canvas(processed)
    from geometry_cleanup import clean_thin_geometry
    scale_stats = json.loads((FLOW_ROOT / "data" / "raw_recovery_scale_stats.json").read_text(encoding="utf-8"))
    pixel_to_meter = float(scale_stats["resplan_meter_scale"]["meters_per_raw_pixel"]["median"])
    restore_info["thin_geometry"] = clean_thin_geometry(processed, pixel_to_meter)
    unfit_after = postprocess.unfit_connected_edges(
        processed,
        adjacency_pairs,
        raw_door_config,
    )
    restore_info["door_length"] = {
        "coordinate_scale": door_scale,
        "minimum_shared_length": raw_door_config.min_door_shared_length,
        "door_length": DOOR_LENGTH_RATIO,
        "window_length": WINDOW_LENGTH_RATIO,
        "repaired": door_repaired,
        "remaining_failures": unfit_after,
    }
    living_id = topology.room_ids_by_type["living"][0]
    door_window_pieces = build_door_window_pieces(
        processed,
        adjacency_pairs,
        living_id,
        rng=rng,
        door_length_ratio=DOOR_LENGTH_RATIO,
        door_thickness_ratio=0.03,
        door_gap_ratio=0.03,
        window_length_ratio=WINDOW_LENGTH_RATIO,
        window_thickness_ratio=0.03,
        window_gap_ratio=0.04,
        pixel_to_meter=pixel_to_meter,
    )
    overlays = {
        "front_door": serialize_door_window_piece(
            door_window_pieces["front_door"]
        ),
        "doors": [
            serialize_door_window_piece(piece)
            for piece in door_window_pieces["doors"]
        ],
        "windows": [
            serialize_door_window_piece(piece)
            for piece in door_window_pieces["windows"]
        ],
    }
    metadata = {
        "prompt": prompt,
        "clip_prompt": apartment["text"],
        "model": "flow",
        "checkpoint": str(checkpoint),
        "seed": seed,
        "inference_steps": steps,
        "discrete_head": USE_DISCRETE,
        "postprocess": USE_POSTPROCESS,
        "door_window_rules": overlays,
        "room_counts": topology.counts.as_dict(),
        "prompt_size": topology.counts.size,
        "area_size_source": size_source,
        "requested_size": requested_size,
        "room_count": int(topology.valid_mask.sum()),
        "corner_counts": {
            str(key): int(value)
            for key, value in apartment["room_counts"].items()
        },
        "edge_lists": [
            [int(room_a), int(room_b)]
            for room_a, room_b in topology.edge_lists
        ],
        "node_type_ids": [
            int(value)
            for value in topology.node_type_ids[
                topology.valid_mask
            ]
        ],
        "postprocess_metrics": postprocess_metrics,
        "restore": restore_info,
        "rooms": postprocess.records_to_dicts(processed),
        "door_window_schema": (
            "opening reference_line + width + normal"
        ),
    }
    return processed, overlays, metadata


def main():
    import argparse
    global INFERENCE_STEPS, SEED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--steps", type=int, default=INFERENCE_STEPS)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--clip-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.checkpoint:
        os.environ["TEXT2REVIT_CHECKPOINT"] = str(args.checkpoint)
    if args.clip_dir:
        os.environ["TEXT2REVIT_CLIP_DIR"] = str(args.clip_dir)
    INFERENCE_STEPS, SEED = args.steps, args.seed
    from backend_cli import default_pixel_to_meter, flatten_openings, validate_door_connections
    from revit_geometry import build_revit_geometry
    from plan_quality import validate_room_areas
    seed = args.seed
    for attempt in range(10):
        try:
            processed, overlays, metadata = generate_plan_data(args.prompt, seed)
            openings = flatten_openings(overlays)
            validate_door_connections(metadata["rooms"], openings, metadata["edge_lists"])
            walls, _ = build_revit_geometry(metadata["rooms"], openings,
                                            default_pixel_to_meter(), metadata["seed"])
            areas = validate_room_areas(metadata["rooms"], default_pixel_to_meter(), walls=walls)
            break
        except ValueError as error:
            if attempt == 9:
                raise ValueError(f"No valid layout after 10 attempts: {error}") from error
            print(f"Layout rejected: {error}. Generating another layout.", flush=True)
            seed = None
            SEED = None
    output_path = resolve_output_path(args.output_dir)
    figure, axis = plt.subplots(1, 1, figsize=(8, 8))
    draw_records(
        axis,
        processed,
        (
            f"{metadata['clip_prompt']}\n"
            f"model=flow | steps={metadata['inference_steps']} | "
            f"size={metadata['restore']['area']['occupancy_group']}"
        ),
        overlays=overlays,
        room_areas_m2=areas,
    )
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        facecolor="white",
    )
    metadata["output"] = str(output_path)
    metadata["room_areas_m2"] = areas
    metadata["total_area_m2"] = sum(areas.values())
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(output_path)


if __name__ == "__main__":
    main()
