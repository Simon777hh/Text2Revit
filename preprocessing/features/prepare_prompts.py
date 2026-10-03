"""Generate user prompts and structured area rules for Flow Matching data.

The full prompt keeps the old user-facing form:

    A large 3-bedroom apartment with 2 bathrooms and 1 balcony,
    featuring a medium living room and small bedrooms.

Only the head before ``, featuring`` is sent to CLIP. The featuring clause
and per-room area levels are stored as structured rules instead.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_RELEASE_ROOT = _Path(__file__).resolve().parents[2]
if str(_RELEASE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_RELEASE_ROOT))


import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data_utils import (
    CANVAS_AREA,
    DATA_DIR,
    DEFAULT_OUT_GT,
    DEFAULT_RAW_MANIFEST,
    PROJECT_ROOT,
    room_polygon_areas,
)


from preprocessing.resplan.prepare_resplan import source_coordinates
from preprocessing.rplan.prepare_rplan import clip_area_plan

DEFAULT_RPLAN_PKL = (
    PROJECT_ROOT / "RPLAN" / "RPLAN_filtered.pkl"
)
DEFAULT_RESPLAN_PKL = PROJECT_ROOT / "ResPlan" / "ResPlan_filtered_canvas.pkl"
DEFAULT_CLIP_RPLAN_PKL = (
    PROJECT_ROOT / "RPLAN" / "RPLAN_filtered.pkl"
)
DEFAULT_CLIP_RESPLAN_PKL = (
    PROJECT_ROOT / "ResPlan" / "ResPlan_filtered_canvas.pkl"
)
DEFAULT_CLIP_AREA_STATS = (
    _RELEASE_ROOT / "data" / "combined_area_stats_natural.json"
)
DEFAULT_OUT_JSON = DATA_DIR / "prompts.json"
DEFAULT_OUT_NPZ = DATA_DIR / "prompt_areas.npz"
DEFAULT_STATS = DATA_DIR / "prompt_area_stats.json"
DEFAULT_PLAN_GT = DEFAULT_OUT_GT
DEFAULT_TOPOLOGY = DATA_DIR / "topology_gt.npz"

ROOM_TYPES = [
    "living",
    "kitchen",
    "bedroom",
    "bathroom",
    "balcony",
    "storage",
    "stair",
]
TYPE_INDEX = {name: index for index, name in enumerate(ROOM_TYPES)}
ALL_ROOM_TYPES = ROOM_TYPES + ["front_door", "door"]
ALL_TYPE_INDEX = {
    name: index for index, name in enumerate(ALL_ROOM_TYPES)
}
FEATURE_ORDER = [
    "living",
    "bedroom",
    "bathroom",
    "kitchen",
    "balcony",
    "storage",
    "stair",
]
SINGULAR = {
    "living": "living room",
    "kitchen": "kitchen",
    "bedroom": "bedroom",
    "bathroom": "bathroom",
    "balcony": "balcony",
    "storage": "storage",
    "stair": "stair",
}
LEVEL_TO_ID = {"small": 0, "medium": 1, "large": 2}


def plural(count: int, word: str) -> str:
    if word == "balcony" and count > 1:
        return "balconies"
    if count > 1 and not word.endswith("s"):
        return word + "s"
    return word


def plan_room_data(plan: dict) -> tuple[dict[str, list[float]], float]:
    conn = plan.get("conn") or {}
    by_type: dict[str, list[float]] = defaultdict(list)
    total = 0.0
    for room_type, area in zip(
        conn.get("room_types") or [],
        conn.get("node_areas") or [],
    ):
        if room_type not in TYPE_INDEX:
            continue
        value = float(area)
        by_type[room_type].append(value)
        total += value
    return by_type, total


def load_plan_order(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_area_ratio_values(
    normalized_areas: np.ndarray,
    node_type_ids: np.ndarray,
) -> dict[str, list[float]]:
    room_values: dict[str, list[float]] = {
        room_type: [] for room_type in ROOM_TYPES
    }
    functional = (
        (node_type_ids >= 0)
        & (node_type_ids < len(ROOM_TYPES))
    )
    ratios = normalized_areas / CANVAS_AREA
    ratios = np.where(functional, ratios, 0.0)
    for type_index, room_type in enumerate(ROOM_TYPES):
        selected = functional & (node_type_ids == type_index)
        room_values[room_type].extend(ratios[selected].tolist())
    return room_values


def quantile_stats(values: list[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "n": 0,
            "mu": 0.0,
            "sigma": 0.0,
            "q30": 0.0,
            "q70": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "n": int(array.size),
        "mu": float(array.mean()),
        "sigma": float(array.std()),
        "q30": float(np.percentile(array, 30)),
        "q70": float(np.percentile(array, 70)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def level_from_stats(
    value: float,
    stats: dict[str, float | int],
) -> str:
    value = float(value)
    if value < float(stats["q30"]):
        return "small"
    if value > float(stats["q70"]):
        return "large"
    return "medium"


def allowed_levels(total_level: str) -> set[str]:
    if total_level == "small":
        return {"large", "medium"}
    if total_level == "large":
        return {"medium", "small"}
    return {"large", "small"}


def build_prompt(
    plan: dict,
    clip_plan: dict,
    normalized_areas: np.ndarray,
    normalized_node_type_ids: np.ndarray,
    area_stats: dict[str, dict],
    total_stats: dict,
) -> dict:
    conn = plan.get("conn") or {}
    node_types = list(conn.get("room_types") or [])
    node_areas = [float(value) for value in (conn.get("node_areas") or [])]
    if not node_types or len(node_types) != len(node_areas):
        raise ValueError("invalid conn room_types/node_areas")
    if len(node_types) > 20:
        raise ValueError("plan has more than 20 rooms")
    if (
        len(normalized_areas) != 20
        or len(normalized_node_type_ids) != 20
    ):
        raise ValueError("normalized area/type rows must contain 20 rooms")

    by_type: dict[str, list[float]] = defaultdict(list)
    for room_type, area in zip(node_types, node_areas):
        if room_type in TYPE_INDEX:
            by_type[room_type].append(area)

    bedroom_count = len(by_type.get("bedroom", []))
    bathroom_count = len(by_type.get("bathroom", []))
    balcony_count = len(by_type.get("balcony", []))
    total_area = float(
        sum(
            node_areas[index]
            for index, name in enumerate(node_types)
            if name in TYPE_INDEX
        )
    )
    functional = (
        (normalized_node_type_ids >= 0)
        & (normalized_node_type_ids < len(ROOM_TYPES))
    )
    normalized_ratios = np.where(
        functional,
        normalized_areas / CANVAS_AREA,
        0.0,
    )
    normalized_total_area = float(normalized_areas[functional].sum())
    _, clip_total_area = plan_room_data(clip_plan)
    total_level = level_from_stats(clip_total_area, total_stats)

    size_word = "" if total_level == "medium" else total_level
    if bedroom_count > 0:
        head = f"A {size_word} {bedroom_count}-bedroom apartment"
    else:
        head = f"A {size_word} apartment"
    head = head.replace("A  ", "A ")
    head += (
        f" with {bathroom_count} "
        f"{plural(bathroom_count, 'bathroom')}"
    )
    if balcony_count > 0:
        head += (
            f" and {balcony_count} "
            f"{plural(balcony_count, 'balcony')}"
        )

    allowed = allowed_levels(total_level)
    featured: list[str] = []
    node_level_ids = np.full(20, -1, dtype=np.int8)
    node_type_ids = np.full(20, -1, dtype=np.int8)
    node_area_ratios = np.zeros(20, dtype=np.float32)
    node_areas_padded = np.zeros(20, dtype=np.float32)
    raw_node_areas_padded = np.zeros(20, dtype=np.float32)
    feature_mask = np.zeros(20, dtype=bool)
    room_counts = np.zeros(len(ROOM_TYPES), dtype=np.int16)

    for room_index, (room_type, area) in enumerate(
        zip(node_types, node_areas)
    ):
        node_areas_padded[room_index] = normalized_areas[room_index]
        raw_node_areas_padded[room_index] = area
        if room_type in ALL_TYPE_INDEX:
            node_type_ids[room_index] = ALL_TYPE_INDEX[room_type]
        if room_type not in TYPE_INDEX:
            continue
        node_area_ratios[room_index] = normalized_ratios[room_index]
        type_index = TYPE_INDEX[room_type]
        room_counts[type_index] += 1
        level = level_from_stats(
            node_area_ratios[room_index],
            area_stats[room_type],
        )
        node_level_ids[room_index] = LEVEL_TO_ID[level]

    for room_type in FEATURE_ORDER:
        values = by_type.get(room_type, [])
        if not values:
            continue
        levels = [
            ("small", "medium", "large")[
                int(node_level_ids[room_index])
            ]
            for room_index, current_type in enumerate(node_types)
            if current_type == room_type
        ]
        level_counts = {
            level: levels.count(level)
            for level in ("large", "medium", "small")
            if level in allowed and levels.count(level) > 0
        }
        if not level_counts:
            continue
        pieces = []
        for level in ("large", "medium", "small"):
            count = level_counts.get(level, 0)
            if count == 0:
                continue
            noun = (
                SINGULAR[room_type]
                if count == 1
                else plural(count, room_type)
            )
            if count == 1:
                pieces.append(f"a {level} {noun}")
            else:
                pieces.append(f"{count} {level} {noun}")
        featured.append(" and ".join(pieces))

    for room_index, room_type in enumerate(node_types):
        if room_type not in TYPE_INDEX:
            continue
        level = level_from_stats(
            node_area_ratios[room_index],
            area_stats[room_type],
        )
        if level in allowed:
            feature_mask[room_index] = True

    area_clause = ""
    if featured:
        area_clause = ", featuring " + " and ".join(featured)
    user_prompt = head + area_clause + "."
    clip_prompt = head + "."

    return {
        "user_prompt": user_prompt,
        "clip_prompt": clip_prompt,
        "area_clause": area_clause,
        "total_area": np.float32(total_area),
        "clip_total_area": float(clip_total_area),
        "normalized_total_area": np.float32(normalized_total_area),
        "total_size_id": np.int8(LEVEL_TO_ID[total_level]),
        "node_areas": node_areas_padded,
        "raw_node_areas": raw_node_areas_padded,
        "node_area_ratios": node_area_ratios,
        "node_size_ids": node_level_ids,
        "node_type_ids": node_type_ids,
        "feature_mask": feature_mask,
        "room_counts": room_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rplan", default=str(DEFAULT_RPLAN_PKL))
    parser.add_argument("--resplan", default=str(DEFAULT_RESPLAN_PKL))
    parser.add_argument(
        "--clip-rplan",
        default=str(DEFAULT_CLIP_RPLAN_PKL),
    )
    parser.add_argument(
        "--clip-resplan",
        default=str(DEFAULT_CLIP_RESPLAN_PKL),
    )
    parser.add_argument(
        "--clip-stats",
        default=str(DEFAULT_CLIP_AREA_STATS),
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_RAW_MANIFEST),
    )
    parser.add_argument("--plan-gt", default=str(DEFAULT_PLAN_GT))
    parser.add_argument("--topology", default=str(DEFAULT_TOPOLOGY))
    parser.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    parser.add_argument("--out-npz", default=str(DEFAULT_OUT_NPZ))
    parser.add_argument("--stats", default=str(DEFAULT_STATS))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    manifest = load_plan_order(args.manifest)
    if args.limit > 0:
        manifest = manifest[: args.limit]

    plan_gt = np.load(args.plan_gt, allow_pickle=True)
    topology = np.load(args.topology, allow_pickle=True)
    normalized_features = plan_gt["features"]
    normalized_node_type_ids = topology["node_type_ids"].astype(np.int16)
    plan_ids_from_gt = plan_gt["plan_ids"].astype(np.int64)
    plan_ids_from_topology = topology["plan_ids"].astype(np.int64)
    if not np.array_equal(plan_ids_from_gt, plan_ids_from_topology):
        raise ValueError("plan GT and topology plan ids are not aligned")
    if len(normalized_features) < len(manifest):
        raise ValueError("plan GT has fewer rows than the manifest")
    normalized_features = normalized_features[: len(manifest)]
    normalized_node_type_ids = normalized_node_type_ids[: len(manifest)]
    normalized_areas = room_polygon_areas(normalized_features)

    room_values = collect_area_ratio_values(
        normalized_areas,
        normalized_node_type_ids,
    )
    area_stats = {
        room_type: quantile_stats(room_values[room_type])
        for room_type in ROOM_TYPES
    }
    functional = (
        (normalized_node_type_ids >= 0)
        & (normalized_node_type_ids < len(ROOM_TYPES))
    )
    normalized_total_values = (
        normalized_areas * functional
    ).sum(axis=1)
    normalized_total_stats = quantile_stats(normalized_total_values)
    with open(args.clip_stats, "r", encoding="utf-8") as handle:
        clip_stats = json.load(handle)
    total_stats = clip_stats["total"]

    count = len(manifest)
    node_areas = np.zeros((count, 20), dtype=np.float32)
    raw_node_areas = np.zeros((count, 20), dtype=np.float32)
    node_area_ratios = np.zeros((count, 20), dtype=np.float32)
    node_size_ids = np.full((count, 20), -1, dtype=np.int8)
    node_type_ids = np.full((count, 20), -1, dtype=np.int8)
    feature_mask = np.zeros((count, 20), dtype=bool)
    room_counts = np.zeros((count, len(ROOM_TYPES)), dtype=np.int16)
    total_area = np.zeros(count, dtype=np.float32)
    normalized_total_area = np.zeros(count, dtype=np.float32)
    clip_total_area = np.zeros(count, dtype=np.float64)
    total_size_ids = np.full(count, -1, dtype=np.int8)
    plan_ids = np.zeros(count, dtype=np.int64)
    sources = np.empty(count, dtype="<U16")
    source_ids = np.zeros(count, dtype=np.int64)
    source_indices = np.zeros(count, dtype=np.int64)
    prompt_rows: list[dict | None] = [None] * count

    for source_name, source_path, clip_path in (
        ("rplan", args.rplan, args.clip_rplan),
        ("resplan", args.resplan, args.clip_resplan),
    ):
        source_positions = [
            index
            for index, row in enumerate(manifest)
            if row["source"] == source_name
        ]
        if not source_positions:
            continue
        with open(source_path, "rb") as handle:
            plans = pickle.load(handle)
        if Path(source_path).resolve() == Path(clip_path).resolve():
            clip_plans = plans
        else:
            with open(clip_path, "rb") as handle:
                clip_plans = pickle.load(handle)
        for position in tqdm(
            source_positions,
            desc=f"building prompts {source_name}",
        ):
            manifest_row = manifest[position]
            source_index = int(manifest_row["source_idx"])
            plan = plans[source_index]
            if source_name == "resplan":
                plan = source_coordinates(plan)
            if int(plan.get("id", -1)) != int(manifest_row["source_id"]):
                raise ValueError(
                    f"manifest/source id mismatch at combined index {position}"
                )
            clip_plan = clip_plans[source_index]
            if source_name == "rplan":
                clip_plan = clip_area_plan(clip_plan)
            if int(clip_plan.get("id", -1)) != int(
                manifest_row["source_id"]
            ):
                raise ValueError(
                    "clip source id mismatch at combined index "
                    f"{position}"
                )
            result = build_prompt(
                plan,
                clip_plan,
                normalized_areas[position],
                normalized_node_type_ids[position],
                area_stats,
                total_stats,
            )
            combined_id = int(manifest_row["id"])
            plan_ids[position] = combined_id
            sources[position] = source_name
            source_ids[position] = int(manifest_row["source_id"])
            source_indices[position] = source_index
            node_areas[position] = result["node_areas"]
            raw_node_areas[position] = result["raw_node_areas"]
            node_area_ratios[position] = result["node_area_ratios"]
            node_size_ids[position] = result["node_size_ids"]
            node_type_ids[position] = result["node_type_ids"]
            feature_mask[position] = result["feature_mask"]
            room_counts[position] = result["room_counts"]
            total_area[position] = result["total_area"]
            normalized_total_area[position] = result[
                "normalized_total_area"
            ]
            clip_total_area[position] = result["clip_total_area"]
            total_size_ids[position] = result["total_size_id"]
            prompt_rows[position] = {
                "id": combined_id,
                "source": source_name,
                "source_id": int(manifest_row["source_id"]),
                "source_idx": source_index,
                "user_prompt": result["user_prompt"],
                "clip": result["clip_prompt"],
                "area_clause": result["area_clause"],
                "clip_total_area": float(result["clip_total_area"]),
                "room_counts": {
                    room_type: int(result["room_counts"][type_index])
                    for type_index, room_type in enumerate(ROOM_TYPES)
                },
            }
        del plans
        del clip_plans

    if any(row is None for row in prompt_rows):
        raise RuntimeError("not every manifest row produced a prompt")
    assert np.array_equal(
        plan_ids,
        np.arange(count, dtype=np.int64),
    )

    out_json = Path(args.out_json)
    out_npz = Path(args.out_npz)
    stats_path = Path(args.stats)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(prompt_rows, handle, ensure_ascii=True, indent=2)
    np.savez_compressed(
        out_npz,
        plan_ids=plan_ids,
        source=sources,
        source_id=source_ids,
        source_idx=source_indices,
        node_areas=node_areas,
        raw_node_areas=raw_node_areas,
        node_area_ratios=node_area_ratios,
        node_size_ids=node_size_ids,
        node_type_ids=node_type_ids,
        feature_mask=feature_mask,
        room_counts=room_counts,
        total_area=total_area,
        normalized_total_area=normalized_total_area,
        clip_total_area=clip_total_area,
        total_size_ids=total_size_ids,
    )
    summary = {
        "plan_count": count,
        "room_type_order": ROOM_TYPES,
        "feature_order": FEATURE_ORDER,
        "size_id_order": {"small": 0, "medium": 1, "large": 2},
        "room": area_stats,
        "total": total_stats,
        "normalized_total": normalized_total_stats,
        "area_constraint_basis": (
            "node_area_ratios = stretched GT room polygon area / "
            f"normalized canvas area ({CANVAS_AREA}); q30/q70 are computed "
            "from the canvas-ratio distribution"
        ),
        "clip_total_area_basis": (
            "scaled RPLAN canvas + ResPlan canvas node areas; "
            "q30/q70 from data/combined_area_stats_natural.json"
        ),
        "clip_total_area_sources": {
            "rplan": str(Path(args.clip_rplan).resolve()),
            "resplan": str(Path(args.clip_resplan).resolve()),
            "stats": str(Path(args.clip_stats).resolve()),
        },
        "clip_rule": (
            "CLIP receives only the head before ', featuring'; "
            "the featuring clause and node area ratios are structured rules"
        ),
        "outputs": {
            "prompts": str(out_json.resolve()),
            "areas": str(out_npz.resolve()),
        },
    }
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    print(
        f"saved {out_json} rows={len(prompt_rows)}; "
        f"saved {out_npz} areas={node_areas.shape}"
    )
    print(f"saved {stats_path}")
    print("sample full prompt:", prompt_rows[0]["user_prompt"])
    print("sample CLIP prompt:", prompt_rows[0]["clip"])


if __name__ == "__main__":
    main()
