"""Prompt-driven topology sampling for standalone floorplan generation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).resolve().parent / "data"
NUM_ROOMS = 20

TYPE_NAMES = (
    "living",
    "kitchen",
    "bedroom",
    "bathroom",
    "balcony",
    "storage",
    "stair",
    "front_door",
    "door",
)
TYPE_TO_ID = {
    name: index
    for index, name in enumerate(TYPE_NAMES)
}
FUNCTIONAL_TYPES = {
    "living",
    "kitchen",
    "bedroom",
    "bathroom",
    "balcony",
    "storage",
}


@dataclass(frozen=True)
class PromptRoomCounts:
    living: int = 1
    kitchen: int = 1
    bedroom: int = 0
    bathroom: int = 0
    balcony: int = 0
    storage: int = 0
    size: str | None = None

    def as_dict(self) -> dict[str, int]:
        return {
            "living": self.living,
            "kitchen": self.kitchen,
            "bedroom": self.bedroom,
            "bathroom": self.bathroom,
            "balcony": self.balcony,
            "storage": self.storage,
        }


@dataclass
class RuleTopology:
    node_type_ids: np.ndarray
    valid_mask: np.ndarray
    adjacency: np.ndarray
    edge_lists: tuple[tuple[int, int], ...]
    counts: PromptRoomCounts
    room_ids_by_type: dict[str, list[int]]
    text: str


_CHINESE_NUMBERS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def _parse_number(value: str) -> int:
    value = value.strip()
    if value.isdigit():
        return int(value)
    if value == "十":
        return 10
    if value.startswith("十"):
        return 10 + _CHINESE_NUMBERS.get(value[1:], 0)
    if value.endswith("十"):
        return 10 * _CHINESE_NUMBERS.get(value[0], 0)
    if "十" in value:
        left, right = value.split("十", 1)
        return (
            10 * _CHINESE_NUMBERS.get(left, 0)
            + _CHINESE_NUMBERS.get(right, 0)
        )
    return _CHINESE_NUMBERS.get(value, 0)


def _search_count(
    text: str,
    patterns: tuple[str, ...],
) -> int | None:
    number = r"(\d+|[零一二两三四五六七八九十]+)"
    for pattern in patterns:
        match = re.search(number + pattern, text)
        if match:
            return _parse_number(match.group(1))
    return None


def parse_prompt_room_counts(prompt: str) -> PromptRoomCounts:
    """Parse dataset-style English apartment prompts.

    Unspecified counts use the deployment defaults living=1, kitchen=1.
    The optional leading size word controls raw-area recovery.
    """
    text = prompt.lower().strip()
    headline = text.split(", featuring", 1)[0]
    size_match = re.search(
        r"\b(small|medium|large)\b",
        headline,
    )
    prompt_size = (
        size_match.group(1)
        if size_match
        else None
    )
    # The dataset writes only small/large explicitly. Its no-size
    # headline group is the medium occupancy group.
    if prompt_size == "medium":
        prompt_size = None
    counts = {
        "living": 1,
        "kitchen": 1,
        "bedroom": 0,
        "bathroom": 0,
        "balcony": 0,
        "storage": 0,
    }
    patterns = {
        "bedroom": (
            r"\s*[- ]?\s*(?:bedrooms?|beds?|卧室|卧房|睡房|房间)",
            r"\s*(?:室|房)(?=.*厅|$)",
        ),
        "bathroom": (
            r"\s*(?:bathrooms?|baths?|卫生间|洗手间|浴室|卫)",
        ),
        "balcony": (
            r"\s*(?:balconies|balcony|阳台)",
        ),
        "kitchen": (
            r"\s*(?:kitchens?|厨房|厨)",
        ),
        "living": (
            r"\s*(?:living rooms?|living|客厅|厅)",
        ),
        "storage": (
            r"\s*(?:storage rooms?|storerooms?|储藏室|储物间|储藏间|仓库)",
        ),
    }
    for room_type, room_patterns in patterns.items():
        value = _search_count(text, room_patterns)
        if value is not None:
            counts[room_type] = value

    # English "3-bedroom apartment" is the common headline form.
    match = re.search(
        r"(\d+)\s*-\s*bedroom",
        text,
    )
    if match:
        counts["bedroom"] = int(match.group(1))

    if counts["living"] < 1:
        counts["living"] = 1
    if counts["kitchen"] < 0:
        counts["kitchen"] = 0
    for key in (
        "bedroom",
        "bathroom",
        "balcony",
        "storage",
    ):
        counts[key] = max(0, int(counts[key]))
    total = sum(counts.values()) + 1  # front door
    if total > NUM_ROOMS:
        raise ValueError(
            f"prompt needs {total} rooms, maximum is {NUM_ROOMS}"
        )
    return PromptRoomCounts(**counts, size=prompt_size)


def _topology_group(bedroom_count: int) -> str:
    if bedroom_count <= 2:
        return "2"
    if bedroom_count == 3:
        return "3"
    if bedroom_count == 4:
        return "4"
    return "5+"


def _choose_unique(
    candidates: list[int],
    used: set[int],
    rng: np.random.Generator,
) -> int | None:
    available = [room_id for room_id in candidates if room_id not in used]
    if not available:
        return None
    return int(rng.choice(available))


def _weighted_type(
    weights: dict[str, float],
    available_types: set[str],
    rng: np.random.Generator,
) -> str:
    entries = [
        (name, weight)
        for name, weight in weights.items()
        if name in available_types and weight > 0.0
    ]
    if not entries:
        return "living"
    names = [item[0] for item in entries]
    probabilities = np.asarray(
        [item[1] for item in entries],
        dtype=float,
    )
    probabilities /= probabilities.sum()
    return str(rng.choice(names, p=probabilities))


def _add_edge(
    adjacency: np.ndarray,
    room_a: int,
    room_b: int,
    node_type_ids: np.ndarray,
) -> None:
    if room_a == room_b:
        return
    type_a = str(TYPE_NAMES[int(node_type_ids[room_a])])
    type_b = str(TYPE_NAMES[int(node_type_ids[room_b])])
    if type_a == "bedroom" and type_b == "bedroom":
        return
    adjacency[room_a, room_b] = True
    adjacency[room_b, room_a] = True


def _connected_rooms(
    adjacency: np.ndarray,
    root: int,
) -> set[int]:
    visited = {int(root)}
    queue = [int(root)]
    while queue:
        current = queue.pop()
        neighbors = np.flatnonzero(adjacency[current]).tolist()
        for neighbor in neighbors:
            neighbor = int(neighbor)
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
    return visited


def _repair_connectivity(
    adjacency: np.ndarray,
    valid_mask: np.ndarray,
    node_type_ids: np.ndarray,
    room_ids_by_type: dict[str, list[int]],
) -> None:
    living = room_ids_by_type["living"][0]
    functional_ids = [
        int(room_id)
        for room_id in np.flatnonzero(valid_mask)
        if int(node_type_ids[room_id]) != TYPE_TO_ID["front_door"]
    ]
    while True:
        connected = _connected_rooms(adjacency, living)
        missing = [
            room_id
            for room_id in functional_ids
            if room_id not in connected
        ]
        if not missing:
            return
        room_id = missing[0]
        room_type = str(TYPE_NAMES[int(node_type_ids[room_id])])
        candidates: list[int] = []
        if room_type in {"bathroom", "balcony"}:
            candidates.extend(
                candidate
                for candidate in room_ids_by_type.get("bedroom", [])
                if candidate in connected
            )
        candidates.extend(
            candidate
            for candidate in room_ids_by_type.get("living", [])
            if candidate in connected
        )
        if not candidates:
            candidates = sorted(connected)
        parent = int(candidates[0])
        _add_edge(
            adjacency,
            room_id,
            parent,
            node_type_ids,
        )


def sample_rule_topology(
    prompt: str,
    rng: np.random.Generator,
) -> RuleTopology:
    counts = parse_prompt_room_counts(prompt)
    count_dict = counts.as_dict()
    ordered_types: list[str] = []
    room_ids_by_type: dict[str, list[int]] = {}
    for room_type in (
        "living",
        "kitchen",
        "bedroom",
        "bathroom",
        "balcony",
        "storage",
    ):
        ids = []
        for _ in range(count_dict[room_type]):
            room_id = len(ordered_types)
            ordered_types.append(room_type)
            ids.append(room_id)
        if ids:
            room_ids_by_type[room_type] = ids
    front_door = len(ordered_types)
    ordered_types.append("front_door")
    room_ids_by_type["front_door"] = [front_door]

    node_type_ids = np.full(NUM_ROOMS, -1, dtype=np.int16)
    for room_id, room_type in enumerate(ordered_types):
        node_type_ids[room_id] = TYPE_TO_ID[room_type]
    valid_mask = np.zeros(NUM_ROOMS, dtype=bool)
    valid_mask[: len(ordered_types)] = True
    adjacency = np.zeros((NUM_ROOMS, NUM_ROOMS), dtype=bool)

    living = room_ids_by_type["living"][0]
    kitchen_ids = room_ids_by_type.get("kitchen", [])
    for kitchen in kitchen_ids:
        if rng.random() < 0.985:
            _add_edge(
                adjacency,
                living,
                kitchen,
                node_type_ids,
            )
    _add_edge(
        adjacency,
        living,
        front_door,
        node_type_ids,
    )

    group = _topology_group(counts.bedroom)
    bedrooms = room_ids_by_type.get("bedroom", [])
    if beds := bedrooms:
        if counts.bedroom <= 3:
            living_bedrooms = beds
        elif counts.bedroom == 4:
            living_bedrooms = list(
                rng.choice(beds, size=3, replace=False)
            )
        else:
            target = max(2, int(np.ceil(0.65 * len(beds))))
            living_bedrooms = list(
                rng.choice(beds, size=target, replace=False)
            )
        for bedroom in living_bedrooms:
            _add_edge(
                adjacency,
                living,
                int(bedroom),
                node_type_ids,
            )

    used_bedroom_attachments: set[int] = set()
    used_kitchen_attachments: set[int] = set()

    bathroom_living = {
        "2": 0.80,
        "3": 0.67,
        "4": 0.43,
        "5+": 0.28,
    }[group]
    bathrooms = room_ids_by_type.get("bathroom", [])
    for bathroom_index, bathroom in enumerate(bathrooms):
        available = {"living"}
        if bedrooms:
            available.add("bedroom")
        if len(bathrooms) >= 2:
            if bathroom_index == 0:
                target_type = "living"
            elif bathroom_index == 1 and bedrooms:
                target_type = "bedroom"
            else:
                target_type = _weighted_type(
                    {
                        "living": bathroom_living,
                        "bedroom": 1.0 - bathroom_living,
                    },
                    available,
                    rng,
                )
        else:
            target_type = _weighted_type(
                {
                    "living": bathroom_living,
                    "bedroom": 1.0 - bathroom_living,
                },
                available,
                rng,
            )
        if target_type == "bedroom":
            target = _choose_unique(
                bedrooms,
                used_bedroom_attachments,
                rng,
            )
            if target is not None:
                used_bedroom_attachments.add(target)
            else:
                target = living
        else:
            target = living
        _add_edge(
            adjacency,
            bathroom,
            int(target),
            node_type_ids,
        )

    balcony_target_weights = {
        "2": {
            "living": 0.53,
            "bedroom": 0.34,
            "kitchen": 0.13,
        },
        "3": {
            "living": 0.55,
            "bedroom": 0.38,
            "kitchen": 0.07,
        },
        "4": {
            "living": 0.45,
            "bedroom": 0.50,
            "kitchen": 0.05,
        },
        "5+": {
            "living": 0.37,
            "bedroom": 0.58,
            "kitchen": 0.05,
        },
    }[group]
    for balcony in room_ids_by_type.get("balcony", []):
        available = {"living"}
        if bedrooms:
            available.add("bedroom")
        if kitchen_ids:
            available.add("kitchen")
        target_type = _weighted_type(
            balcony_target_weights,
            available,
            rng,
        )
        if target_type == "bedroom":
            target = _choose_unique(
                bedrooms,
                used_bedroom_attachments,
                rng,
            )
        elif target_type == "kitchen":
            target = _choose_unique(
                kitchen_ids,
                used_kitchen_attachments,
                rng,
            )
        else:
            target = living
        if target is None:
            target = living
        if target_type == "bedroom" and target is not None:
            used_bedroom_attachments.add(int(target))
        if target_type == "kitchen" and target is not None:
            used_kitchen_attachments.add(int(target))
        _add_edge(
            adjacency,
            balcony,
            int(target),
            node_type_ids,
        )

    storage_weights = {
        "living": 0.72,
        "bedroom": 0.19,
        "kitchen": 0.09,
    }
    for storage in room_ids_by_type.get("storage", []):
        available = {"living"}
        if bedrooms:
            available.add("bedroom")
        if kitchen_ids:
            available.add("kitchen")
        target_type = _weighted_type(
            storage_weights,
            available,
            rng,
        )
        if target_type == "bedroom":
            target = _choose_unique(
                bedrooms,
                used_bedroom_attachments,
                rng,
            )
        elif target_type == "kitchen":
            target = _choose_unique(
                kitchen_ids,
                used_kitchen_attachments,
                rng,
            )
        else:
            target = living
        if target is None:
            target = living
        if target_type == "bedroom" and target is not None:
            used_bedroom_attachments.add(int(target))
        if target_type == "kitchen" and target is not None:
            used_kitchen_attachments.add(int(target))
        _add_edge(
            adjacency,
            storage,
            int(target),
            node_type_ids,
        )

    _repair_connectivity(
        adjacency,
        valid_mask,
        node_type_ids,
        room_ids_by_type,
    )

    edge_lists = tuple(
        (int(room_a), int(room_b))
        for room_a in range(NUM_ROOMS)
        for room_b in range(room_a + 1, NUM_ROOMS)
        if valid_mask[room_a]
        and valid_mask[room_b]
        and adjacency[room_a, room_b]
    )
    text = make_clip_prompt(counts)
    return RuleTopology(
        node_type_ids=node_type_ids,
        valid_mask=valid_mask,
        adjacency=adjacency,
        edge_lists=edge_lists,
        counts=counts,
        room_ids_by_type=room_ids_by_type,
        text=text,
    )


def make_clip_prompt(counts: PromptRoomCounts) -> str:
    size_prefix = (
        f"{counts.size} "
        if counts.size in {"small", "large"}
        else ""
    )
    bathroom_word = (
        "bathroom" if counts.bathroom <= 1 else "bathrooms"
    )
    bedroom_prefix = (
        f"{counts.bedroom}-bedroom "
        if counts.bedroom > 0
        else ""
    )
    prompt = (
        f"A {size_prefix}{bedroom_prefix}apartment "
        f"with {counts.bathroom} {bathroom_word}"
    )
    if counts.balcony:
        balcony_word = (
            "balcony" if counts.balcony == 1 else "balconies"
        )
        prompt += (
            f" and {counts.balcony} {balcony_word}"
        )
    return prompt + "."


class CornerCountRules:
    def __init__(self, path: Path | None = None):
        path = path or (
            DATA_DIR / "corner_count_distributions.json"
        )
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.type_names = tuple(payload["type_names"])
        self.source_counts = payload["source_counts"]
        self.by_type = payload["by_type"]
        self.by_source_type = payload.get("by_source_type", {})

    def sample_source(
        self,
        rng: np.random.Generator,
    ) -> str:
        names = sorted(self.source_counts)
        weights = np.asarray(
            [self.source_counts[name] for name in names],
            dtype=float,
        )
        weights /= weights.sum()
        return str(rng.choice(names, p=weights))

    def sample(
        self,
        room_type: str,
        rng: np.random.Generator,
        source: str | None = None,
    ) -> int:
        if room_type == "front_door":
            return 4
        distribution = None
        if source:
            distribution = self.by_source_type.get(
                f"{source}:{room_type}"
            )
        if distribution is None:
            distribution = self.by_type.get(room_type)
        if not distribution:
            raise KeyError(f"unknown room type: {room_type}")
        values = np.asarray(
            [int(value) for value in distribution],
            dtype=np.int16,
        )
        weights = np.asarray(
            [distribution[str(value)] for value in values],
            dtype=float,
        )
        weights /= weights.sum()
        return int(rng.choice(values, p=weights))


_DEFAULT_CORNER_RULES = None


def get_default_corner_count_rules() -> CornerCountRules:
    global _DEFAULT_CORNER_RULES
    if _DEFAULT_CORNER_RULES is None:
        _DEFAULT_CORNER_RULES = CornerCountRules()
    return _DEFAULT_CORNER_RULES
