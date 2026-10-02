"""Entrance-wall exclusion and bedroom balcony window regression checks."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from shapely.geometry import LineString, Polygon, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from door_window_rules import (
    DoorWindowPiece, _front_door_wall_edges, _polygon_edges,
    build_door_window_pieces, build_windows,
)


def room(room_id, kind, polygon):
    return SimpleNamespace(room_id=room_id, type_name=kind, polygon=polygon)


def entrance(start=(1, 0), end=(3, 0)):
    line = LineString([start, end])
    return DoorWindowPiece(
        kind="front_door", owner_room_id=0, polygon=line.buffer(0.1),
        anchor=tuple(line.centroid.coords[0]), long_start=start, long_end=end,
    )


def windows(records, seed=0, **kwargs):
    return build_windows(records, 0.15, 0.03, 0.04,
                         np.random.default_rng(seed), **kwargs)


class OpeningPlacementPolicyTests(unittest.TestCase):
    def test_living_windows_avoid_entire_entrance_wall(self):
        records = [room(0, "living", box(0, 0, 20, 10))]
        for seed in range(40):
            result = windows(records, seed, front_door=entrance())
            self.assertTrue(result)
            for piece in result:
                self.assertFalse(abs(piece.long_start[1]) < 1e-7
                                 and abs(piece.long_end[1]) < 1e-7)

    def test_collinear_segments_and_ring_wrap_are_one_wall(self):
        polygon = Polygon([(4, 0), (8, 0), (12, 0), (12, 8), (0, 8), (0, 0)])
        edges = list(_polygon_edges(polygon))
        self.assertEqual(_front_door_wall_edges(edges, entrance((1, 0), (2, 0))),
                         {0, 1, 5})
        for seed in range(20):
            result = windows([room(0, "living", polygon)], seed, front_door=entrance())
            self.assertTrue(result)
            self.assertTrue(all(abs(p.anchor[1]) > 1e-7 for p in result))

    def test_reversed_polygon_and_vertical_entrance(self):
        polygon = Polygon(list(box(0, 0, 10, 20).exterior.coords)[::-1])
        for seed in range(20):
            result = windows([room(0, "living", polygon)], seed,
                             front_door=entrance((0, 2), (0, 4)))
            self.assertTrue(result)
            self.assertTrue(all(abs(p.anchor[0]) > 1e-7 for p in result))

    def test_parallel_recess_is_a_different_wall(self):
        polygon = Polygon([(0, 0), (8, 0), (8, 4), (12, 4), (12, 8), (0, 8)])
        self.assertEqual(_front_door_wall_edges(list(_polygon_edges(polygon)), entrance()), {0})

    def test_no_window_when_only_entrance_wall_is_exposed(self):
        records = [room(0, "living", box(0, 0, 10, 10)),
                   room(1, "bedroom", box(-10, 0, 0, 10)),
                   room(2, "kitchen", box(10, 0, 20, 10)),
                   room(3, "bathroom", box(0, 10, 10, 20))]
        result = windows(records, front_door=entrance())
        self.assertFalse(any(p.owner_room_id == 0 for p in result))

    def test_bedroom_connected_to_balcony_has_no_window(self):
        records = [room(0, "living", box(0, 0, 10, 10)),
                   room(1, "bedroom", box(10, 0, 20, 10)),
                   room(2, "balcony", box(20, 0, 24, 10))]
        for balcony_edge in [(1, 2), (2, 1)]:
            pieces = build_door_window_pieces(records, [(0, 1), balcony_edge], 0,
                                             rng=np.random.default_rng(0))
            self.assertFalse(any(p.owner_room_id == 1 for p in pieces["windows"]))
            self.assertTrue(any({p.owner_room_id, p.target_room_id} == {1, 2}
                                for p in pieces["doors"]))

    def test_balcony_on_living_does_not_disable_bedroom_window(self):
        records = [room(0, "living", box(0, 0, 10, 10)),
                   room(1, "bedroom", box(10, 0, 20, 10)),
                   room(2, "balcony", box(-4, 0, 0, 10))]
        pieces = build_door_window_pieces(records, [(0, 1), (0, 2)], 0,
                                         rng=np.random.default_rng(0))
        self.assertTrue(any(p.owner_room_id == 1 for p in pieces["windows"]))

    def test_direct_window_builder_honors_bedroom_balcony_count(self):
        records = [room(1, "bedroom", box(0, 0, 10, 10))]
        self.assertFalse(windows(records, balcony_count_by_room={1: 1}))
        self.assertTrue(windows(records))

    def test_all_room_windows_avoid_continuous_entrance_facade(self):
        records = [room(0, "living", box(0, 0, 10, 10)),
                   room(1, "bedroom", box(10, 0, 20, 10)),
                   room(2, "kitchen", box(20, 0, 30, 10)),
                   room(3, "bathroom", box(30, 0, 40, 10))]
        for seed in range(10):
            result = windows(records, seed, front_door=entrance())
            self.assertTrue(result)
            self.assertTrue(all(abs(piece.anchor[1]) > 1e-7 for piece in result))
            # The opposite facade is parallel and remains usable.
            self.assertTrue(any(abs(piece.anchor[1] - 10) < 1e-7 for piece in result))


if __name__ == "__main__":
    unittest.main()
