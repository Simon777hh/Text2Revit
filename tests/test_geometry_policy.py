"""Balcony access and inclusive 100 mm window sampling regression checks."""
import sys
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import LineString
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from revit_geometry import build_revit_geometry, sample_window_dimensions, WINDOW_RANGES_MM


class GeometryPolicyTests(unittest.TestCase):
    def test_open_balcony_preserves_access_wall_and_door(self):
        rooms = [dict(room_id=1, type="living", polygon=[[0, 0], [4, 0], [4, 4], [0, 4]]),
                 dict(room_id=2, type="balcony", polygon=[[4, 0], [6, 0], [6, 4], [4, 4]])]
        openings = [dict(type="door", owner_room_id=1, target_room_id=2,
                         host="shared_wall", reference_line=[[4, 1], [4, 2]], width=1)]
        walls, rails = build_revit_geometry(rooms, openings, 1, 123)
        self.assertEqual(len(rails), 3)
        self.assertFalse(any(w["owner_room_ids"] == [2] for w in walls))
        host = next(w for w in walls if w["wall_id"] == openings[0]["wall_id"])
        self.assertEqual(host["owner_room_ids"], [1, 2])
        self.assertEqual(host["height_m"], 3.3)
        self.assertEqual(sum(LineString(r["reference_line"]).length for r in rails), 8)

    def test_window_dimensions_reach_only_allowed_100mm_values(self):
        for kind, ranges in WINDOW_RANGES_MM.items():
            values = [sample_window_dimensions(kind, np.random.default_rng(seed)) for seed in range(500)]
            for index, (low, high) in enumerate(ranges):
                actual = {round(v[index] * 1000) for v in values}
                self.assertEqual(actual, set(range(low, high + 1, 100)), kind)


if __name__ == "__main__":
    unittest.main()
