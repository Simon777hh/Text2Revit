"""Clear-area thresholds and the exclusion of unrelated room types."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plan_quality import PlanQualityError, validate_room_areas


def room(kind, width, height):
    return {"room_id": 0, "type": kind,
            "polygon": [[0, 0], [width, 0], [width, height], [0, height]]}


class PlanQualityTests(unittest.TestCase):
    def test_kitchen_and_bathroom_below_two_square_metres_are_rejected(self):
        for kind in ("kitchen", "bathroom"):
            with self.assertRaises(PlanQualityError):
                validate_room_areas([room(kind, 1, 1.9)], 1)
            self.assertEqual(validate_room_areas([room(kind, 1, 2)], 1)["0"], 2)

    def test_other_rooms_do_not_trigger_area_rejection(self):
        for kind in ("bedroom", "living", "balcony", "storage"):
            self.assertAlmostEqual(validate_room_areas([room(kind, 1, .1)], 1)["0"], .1)

    def test_area_uses_square_of_metre_scale(self):
        self.assertAlmostEqual(validate_room_areas([room("bathroom", 10, 20)], .1)["0"], 2)

    def test_wall_footprints_are_excluded(self):
        walls = [{"reference_line": [[0, 0], [0, 2]], "thickness_m": .4}]
        with self.assertRaises(PlanQualityError):
            validate_room_areas([room("bathroom", 1.1, 2)], 1, walls=walls)


if __name__ == "__main__":
    unittest.main()
