"""Clear-area thresholds and the exclusion of unrelated room types."""
import sys
import json
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plan_quality import PlanQualityError, load_physical_scale, measure_room_areas, validate_room_areas


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

    def test_output_calibration_uses_length_sqrt_and_keeps_wall_thickness(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            (data / "raw_recovery_scale_stats.json").write_text(json.dumps({
                "resplan_meter_scale": {"meters_per_raw_pixel": {"median": .5}}}))
            (data / "geometry_policy.json").write_text(json.dumps({
                "output_area_multipliers": {"large": 1.21}}))
            scale = load_physical_scale("large", data)["pixel_to_meter"]
            self.assertAlmostEqual(scale, .55)
            walls = [{"reference_line": [[0, 0], [0, 4]], "thickness_m": .4}]
            area = measure_room_areas([room("living", 4, 4)], scale, walls)["0"]
            self.assertAlmostEqual(area, 2.2 * (2.2 - .2))
            self.assertEqual(load_physical_scale("small", data)["pixel_to_meter"], .5)

    def test_invalid_output_calibration_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            for value in (0, -1, float("inf"), float("nan")):
                (data / "geometry_policy.json").write_text(json.dumps({
                    "output_area_multipliers": {"large": value}}))
                with self.assertRaises(ValueError):
                    load_physical_scale("large", data)
            with self.assertRaises(ValueError):
                load_physical_scale("auto", data)


if __name__ == "__main__":
    unittest.main()
