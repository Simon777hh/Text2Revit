"""Regression checks for the single ResPlan dataset and coordinate contract."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import MultiPolygon, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.resplan.prepare_resplan import (
    clean_reason, fit_plan, plan_bounds, source_coordinates,
)
from preprocessing.features.extract_raw_data import DEFAULT_RESPLAN_PKL
from preprocessing.features.prepare_prompts import DEFAULT_CLIP_RESPLAN_PKL


class CanvasContract(unittest.TestCase):
    def setUp(self):
        self.plan = {
            "id": 42, "living": box(-50, 20, 350, 220),
            "inner": box(-50, 20, 350, 220),
            "wall_depth": 4.0, "rooms_area": 80000.0,
            "area": 72.0, "neighbor": box(1000, 0, 1200, 100),
            "conn": {"node_names": ["living_0"],
                     "room_types": ["living"], "node_areas": [80000.0],
                     "edge_list": [], "edge_type_names": []},
        }

    def test_owned_geometry_fits_and_aspect_is_preserved(self):
        fitted = fit_plan(self.plan)
        bounds = plan_bounds(fitted)
        self.assertAlmostEqual(bounds["min_x"], 0)
        self.assertAlmostEqual(bounds["max_x"], 255)
        self.assertAlmostEqual(bounds["max_y"] - bounds["min_y"], 127.5)
        self.assertGreater(fitted["neighbor"].bounds[2], 255)

    def test_pixel_areas_depth_and_physical_area_units(self):
        fitted = fit_plan(self.plan)
        self.assertAlmostEqual(fitted["rooms_area"], 80000 * (255 / 400)**2)
        self.assertEqual(fitted["rooms_area"], fitted["conn"]["node_areas"][0])
        self.assertAlmostEqual(fitted["wall_depth"], 4 * 255 / 400)
        self.assertEqual(fitted["area"], 72.0)

    def test_round_trip_and_no_input_mutation(self):
        before = copy.deepcopy(self.plan)
        restored = source_coordinates(fit_plan(self.plan))
        for key in ("living", "inner", "neighbor"):
            self.assertTrue(restored[key].equals_exact(before[key], 1e-8))
            self.assertTrue(self.plan[key].equals_exact(before[key], 0))
        np.testing.assert_allclose(restored["conn"]["node_areas"], [80000])
        self.assertEqual(self.plan["conn"], before["conn"])
        self.assertNotIn("canvas_transform", restored)
        self.assertAlmostEqual(restored["wall_depth"], 4)

    def test_legacy_source_without_transform_keeps_units(self):
        self.assertIs(source_coordinates(self.plan), self.plan)

    def test_invalid_canvas_or_empty_geometry_rejected(self):
        for canvas in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                fit_plan(self.plan, canvas)
        with self.assertRaises(ValueError):
            fit_plan({})

    def test_invalid_recovery_scale_rejected(self):
        fitted = fit_plan(self.plan)
        fitted["canvas_transform"]["scale"] = 0
        with self.assertRaises(ValueError):
            source_coordinates(fitted)

    def test_clean_filter_retains_multiple_kitchen_guard(self):
        self.assertEqual(clean_reason({"kitchen": MultiPolygon([
            box(0, 0, 10, 10), box(20, 0, 30, 10)])}), "multiple_kitchens")

    def test_gt_and_clip_share_one_resplan_dataset(self):
        self.assertEqual(DEFAULT_RESPLAN_PKL, DEFAULT_CLIP_RESPLAN_PKL)
        self.assertEqual(DEFAULT_RESPLAN_PKL.name, "ResPlan_filtered_canvas.pkl")


if __name__ == "__main__":
    unittest.main()
