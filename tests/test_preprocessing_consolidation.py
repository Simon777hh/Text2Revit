"""Check the consolidated RPLAN and static training-data contracts."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.rplan.prepare_rplan import area_scale, clip_area_plan, filter_reasons
from preprocessing.features.extract_raw_data import DEFAULT_RPLAN_PKL
from preprocessing.features.prepare_prompts import DEFAULT_CLIP_RPLAN_PKL
from preprocessing.features.prepare_topology_gt import build_topology_gt
from preprocessing.features.prepare_attention import build_corner_masks, build_mapping


def apartment():
    return {"id": 1, "living": box(0, 0, 10, 10), "bedroom": [box(10, 0, 20, 10)],
            "inner": box(0, 0, 20, 10),
            "conn": {"node_names": ["living_0", "bedroom_0", "front_door_0"],
                     "room_types": ["living", "bedroom", "front_door"],
                     "node_areas": [100.0, 100.0, 0.0],
                     "edge_list": [[0, 1], [0, 2]],
                     "edge_type_names": ["via_door", "direct"]}}


class RplanContracts(unittest.TestCase):
    def test_valid_list_geometry_and_entrance_pass(self):
        self.assertEqual(filter_reasons(apartment()), [])

    def test_entrance_to_bedroom_is_rejected(self):
        plan = apartment()
        plan["conn"]["edge_list"][-1] = [1, 2]
        self.assertIn("entrance_connection", filter_reasons(plan))

    def test_room_coverage_and_door_wall_are_checked(self):
        plan = apartment()
        plan["bedroom"] = [box(11, 0, 20, 10)]
        reasons = filter_reasons(plan)
        self.assertIn("uncovered_inner_area", reasons)
        self.assertIn("door_without_shared_wall", reasons)

    def test_clip_scaling_preserves_gt_geometry_and_input_areas(self):
        plan = apartment()
        plan["clip_area_scale"] = 2.5
        before = copy.deepcopy(plan)
        result = clip_area_plan(plan)
        self.assertEqual(result["conn"]["node_areas"], [250, 250, 0])
        self.assertEqual(plan["conn"], before["conn"])
        self.assertIs(result["living"], plan["living"])

    def test_area_calibration_matches_median_rule_and_rejects_empty(self):
        self.assertEqual(area_scale([apartment()], [apartment()]), 1.0)
        with self.assertRaises(ValueError):
            area_scale([], [apartment()])
        with self.assertRaises(ValueError):
            clip_area_plan(dict(apartment(), clip_area_scale=float("nan")))

    def test_gt_and_clip_share_one_rplan_file(self):
        self.assertEqual(DEFAULT_RPLAN_PKL, DEFAULT_CLIP_RPLAN_PKL)
        self.assertEqual(DEFAULT_RPLAN_PKL.name, "RPLAN_filtered.pkl")


class TrainingContracts(unittest.TestCase):
    def test_topology_build_removes_bedroom_edges_before_writing(self):
        features = np.zeros((1, 20, 61), dtype=np.float32)
        valid = np.zeros((1, 20), dtype=bool)
        valid[0, :3] = True
        for i, kind in enumerate([2, 2, 0]):
            features[0, i, i] = 1
            features[0, i, 20 + kind] = 1
        adjacency = np.zeros((1, 20, 20), dtype=bool)
        adjacency[0, 0, 1] = adjacency[0, 1, 0] = True
        adjacency[0, 1, 2] = adjacency[0, 2, 1] = True
        edges = np.empty(1, dtype=object)
        edges[0] = [(0, 1), (1, 2)]
        topo = {"features": features, "valid_mask": valid,
                "plan_ids": np.array([7]), "num_rooms": np.array([3])}
        data = build_topology_gt(topo, {"plan_ids": np.array([7]), "adjacency": adjacency,
                                       "edge_lists": edges, "num_edges": np.array([2])})
        self.assertEqual(data["edge_lists"][0], [(1, 2)])
        self.assertEqual(data["num_edges"][0], 1)
        self.assertFalse(data["adjacency"][0, 0, 1])
        self.assertTrue(data["adjacency"][0, 1, 2])
        self.assertTrue(np.isfinite(data["features"]).all())
        self.assertGreater(np.abs(data["features"][0, :, 29:]).sum(), 0)

    def test_attention_and_mapping_respect_room_rings_and_padding(self):
        features = np.zeros((128, 52), dtype=np.float32)
        for token in range(8):
            features[token, 4 + token // 4] = 1
            features[token, 24 + token % 4] = 1
        valid, following = build_mapping(features)
        np.testing.assert_array_equal(following[:8], [1, 2, 3, 0, 5, 6, 7, 4])
        self.assertEqual(valid.sum(), 8)
        adjacency = np.zeros((20, 20), dtype=bool)
        room, conn, global_mask = build_corner_masks(features, adjacency)
        self.assertFalse(conn[0, 4])
        self.assertTrue(global_mask[0, 4])
        self.assertTrue(room[0, 3])
        self.assertFalse(global_mask[0, 8])
        adjacency[0, 1] = adjacency[1, 0] = True
        self.assertTrue(build_corner_masks(features, adjacency)[1][0, 4])


if __name__ == "__main__":
    unittest.main()
