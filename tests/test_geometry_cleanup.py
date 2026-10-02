"""Thin bedroom tabs, narrow gaps and opposite-door regression checks."""
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union
from geometry_cleanup import clean_thin_geometry
from door_window_rules import opening_lines_conflict, build_internal_doors
from revit_geometry import validate_opening_clearances


def room(room_id, kind, polygon):
    return SimpleNamespace(room_id=room_id, type_name=kind, polygon=polygon)


class GeometryCleanupTests(unittest.TestCase):
    def test_narrow_bedroom_tab_is_transferred_to_bathroom(self):
        rooms = [room(0,"bedroom",Polygon([(0,0),(5,0),(5,7),(4.6,7),(4.6,5),(0,5)])),
                 room(1,"bathroom",box(1,5,4.6,7))]
        original = unary_union([r.polygon for r in rooms])
        report = clean_thin_geometry(rooms,1)
        self.assertEqual(report["transferred_strips"],1)
        self.assertLess(rooms[0].polygon.symmetric_difference(box(0,0,5,5)).area,1e-8)
        self.assertLess(rooms[1].polygon.symmetric_difference(box(1,5,5,7)).area,1e-8)
        self.assertLess(original.symmetric_difference(unary_union([r.polygon for r in rooms])).area,1e-8)

    def test_small_gap_is_filled_without_room_overlap(self):
        rooms = [room(0,"living",box(0,0,4,4)),room(1,"bedroom",box(4.3,0,8,4))]
        clean_thin_geometry(rooms,1)
        self.assertLess(rooms[0].polygon.intersection(rooms[1].polygon).area,1e-8)
        self.assertAlmostEqual(unary_union([r.polygon for r in rooms]).area,32)

    def test_large_or_disconnected_repair_is_rejected(self):
        with self.assertRaises(ValueError):
            clean_thin_geometry([room(0,"bedroom",box(0,0,.4,6))],1)

    def test_large_narrow_gap_is_rejected_instead_of_left_unfilled(self):
        with self.assertRaises(ValueError):
            clean_thin_geometry([room(0,"living",box(0,0,4,8)),
                                 room(1,"bedroom",box(4.4,0,8,8))],1)

    def test_opposite_doors_with_overlapping_leaves_are_rejected(self):
        first=LineString([(0,0),(0,.95)]);second=LineString([(.37,0),(.37,.95)])
        self.assertTrue(opening_lines_conflict(first,second,1,common_room=True))
        self.assertFalse(opening_lines_conflict(first,LineString([(2,0),(2,.95)]),1,common_room=True))

    def test_separate_collinear_doors_remain_allowed(self):
        first=LineString([(0,0),(1,0)])
        self.assertFalse(opening_lines_conflict(first,LineString([(1.3,0),(2.3,0)]),1,True))
        self.assertTrue(opening_lines_conflict(first,LineString([(1.05,0),(2.05,0)]),1,True))

    def test_validator_checks_doors_on_different_host_walls(self):
        openings=[dict(type="door",owner_room_id=0,target_room_id=i+1,
                       reference_line=[[x,0],[x,.95]]) for i,x in enumerate([0,.37])]
        with self.assertRaises(ValueError): validate_opening_clearances(openings,1)
