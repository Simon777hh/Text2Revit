"""Remove narrow room appendages and small gaps before opening placement."""
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union


def _parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    return [g for g in getattr(geometry, "geoms", ()) if g.geom_type == "Polygon"]


def clean_thin_geometry(records, pixel_to_meter, minimum_width_m=.6, maximum_patch_m2=2.):
    """Transfer small sub-600 mm tabs to neighbors; reject unsafe changes.

    This operates in physical units after scale recovery. Whole narrow rooms or
    disconnected room bodies are rejected rather than silently deleting rooms.
    """
    functional = [r for r in records if r.type_name not in {"front_door", "door"}]
    radius = minimum_width_m / (2 * pixel_to_meter)
    original = {r.room_id: r.polygon for r in functional}
    patches = []
    for room in functional:
        polygon = room.polygon
        core = polygon.buffer(-radius, join_style=2).buffer(radius, join_style=2).intersection(polygon)
        removed = polygon.difference(core)
        if removed.area * pixel_to_meter ** 2 < 1e-6:
            continue
        if core.is_empty or core.geom_type != "Polygon" or core.interiors:
            raise ValueError(f"Room {room.room_id} has a narrow or disconnected body")
        if removed.area * pixel_to_meter ** 2 > maximum_patch_m2 or removed.area > polygon.area * .2:
            raise ValueError(f"Room {room.room_id} requires excessive thin-strip repair")
        room.polygon = orient(core, sign=1.)
        patches.extend((room.room_id, part) for part in _parts(removed))
    transferred = 0
    for owner, patch in patches:
        candidates = sorted((r for r in functional if r.room_id != owner),
                            key=lambda r: patch.boundary.intersection(r.polygon.boundary).length,
                            reverse=True)
        assigned = False
        for neighbor in candidates:
            if patch.boundary.intersection(neighbor.polygon.boundary).length < 1e-5:
                continue
            merged = neighbor.polygon.union(patch)
            if merged.geom_type != "Polygon" or merged.interiors or not merged.is_valid:
                continue
            neighbor.polygon = orient(merged, sign=1.)
            transferred += 1
            assigned = True
            break
        if not assigned:
            raise ValueError("A narrow strip cannot be safely assigned to a neighbor")
    envelope = unary_union([r.polygon for r in functional])
    gaps = envelope.buffer(radius, join_style=2).buffer(-radius, join_style=2).difference(envelope)
    for poly in _parts(envelope):
        gaps = gaps.union(unary_union([Polygon(ring) for ring in poly.interiors]))
    filled = 0
    for patch in _parts(gaps):
        if patch.area * pixel_to_meter ** 2 < 1e-6:
            continue
        rectangle = patch.minimum_rotated_rectangle
        coords = list(rectangle.exterior.coords)
        short = min(((a[0]-b[0])**2+(a[1]-b[1])**2)**.5 for a,b in zip(coords[:-1],coords[1:]))
        if short * pixel_to_meter > minimum_width_m + 1e-6:
            continue
        if patch.area * pixel_to_meter ** 2 > maximum_patch_m2:
            raise ValueError("A narrow gap is too large to repair safely")
        assigned = False
        for room in sorted(functional, key=lambda r: patch.boundary.intersection(r.polygon.boundary).length, reverse=True):
            if patch.boundary.intersection(room.polygon.boundary).length < 1e-5:
                continue
            merged = room.polygon.union(patch)
            if merged.geom_type == "Polygon" and merged.is_valid and not merged.interiors:
                room.polygon = orient(merged, sign=1.)
                filled += 1
                assigned = True
                break
        if not assigned:
            raise ValueError("A narrow gap cannot be filled without invalid room geometry")
    final_envelope = unary_union([r.polygon for r in functional])
    if unary_union(list(original.values())).difference(final_envelope).area * pixel_to_meter ** 2 > 1e-5:
        raise ValueError("Thin-strip cleanup lost occupied floor area")
    for room in functional:
        core = room.polygon.buffer(-radius, join_style=2).buffer(radius, join_style=2)
        if room.polygon.difference(core).area * pixel_to_meter ** 2 > 1e-5:
            raise ValueError(f"Room {room.room_id} still contains a narrow appendage")
    return {"transferred_strips": transferred, "filled_gaps": filled,
            "minimum_width_m": minimum_width_m}
