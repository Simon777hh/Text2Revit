# Verified example

Both images show the same generated layout, constructed with the current C#
builders in Revit 2022. `floorplan.png` is a native Revit export with every room
area and the total; `model3d.png` is the matching shaded 3D export. Balcony outer
edges use railings, with room-separation lines for native area measurement.

Prompt: `A large 3-bedroom apartment with 2 bathrooms and 2 balconies.`
Accepted seed: `3968430468`. Native total area including balconies: **106.2 m²**.
Nine rooms, 28 wall segments, nine doors, four windows and five railing segments.
Kitchen: 8.9 m²; bathrooms: 3.2 m² and 2.2 m². All windows avoid the continuous
entrance facade; the bedroom connected to the balcony has no window.

`plan.json` is the accepted backend result. Its areas estimate clear floor area
by subtracting wall footprints; native Revit areas can differ slightly at joins.
`model-metadata.json` records native room areas and element counts.

The narrow bedroom appendage was transferred to the living room before door placement. The two reported nearby doors now lie on different boundary runs with approximately 604 mm between their opening segments.
