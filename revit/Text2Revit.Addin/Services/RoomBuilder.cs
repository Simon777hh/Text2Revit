using System;
using System.Collections.Generic;
using System.Linq;
using Autodesk.Revit.DB;
using Autodesk.Revit.DB.Architecture;
using Text2Revit.Addin.Models;

namespace Text2Revit.Addin.Services
{
    public sealed class RoomBuilder
    {
        private readonly Document document;
        private readonly ViewPlan view;
        private readonly Level level;
        private readonly double pixelToMeter;
        private readonly XYZ offset;

        public RoomBuilder(
            Document document,
            ViewPlan view,
            Level level,
            double pixelToMeter,XYZ offset=null)
        {
            this.document = document;
            this.view = view;
            this.level = level;
            this.pixelToMeter = pixelToMeter;
            this.offset=offset??XYZ.Zero;
        }

        public void Build(IEnumerable<RoomData> rooms)
        {
            int index = 1;
            var placedRooms = new List<Room>();
            foreach (RoomData data in rooms)
            {
                if (
                    string.Equals(
                        data.Type,
                        "front_door",
                        StringComparison.OrdinalIgnoreCase
                    ) ||
                    string.Equals(
                        data.Type,
                        "door",
                        StringComparison.OrdinalIgnoreCase
                    )
                )
                {
                    continue;
                }
                if (data.Polygon == null || data.Polygon.Count < 3)
                {
                    throw new InvalidOperationException(
                        $"Room {data.RoomId} has an invalid polygon."
                    );
                }
                List<double> point = data.InteriorPoint ?? PolygonCentroid(data.Polygon);
                UV location = new UV(
                    UnitConverter.PixelsToFeet(
                        point[0],
                        pixelToMeter
                    )+offset.X,
                    UnitConverter.PixelsToFeet(
                        point[1],
                        pixelToMeter
                    )+offset.Y
                );
                Room room;
                try
                {
                    room = document.Create.NewRoom(level, location);
                }
                catch (Exception exception)
                {
                    throw new InvalidOperationException(
                        $"Room {data.RoomId} ({data.Type}) could not be created.",
                        exception
                    );
                }
                if (room == null)
                {
                    throw new InvalidOperationException(
                        $"Room {data.RoomId} ({data.Type}) was not created."
                    );
                }
                SetParameter(
                    room,
                    BuiltInParameter.ROOM_NAME,
                    MapRoomName(data.Type)
                );
                SetParameter(
                    room,
                    BuiltInParameter.ROOM_NUMBER,
                    index.ToString("D3")
                );
                placedRooms.Add(room);
                try
                {
                    document.Create.NewRoomTag(
                        new LinkElementId(room.Id),
                        location,
                        view.Id
                    );
                }
                catch (Exception exception)
                {
                    throw new InvalidOperationException(
                        $"Room {data.RoomId} ({data.Type}) was created, " +
                        "but its tag could not be placed.",
                        exception
                    );
                }
                index++;
            }
            document.Regenerate();
            var noteType = new FilteredElementCollector(document).OfClass(typeof(TextNoteType))
                .Cast<TextNoteType>().FirstOrDefault();
            if (noteType == null) throw new InvalidOperationException("The project needs a text note type for area labels.");
            double totalArea = 0;
            foreach (Room room in placedRooms)
            {
                double squareMetres = room.Area * 0.09290304;
                if (squareMetres <= 0) throw new InvalidOperationException("A generated room has no measurable floor area.");
                totalArea += squareMetres;
                var location = ((LocationPoint)room.Location).Point;
                TextNote.Create(document, view.Id, location + new XYZ(0, -UnitConverter.MetersToFeet(.35), 0),
                    squareMetres.ToString("0.0", System.Globalization.CultureInfo.InvariantCulture) + " m²", noteType.Id);
            }
            if (placedRooms.Count > 0)
            {
                var bounds = placedRooms.Select(r => r.get_BoundingBox(view)).Where(b => b != null).ToList();
                if (bounds.Count > 0)
                    TextNote.Create(document, view.Id,
                        new XYZ(bounds.Min(b => b.Min.X), bounds.Min(b => b.Min.Y) - UnitConverter.MetersToFeet(1), level.Elevation),
                        UiLanguage.Text("Total floor area (including balconies): ", "总使用面积（含阳台）：") +
                        totalArea.ToString("0.0", System.Globalization.CultureInfo.InvariantCulture) + " m²", noteType.Id);
            }
        }

        private static void SetParameter(
            Room room,
            BuiltInParameter parameter,
            string value)
        {
            Parameter target = room.get_Parameter(parameter);
            if (target == null)
            {
                throw new InvalidOperationException(
                    $"Room parameter {parameter} is unavailable."
                );
            }
            if (target.IsReadOnly)
            {
                throw new InvalidOperationException(
                    $"Room parameter {parameter} is read-only."
                );
            }
            target.Set(value);
        }

        private static string MapRoomName(string type)
        {
            switch ((type ?? string.Empty).ToLowerInvariant())
            {
                case "living":
                    return UiLanguage.Text("Living","客厅");
                case "bedroom":
                    return UiLanguage.Text("Bedroom","卧室");
                case "bathroom":
                    return UiLanguage.Text("Bathroom","卫生间");
                case "kitchen":
                    return UiLanguage.Text("Kitchen","厨房");
                case "balcony":
                    return UiLanguage.Text("Balcony","阳台");
                case "storage":
                    return UiLanguage.Text("Storage","储藏室");
                default:
                    return UiLanguage.Text("Room","房间");
            }
        }

        private static List<double> PolygonCentroid(
            List<List<double>> polygon)
        {
            double signedArea = 0.0;
            double x = 0.0;
            double y = 0.0;
            for (int index = 0; index < polygon.Count; index++)
            {
                List<double> current = polygon[index];
                List<double> next = polygon[
                    (index + 1) % polygon.Count
                ];
                double cross =
                    current[0] * next[1] -
                    next[0] * current[1];
                signedArea += cross;
                x += (current[0] + next[0]) * cross;
                y += (current[1] + next[1]) * cross;
            }
            if (Math.Abs(signedArea) <= 1e-9)
            {
                return new List<double>
                {
                    polygon.Average(item => item[0]),
                    polygon.Average(item => item[1])
                };
            }
            signedArea *= 0.5;
            return new List<double>
            {
                x / (6.0 * signedArea),
                y / (6.0 * signedArea)
            };
        }
    }
}
