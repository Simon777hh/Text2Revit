using System;
using System.Collections.Generic;
using System.Linq;
using Autodesk.Revit.DB;

namespace Text2Revit.Addin.Services
{
    public static class OpeningGeometryValidator
    {
        public static void ValidateWindow(FamilyInstance window,Level level,double sillM,double heightM)
        {
            var options=new Options { DetailLevel=ViewDetailLevel.Fine };
            var points=Vertices(window.get_Geometry(options)).ToList();
            if (points.Count==0) throw new InvalidOperationException("The generated window has no solid geometry.");
            double bottom=UnitConverter.MetersToFeet(sillM)+level.Elevation;
            double top=bottom+UnitConverter.MetersToFeet(heightM);
            double tolerance=UnitConverter.MillimetersToFeet(1);
            if (Math.Abs(points.Min(p=>p.Z)-bottom)>tolerance || Math.Abs(points.Max(p=>p.Z)-top)>tolerance)
                throw new InvalidOperationException(UiLanguage.Text("The window's actual sill or height differs from the requested dimensions. Check the family template insertion origin.","窗户实体的窗台高或窗高与请求不一致，请检查窗族模板的插入原点。"));
        }
        private static IEnumerable<XYZ> Vertices(GeometryElement geometry)
        {
            foreach (var item in geometry)
            {
                if (item is GeometryInstance instance)
                {
                    foreach (var point in Vertices(instance.GetInstanceGeometry())) yield return point;
                }
                else if (item is Solid solid && solid.Volume>1e-9)
                {
                    foreach (Face face in solid.Faces)
                        foreach (XYZ point in face.Triangulate().Vertices) yield return point;
                }
            }
        }
    }
}
