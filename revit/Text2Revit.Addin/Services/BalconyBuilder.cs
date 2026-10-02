using System;
using System.Collections.Generic;
using System.Linq;
using Autodesk.Revit.DB;
using Autodesk.Revit.DB.Architecture;
using Text2Revit.Addin.Models;

namespace Text2Revit.Addin.Services
{
    public sealed class BalconyBuilder
    {
        private readonly Document document;
        private readonly ViewPlan view;
        private readonly double scale;
        private readonly XYZ offset;
        public BalconyBuilder(Document document, ViewPlan view, double scale, XYZ offset)
        { this.document=document; this.view=view; this.scale=scale; this.offset=offset; }

        public List<ElementId> Build(IEnumerable<RailingData> railings)
        {
            var result=new List<ElementId>();
            var segments=(railings??Enumerable.Empty<RailingData>()).ToList();
            if (segments.Count==0) return result;
            var sketch=SketchPlane.Create(document, Plane.CreateByNormalAndOrigin(XYZ.BasisZ,new XYZ(0,0,view.GenLevel.Elevation)));
            foreach (var data in segments)
            {
                var points=data.ReferenceLine.Select(p=>UnitConverter.PointToXyz(p[0],p[1],0,scale)+new XYZ(offset.X,offset.Y,view.GenLevel.Elevation)).ToList();
                var line=Line.CreateBound(points[0],points[1]);
                var path=new CurveLoop();
                path.Append(line);
                var railing=Railing.Create(document,path,EnsureType(data.HeightM).Id,view.GenLevel.Id);
                if (railing==null) throw new InvalidOperationException("The balcony railing could not be created.");
                result.Add(railing.Id);
                // Railings do not enclose rooms. Separation lines preserve the balcony area.
                var boundary=new CurveArray();
                boundary.Append(line);
                document.Create.NewRoomBoundaryLines(sketch,boundary,view);
            }
            return result;
        }

        private RailingType EnsureType(double heightM)
        {
            string name="Text2Revit Balcony "+Math.Round(heightM*1000)+"mm";
            var types=new FilteredElementCollector(document).OfClass(typeof(RailingType)).Cast<RailingType>().ToList();
            var existing=types.FirstOrDefault(t=>t.Name==name);
            if (existing!=null) return existing;
            var source=types.OrderByDescending(t=>t.RailStructure.GetNonContinuousRailCount()).FirstOrDefault();
            if (source==null) throw new InvalidOperationException(UiLanguage.Text("The project template needs a railing type for open balconies.","项目模板需要包含阳台栏杆类型。"));
            var type=(RailingType)source.Duplicate(name);
            type.TopRailHeight=UnitConverter.MetersToFeet(heightM);
            var height=type.get_Parameter(BuiltInParameter.STAIRS_RAILING_HEIGHT);
            if (height!=null && !height.IsReadOnly) height.Set(UnitConverter.MetersToFeet(heightM));
            return type;
        }
    }
}
