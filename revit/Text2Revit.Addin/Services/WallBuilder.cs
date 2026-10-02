using System;
using System.Collections.Generic;
using System.Linq;
using Autodesk.Revit.DB;
using Text2Revit.Addin.Models;
namespace Text2Revit.Addin.Services
{
    public sealed class WallHost
    {
        public string WallId { get; set; }
        public Wall Wall { get; set; }
        public List<List<double>> ReferenceLine { get; set; }
        public bool IsExterior { get; set; }
    }
    public sealed class WallBuilder
    {
        private readonly Document document; private readonly Level level; private readonly double pixelToMeter; private readonly XYZ offset;
        public WallBuilder(Document document,Level level,double pixelToMeter,XYZ offset=null) { this.document=document;this.level=level;this.pixelToMeter=pixelToMeter;this.offset=offset??XYZ.Zero; }
        public List<WallHost> Build(PlanData plan)
        {
            var result=new List<WallHost>();
            foreach (var segment in plan.Walls)
            {
                var a=segment.ReferenceLine[0]; var b=segment.ReferenceLine[1];
                var start=UnitConverter.PointToXyz(a[0],a[1],0,pixelToMeter)+new XYZ(offset.X,offset.Y,level.Elevation);
                var end=UnitConverter.PointToXyz(b[0],b[1],0,pixelToMeter)+new XYZ(offset.X,offset.Y,level.Elevation);
                if (start.DistanceTo(end)<document.Application.ShortCurveTolerance) throw new InvalidOperationException(UiLanguage.Text("A wall segment is shorter than the Revit tolerance.","墙段太短，无法在 Revit 中创建。"));
                bool exterior=segment.Host=="exterior_wall";
                var type=EnsureWallType(exterior ? "Text2Revit Exterior 400mm" : "Text2Revit Interior 200mm", exterior ? 0.4 : 0.2);
                var wall=Wall.Create(document,Line.CreateBound(start,end),type.Id,level.Id,UnitConverter.MetersToFeet(segment.HeightM),0,false,false);
                result.Add(new WallHost { WallId=segment.WallId,Wall=wall,ReferenceLine=segment.ReferenceLine,IsExterior=exterior });
            }
            return result;
        }
        private WallType EnsureWallType(string name,double thicknessM)
        {
            var types=new FilteredElementCollector(document).OfClass(typeof(WallType)).Cast<WallType>();
            var existing=types.FirstOrDefault(t=>t.Name==name);
            if (existing != null) return existing;
            var source=types.FirstOrDefault(t=>t.Kind==WallKind.Basic) ?? throw new InvalidOperationException(UiLanguage.Text("The project has no basic wall type.","项目中没有基本墙类型。"));
            var type=(WallType)source.Duplicate(name);
            var old=source.GetCompoundStructure();
            var material=old?.GetLayers().FirstOrDefault(l=>l.Width>0)?.MaterialId ?? ElementId.InvalidElementId;
            var layers=new List<CompoundStructureLayer> { new CompoundStructureLayer(UnitConverter.MetersToFeet(thicknessM),MaterialFunctionAssignment.Structure,material) };
            type.SetCompoundStructure(CompoundStructure.CreateSimpleCompoundStructure(layers)); return type;
        }
    }
}
