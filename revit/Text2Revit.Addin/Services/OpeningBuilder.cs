using System;
using System.Collections.Generic;
using System.Linq;
using Autodesk.Revit.DB;
using Autodesk.Revit.DB.Structure;
using Text2Revit.Addin.Models;
namespace Text2Revit.Addin.Services
{
    public sealed class OpeningBuilder
    {
        private readonly Document document; private readonly Level level;
        private readonly Dictionary<string,WallHost> walls; private readonly double scale;
        private readonly Dictionary<string,FamilySymbol> symbols;
        private readonly XYZ offset;
        public OpeningBuilder(Document document,Level level,List<WallHost> walls,double scale,Dictionary<string,FamilySymbol> symbols,XYZ offset=null)
        { this.document=document;this.level=level;this.walls=walls.ToDictionary(w=>w.WallId);this.scale=scale;this.symbols=symbols;this.offset=offset??XYZ.Zero; }
        public void Build(IEnumerable<OpeningData> openings)
        {
            var windows=new List<Tuple<FamilyInstance,OpeningData>>();
            foreach (var opening in openings)
            {
                if (!walls.TryGetValue(opening.WallId,out WallHost host)) throw new InvalidOperationException(UiLanguage.Text("A door/window has no host wall.","门窗缺少宿主墙。"));
                var symbol=symbols[FamilyLibrary.Key(opening)];
                if (!symbol.IsActive) { symbol.Activate();document.Regenerate(); }
                var point=UnitConverter.PointToXyz(opening.Center[0],opening.Center[1],opening.SillM,scale)+new XYZ(offset.X,offset.Y,level.Elevation);
                var instance=document.Create.NewFamilyInstance(point,symbol,host.Wall,level,StructuralType.NonStructural);
                var sill=instance.get_Parameter(BuiltInParameter.INSTANCE_SILL_HEIGHT_PARAM);
                if (sill != null && !sill.IsReadOnly && opening.Type=="window") sill.Set(UnitConverter.MetersToFeet(opening.SillM));
                if (opening.Type=="window") windows.Add(Tuple.Create(instance,opening));
            }
            document.Regenerate();
            foreach (var window in windows)
                OpeningGeometryValidator.ValidateWindow(window.Item1,level,window.Item2.SillM,window.Item2.HeightM);
        }
    }
}
