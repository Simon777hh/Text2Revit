using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Autodesk.Revit.DB;
using Text2Revit.Addin.Models;
namespace Text2Revit.Addin.Services
{
    public sealed class FamilyLibrary
    {
        private readonly Document project;
        private readonly Autodesk.Revit.ApplicationServices.Application application;
        public FamilyLibrary(Document project,Autodesk.Revit.ApplicationServices.Application application) { this.project=project;this.application=application; }
        public static string Key(OpeningData o)
        {
            // New window geometry must not reuse the old zero-based cached families.
            return "Text2Revit_"+(o.Type=="window" ? "v3_Window" : "v2_Door")+"_"+
                Math.Round(o.WidthM*1000).ToString(CultureInfo.InvariantCulture)+"x"+Math.Round(o.HeightM*1000).ToString(CultureInfo.InvariantCulture);
        }
        public Dictionary<string,FamilySymbol> Prepare(IEnumerable<OpeningData> openings)
        {
            var symbols=new Dictionary<string,FamilySymbol>();
            var cache=Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),"Text2Revit","families",application.VersionNumber);
            Directory.CreateDirectory(cache);
            foreach (var group in openings.GroupBy(Key))
            {
                string name=group.Key;var o=group.First();
                var existing=new FilteredElementCollector(project).OfClass(typeof(FamilySymbol)).Cast<FamilySymbol>().FirstOrDefault(s=>s.FamilyName==name);
                if (existing != null) { symbols[name]=existing;continue; }
                string path=Path.Combine(cache,name+".rfa");
                if (!File.Exists(path)) CreateFamily(path,o);
                Family family;
                using (var transaction=new Transaction(project,UiLanguage.Text("Text2Revit Load Door and Window Families","Text2Revit 加载门窗族")))
                {
                    transaction.Start();
                    if (!project.LoadFamily(path,new LoadOptions(),out family)) throw new InvalidOperationException(UiLanguage.Text("Could not load the door/window family: ","无法加载门窗族：")+name);
                    transaction.Commit();
                }
                symbols[name]=(FamilySymbol)project.GetElement(family.GetFamilySymbolIds().First());
            }
            return symbols;
        }
        private void CreateFamily(string path,OpeningData o)
        {
            bool window=o.Type=="window";
            string template=FindTemplate(window);
            using (Document family=application.NewFamilyDocument(template))
            {
                double width=UnitConverter.MetersToFeet(Math.Round(o.WidthM*1000)/1000.0), height=UnitConverter.MetersToFeet(Math.Round(o.HeightM*1000)/1000.0);
                double frame=UnitConverter.MetersToFeet(0.045);
                using (var transaction=new Transaction(family,UiLanguage.Text("Create Text2Revit Doors and Windows","创建 Text2Revit 门窗")))
                {
                    transaction.Start();
                    var manager=family.FamilyManager;
                    if (manager.CurrentType == null) manager.NewType("Default");
                    SetDimension(manager,BuiltInParameter.FAMILY_WIDTH_PARAM,width);
                    SetDimension(manager,BuiltInParameter.FAMILY_HEIGHT_PARAM,height);
                    family.Regenerate();
                    double baseZ=InsertionHeight(family,window);
                    var wall=new FilteredElementCollector(family).OfClass(typeof(Wall)).Cast<Wall>().FirstOrDefault();
                    if (wall == null) throw new InvalidOperationException(UiLanguage.Text("The door/window template has no host wall: ","门窗模板没有宿主墙：")+template);
                    // Replace the template opening with one sized to this generated type.
                    foreach (var id in new FilteredElementCollector(family).OfClass(typeof(Opening)).ToElementIds()) family.Delete(id);
                    var profile=Loop(new[] { new XYZ(-width/2,0,baseZ),new XYZ(width/2,0,baseZ),new XYZ(width/2,0,baseZ+height),new XYZ(-width/2,0,baseZ+height) });
                    family.FamilyCreate.NewOpening(wall,profile);
                    var plane=SketchPlane.Create(family,Plane.CreateByNormalAndOrigin(new XYZ(0,-1,0),new XYZ(0,0,baseZ)));
                    // Each frame side is its own solid, leaving the center clear.
                    Solid(family,plane,-width/2,0,-width/2+frame,height,0.12,null);
                    Solid(family,plane,width/2-frame,0,width/2,height,0.12,null);
                    Solid(family,plane,-width/2,height-frame,width/2,height,0.12,null);
                    if (window)
                    {
                        Solid(family,plane,-width/2,0,width/2,frame,0.12,null);
                        var materialId=Material.Create(family,"Text2Revit Glass");
                        var material=(Material)family.GetElement(materialId);material.Transparency=80;material.Color=new Color(145,195,220);
                        Solid(family,plane,-width/2+frame,frame,width/2-frame,height-frame,0.018,materialId);
                        Solid(family,plane,-frame/2,frame,frame/2,height-frame,0.10,null);
                    }
                    else
                    {
                        // Door leaf plus jambs; a separate family is created for each generated dimension.
                        var leaf=Solid(family,plane,-width/2+frame,0,width/2-frame,height-frame,0.04,null);
                        var visibility=new FamilyElementVisibility(FamilyElementVisibilityType.Model);
                        visibility.IsShownInPlanRCPCut=false;visibility.IsShownInTopBottom=false;leaf.SetVisibility(visibility);
                        var planPlane=SketchPlane.Create(family,Plane.CreateByNormalAndOrigin(XYZ.BasisZ,XYZ.Zero));
                        var hinge=new XYZ(-width/2+frame,0,0);double clear=width-2*frame;
                        family.FamilyCreate.NewSymbolicCurve(Line.CreateBound(hinge,hinge-new XYZ(0,clear,0)),planPlane);
                        family.FamilyCreate.NewSymbolicCurve(Arc.Create(hinge,clear,0,Math.PI/2,XYZ.BasisX,-XYZ.BasisY),planPlane);
                    }
                    transaction.Commit();
                }
                family.SaveAs(path,new SaveAsOptions { OverwriteExistingFile=true });
            }
        }
        private static void SetDimension(FamilyManager manager,BuiltInParameter id,double value)
        {
            var parameter=manager.get_Parameter(id);
            if (parameter != null && !parameter.IsReadOnly && !parameter.IsDeterminedByFormula) manager.Set(parameter,value);
        }
        private static double InsertionHeight(Document family,bool window)
        {
            if (!window) return 0;
            // The horizontal origin plane can be above zero (e.g. 800 mm in the Chinese template).
            // Use the API flag and normal, independent of localized reference-plane names.
            var origins=new FilteredElementCollector(family).OfClass(typeof(ReferencePlane)).Cast<ReferencePlane>()
                .Where(p=>p.get_Parameter(BuiltInParameter.DATUM_PLANE_DEFINES_ORIGIN)?.AsInteger()==1)
                .Select(p=>p.GetPlane()).Where(p=>Math.Abs(p.Normal.Z)>0.999).ToList();
            if (origins.Count!=1) throw new InvalidOperationException(UiLanguage.Text("The window template needs one horizontal insertion-origin plane.","窗族模板需要一个定义插入原点的水平参考平面。"));
            return origins[0].Origin.Z;
        }
        private static CurveArray Loop(XYZ[] points)
        {
            var curves=new CurveArray();for (int i=0;i<points.Length;i++) curves.Append(Line.CreateBound(points[i],points[(i+1)%points.Length]));return curves;
        }
        private static Extrusion Solid(Document family,SketchPlane plane,double x0,double z0,double x1,double z1,double depth,ElementId material)
        {
            double baseZ=plane.GetPlane().Origin.Z;
            z0+=baseZ;z1+=baseZ;
            var loops=new CurveArrArray();loops.Append(Loop(new[] { new XYZ(x0,0,z0),new XYZ(x1,0,z0),new XYZ(x1,0,z1),new XYZ(x0,0,z1) }));
            var extrusion=family.FamilyCreate.NewExtrusion(true,loops,plane,UnitConverter.MetersToFeet(depth));
            if (material != null) extrusion.get_Parameter(BuiltInParameter.MATERIAL_ID_PARAM)?.Set(material);
            return extrusion;
        }
        private string FindTemplate(bool window)
        {
            var names=window ? new[] { "Metric Window.rft","公制窗.rft","Window.rft","公制窗戶.rft" } : new[] { "Metric Door.rft","公制门.rft","Door.rft","公制門.rft" };
            var roots=new[] { application.FamilyTemplatePath,Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData),"Autodesk","RVT "+application.VersionNumber,"Family Templates") };
            foreach (var root in roots.Distinct())
                if (Directory.Exists(root)) foreach (var name in names)
                {
                    var file=Directory.EnumerateFiles(root,name,SearchOption.AllDirectories).FirstOrDefault();if (file != null) return file;
                }
            throw new FileNotFoundException(UiLanguage.Text("The Revit door/window templates are missing. Repair the core content for this Revit version.","Revit 的门窗族模板不可用。请安装对应年份的 Revit 内容库。"));
        }
        private sealed class LoadOptions : IFamilyLoadOptions
        {
            public bool OnFamilyFound(bool inUse,out bool overwrite) { overwrite=true;return true; }
            public bool OnSharedFamilyFound(Family family,bool inUse,out FamilySource source,out bool overwrite) { source=FamilySource.Family;overwrite=true;return true; }
        }
    }
}
