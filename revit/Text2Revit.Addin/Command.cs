using System;
using System.Collections.Generic;
using System.Linq;
using Autodesk.Revit.Attributes;
using Autodesk.Revit.DB;
using Autodesk.Revit.UI;
using Text2Revit.Addin.Services;
using Text2Revit.Addin.UI;
namespace Text2Revit.Addin
{
    [Transaction(TransactionMode.Manual)]
    public sealed class Command : IExternalCommand
    {
        public Result Execute(ExternalCommandData data, ref string message, ElementSet elements)
        {
            try
            {
                var ui=data.Application.ActiveUIDocument;
                var view=ui == null ? null : ModelViewBuilder.FindSourcePlan(ui.Document,ui.ActiveView);
                if (view?.GenLevel == null || ui.Document.IsFamilyDocument) { message=UiLanguage.Text("Open a floor plan view in a project first.","请先打开项目中的楼层平面视图。"); return Result.Failed; }
                var plan=PromptDialog.Generate(data.Application.MainWindowHandle);
                if (plan == null) return Result.Cancelled;
                var doc=ui.Document;
                var boxes=new FilteredElementCollector(doc,view.Id).OfClass(typeof(Wall))
                    .Select(w=>w.get_BoundingBox(view)).Where(b=>b!=null).ToList();
                var points=plan.Walls.SelectMany(w=>w.ReferenceLine)
                    .Concat((plan.Railings??new List<Text2Revit.Addin.Models.RailingData>()).SelectMany(r=>r.ReferenceLine)).ToList();
                var offset=boxes.Count==0 ? XYZ.Zero : new XYZ(
                    boxes.Max(b=>b.Max.X)+UnitConverter.MetersToFeet(2)-UnitConverter.PixelsToFeet(points.Min(p=>p[0]),plan.PixelToMeter),
                    boxes.Min(b=>b.Min.Y)-UnitConverter.PixelsToFeet(points.Min(p=>p[1]),plan.PixelToMeter),0);
                var created=new List<ElementId>();
                View3D modelView=null;
                using(var group=new TransactionGroup(doc,UiLanguage.Text("Text2Revit Generate Model","Text2Revit 生成模型")))
                {
                group.Start();
                // Family loading and construction are combined into one undo operation.
                var symbols=new FamilyLibrary(doc, data.Application.Application).Prepare(plan.Openings);
                using (var transaction=new Transaction(doc,UiLanguage.Text("Text2Revit Generate Model","Text2Revit 生成模型")))
                {
                    transaction.Start();
                    var hosts=new WallBuilder(doc,view.GenLevel,plan.PixelToMeter,offset).Build(plan);
                    created.AddRange(hosts.Select(h=>h.Wall.Id));
                    created.AddRange(new BalconyBuilder(doc,view,plan.PixelToMeter,offset).Build(plan.Railings));
                    doc.Regenerate();
                    new OpeningBuilder(doc,view.GenLevel,hosts,plan.PixelToMeter,symbols,offset).Build(plan.Openings);
                    doc.Regenerate();
                    new RoomBuilder(doc,view,view.GenLevel,plan.PixelToMeter,offset).Build(plan.Rooms);
                    modelView=ModelViewBuilder.Build(doc,view,created);
                    var result=transaction.Commit();
                    if (result != TransactionStatus.Committed) throw new InvalidOperationException(UiLanguage.Text("Revit could not commit the generated plan.","Revit 未能提交生成结果。"));
                }
                group.Assimilate();
                }
                ui.ActiveView=modelView;
                ui.ShowElements(created);
                return Result.Succeeded;
            }
            catch (Exception error) { message=error.Message; return Result.Failed; }
        }
    }
}
