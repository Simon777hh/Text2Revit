using System;
using System.Collections.Generic;
using System.Linq;
using Autodesk.Revit.DB;
using Autodesk.Revit.DB.ExtensibleStorage;
namespace Text2Revit.Addin.Services
{
    public static class ModelViewBuilder
    {
        private static readonly Guid SourceSchemaId = new Guid("AC65B084-04C9-44A5-A270-501828951FEB");
        private const string SourceField = "SourcePlanUniqueId";
        public static ViewPlan FindSourcePlan(Document document, View activeView)
        {
            var plan = activeView as ViewPlan;
            if (plan != null) return plan.GenLevel != null ? plan : null;
            if (!(activeView is View3D)) return null;
            var schema = Schema.Lookup(SourceSchemaId);
            if (schema == null) return null;
            var entity = activeView.GetEntity(schema);
            if (!entity.IsValid()) return null;
            return document.GetElement(entity.Get<string>(schema.GetField(SourceField))) as ViewPlan;
        }
        // Called inside the construction transaction; the active UI view is changed after commit.
        public static View3D Build(Document document, ViewPlan sourcePlan, ICollection<ElementId> modelIds)
        {
            var schema = Schema.Lookup(SourceSchemaId);
            if (schema == null)
            {
                var builder = new SchemaBuilder(SourceSchemaId);
                builder.SetSchemaName("Text2RevitSourcePlan");
                builder.SetReadAccessLevel(AccessLevel.Public);
                builder.SetWriteAccessLevel(AccessLevel.Public);
                builder.AddSimpleField(SourceField, typeof(string));
                schema = builder.Finish();
            }
            var view = new FilteredElementCollector(document).OfClass(typeof(View3D)).Cast<View3D>()
                .FirstOrDefault(v => !v.IsTemplate && FindSourcePlan(document, v)?.UniqueId == sourcePlan.UniqueId);
            if (view == null)
            {
                var type = new FilteredElementCollector(document).OfClass(typeof(ViewFamilyType)).Cast<ViewFamilyType>()
                    .First(t => t.ViewFamily == ViewFamily.ThreeDimensional);
                view = View3D.CreateIsometric(document, type.Id);
                var names = new HashSet<string>(new FilteredElementCollector(document).OfClass(typeof(View)).Cast<View>().Select(v => v.Name));
                string name = "Text2Revit 3D", candidate = name;
                for (int i = 2; names.Contains(candidate); i++) candidate = name + " " + i;
                view.Name = candidate;
                var entity = new Entity(schema);
                entity.Set(schema.GetField(SourceField), sourcePlan.UniqueId);
                view.SetEntity(entity);
            }
            document.Regenerate();
            var boxes = modelIds.Select(id => document.GetElement(id).get_BoundingBox(null)).Where(b => b != null).ToList();
            double padding = UnitConverter.MetersToFeet(0.8);
            var bounds = new BoundingBoxXYZ
            {
                Min = new XYZ(boxes.Min(b => b.Min.X) - padding, boxes.Min(b => b.Min.Y) - padding, boxes.Min(b => b.Min.Z) - padding),
                Max = new XYZ(boxes.Max(b => b.Max.X) + padding, boxes.Max(b => b.Max.Y) + padding, boxes.Max(b => b.Max.Z) + padding)
            };
            view.SetSectionBox(bounds);
            view.IsSectionBoxActive = true;
            view.DetailLevel = ViewDetailLevel.Fine;
            view.DisplayStyle = DisplayStyle.ShadingWithEdges;
            foreach (var category in new[] { BuiltInCategory.OST_Levels, BuiltInCategory.OST_SectionBox })
            {
                var id = new ElementId(category);
                if (view.CanCategoryBeHidden(id)) view.SetCategoryHidden(id, true);
            }
            var forward = new XYZ(-1, 1, -1).Normalize();
            var up = forward.CrossProduct(XYZ.BasisZ).Normalize().CrossProduct(forward).Normalize();
            var center = (bounds.Min + bounds.Max) / 2;
            view.SetOrientation(new ViewOrientation3D(center - forward * UnitConverter.MetersToFeet(20), up, forward));
            return view;
        }
    }
}
