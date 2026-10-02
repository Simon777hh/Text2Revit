using System.IO;
using System.Reflection;
using System.Windows.Media.Imaging;
using Autodesk.Revit.UI;
using Autodesk.Revit.DB;
using Text2Revit.Addin.Services;
namespace Text2Revit.Addin
{
    public sealed class App : IExternalApplication
    {
        public Result OnStartup(UIControlledApplication app)
        {
            try { app.CreateRibbonTab("Text2Revit"); } catch (Autodesk.Revit.Exceptions.ArgumentException) { }
            var panel = app.CreateRibbonPanel("Text2Revit", UiLanguage.Text("Apartment models","户型模型"));
            var data = new PushButtonData("Text2Revit.Generate", UiLanguage.Text("Generate\nModel","生成\n模型"), Assembly.GetExecutingAssembly().Location, typeof(Command).FullName);
            data.AvailabilityClassName = typeof(PlanAvailability).FullName;
            var button = (PushButton)panel.AddItem(data);
            button.ToolTip = UiLanguage.Text("Generate a 3D apartment with walls, doors, windows and rooms from a prompt.","输入户型描述，自动生成三维墙、门窗和房间。");
            button.LongDescription = UiLanguage.Text("Change small / large (omit for medium) and the bedroom, bathroom and balcony counts. Open a floor plan view first.","支持 small / large（省略时默认中等）和卧室、卫生间、阳台数量。请先打开楼层平面视图。");
            button.LargeImage = Icon("icon32.png"); button.Image = Icon("icon16.png");
            return Result.Succeeded;
        }
        public Result OnShutdown(UIControlledApplication app) { return Result.Succeeded; }
        private static BitmapSource Icon(string name)
        {
            using (Stream stream = Assembly.GetExecutingAssembly().GetManifestResourceStream("Text2Revit.Addin.Resources." + name))
            {
                var bitmap = BitmapFrame.Create(stream, BitmapCreateOptions.None, BitmapCacheOption.OnLoad);
                bitmap.Freeze(); return bitmap;
            }
        }
    }
    public sealed class PlanAvailability : IExternalCommandAvailability
    {
        public bool IsCommandAvailable(UIApplication app, CategorySet categories)
        {
            var doc = app.ActiveUIDocument;
            return doc != null && !doc.Document.IsFamilyDocument && ModelViewBuilder.FindSourcePlan(doc.Document,doc.ActiveView) != null;
        }
    }
}
