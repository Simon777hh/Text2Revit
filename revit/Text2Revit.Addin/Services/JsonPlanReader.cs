using System;
using System.IO;
using System.Runtime.Serialization.Json;
using Text2Revit.Addin.Models;
namespace Text2Revit.Addin.Services
{
    public static class JsonPlanReader
    {
        public static T Read<T>(string path)
        {
            using (var stream = new FileStream(path,FileMode.Open,FileAccess.Read,FileShare.ReadWrite|FileShare.Delete))
                return (T)new DataContractJsonSerializer(typeof(T)).ReadObject(stream);
        }
        public static PlanData Read(string path)
        {
            var plan = Read<PlanData>(path);
            if (plan == null || plan.SchemaVersion != "1.0" || plan.CoordinateSystem != "raw_canvas_px"
                || double.IsNaN(plan.PixelToMeter) || double.IsInfinity(plan.PixelToMeter) || plan.PixelToMeter <= 0
                || plan.Rooms == null || plan.Rooms.Count == 0 || plan.Walls == null || plan.Walls.Count == 0 || plan.Openings == null)
                throw new InvalidDataException(UiLanguage.Text("The backend result is invalid. Generate another plan.","后端结果格式不正确，请重新生成。"));
            return plan;
        }
    }
}
