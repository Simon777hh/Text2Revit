using System;
using System.IO;
namespace Text2Revit
{
    internal static class UiLanguage
    {
        static readonly string Preference=Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),"Text2Revit","preferences.language");
        public static bool IsChinese { get; private set; }=ReadPreference();
        public static string Text(string english,string chinese) { return IsChinese ? chinese : english; }
        static bool ReadPreference()
        {
            try { return File.Exists(Preference)&&File.ReadAllText(Preference).Trim()=="zh"; } catch(IOException) { return false; } catch(UnauthorizedAccessException) { return false; }
        }
        public static void Set(bool chinese)
        {
            IsChinese=chinese;
            try { Directory.CreateDirectory(Path.GetDirectoryName(Preference));File.WriteAllText(Preference,chinese?"zh":"en"); } catch(IOException) { } catch(UnauthorizedAccessException) { }
        }
    }
}
