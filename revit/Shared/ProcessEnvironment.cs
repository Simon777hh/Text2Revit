using System;
using System.Collections;
using System.Collections.Specialized;
using System.Diagnostics;
using System.IO;
using System.Text;
namespace Text2Revit
{
    internal static class ProcessEnvironment
    {
        public static void ConfigurePython(ProcessStartInfo info,string runtime)
        {
            if(info.RedirectStandardOutput) info.StandardOutputEncoding=Encoding.UTF8;
            if(info.RedirectStandardError) info.StandardErrorEncoding=Encoding.UTF8;
            StringDictionary variables;
            try { variables=info.EnvironmentVariables; }
            catch(ArgumentException)
            {
                // Framework assigns its dictionary before populating it; retry after duplicate-key failure.
                variables=info.EnvironmentVariables;
            }
            variables.Clear();
            foreach(DictionaryEntry entry in Environment.GetEnvironmentVariables())
            {
                string key=(string)entry.Key;
                variables[key]=Environment.GetEnvironmentVariable(key)??(string)entry.Value;
            }
            variables["PATH"]=runtime+";"+Path.Combine(runtime,"Library","bin")+";"+Path.Combine(runtime,"DLLs")+";"+variables["PATH"];
            variables["PYTHONNOUSERSITE"]="1";variables["PYTHONIOENCODING"]="utf-8";
            variables.Remove("PYTHONHOME");variables.Remove("PYTHONPATH");
        }
    }
}
