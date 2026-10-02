using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.Serialization.Json;
using System.Threading;
using System.Threading.Tasks;
using Text2Revit.Addin.Models;
namespace Text2Revit.Addin.Services
{
    public sealed class PythonBackendRunner
    {
        public string WorkDirectory { get; private set; }
        public string StatusPath { get { return Path.Combine(WorkDirectory, "status.json"); } }
        public static string InstallRoot { get { return Directory.GetParent(Directory.GetParent(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)).FullName).FullName; } }
        public PythonBackendRunner()
        {
            WorkDirectory = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "Text2Revit", "jobs", DateTime.Now.ToString("yyyyMMdd_HHmmss") + "_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(WorkDirectory);
        }
        public PlanData Run(string prompt, CancellationToken cancellation)
        {
            string root = InstallRoot, runtime = Path.Combine(root, "runtime");
            string python = Path.Combine(runtime, "python.exe"), backend = Path.Combine(root, "backend", "backend_cli.py");
            if (!File.Exists(python) || !File.Exists(backend)) throw new FileNotFoundException(UiLanguage.Text("Backend files are missing. Run the installer again.","后端文件不完整，请重新运行安装程序。"));
            string request = Path.Combine(WorkDirectory, "request.json"), response = Path.Combine(WorkDirectory, "response.json");
            using (var stream = File.Create(request)) new DataContractJsonSerializer(typeof(BackendRequest)).WriteObject(stream, new BackendRequest { Prompt = prompt,Language=UiLanguage.IsChinese?"zh":"en" });
            var start = new ProcessStartInfo(python, Quote(backend) + " --request " + Quote(request) + " --response " + Quote(response) + " --status " + Quote(StatusPath))
            { UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true, RedirectStandardError = true, WorkingDirectory = root };
            ProcessEnvironment.ConfigurePython(start,runtime);
            using (var process = Process.Start(start))
            {
                if (process == null) throw new InvalidOperationException(UiLanguage.Text("Could not start the Python backend.","无法启动 Python 后端。"));
                Task<string> stdout = process.StandardOutput.ReadToEndAsync(), stderr = process.StandardError.ReadToEndAsync();
                var timer = Stopwatch.StartNew();
                while (!process.WaitForExit(200))
                {
                    if (cancellation.IsCancellationRequested || timer.Elapsed > TimeSpan.FromMinutes(30))
                    {
                        process.Kill(); process.WaitForExit();
                        if (cancellation.IsCancellationRequested) throw new OperationCanceledException();
                        throw new TimeoutException(UiLanguage.Text("Generation exceeded 30 minutes and was stopped.","生成超过 30 分钟，已停止。"));
                    }
                }
                Task.WaitAll(stdout, stderr);
                File.WriteAllText(Path.Combine(WorkDirectory, "backend.log"), stdout.Result + "\n" + stderr.Result);
                cancellation.ThrowIfCancellationRequested();
                if (process.ExitCode != 0) throw new InvalidOperationException(UiLanguage.Text("Generation failed. Details: ","生成失败，详细日志：") + Path.Combine(WorkDirectory, "backend.log") + "\n" + stderr.Result);
            }
            return JsonPlanReader.Read(response);
        }
        private static string Quote(string value) { return "\"" + value + "\""; }
    }
}
