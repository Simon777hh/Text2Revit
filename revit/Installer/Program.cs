using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Xml.Linq;
using Microsoft.Win32;
namespace Text2Revit.Installer
{
    [DataContract] public sealed class Release
    {
        [DataMember(Name="version")] public string Version { get; set; }
        [DataMember(Name="revit_versions")] public int[] RevitVersions { get; set; }
        [DataMember(Name="required_files")] public string[] RequiredFiles { get; set; }
        [DataMember(Name="required_free_bytes")] public long RequiredFreeBytes { get; set; }
    }
    internal static class Program
    {
        const string RegistryPath=@"Software\Microsoft\Windows\CurrentVersion\Uninstall\Text2Revit";
        static string Root=Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),"Text2Revit");
        static string Addins=Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),"Autodesk","Revit","Addins");
        [STAThread] static void Main(string[] args)
        {
            Application.EnableVisualStyles(); Application.SetCompatibleTextRenderingDefault(false);
            try
            {
                if (args.Length==2 && (args[0]=="--test-root" || args[0]=="--test-uninstall")) { Root=Path.GetFullPath(args[1]); Addins=Path.Combine(Root,"test-addins"); if(args[0]=="--test-uninstall") Uninstall(s=>Console.WriteLine(s),false); else Install(s=>Console.WriteLine(s),false); return; }
                if (args.Contains("--verify-only")) { using (var stream=Payload()) using (var zip=new ZipArchive(stream,ZipArchiveMode.Read)) { ReadRelease(zip); foreach (var e in zip.Entries) SafePath(Root,e.FullName); } return; }
                bool uninstall=args.Contains("--uninstall");
                using (var form=new Form { Text=uninstall ? UiLanguage.Text("Uninstall Text2Revit","卸载 Text2Revit") : UiLanguage.Text("Install Text2Revit","安装 Text2Revit"),ClientSize=new Size(600,300),StartPosition=FormStartPosition.CenterScreen,
                    FormBorderStyle=FormBorderStyle.FixedDialog,MaximizeBox=false,Font=new Font("Segoe UI",10) })
                {
                    var heading=new Label { Left=24,Top=24,Width=405,Height=38,Text=uninstall ? UiLanguage.Text("Uninstall Text2Revit","卸载 Text2Revit") : UiLanguage.Text("Text2Revit · Setup","Text2Revit · 自动安装"),Font=new Font(form.Font.FontFamily,17,FontStyle.Bold) };
                    var language=new ComboBox { Left=446,Top=31,Width=130,DropDownStyle=ComboBoxStyle.DropDownList };
                    language.Items.AddRange(new object[]{"English","中文"});language.SelectedIndex=UiLanguage.IsChinese?1:0;
                    Func<string> introduction=()=>uninstall ? UiLanguage.Text("Removes the add-in and runtime. Generation records are kept.\nClose Revit before continuing.","将移除工具栏插件和运行环境，保留生成记录。请先关闭 Revit。") : UiLanguage.Text("Installs the Revit add-in, Python runtime and models automatically.\nSupports Revit 2020–2026. No Python configuration or file copying.\nClose Revit and allow at least 12 GB of free disk space.","自动安装 Revit 插件、Python 后端和模型。\n支持 Revit 2020–2026，无需配置 Python 或移动文件。\n请先关闭 Revit，并预留至少 12 GB 磁盘空间。");
                    var label=new Label { Left=24,Top=78,Width=552,Height=110,Text=introduction() };
                    var progress=new ProgressBar { Left=24,Top=195,Width=552,Height=20,Visible=false,Style=ProgressBarStyle.Marquee };
                    var button=new Button { Left=440,Top=242,Width=136,Height=36,Text=uninstall ? UiLanguage.Text("Uninstall","卸载") : UiLanguage.Text("Install","安装") };
                    form.Controls.AddRange(new Control[] { heading,language,label,progress,button }); bool busy=false,finished=false;
                    language.SelectedIndexChanged+=(s,e)=>{UiLanguage.Set(language.SelectedIndex==1);form.Text=uninstall ? UiLanguage.Text("Uninstall Text2Revit","卸载 Text2Revit") : UiLanguage.Text("Install Text2Revit","安装 Text2Revit");heading.Text=uninstall ? UiLanguage.Text("Uninstall Text2Revit","卸载 Text2Revit") : UiLanguage.Text("Text2Revit · Setup","Text2Revit · 自动安装");label.Text=introduction();button.Text=uninstall ? UiLanguage.Text("Uninstall","卸载") : UiLanguage.Text("Install","安装");};
                    form.FormClosing+=(s,e)=>{ if (busy) e.Cancel=true; };
                    button.Click+=async (s,e)=>{
                        if (finished) { form.Close();return; }
                        if (Process.GetProcessesByName("Revit").Length>0) { MessageBox.Show(form,UiLanguage.Text("Close all Revit windows before continuing.","请先关闭所有 Revit 窗口，再继续。"),UiLanguage.Text("Close Revit","提示"));return; }
                        busy=true;language.Enabled=false;button.Enabled=false;progress.Visible=true;
                        Action<string> update=text=>form.BeginInvoke(new Action(()=>label.Text=text));
                        try
                        {
                            await Task.Run(()=>{ if (uninstall) Uninstall(update); else Install(update,true); });
                            label.Text=uninstall ? UiLanguage.Text("Uninstall complete. Generation records have been kept.","卸载完成。生成记录已保留。") : UiLanguage.Text("Installation complete.\nStart Revit and open a floor plan view.\nClick Generate Model on the Text2Revit tab and enter your prompt.","安装完成。\n启动 Revit，打开楼层平面视图。\n在 Text2Revit 选项卡中点击“生成模型”，输入 prompt 后即可生成三维模型。");
                            button.Text=UiLanguage.Text("Finish","完成");finished=true;
                        }
                        catch (Exception error) { label.Text=UiLanguage.Text("Setup did not finish:\n","安装未完成：\n")+error.Message;button.Text=UiLanguage.Text("Close","关闭");finished=true; }
                        finally { busy=false;progress.Visible=false;button.Enabled=true; }
                    };
                    Application.Run(form);
                }
            }
            catch (Exception error) { File.WriteAllText(Path.Combine(Path.GetTempPath(),"Text2Revit-setup-error.txt"),error.ToString());Environment.ExitCode=1; if (!args.Any(a=>a.StartsWith("--"))) MessageBox.Show(error.Message,UiLanguage.Text("Text2Revit Setup Failed","Text2Revit 安装失败")); }
        }
        static void Install(Action<string> update,bool register)
        {
            update(UiLanguage.Text("Verifying the installer…","正在校验安装包…"));
            using (var stream=Payload()) using (var zip=new ZipArchive(stream,ZipArchiveMode.Read))
            {
                var release=ReadRelease(zip);
                if (new DriveInfo(Path.GetPathRoot(Root)).AvailableFreeSpace < release.RequiredFreeBytes) throw new IOException(UiLanguage.Text("Not enough disk space. Allow at least ","磁盘空间不足，请预留至少 ")+Math.Ceiling(release.RequiredFreeBytes/1024.0/1024/1024)+" GB.");
                string destination=SafePath(Path.Combine(Root,"releases"),release.Version);
                if (Directory.Exists(destination)) throw new IOException(UiLanguage.Text("This release is already installed. Uninstall it before reinstalling.","此版本已经安装。请先卸载后重新安装。"));
                var backups=new Dictionary<string,byte[]>();
                try
                {
                    Directory.CreateDirectory(destination);
                    foreach (var entry in zip.Entries) Extract(entry,destination);
                    string runtime=Path.Combine(destination,"runtime");Directory.CreateDirectory(runtime);
                    update(UiLanguage.Text("Installing the Python runtime. This can take several minutes…","正在安装 Python 运行环境，可能需要几分钟…"));
                    using (var packed=ZipFile.OpenRead(Path.Combine(destination,"runtime.zip")))
                        foreach (var entry in packed.Entries) Extract(entry,runtime);
                    File.Delete(Path.Combine(destination,"runtime.zip"));
                    update(UiLanguage.Text("Initializing the Python runtime…","正在初始化 Python 环境…"));
                    string unpack=Path.Combine(runtime,"Scripts","conda-unpack-script.py");
                    if (!File.Exists(unpack)) unpack=Path.Combine(runtime,"Scripts","conda-unpack");
                    if (!File.Exists(unpack)) throw new FileNotFoundException(UiLanguage.Text("The runtime relocation script is missing.","运行环境缺少路径修复脚本。"));
                    RunPython(runtime,Quote(unpack),destination);
                    update(UiLanguage.Text("Checking inference dependencies and models…","正在检查推理依赖和模型文件…"));
                    foreach (string file in release.RequiredFiles) if (!File.Exists(SafePath(destination,file))) throw new FileNotFoundException(UiLanguage.Text("The installer is missing a required file: ","安装包缺少文件：")+file);
                    RunPython(runtime,Quote(Path.Combine(destination,"backend","healthcheck.py")),destination);
                        foreach (int year in release.RevitVersions)
                        {
                            string folder=Path.Combine(Addins,year.ToString());Directory.CreateDirectory(folder);
                            string path=Path.Combine(folder,"Text2Revit.addin");backups[path]=File.Exists(path) ? File.ReadAllBytes(path) : null;
                            var xml=new XDocument(new XElement("RevitAddIns",new XElement("AddIn",new XAttribute("Type","Application"),
                                new XElement("Name","Text2Revit"),new XElement("Assembly",Path.Combine(destination,"addin",year.ToString(),"Text2Revit.Addin.dll")),
                                new XElement("AddInId","58D0137C-7A55-4A1B-9375-5B4D947A4A4E"),new XElement("FullClassName","Text2Revit.Addin.App"),
                                new XElement("VendorId","T2R"),new XElement("VendorDescription","Text2Revit floorplan generation"))));
                            xml.Save(path);
                        }
                    if (register)
                    {
                        // The small uninstaller has no appended payload.
                        CopyStub(Path.Combine(Root,"Uninstall.exe"));
                        using (var key=Registry.CurrentUser.CreateSubKey(RegistryPath))
                        {
                            key.SetValue("DisplayName","Text2Revit");key.SetValue("DisplayVersion",release.Version);
                            key.SetValue("Publisher","Text2Revit");key.SetValue("InstallLocation",Root);
                            key.SetValue("UninstallString",Quote(Path.Combine(Root,"Uninstall.exe"))+" --uninstall");
                            key.SetValue("NoModify",1);key.SetValue("NoRepair",1);
                        }
                    }
                    File.WriteAllText(Path.Combine(destination,"installed.txt"),DateTime.UtcNow.ToString("O"));
                    update(UiLanguage.Text("Installation complete.","安装完成。"));
                }
                catch
                {
                    foreach (var pair in backups) { if (pair.Value==null) File.Delete(pair.Key);else File.WriteAllBytes(pair.Key,pair.Value); }
                    if (Directory.Exists(destination)) DeleteOwned(destination,Path.Combine(Root,"releases"));
                    throw;
                }
            }
        }
        static void Uninstall(Action<string> update,bool register=true)
        {
            update(UiLanguage.Text("Removing add-in registrations…","正在移除插件注册…"));
            for (int year=2020;year<=2026;year++)
            {
                string file=Path.Combine(Addins,year.ToString(),"Text2Revit.addin");
                if (File.Exists(file) && XDocument.Load(file).Descendants("Assembly").Any(e=>Path.GetFullPath(e.Value).StartsWith(Path.GetFullPath(Root)+Path.DirectorySeparatorChar,StringComparison.OrdinalIgnoreCase))) File.Delete(file);
            }
            string releases=Path.Combine(Root,"releases");
            if (Directory.Exists(releases)) foreach (string folder in Directory.GetDirectories(releases)) DeleteOwned(folder,releases);
            if(register) Registry.CurrentUser.DeleteSubKeyTree(RegistryPath,false);
        }
        static void DeleteOwned(string directory,string allowedRoot)
        {
            string full=Path.GetFullPath(directory),root=Path.GetFullPath(allowedRoot)+Path.DirectorySeparatorChar;
            if (!full.StartsWith(root,StringComparison.OrdinalIgnoreCase) || (File.GetAttributes(full)&FileAttributes.ReparsePoint)!=0) throw new IOException(UiLanguage.Text("Refusing to remove files outside the product releases directory.","拒绝清理安装目录之外的文件。"));
            // Check every child before traversing; never follow user-created junctions.
            var pending=new Stack<string>();pending.Push(full);
            while(pending.Count>0)
                foreach(string child in Directory.EnumerateFileSystemEntries(pending.Pop()))
                {
                    var attributes=File.GetAttributes(child);
                    if((attributes&FileAttributes.ReparsePoint)!=0) throw new IOException(UiLanguage.Text("Cleanup stopped because the installation directory contains a link: ","安装目录包含链接，已停止清理：")+child);
                    if((attributes&FileAttributes.Directory)!=0) pending.Push(child);
                }
            Directory.Delete(full,true);
        }
        static string SafePath(string root,string relative)
        {
            if (string.IsNullOrEmpty(relative) || Path.IsPathRooted(relative) || relative.Contains(":")) throw new InvalidDataException(UiLanguage.Text("Invalid path in the installer payload.","安装包路径无效。"));
            string prefix=Path.GetFullPath(root)+Path.DirectorySeparatorChar;
            string full=Path.GetFullPath(Path.Combine(root,relative.Replace('/',Path.DirectorySeparatorChar)));
            if (!full.StartsWith(prefix,StringComparison.OrdinalIgnoreCase)) throw new InvalidDataException(UiLanguage.Text("A payload path is outside the installation directory.","安装包路径越界。"));return full;
        }
        static void Extract(ZipArchiveEntry entry,string root)
        {
            string file=SafePath(root,entry.FullName);
            if (entry.FullName.EndsWith("/")) { Directory.CreateDirectory(file);return; }
            Directory.CreateDirectory(Path.GetDirectoryName(file));
            using (var input=entry.Open()) using (var output=File.Create(file)) input.CopyTo(output);
        }
        static Release ReadRelease(ZipArchive zip)
        {
            var entry=zip.GetEntry("release.json") ?? throw new InvalidDataException(UiLanguage.Text("The installer has no release manifest.","安装包没有发行信息。"));
            Release release;using (var stream=entry.Open()) release=(Release)new DataContractJsonSerializer(typeof(Release)).ReadObject(stream);
            if (release==null || !Regex.IsMatch(release.Version ?? "","^[a-zA-Z0-9._-]+$") || release.RevitVersions==null || release.RequiredFiles==null || release.RevitVersions.Any(y=>y<2020||y>2026)) throw new InvalidDataException(UiLanguage.Text("Invalid release manifest.","发行信息无效。"));return release;
        }
        static void RunPython(string runtime,string arguments,string workingDirectory)
        {
            var info=new ProcessStartInfo(Path.Combine(runtime,"python.exe"),arguments) { WorkingDirectory=workingDirectory,UseShellExecute=false,CreateNoWindow=true,RedirectStandardOutput=true,RedirectStandardError=true };
            ProcessEnvironment.ConfigurePython(info,runtime);
            using (var process=Process.Start(info))
            {
                var output=process.StandardOutput.ReadToEndAsync();var error=process.StandardError.ReadToEndAsync();
                if (!process.WaitForExit(300000)) { process.Kill();process.WaitForExit();throw new TimeoutException(UiLanguage.Text("Backend initialization timed out.","后端初始化超时。")); }
                Task.WaitAll(output,error);
                File.AppendAllText(Path.Combine(workingDirectory,"setup.log"),output.Result+"\n"+error.Result);
                if (process.ExitCode!=0) throw new InvalidOperationException(UiLanguage.Text("Python initialization failed: ","Python 初始化失败：")+error.Result);
            }
        }
        static Stream Payload()
        {
            var file=File.OpenRead(Application.ExecutablePath);
            try
            {
                if (file.Length<56) throw new InvalidDataException(UiLanguage.Text("The installer is incomplete.","安装包不完整。"));
                file.Position=file.Length-56;var reader=new BinaryReader(file);
                if (Encoding.ASCII.GetString(reader.ReadBytes(8))!="T2RPKG01") throw new InvalidDataException(UiLanguage.Text("This executable has no product payload. Run Build-Release.cmd first.","安装程序没有打包产品文件，请运行 Build-Release.cmd。"));
                long offset=reader.ReadInt64(),length=reader.ReadInt64();byte[] expected=reader.ReadBytes(32);
                if (offset<0 || length<0 || offset+length!=file.Length-56) throw new InvalidDataException(UiLanguage.Text("The installer payload has an invalid length.","安装包长度不正确。"));
                var stream=new SliceStream(file,offset,length);
                using (var sha=SHA256.Create()) if (!sha.ComputeHash(stream).SequenceEqual(expected)) throw new InvalidDataException(UiLanguage.Text("Installer verification failed. Obtain a new copy.","安装包校验失败，请重新获取安装程序。"));
                stream.Position=0;return stream;
            }
            catch { file.Dispose();throw; }
        }
        static void CopyStub(string output)
        {
            using (var source=File.OpenRead(Application.ExecutablePath))
            {
                source.Position=source.Length-48;long length=new BinaryReader(source).ReadInt64();source.Position=0;
                using (var target=File.Create(output)) { var bytes=new byte[65536];while (length>0) { int n=source.Read(bytes,0,(int)Math.Min(bytes.Length,length));if(n==0)throw new EndOfStreamException();target.Write(bytes,0,n);length-=n; } }
            }
        }
        static string Quote(string s) { return "\""+s+"\""; }
        sealed class SliceStream : Stream
        {
            readonly Stream inner;readonly long start,length;
            public SliceStream(Stream inner,long start,long length) { this.inner=inner;this.start=start;this.length=length;Position=0; }
            public override bool CanRead=>true;public override bool CanSeek=>true;public override bool CanWrite=>false;
            public override long Length=>length;public override long Position { get=>inner.Position-start;set { if(value<0||value>length)throw new IOException();inner.Position=start+value; } }
            public override int Read(byte[] buffer,int offset,int count)=>inner.Read(buffer,offset,(int)Math.Min(count,length-Position));
            public override long Seek(long offset,SeekOrigin origin) { Position=(origin==SeekOrigin.Begin ? 0 : origin==SeekOrigin.Current ? Position : length)+offset;return Position; }
            public override void Flush() { }public override void SetLength(long value)=>throw new NotSupportedException();public override void Write(byte[] b,int o,int c)=>throw new NotSupportedException();
            protected override void Dispose(bool disposing) { if(disposing)inner.Dispose();base.Dispose(disposing); }
        }
    }
}
