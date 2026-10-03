using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
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
        [DataMember(Name="environment_format")] public string EnvironmentFormat { get; set; }
        [DataMember(Name="remote_environment")] public RemoteEnvironment RemoteEnvironment { get; set; }
    }
    [DataContract] public sealed class DownloadPart
    {
        [DataMember(Name="name")] public string Name { get; set; }
        [DataMember(Name="url")] public string Url { get; set; }
        [DataMember(Name="size")] public long Size { get; set; }
        [DataMember(Name="sha256")] public string Sha256 { get; set; }
    }
    [DataContract] public sealed class RemoteEnvironment
    {
        [DataMember(Name="parts")] public DownloadPart[] Parts { get; set; }
        [DataMember(Name="size")] public long Size { get; set; }
        [DataMember(Name="sha256")] public string Sha256 { get; set; }
    }
    [DataContract] public sealed class PackedFile
    {
        [DataMember(Name="path")] public string Path { get; set; }
        [DataMember(Name="size")] public long Size { get; set; }
        [DataMember(Name="sha256")] public string Sha256 { get; set; }
    }
    [DataContract] public sealed class EnvironmentInventory
    {
        [DataMember(Name="files")] public PackedFile[] Files { get; set; }
    }
    internal static class Program
    {
        const string RegistryPath=@"Software\Microsoft\Windows\CurrentVersion\Uninstall\Text2Revit";
        static string Root;
        static string Addins;
        static bool Testing;
        static Program()
        {
            AppContext.SetSwitch("Switch.System.IO.UseLegacyPathHandling",false);
            AppContext.SetSwitch("Switch.System.IO.BlockLongPaths",false);
            Root=Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),"Text2Revit");
            Addins=Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),"Autodesk","Revit","Addins");
        }
        [STAThread] static void Main(string[] args)
        {
            Application.EnableVisualStyles(); Application.SetCompatibleTextRenderingDefault(false);
            try
            {
                if (args.Length==2 && (args[0]=="--test-root" || args[0]=="--test-uninstall")) { Testing=true; Root=Path.GetFullPath(args[1]); Addins=Path.Combine(Root,"test-addins"); if(args[0]=="--test-uninstall") Uninstall(s=>Console.WriteLine(s),false); else Install(s=>Console.WriteLine(s),false); return; }
                if (args.Contains("--verify-only")) { using (var stream=Payload()) using (var zip=new ZipArchive(stream,ZipArchiveMode.Read)) { var release=ReadRelease(zip); foreach (var e in zip.Entries) SafePath(Root,e.FullName); if(release.EnvironmentFormat=="7z") ReadInventory(zip,Root); } return; }
#if STANDALONE_UNINSTALLER
                bool uninstall=true;
#else
                bool uninstall=args.Contains("--uninstall") || string.Equals(Path.GetFileNameWithoutExtension(Application.ExecutablePath),"Uninstall",StringComparison.OrdinalIgnoreCase);
#endif
                using (var form=new Form { Text=uninstall ? UiLanguage.Text("Uninstall Text2Revit","卸载 Text2Revit") : UiLanguage.Text("Install Text2Revit","安装 Text2Revit"),ClientSize=new Size(600,300),StartPosition=FormStartPosition.CenterScreen,
                    FormBorderStyle=FormBorderStyle.FixedDialog,MaximizeBox=false,Font=new Font("Segoe UI",10) })
                {
                    var heading=new Label { Left=24,Top=24,Width=405,Height=38,Text=uninstall ? UiLanguage.Text("Uninstall Text2Revit","卸载 Text2Revit") : UiLanguage.Text("Text2Revit · Setup","Text2Revit · 自动安装"),Font=new Font(form.Font.FontFamily,17,FontStyle.Bold) };
                    var language=new ComboBox { Left=446,Top=31,Width=130,DropDownStyle=ComboBoxStyle.DropDownList };
                    language.Items.AddRange(new object[]{"English","中文"});language.SelectedIndex=UiLanguage.IsChinese?1:0;
                    bool online=false;
                    if(!uninstall) using(var stream=Payload()) using(var zip=new ZipArchive(stream,ZipArchiveMode.Read)) online=ReadRelease(zip).RemoteEnvironment!=null;
                    Func<string> introduction=()=>uninstall ? UiLanguage.Text("Removes the add-in and runtime. Generation records are kept.\nClose Revit before continuing.","将移除工具栏插件和运行环境，保留生成记录。请先关闭 Revit。") : online ? UiLanguage.Text("Downloads and installs the Revit add-in, runtime and models.\nInternet is required for installation; inference then works offline.\nClose Revit and allow at least 12 GB of free disk space.","自动下载并安装 Revit 插件、运行环境和模型。\n安装需要联网，之后可以离线推理。\n请先关闭 Revit，并预留至少 12 GB 磁盘空间。") : UiLanguage.Text("Installs the Revit add-in, Python runtime and models automatically.\nSupports Revit 2020–2026. No Python configuration or file copying.\nClose Revit and allow at least 12 GB of free disk space.","自动安装 Revit 插件、Python 后端和模型。\n支持 Revit 2020–2026，无需配置 Python 或移动文件。\n请先关闭 Revit，并预留至少 12 GB 磁盘空间。");
                    var label=new Label { Left=24,Top=78,Width=552,Height=110,Text=introduction() };
                    var progress=new ProgressBar { Left=24,Top=195,Width=552,Height=20,Visible=false,Style=ProgressBarStyle.Marquee };
                    var button=new Button { Left=440,Top=242,Width=136,Height=36,Text=uninstall ? UiLanguage.Text("Uninstall","卸载") : UiLanguage.Text("Install","安装") };
                    var cancel=new Button { Left=292,Top=242,Width=136,Height=36,Text=UiLanguage.Text("Cancel","取消") };
                    form.Controls.AddRange(new Control[] { heading,language,label,progress,button,cancel }); bool busy=false,finished=false;
                    CancellationTokenSource cancellation=null;
                    Action requestCancel=()=>{ if(cancellation!=null && !cancellation.IsCancellationRequested) { cancellation.Cancel();cancel.Enabled=false;label.Text=UiLanguage.Text("Cancelling and cleaning up…","正在取消并清理未完成的安装…"); } };
                    cancel.Click+=(s,e)=>{ if(busy) requestCancel();else form.Close(); };
                    language.SelectedIndexChanged+=(s,e)=>{UiLanguage.Set(language.SelectedIndex==1);cancel.Text=UiLanguage.Text("Cancel","取消");form.Text=uninstall ? UiLanguage.Text("Uninstall Text2Revit","卸载 Text2Revit") : UiLanguage.Text("Install Text2Revit","安装 Text2Revit");heading.Text=uninstall ? UiLanguage.Text("Uninstall Text2Revit","卸载 Text2Revit") : UiLanguage.Text("Text2Revit · Setup","Text2Revit · 自动安装");label.Text=introduction();button.Text=uninstall ? UiLanguage.Text("Uninstall","卸载") : UiLanguage.Text("Install","安装");};
                    form.FormClosing+=(s,e)=>{ if (busy) { e.Cancel=true;if(!uninstall) requestCancel(); } };
                    button.Click+=async (s,e)=>{
                        if (finished) { form.Close();return; }
                        if (Process.GetProcessesByName("Revit").Length>0) { MessageBox.Show(form,UiLanguage.Text("Close all Revit windows before continuing.","请先关闭所有 Revit 窗口，再继续。"),UiLanguage.Text("Close Revit","提示"));return; }
                        busy=true;cancellation=new CancellationTokenSource();language.Enabled=false;button.Enabled=false;cancel.Enabled=!uninstall;progress.Visible=true;
                        Action<string> update=text=>form.BeginInvoke(new Action(()=>{ if(!cancellation.IsCancellationRequested && busy) label.Text=text; }));
                        try
                        {
                            await Task.Run(()=>{ if (uninstall) Uninstall(update); else Install(update,true,cancellation.Token); });
                            label.Text=uninstall ? UiLanguage.Text("Uninstall complete. Generation records have been kept.","卸载完成。生成记录已保留。") : UiLanguage.Text("Installation complete.\nStart Revit and open a floor plan view.\nClick Generate Model on the Text2Revit tab and enter your prompt.","安装完成。\n启动 Revit，打开楼层平面视图。\n在 Text2Revit 选项卡中点击“生成模型”，输入 prompt 后即可生成三维模型。");
                            button.Text=UiLanguage.Text("Finish","完成");finished=true;
                        }
                        catch (OperationCanceledException) { label.Text=UiLanguage.Text("Installation cancelled. Incomplete installation files were removed. Downloaded data is kept for retry.","安装已取消。未完成的安装文件已清理，下载缓存保留供重试。");button.Text=UiLanguage.Text("Close","关闭");finished=true; }
                        catch (Exception error) { label.Text=UiLanguage.Text("Setup did not finish:\n","安装未完成：\n")+error.Message;button.Text=UiLanguage.Text("Close","关闭");finished=true; }
                        finally { busy=false;progress.Visible=false;button.Enabled=true;cancel.Visible=false;cancellation.Dispose(); }
                    };
                    Application.Run(form);
                }
            }
            catch (Exception error) { File.WriteAllText(Path.Combine(Path.GetTempPath(),"Text2Revit-setup-error.txt"),error.ToString());Console.Error.WriteLine(error.Message);Environment.ExitCode=1; if (!args.Any(a=>a.StartsWith("--"))) MessageBox.Show(error.Message,UiLanguage.Text("Text2Revit Setup Failed","Text2Revit 安装失败")); }
        }
        static void Install(Action<string> update,bool register,CancellationToken token=default(CancellationToken))
        {
            update(UiLanguage.Text("Verifying the installer…","正在校验安装包…"));
            using (var stream=Payload(token)) using (var zip=new ZipArchive(stream,ZipArchiveMode.Read))
            {
                token.ThrowIfCancellationRequested();
                var release=ReadRelease(zip);
                if (new DriveInfo(Path.GetPathRoot(Root)).AvailableFreeSpace < release.RequiredFreeBytes) throw new IOException(UiLanguage.Text("Not enough disk space. Allow at least ","磁盘空间不足，请预留至少 ")+Math.Ceiling(release.RequiredFreeBytes/1024.0/1024/1024)+" GB.");
                string destination=SafePath(Path.Combine(Root,"releases"),release.Version);
                if (Directory.Exists(destination)) throw new IOException(UiLanguage.Text("This release is already installed. Uninstall it before reinstalling.","此版本已经安装。请先卸载后重新安装。"));
                var backups=new Dictionary<string,byte[]>();
                try
                {
                    Directory.CreateDirectory(destination);
                    foreach (var entry in zip.Entries) Extract(entry,destination,token);
                    if(release.RemoteEnvironment!=null) DownloadEnvironment(release,destination,update,token);
                    string runtime=Path.Combine(destination,"runtime");Directory.CreateDirectory(runtime);
                    update(UiLanguage.Text("Installing the Python runtime. This can take several minutes…","正在安装 Python 运行环境，可能需要几分钟…"));
                    if (release.EnvironmentFormat=="7z")
                        ExtractEnvironment(destination,ReadInventory(zip,destination),update,token);
                    else
                    {
                        using (var packed=ZipFile.OpenRead(Path.Combine(destination,"runtime.zip")))
                            foreach (var entry in packed.Entries) Extract(entry,runtime,token);
                        File.Delete(Path.Combine(destination,"runtime.zip"));
                    }
                    update(UiLanguage.Text("Initializing the Python runtime…","正在初始化 Python 环境…"));
                    string unpack=Path.Combine(runtime,"Scripts","conda-unpack-script.py");
                    if (!File.Exists(unpack)) unpack=Path.Combine(runtime,"Scripts","conda-unpack");
                    if (!File.Exists(unpack)) throw new FileNotFoundException(UiLanguage.Text("The runtime relocation script is missing.","运行环境缺少路径修复脚本。"));
                    RunPython(runtime,Quote(unpack),destination,token);
                    update(UiLanguage.Text("Checking inference dependencies and models…","正在检查推理依赖和模型文件…"));
                    foreach (string file in release.RequiredFiles) if (!File.Exists(SafePath(destination,file))) throw new FileNotFoundException(UiLanguage.Text("The installer is missing a required file: ","安装包缺少文件：")+file);
                    RunPython(runtime,Quote(Path.Combine(destination,"backend","healthcheck.py")),destination,token);
                    // Commit registrations only after every cancellable stage has completed.
                    token.ThrowIfCancellationRequested();
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
                    if(release.RemoteEnvironment!=null)
                    {
                        string downloads=Path.Combine(Root,"downloads"),cache=SafePath(downloads,release.Version);
                        try { if(Directory.Exists(cache)) DeleteOwned(cache,downloads); }
                        catch(IOException) { }
                        catch(UnauthorizedAccessException) { }
                    }
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
            if(register && !Testing) Registry.CurrentUser.DeleteSubKeyTree(RegistryPath,false);
            string downloads=Path.Combine(Root,"downloads");
            if(Directory.Exists(downloads)) foreach(string folder in Directory.GetDirectories(downloads)) DeleteOwned(folder,downloads);
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
        static void Extract(ZipArchiveEntry entry,string root,CancellationToken token=default(CancellationToken))
        {
            token.ThrowIfCancellationRequested();
            string file=SafePath(root,entry.FullName);
            if (entry.FullName.EndsWith("/")) { Directory.CreateDirectory(file);return; }
            Directory.CreateDirectory(Path.GetDirectoryName(file));
            using (var input=entry.Open()) using (var output=File.Create(file)) Copy(input,output,token);
        }
        static Release ReadRelease(ZipArchive zip)
        {
            var entry=zip.GetEntry("release.json") ?? throw new InvalidDataException(UiLanguage.Text("The installer has no release manifest.","安装包没有发行信息。"));
            Release release;using (var stream=entry.Open()) release=(Release)new DataContractJsonSerializer(typeof(Release)).ReadObject(stream);
            if (release==null || !Regex.IsMatch(release.Version ?? "","^[a-zA-Z0-9._-]+$") || release.RevitVersions==null || release.RequiredFiles==null || release.RevitVersions.Any(y=>y<2020||y>2026) || (release.EnvironmentFormat!=null && release.EnvironmentFormat!="zip" && release.EnvironmentFormat!="7z")) throw new InvalidDataException(UiLanguage.Text("Invalid release manifest.","发行信息无效。"));
            if(release.RemoteEnvironment!=null)
            {
                var remote=release.RemoteEnvironment;
                if(release.EnvironmentFormat!="7z" || remote.Parts==null || remote.Parts.Length==0 || remote.Parts.Length>1000 || remote.Size<=0 || !Regex.IsMatch(remote.Sha256 ?? "","^[a-f0-9]{64}$")) throw new InvalidDataException("Invalid download manifest.");
                long total=0;var names=new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                foreach(var part in remote.Parts)
                {
                    Uri uri;
                    if(part==null || !Regex.IsMatch(part.Name ?? "","^[a-zA-Z0-9][a-zA-Z0-9._-]*$") || !names.Add(part.Name) || part.Size<=0 || part.Size>=2147483648L || !Regex.IsMatch(part.Sha256 ?? "","^[a-f0-9]{64}$") || !Uri.TryCreate(part.Url,UriKind.Absolute,out uri) || !(uri.Scheme==Uri.UriSchemeHttps || (Testing && uri.Scheme==Uri.UriSchemeHttp && uri.IsLoopback))) throw new InvalidDataException("Invalid download part.");
                    total=checked(total+part.Size);
                }
                if(total!=remote.Size) throw new InvalidDataException("Download sizes do not match.");
            }
            return release;
        }
        static EnvironmentInventory ReadInventory(ZipArchive zip,string destination)
        {
            var entry=zip.GetEntry("environment-files.json");
            if(entry==null || (zip.GetEntry("environment.7z")==null && ReadRelease(zip).RemoteEnvironment==null) || zip.GetEntry("tools/7zr.exe")==null) throw new InvalidDataException("Missing compressed environment or extractor.");
            EnvironmentInventory inventory;
            using(var stream=entry.Open()) inventory=(EnvironmentInventory)new DataContractJsonSerializer(typeof(EnvironmentInventory)).ReadObject(stream);
            if(inventory==null || inventory.Files==null || inventory.Files.Length==0) throw new InvalidDataException("Invalid environment inventory.");
            var seen=new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach(var file in inventory.Files)
            {
                if(file==null || file.Path==null || !(file.Path.StartsWith("runtime/",StringComparison.Ordinal) || file.Path.StartsWith("models/",StringComparison.Ordinal)) || file.Size<0 || !Regex.IsMatch(file.Sha256 ?? "","^[a-f0-9]{64}$") || !seen.Add(SafePath(destination,file.Path))) throw new InvalidDataException("Invalid environment file record.");
            }
            return inventory;
        }
        static string FileHash(string path,CancellationToken token=default(CancellationToken))
        {
            using(var input=File.OpenRead(path)) using(var sha=SHA256.Create()) return BitConverter.ToString(Hash(input,sha,token)).Replace("-","").ToLowerInvariant();
        }
        static void DownloadEnvironment(Release release,string destination,Action<string> update,CancellationToken token)
        {
            ServicePointManager.SecurityProtocol|=SecurityProtocolType.Tls12;
            string cache=SafePath(Path.Combine(Root,"downloads"),release.Version);Directory.CreateDirectory(cache);
            long completed=0;
            foreach(var part in release.RemoteEnvironment.Parts)
            {
                string target=SafePath(cache,part.Name);
                if(!File.Exists(target) || new FileInfo(target).Length!=part.Size || FileHash(target,token)!=part.Sha256)
                    DownloadPartFile(part,target,completed,release.RemoteEnvironment.Size,update,token);
                completed+=part.Size;
            }
            update(UiLanguage.Text("Verifying downloaded runtime and models…","正在校验已下载的运行环境和模型…"));
            string archive=Path.Combine(destination,"environment.7z"),temporary=archive+".partial";
            using(var output=File.Create(temporary))
                foreach(var part in release.RemoteEnvironment.Parts)
                    using(var input=File.OpenRead(SafePath(cache,part.Name))) Copy(input,output,token);
            if(new FileInfo(temporary).Length!=release.RemoteEnvironment.Size || FileHash(temporary,token)!=release.RemoteEnvironment.Sha256) throw new InvalidDataException(UiLanguage.Text("Downloaded archive checksum differs. Run the installer again.","下载数据校验失败，请重新运行安装器。"));
            File.Move(temporary,archive);
        }
        static void DownloadPartFile(DownloadPart part,string target,long completed,long total,Action<string> update,CancellationToken token)
        {
            string partial=target+".partial";
            for(int attempt=0;attempt<2;attempt++)
            {
                token.ThrowIfCancellationRequested();
                long offset=File.Exists(partial)?new FileInfo(partial).Length:0;
                if(offset>part.Size) { File.Delete(partial);offset=0; }
                if(offset<part.Size)
                {
                    var request=(HttpWebRequest)WebRequest.Create(part.Url);
                    request.Timeout=60000;request.ReadWriteTimeout=120000;request.UserAgent="Text2Revit-Setup/1.0";
                    if(Testing && request.RequestUri.IsLoopback) request.Proxy=null;
                    if(offset>0) request.AddRange(offset);
                    using(token.Register(request.Abort))
                    try
                    {
                        token.ThrowIfCancellationRequested();
                        using(var response=(HttpWebResponse)request.GetResponse())
                        {
                            if(response.StatusCode==HttpStatusCode.OK) offset=0;
                            else if(response.StatusCode!=HttpStatusCode.PartialContent || !Regex.IsMatch(response.Headers["Content-Range"] ?? "","^bytes "+offset+"-[0-9]+/"+part.Size+"$")) throw new InvalidDataException("Invalid partial download response.");
                            using(var input=response.GetResponseStream()) using(var output=new FileStream(partial,offset==0?FileMode.Create:FileMode.Append,FileAccess.Write))
                            {
                                byte[] buffer=new byte[65536];long received=offset;var timer=Stopwatch.StartNew();
                                int count;
                                while((count=input.Read(buffer,0,buffer.Length))>0)
                                {
                                    token.ThrowIfCancellationRequested();
                                    if(received+count>part.Size) throw new InvalidDataException("Download exceeds the declared size.");
                                    output.Write(buffer,0,count);received+=count;
                                    if(timer.ElapsedMilliseconds>=500 || received==part.Size)
                                    {
                                        update(UiLanguage.Text("Downloading runtime and models: ","正在下载运行环境和模型：")+((completed+received)*100.0/total).ToString("F1")+"%\n"+part.Name);timer.Restart();
                                    }
                                }
                            }
                        }
                    }
                    catch(Exception) when(token.IsCancellationRequested) { throw new OperationCanceledException(token); }
                    catch(WebException error) { throw new IOException(UiLanguage.Text("Download failed for ","下载失败：")+part.Name+UiLanguage.Text(". Check your connection and run the installer again; downloaded data is kept.","。请检查网络后重新运行安装器，已下载的数据会保留。"),error); }
                }
                if(new FileInfo(partial).Length==part.Size && FileHash(partial,token)==part.Sha256)
                {
                    if(File.Exists(target)) File.Delete(target);
                    File.Move(partial,target);return;
                }
                File.Delete(partial);
            }
            throw new InvalidDataException(UiLanguage.Text("Downloaded file checksum differs: ","下载文件校验失败：")+part.Name);
        }
        static void ExtractEnvironment(string destination,EnvironmentInventory inventory,Action<string> update,CancellationToken token)
        {
            string tool=Path.Combine(destination,"tools","7zr.exe"),archive=Path.Combine(destination,"environment.7z");
            var expected=new HashSet<string>(inventory.Files.Select(f=>SafePath(destination,f.Path)),StringComparer.OrdinalIgnoreCase);
            string listing=RunExtractor(tool,"l -slt -ba -sccUTF-8 "+Quote(archive),destination,token);
            var listed=new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach(string line in listing.Split('\n'))
            {
                if(line.StartsWith("Symbolic Link = ",StringComparison.Ordinal) || line.StartsWith("Hard Link = ",StringComparison.Ordinal)) throw new InvalidDataException("Links are not allowed in the packed environment.");
                if(!line.StartsWith("Path = ",StringComparison.Ordinal)) continue;
                string full=SafePath(destination,line.Substring(7).TrimEnd('\r'));
                if(!expected.Contains(full) || !listed.Add(full)) throw new InvalidDataException("Unexpected file in the packed environment.");
            }
            if(!listed.SetEquals(expected)) throw new InvalidDataException("The packed environment does not match its inventory.");
            RunExtractor(tool,"x "+Quote(archive)+" -o"+Quote(destination)+" -y -bsp0 -bso0",destination,token);
            update(UiLanguage.Text("Verifying extracted runtime and model files…","正在校验解压后的运行环境和模型…"));
            foreach(var file in inventory.Files)
            {
                string path=SafePath(destination,file.Path);
                if(!File.Exists(path) || new FileInfo(path).Length!=file.Size || (File.GetAttributes(path)&FileAttributes.ReparsePoint)!=0) throw new InvalidDataException("Invalid extracted file: "+file.Path);
                using(var input=File.OpenRead(path)) using(var sha=SHA256.Create())
                    if(!string.Equals(BitConverter.ToString(Hash(input,sha,token)).Replace("-","").ToLowerInvariant(),file.Sha256,StringComparison.Ordinal)) throw new InvalidDataException("Extracted file checksum differs: "+file.Path);
            }
            File.Delete(archive);
        }
        static string RunExtractor(string tool,string arguments,string workingDirectory,CancellationToken token=default(CancellationToken))
        {
            var info=new ProcessStartInfo(tool,arguments) { WorkingDirectory=workingDirectory,UseShellExecute=false,CreateNoWindow=true,RedirectStandardOutput=true,RedirectStandardError=true,StandardOutputEncoding=Encoding.UTF8,StandardErrorEncoding=Encoding.UTF8 };
            using(var process=Process.Start(info))
            {
                var output=process.StandardOutput.ReadToEndAsync();var error=process.StandardError.ReadToEndAsync();
                WaitForProcess(process,1800000,token,UiLanguage.Text("Runtime extraction timed out.","运行环境解压超时。"));
                Task.WaitAll(output,error);
                if(process.ExitCode!=0) throw new InvalidDataException(UiLanguage.Text("Runtime extraction failed: ","运行环境解压失败：")+error.Result);
                return output.Result;
            }
        }
        static void RunPython(string runtime,string arguments,string workingDirectory,CancellationToken token)
        {
            var info=new ProcessStartInfo(Path.Combine(runtime,"python.exe"),arguments) { WorkingDirectory=workingDirectory,UseShellExecute=false,CreateNoWindow=true,RedirectStandardOutput=true,RedirectStandardError=true };
            ProcessEnvironment.ConfigurePython(info,runtime);
            using (var process=Process.Start(info))
            {
                var output=process.StandardOutput.ReadToEndAsync();var error=process.StandardError.ReadToEndAsync();
                WaitForProcess(process,300000,token,UiLanguage.Text("Backend initialization timed out.","后端初始化超时。"));
                Task.WaitAll(output,error);
                File.AppendAllText(Path.Combine(workingDirectory,"setup.log"),output.Result+"\n"+error.Result);
                if (process.ExitCode!=0) throw new InvalidOperationException(UiLanguage.Text("Python initialization failed: ","Python 初始化失败：")+error.Result);
            }
        }
        static Stream Payload(CancellationToken token=default(CancellationToken))
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
                using (var sha=SHA256.Create()) if (!Hash(stream,sha,token).SequenceEqual(expected)) throw new InvalidDataException(UiLanguage.Text("Installer verification failed. Obtain a new copy.","安装包校验失败，请重新获取安装程序。"));
                stream.Position=0;return stream;
            }
            catch { file.Dispose();throw; }
        }
        static void Copy(Stream input,Stream output,CancellationToken token)
        {
            byte[] buffer=new byte[65536];int count;
            while(true) { token.ThrowIfCancellationRequested();count=input.Read(buffer,0,buffer.Length);if(count==0)break;output.Write(buffer,0,count); }
        }
        static byte[] Hash(Stream input,HashAlgorithm hash,CancellationToken token)
        {
            byte[] buffer=new byte[65536];int count;
            while(true) { token.ThrowIfCancellationRequested();count=input.Read(buffer,0,buffer.Length);if(count==0)break;hash.TransformBlock(buffer,0,count,buffer,0); }
            hash.TransformFinalBlock(new byte[0],0,0);return hash.Hash;
        }
        static void WaitForProcess(Process process,int timeout,CancellationToken token,string timeoutMessage)
        {
            var timer=Stopwatch.StartNew();
            try
            {
                while(!process.WaitForExit(100)) { token.ThrowIfCancellationRequested();if(timer.ElapsedMilliseconds>=timeout)throw new TimeoutException(timeoutMessage); }
                token.ThrowIfCancellationRequested();
            }
            catch
            {
                if(!process.HasExited) { try { process.Kill(); } catch(InvalidOperationException) { } }
                process.WaitForExit();throw;
            }
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
