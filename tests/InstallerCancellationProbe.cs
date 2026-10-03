using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Windows.Forms;

// Runs the real setup form against an appended, isolated loopback fixture.
internal static class InstallerCancellationProbe
{
    sealed class CancellingStream : MemoryStream
    {
        readonly CancellationTokenSource cancellation;
        public CancellingStream(CancellationTokenSource source) : base(new byte[1024*1024]) { cancellation=source; }
        public override int Read(byte[] buffer,int offset,int count) { int read=base.Read(buffer,offset,count);cancellation.Cancel();return read; }
    }
    [STAThread] static int Main(string[] args)
    {
        Console.OutputEncoding=System.Text.Encoding.UTF8;
        if(args[0]=="--worker") { File.WriteAllText(args[1],Process.GetCurrentProcess().Id.ToString());Thread.Sleep(30000);return 0; }
        var assembly=Assembly.LoadFrom(args[0]);
        var program=assembly.GetType("Text2Revit.Installer.Program",true);
        var flags=BindingFlags.Static|BindingFlags.NonPublic;
        program.GetField("Testing",flags).SetValue(null,true);
        program.GetField("Root",flags).SetValue(null,args[1]);
        program.GetField("Addins",flags).SetValue(null,Path.Combine(args[1],"test-addins"));
        if(args.Length>2 && (args[2]=="copy" || args[2]=="hash"))
        {
            using(var cancellation=new CancellationTokenSource())
            using(var input=new CancellingStream(cancellation))
            using(var output=new MemoryStream())
            using(var hash=System.Security.Cryptography.SHA256.Create())
            {
                bool cancelled=false;
                try {
                    object sink=args[2]=="copy" ? (object)output : hash;
                    program.GetMethod(args[2]=="copy" ? "Copy" : "Hash",flags).Invoke(null,new object[]{input,sink,cancellation.Token});
                }
                catch(TargetInvocationException error) { if(!(error.InnerException is OperationCanceledException)) throw;cancelled=true; }
                if(!cancelled || input.Position>65536) return 1;
                Console.WriteLine("PASS: cancellation interrupts "+args[2]+" between chunks");return 0;
            }
        }
        if(args.Length>2 && (args[2]=="extractor" || args[2]=="python"))
        {
            Directory.CreateDirectory(args[1]);
            string pidFile=Path.Combine(args[1],"child.pid");
            string executable=Application.ExecutablePath;
            if(args[2]=="python") { executable=Path.Combine(args[1],"python.exe");File.Copy(Application.ExecutablePath,executable); }
            using(var cancellation=new CancellationTokenSource())
            {
                var worker=new Thread(()=>{ while(!File.Exists(pidFile)) Thread.Sleep(20);cancellation.Cancel(); });
                worker.IsBackground=true;worker.Start();
                var watch=Stopwatch.StartNew();bool cancelled=false;
                try {
                    if(args[2]=="extractor") program.GetMethod("RunExtractor",flags).Invoke(null,new object[]{executable,"--worker \""+pidFile+"\"",args[1],cancellation.Token});
                    else program.GetMethod("RunPython",flags).Invoke(null,new object[]{args[1],"--worker \""+pidFile+"\"",args[1],cancellation.Token});
                }
                catch(TargetInvocationException error) { if(!(error.InnerException is OperationCanceledException)) throw;cancelled=true; }
                if(!cancelled || watch.Elapsed.TotalSeconds>5) return 1;
                int pid=int.Parse(File.ReadAllText(pidFile));
                try { using(var child=Process.GetProcessById(pid)) if(!child.HasExited) return 1; } catch(ArgumentException) { }
                Console.WriteLine("PASS: cancellation stops "+args[2]+" child process");return 0;
            }
        }
        int phase=0; bool passed=false;
        var clock=Stopwatch.StartNew();
        var timer=new System.Windows.Forms.Timer { Interval=100 };
        timer.Tick+=(s,e)=> {
            if(clock.Elapsed.TotalSeconds>10) { Console.Error.WriteLine("FAIL: cancellation did not complete within 10 seconds");Environment.Exit(1); }
            var form=Application.OpenForms.Cast<Form>().FirstOrDefault();
            if(form==null) return;
            var buttons=form.Controls.OfType<Button>().ToArray();
            if(phase==0) { phase=1;buttons.First(b=>b.Text=="Install" || b.Text=="安装").PerformClick(); }
            else if(phase==1 && File.Exists(Path.Combine(args[1],"downloads","cancel-probe","environment.7z.001.partial"))) {
                if(args.Length>2 && args[2]=="close") form.Close();
                else {
                    var cancel=buttons.FirstOrDefault(b=>b.Enabled && (b.Text=="Cancel" || b.Text=="取消"));
                    if(cancel==null) { Console.Error.WriteLine("FAIL: busy installer has no enabled Cancel button");Environment.Exit(1); }
                    cancel.PerformClick();
                }
                phase=2;
            }
            else if(phase==2 && form.Controls.OfType<Label>().Any(l=>l.Text.Contains("cancelled") || l.Text.Contains("已取消"))) {
                passed=true;timer.Stop();form.Close();
            }
        };
        EventHandler idle=null;
        idle=(s,e)=>{Application.Idle-=idle;timer.Start();};
        Application.Idle+=idle;
        try { program.GetMethod("Main",flags).Invoke(null,new object[]{new string[0]}); }
        catch(Exception error) { while(error.InnerException!=null) error=error.InnerException;Console.Error.WriteLine(error.GetType().Name+": "+error.Message);return 1; }
        timer.Dispose();
        if(!passed || Directory.Exists(Path.Combine(args[1],"releases","cancel-probe"))) return 1;
        Console.WriteLine("PASS: cancellation finishes and rolls back the isolated release");
        return 0;
    }
}
