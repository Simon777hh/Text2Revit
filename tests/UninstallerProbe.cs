using System;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Windows.Forms;

internal static class UninstallerProbe
{
    [STAThread] static int Main(string[] args)
    {
        var program=Assembly.LoadFrom(args[0]).GetType("Text2Revit.Installer.Program",true);
        var flags=BindingFlags.Static|BindingFlags.NonPublic;
        program.GetField("Testing",flags).SetValue(null,true);
        program.GetField("Root",flags).SetValue(null,args[1]);
        program.GetField("Addins",flags).SetValue(null,System.IO.Path.Combine(args[1],"test-addins"));
        var clock=Stopwatch.StartNew();bool clicked=false,passed=false;
        var timer=new Timer { Interval=100 };
        timer.Tick+=(s,e)=>{
            if(clock.Elapsed.TotalSeconds>10) { Console.Error.WriteLine("FAIL: uninstall form timed out");Environment.Exit(1); }
            var form=Application.OpenForms.Cast<Form>().FirstOrDefault();if(form==null)return;
            if(!clicked)
            {
                var button=form.Controls.OfType<Button>().FirstOrDefault(b=>b.Text=="Uninstall" || b.Text=="卸载");
                if(button==null) { Console.Error.WriteLine("FAIL: standalone tool did not open uninstall mode");Environment.Exit(1); }
                clicked=true;button.PerformClick();
            }
            else if(form.Controls.OfType<Label>().Any(l=>l.Text.Contains("Uninstall complete") || l.Text.Contains("卸载完成")))
            { passed=true;timer.Stop();form.Close(); }
        };
        EventHandler idle=null;idle=(s,e)=>{Application.Idle-=idle;timer.Start();};Application.Idle+=idle;
        program.GetMethod("Main",flags).Invoke(null,new object[]{new string[0]});
        timer.Dispose();return passed?0:1;
    }
}
