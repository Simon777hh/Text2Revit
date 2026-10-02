using System;
using System.Drawing;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using Text2Revit.Addin.Models;
using Text2Revit.Addin.Services;
namespace Text2Revit.Addin.UI
{
    internal sealed class RevitWindow : IWin32Window
    {
        public IntPtr Handle { get; private set; }
        public RevitWindow(IntPtr handle) { Handle = handle; }
    }
    internal static class PromptDialog
    {
        public static PlanData Generate(IntPtr owner)
        {
            using (var form = new Form { Text=UiLanguage.Text("Text2Revit · Generate apartment model","Text2Revit · 生成户型模型"), ClientSize=new Size(720,430),
                StartPosition=FormStartPosition.CenterParent, FormBorderStyle=FormBorderStyle.FixedDialog,
                MaximizeBox=false, MinimizeBox=false, Font=new Font("Segoe UI",10), AutoScaleMode=AutoScaleMode.Dpi })
            using (var cancellation = new CancellationTokenSource())
            using (var timer = new System.Windows.Forms.Timer { Interval=400 })
            {
                var title = new Label { Text=UiLanguage.Text("Describe your apartment","描述你想生成的户型"), Left=24,Top=22,Width=510,Height=30,Font=new Font(form.Font.FontFamily,15,FontStyle.Bold) };
                var language=new ComboBox { Left=566,Top=24,Width=130,DropDownStyle=ComboBoxStyle.DropDownList };
                language.Items.AddRange(new object[]{"English","中文"});language.SelectedIndex=UiLanguage.IsChinese?1:0;
                var input = new TextBox { Left=24,Top=66,Width=672,Height=56,Multiline=true,Text="A 3-bedroom apartment with 2 bathrooms and 1 balcony." };
                var hint = new Label { Left=24,Top=135,Width=672,Height=52,
                    Text=Hint() };
                var examples = new ListBox { Left=24,Top=196,Width=672,Height=85 };
                examples.Items.AddRange(new object[] { "A small 2-bedroom apartment with 1 bathroom and 1 balcony.",
                    "A 3-bedroom apartment with 2 bathrooms and 1 balcony.",
                    "A large 4-bedroom apartment with 2 bathrooms and 2 balconies." });
                examples.SelectedIndexChanged += (s,e) => { if (examples.SelectedItem != null && input.Enabled) input.Text=examples.SelectedItem.ToString(); };
                var status = new Label { Left=24,Top=298,Width=672,Height=40, Text=Ready() };
                var progress = new ProgressBar { Left=24,Top=345,Width=440,Height=22,Visible=false,Style=ProgressBarStyle.Marquee };
                var generate = new Button { Left=492,Top=365,Width=98,Height=36,Text=UiLanguage.Text("Generate","生成") };
                var cancel = new Button { Left=598,Top=365,Width=98,Height=36,Text=UiLanguage.Text("Close","关闭") };
                form.Controls.AddRange(new Control[] { title,language,input,hint,examples,status,progress,generate,cancel });
                PlanData plan=null; bool running=false; PythonBackendRunner runner=null;
                language.SelectedIndexChanged+=(s,e)=>{UiLanguage.Set(language.SelectedIndex==1);form.Text=UiLanguage.Text("Text2Revit · Generate apartment model","Text2Revit · 生成户型模型");title.Text=UiLanguage.Text("Describe your apartment","描述你想生成的户型");hint.Text=Hint();status.Text=Ready();generate.Text=UiLanguage.Text("Generate","生成");cancel.Text=UiLanguage.Text("Close","关闭");};
                cancel.Click += (s,e) => { if (running) { cancellation.Cancel(); cancel.Enabled=false; status.Text=UiLanguage.Text("Stopping…","正在停止…"); } else form.Close(); };
                form.FormClosing += (s,e) => { if (running) { e.Cancel=true; cancellation.Cancel(); status.Text=UiLanguage.Text("Stopping…","正在停止…"); } };
                timer.Tick += (s,e) => {
                    if (runner != null && System.IO.File.Exists(runner.StatusPath))
                        try { status.Text=JsonPlanReader.Read<BackendStatus>(runner.StatusPath).Message; } catch (System.IO.IOException) { } catch (System.Runtime.Serialization.SerializationException) { }
                };
                generate.Click += async (s,e) => {
                    var match=Regex.Match(input.Text.Trim(), @"^A\s+(?:(small|medium|large)\s+)?(\d+)-bedroom\s+apartment\s+with\s+(\d+)\s+bathrooms?\s+and\s+(\d+)\s+(?:balcony|balconies)\.?$", RegexOptions.IgnoreCase);
                    if (!match.Success) { MessageBox.Show(form,UiLanguage.Text("Use the example format. Change only the size and room counts.","请使用示例格式，只修改大小档位和房间数量。"),UiLanguage.Text("Check your prompt","输入提示")); return; }
                    int beds,baths,balconies;
                    if (!int.TryParse(match.Groups[2].Value,out beds) || !int.TryParse(match.Groups[3].Value,out baths) || !int.TryParse(match.Groups[4].Value,out balconies)) { MessageBox.Show(form,UiLanguage.Text("Room counts are out of range.","房间数量超出范围。"),UiLanguage.Text("Check your prompt","输入提示")); return; }
                    if (beds<1 || baths<1 || (long)beds+baths+balconies+3>20) { MessageBox.Show(form,UiLanguage.Text("Use at least 1 bedroom and 1 bathroom, with at most 20 nodes in total (including the living room, kitchen and entry).","至少 1 间卧室和 1 间卫生间，总房间节点不能超过 20。"),UiLanguage.Text("Check your prompt","输入提示")); return; }
                    running=true; language.Enabled=false; input.Enabled=false; examples.Enabled=false; generate.Enabled=false; cancel.Text=UiLanguage.Text("Cancel","取消");
                    progress.Visible=true; runner=new PythonBackendRunner(); timer.Start();
                    string prompt=input.Text.Trim();
                    try { plan=await Task.Run(() => runner.Run(prompt,cancellation.Token)); running=false; form.Close(); }
                    catch (OperationCanceledException) { running=false; form.Close(); }
                    catch (Exception error) { running=false; MessageBox.Show(form,error.Message,UiLanguage.Text("Generation failed","生成失败")); form.Close(); }
                    finally { timer.Stop(); }
                };
                form.ShowDialog(new RevitWindow(owner)); return plan;
            }
        }
        static string Hint() { return UiLanguage.Text("Change: small / large (omit for medium), bedrooms, bathrooms and balconies.\nIncludes 1 living room and 1 kitchen. Use the English examples below.","可修改：small / large（省略时默认中等）、卧室数、卫生间数、阳台数。\n自动包含 1 个客厅和 1 个厨房。请按下方英文示例填写。"); }
        static string Ready() { return UiLanguage.Text("Creates 3D walls, doors, windows and rooms, then opens the 3D model.\nThe first generation loads the bundled models.","生成三维墙、门窗和房间，完成后自动打开三维模型。\n首次生成需要加载模型。"); }
    }
}
