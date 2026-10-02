# Text2Revit 用户安装与开发说明

[English](guide.md)

## 给用户

关闭 Revit，双击 `Text2Revit-Setup.exe`，点击安装。
在线安装程序自动下载、校验并安装 Python、模型和各年插件，并生成插件注册文件。用户只需下载 EXE，无需手动下载或解压数据附件。下载中断后重新运行安装器即可复用已验证的数据，并继续未完成的下载。
安装完成后启动 Revit，打开项目中的楼层平面，在 Text2Revit 选项卡点击 Generate Model。
按弹窗示例输入英文 prompt，点击生成。可以修改整体大小、卧室数、卫生间数和阳台数。
生成三维墙、真实门窗族、开放阳台栏杆、房间及标记，完成后自动打开三维视图。在该三维视图中也可再次点击生成。
中等户型默认省略大小词，例如 `A 3-bedroom apartment with 2 bathrooms and 1 balcony.`；小户型和大户型分别加 `small`、`large`。
无需安装 Conda、设置环境变量或手动复制文件。CPU 可用，有兼容的 NVIDIA 显卡时自动加速。
安装按当前 Windows 用户生效；同机其他 Windows 用户需分别安装。
Windows 的“已安装的应用”中可以卸载 Text2Revit。卸载保留生成记录。
在线安装需要联网，之后推理离线运行。安装过程建议预留 12 GB 空间，无需另装压缩软件。完整离线版也可单独提供，无需在安装时下载数据。
首次启动 Revit 可能显示 Autodesk 的插件信任提示，选择“始终加载”即可。
目前已在 Revit 2022 实际验证生成；2020、2021、2023–2026 已单独编译，仍需对应年份实机测试。

## 自动生成的尺寸

| 构件 | 规则 |
| --- | --- |
| 外墙 / 内墙 | 400 mm / 200 mm |
| 墙高 | 3.3 m |
| 阳台外边界 | 1100 mm 高开放式 Revit 栏杆；保留与室内相接的墙和门 |
| 门高 | 2.1 m |
| 门宽 / 窗宽 | Python 最终门窗线段长度乘米制比例，不另设固定宽度 |
| 客厅、卧室窗 | 窗高 1.5–1.7 m，窗台高 0.9–1.0 m |
| 厨房窗 | 窗高 1.2–1.5 m，窗台高 0.9–1.0 m |
| 卫生间窗 | 窗高 0.6–1.2 m，窗台高 1.2–1.5 m |

窗高和窗台高按 100 mm 步长抽样，包含范围的两端值。以上参数由代码自动使用，第一版弹窗仅开放四类 prompt 条件。阳台使用房间分隔线保持房间面积，外侧不生成全高墙。项目模板需包含栏杆类型，插件复制该类型用于阳台。
窗框和墙洞按窗族模板的实际插入原点生成；放置后检查实体窗底、窗顶与请求的窗台高和窗高一致。修正后的窗族使用新的缓存版本，避免复用旧几何。
图标为原创的蓝色户型线稿和金色闪光，提供 16 px / 32 px 两种尺寸；选型调研参考 [Lucide house](https://lucide.dev/icons/house)。

界面默认英文，安装器和弹窗可选择“中文”，并记住选择。工具栏文案在下次 Revit 启动时应用该语言。两种界面都使用模型支持的英文 prompt。
英文工具栏生成、开放阳台栏杆及三维视图已在 Revit 2022 验证。窗族插入原点偏差已修正，实体窗底高度及非零楼层的新窗放置均通过检查。其他年份仍需实机验证。

## 开发文件放哪里、有什么用

所有文件已放在本项目 `revit` 目录，保持目录结构即可，不需要你手工移动。

| 文件/目录 | 作用 |
| --- | --- |
| `Build-Release.cmd` | 开发者双击此文件，构建完整用户安装程序 |
| `build_release.py` | 编译各年 C# 插件、导出模型、打包环境、生成单文件安装程序 |
| `Text2Revit.Addin/App.cs` | 启动 Revit 时创建工具栏选项卡和图标 |
| `Text2Revit.Addin/Command.cs` | 串联弹窗、推理、门窗族加载和 Revit 事务 |
| `Text2Revit.Addin/UI/PromptDialog.cs` | 输入 prompt、示例、等待状态和取消 |
| `Text2Revit.Addin/Services/PythonBackendRunner.cs` | 启动安装包自带的 Python，传入请求、读取结果 |
| `Text2Revit.Addin/Services/WallBuilder.cs` | 创建 400 mm 外墙、200 mm 内墙、高 3.3 m |
| `Text2Revit.Addin/Services/BalconyBuilder.cs` | 创建开放阳台栏杆及房间分隔线 |
| `Text2Revit.Addin/Services/ModelViewBuilder.cs` | 创建三维视图并关联原楼层平面，自动调整显示范围 |
| `Text2Revit.Addin/Services/FamilyLibrary.cs` | 自动创建/缓存真正的门窗族，包含门扇、窗框和玻璃 |
| `Text2Revit.Addin/Services/OpeningBuilder.cs` | 按墙 ID 放置门窗族实例 |
| `Text2Revit.Addin/Services/RoomBuilder.cs` | 创建房间和房间标记 |
| `Text2Revit.Addin/Models/PlanData.cs` | C# 读取的 JSON 字段定义 |
| `Installer/` | 自动解压、初始化、注册和卸载的 Windows 安装程序源码 |
| `build_tools/` | 只供开发构建的 SDK、API 引用和 conda-pack，不交付用户 |
| `build/` | 导出的精简模型、环境压缩包等中间产物，不交付用户 |
| `dist/` | 最终用户安装程序和 SHA256 校验值 |

Python 推理代码在仓库根目录，打包时自动复制所需模块及小型统计文件。
`inference_runtime.py` 独立运行推理，不导入训练器或加载训练数据。
`revit_geometry.py` 输出唯一墙段、宿主墙 ID、门窗实际宽高及窗台高。
`backend_cli.py` 输出 JSON，记录状态，几何检查失败时最多尝试十次。

## 打包

在现有开发电脑双击 `Build-Release.cmd`。
首次构建需要网络，用于下载编译 SDK、各年 API 引用和打包工具。
用户安装和推理不需要下载模型。
如果更换训练权重，可运行：

```powershell
python build_release.py --checkpoint "你的模型路径"
```

第一次打包耗时较长，之后自动复用 SDK、运行环境压缩包和未改变的模型导出。
默认构建 Revit 2020–2026，每个年份单独 DLL，共享一个 Python 后端。
编译成功表示 API 调用可以编译；实际运行兼容性需要在对应 Revit 年份中验证。
门窗族首次生成使用对应年份 Revit 的门窗模板，之后自动缓存。
基本族模板属于正常 Revit 安装的核心内容；发行版无需用户自行寻找门窗 RFA。
门族包含三维门扇/门框及平面开启弧线，窗族包含窗框、玻璃和中梃。
房间名称按所选界面语言标记；已有墙时，新户型自动放到右侧，避免重复生成重叠。

## 自动安装位置

- 产品文件：`%LocalAppData%/Text2Revit/releases/发行版本/`
- 插件注册：`%AppData%/Autodesk/Revit/Addins/年份/Text2Revit.addin`
- 生成 JSON/日志：`%LocalAppData%/Text2Revit/jobs/`
- 自动门窗族缓存：`%LocalAppData%/Text2Revit/families/年份/`

以上路径都由程序管理，用户不需要手工放文件。

所有房间的窗户避开入户门所在的连续外墙，另一侧平行墙可以开窗。连接阳台的卧室不加窗。厨房和卫生间净面积不得小于 2㎡，不合格则重试。预览和平面图显示每个房间及总面积，总面积含阳台。
