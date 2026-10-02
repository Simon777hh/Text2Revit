# Text2Revit

[English](README.md)

**从文字生成户型，并创建可编辑的 Revit 模型。**

Text2Revit 使用条件 Flow Matching 模型，根据支持的英文提示词生成房间布局。Python 流程处理几何并布置门窗，C# 插件通过 Revit API 创建原生建筑构件。仓库包含数据预处理、模型训练、推理和 Revit 集成代码。

## 示例

```text
A large 3-bedroom apartment with 2 bathrooms and 2 balconies.
```

以下两张图展示同一个在 Revit 2022 中生成的户型：**3 个卧室、2 个卫生间、2 个阳台，总面积 106.2 m²**，含阳台。

![带房间面积及总面积的生成平面图](sample/floorplan.png)
![同一户型的可编辑 Revit 三维模型](sample/model3d.png)

[示例目录](sample/README.md) 包含生成的 JSON 和 Revit 原生面积记录。

## 生成内容

- 可编辑的三维墙体、真实门窗族、房间和开放阳台栏杆。
- 平面图中的各房间面积、总面积，以及着色三维视图。
- 按房间类型设置的窗高和窗台高，以 100 mm 为步长采样。
- 几何检查及不合格户型重新生成，包括厨房或卫生间面积小于 2 m² 的情况。

所有窗户避开入户门所在的连续外墙。连接阳台的卧室保留阳台门，不再添加窗户。门窗布置前修复狭窄几何，并检查洞口间距。具体尺寸见 [使用指南](revit/guide.zh.md)，检查范围及局限见 [验证报告](docs/audit.md)。

## 在 Revit 中使用

1. 从 [Windows 发行版](https://github.com/Simon777hh/Text2Revit/releases) 下载 `Text2Revit-Setup.exe`。关闭 Revit，运行安装器，选择 **Install**。
2. 启动 Revit，打开项目平面视图，选择 **Text2Revit → Generate Model**。
3. 输入支持的提示词，选择 **Generate**。插件创建模型并打开三维视图。

Windows 在线安装器自动下载并安装 Python、PyTorch、CLIP 和训练好的模型。用户只需手动下载安装 EXE，无需另外安装 Python 或 Conda。安装需要联网，之后推理在本地离线运行，兼容的 NVIDIA 显卡自动加速，同时支持 CPU 回退。完整离线安装包也可单独提供下载。

界面默认英文，可在安装器和提示词窗口中选择中文。两种界面均使用相同的英文提示词格式。

### 提示词格式

```text
A small 2-bedroom apartment with 1 bathroom and 1 balcony.
A 3-bedroom apartment with 2 bathrooms and 1 balcony.
A large 4-bedroom apartment with 2 bathrooms and 2 balconies.
```

| 条件 | 写法 |
| --- | --- |
| 户型大小 | 使用 `small` 或 `large`；中等户型省略大小 |
| 卧室数 | 修改 bedroom 数量 |
| 卫生间数 | 修改 bathroom 数量 |
| 阳台数 | 修改 balcony 数量 |

每个户型包含一个客厅和一个厨房。目前提示词界面支持这四项条件，其他设计要求尚未作为可调条件提供。

### 兼容性

安装包包含 **Revit 2020–2026** 各版本对应的插件，并分别注册到匹配版本，无需手动选择 DLL。七个版本均已编译；**Revit 2022** 已验证原生模型生成。其他版本的运行验证和全新外部电脑的安装验证仍待完成。

## 从源码运行推理

本地已验证的环境为 Windows x64、Python 3.14 和 PyTorch 2.11。源码仓库不包含模型权重。从源码推理需自行提供兼容的训练权重，放在 `checkpoints/flow_matching_best.pth`，或通过 `--checkpoint` 指定路径。Windows 安装器会自动提供在 Revit 中使用所需的权重。

在 Python 环境中，从仓库根目录运行以下命令：

```powershell
python -m pip install -r requirements.txt
python scripts/setup_models.py
python pipeline_generate.py --prompt "A 3-bedroom apartment with 2 bathrooms and 1 balcony."
```

初始化脚本首次下载 CLIP 文本编码器。推理读取 `checkpoints/flow_matching_best.pth`，将 JSON 和预览图写入 `outputs/`。可通过 `--checkpoint` 和 `--clip-dir` 指定其他模型路径。权重来源及校验值见 [权重说明](checkpoints/README.md)。

## 数据、训练与开发

| 位置 | 内容 |
| --- | --- |
| `preprocessing/resplan/`、`preprocessing/rplan/` | 数据清洗、墙体重建、过滤和坐标准备 |
| `preprocessing/features/` | GT、拓扑、提示词、mask、边映射、CLIP 特征及验证 |
| `model.py`、`train.py`、`losses.py` | 模型结构与 Flow Matching 训练 |
| `pipeline_generate.py`、`backend_cli.py` | 推理与结果输出 |
| `geometry_cleanup.py`、`plan_quality.py`、`door_window_rules.py`、`revit_geometry.py` | 几何修复、面积检查、门窗布置和构件尺寸 |
| `revit/` | C# 插件、安装器和发行版构建程序 |
| `sample/`、`tests/`、`docs/` | 示例、回归检查和文档 |

训练数据集需另行获取。[数据流程](docs/data_pipeline.md) 介绍数据准备和验证；[开发指南](docs/development.md) 介绍 C# 实现、安装包构建和检查命令；[验证报告](docs/audit.md) 记录已验证行为及尚存局限。

## 许可

Copyright (c) 2026 Simon H.

Text2Revit 采用 [源码可见评估许可](LICENSE)。允许克隆、学习，并为个人非商业评估私下运行或修改。商用、在其他项目中大量复用，以及超出许可范围的再分发，需事先取得书面授权。

第三方组件和数据集保留其原有条款，详见 [第三方声明](THIRD_PARTY_NOTICES.md)。
