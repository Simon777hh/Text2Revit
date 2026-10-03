# 数据文件

[English](README.md)

仓库保留的六个数据文件共约 **1.13 MiB**，是代码使用的统计先验和配置，
不是完整户型数据集、训练张量或模型权重。

| 文件 | 用途 |
| --- | --- |
| `combined_area_stats_natural.json` | canvas 房间及总面积分位数，用于训练提示词的大中小标签 |
| `corner_count_distributions.json` | 房间角点数量分布，用于拓扑条件采样 |
| `geometry_policy.json` | 厨房和卫生间最低面积，单位 m² |
| `raw_recovery_scale_stats.json` | 坐标到米的尺度恢复及几何统计 |
| `room_aspect_ranges.json` | 几何恢复时使用的房间长宽比范围 |
| `room_aspect_samples.npz` | 几何恢复时使用的房间长宽比样本 |

安装包包含后五项推理文件；第一项用于生成训练提示词。六项都有当前调用方，
需要保留。修改它们可能改变生成户型或训练标签。

本地准备数据时也会使用这个目录，但大文件不进入 Git：

- `sources/ResPlan/ResPlan.pkl`：另行获取的原始数据。
- `sources/ResPlan/ResPlan_filtered_canvas.pkl`：唯一保留的 ResPlan 处理结果。
- `sources/RPLAN/dataset/floorplan_dataset/`：另行获取的 RPLAN 原始 PNG。
- `sources/RPLAN/RPLAN_filtered.pkl`：唯一保留的 RPLAN 处理结果，使用
  `clip_area_scale` 记录 CLIP 标签的面积倍率，不再保存多份几何文件。
- 生成的 `.npz`、`prompts.json`、`raw_manifest.json`、`clip/` 和
  `mmap_cache/`：训练数据及缓存，由 `.gitignore` 排除。

准备及验证方法见 [数据流程](../docs/data_pipeline.md)。模型权重放在
[`checkpoints/`](../checkpoints/README.md)。
