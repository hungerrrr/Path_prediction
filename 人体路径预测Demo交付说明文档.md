# 人体路径预测Demo交付说明文档
## 1. 文档概述
本文档为人体路径预测Demo的交付说明，该Demo基于人体骨架关键点数据，通过卡尔曼滤波平滑、匀速外推预测等算法实现人体路径预测，并输出标准化的预测结果文件及可视化视频。

## 2. 交付文件清单
| 文件名称                  | 格式 | 说明                                                         |
| ------------------------- | ---- | ------------------------------------------------------------ |
| standard_pred_output.json | JSON | 符合自定义标准Schema的路径预测结果，包含每帧的人体位置、预测路径、走廊多边形等核心信息 |
| standard_pred_output.csv  | CSV  | JSON文件的表格化版本，便于数据分析工具（如Excel、Python Pandas）读取 |
| prediction_visual.mp4     | MP4  | 路径预测结果可视化视频，直观展示真实轨迹、预测路径及走廊区域 |

所有交付文件默认输出至项目根目录下的`outputs/`文件夹中。

## 3. 坐标系统约定
### 3.1 基础坐标定义
预测结果中所有坐标均遵循**原始视频像素坐标系**（配置项`COORD_SYSTEM = "original_video_pixels"`），具体规则：
- 坐标原点：视频画面**左上角**；
- X轴：水平向右为正方向；
- Y轴：垂直向下为正方向；
- 有效坐标范围：
  - X轴：[320, 640]
  - Y轴：[0, 240]

## 4. JSON格式字段说明（standard_pred_output.json）
JSON文件采用层级化结构，核心字段说明如下：

### 4.1 顶层字段
| 字段名            | 类型   | 说明                                             |
| ----------------- | ------ | ------------------------------------------------ |
| schema_version    | string | Schema版本号，固定为`path_prediction.v1`         |
| coordinate_system | string | 坐标系统说明，固定为`original_video_pixels`      |
| video             | string | 输入视频名称，固定为`adl-01-cam0.mp4`            |
| frames            | array  | 按帧索引排序的预测结果列表，每一项对应视频的一帧 |

### 4.2 frames数组元素字段
| 字段名           | 类型                | 说明                                                         |
| ---------------- | ------------------- | ------------------------------------------------------------ |
| frame_index      | int                 | 视频帧索引（从0开始）                                        |
| timestamp_sec    | float               | 帧对应的时间戳（单位：秒），计算方式：frame_index / FPS（FPS=30） |
| person_id        | string              | 人体标识ID，固定为`person_001`                               |
| current_position | array[float]        | 当前帧人体中心坐标，格式：[x, y]                             |
| predicted_path   | array[array[float]] | 未来3秒的预测路径坐标列表，每个元素为[x, y]，共90个点（30FPS*3秒） |
| path_corridor    | array[array[float]] | 预测路径的安全走廊多边形，由4个顶点组成的顺时针闭合多边形，格式：[[x1,y1], [x2,y2], [x3,y3], [x4,y4]] |
| confidence       | float               | 预测置信度（0-1），由MSE误差计算得出：1 - min(MSE/100, 0.3)，值越高预测越可靠 |

### 4.3 JSON示例片段
```json
{
  "schema_version": "path_prediction.v1",
  "coordinate_system": "original_video_pixels",
  "video": "adl-01-cam0.mp4",
  "frames": [
    {
      "frame_index": 7,
      "timestamp_sec": 0.23,
      "person_id": "person_001",
      "current_position": [561,112],
      "predicted_path": [[534.4123222748815, 106.69194312796208], [534.4123222748815, 106.69194312796208], ...],
      "path_corridor": [[514, 51], [554, 51], [554, 162], [514, 162]],
      "confidence": 0.7
    }
  ]
}
```

## 5. CSV格式说明（standard_pred_output.csv）
CSV文件为JSON中`frames`数组的表格化转换，便于数据解析和统计，字段对应关系及格式说明如下：

### 5.1 字段映射
| CSV列名          | 对应JSON字段               | 格式说明                                                     |
| ---------------- | -------------------------- | ------------------------------------------------------------ |
| frame_index      | frames[*].frame_index      | 整数                                                         |
| timestamp_sec    | frames[*].timestamp_sec    | 浮点数（保留2位小数）                                        |
| person_id        | frames[*].person_id        | 字符串                                                       |
| current_position | frames[*].current_position | 字符串格式的列表，如"[350.0, 120.0]"                         |
| predicted_path   | frames[*].predicted_path   | 字符串格式的二维列表，如"[[350.0,120.0],[350.5,120.2],...]"  |
| path_corridor    | frames[*].path_corridor    | 字符串格式的二维列表，如"[[330,100],[370,100],[370,140],[330,140]]" |
| confidence       | frames[*].confidence       | 浮点数（保留2位小数）                                        |

### 5.2 读取说明
由于CSV不支持嵌套列表，所有数组类型字段均转换为字符串格式，使用Python读取时可通过`eval()`函数还原为列表：
```python
import pandas as pd

df = pd.read_csv("standard_pred_output.csv", encoding="utf-8-sig")
# 还原current_position为列表
df["current_position"] = df["current_position"].apply(eval)
# 还原predicted_path为二维列表
df["predicted_path"] = df["predicted_path"].apply(eval)
```

## 6. 可视化视频说明（prediction_visual.mp4）
### 6.1 视频内容
视频为原始输入视频的缩版本（分辨率640*240），叠加以下可视化元素：
- **白色实线**：人体真实历史轨迹；
- **红色虚线**：未来3秒的预测路径；
- **蓝色半透明多边形**：预测路径的安全走廊区域；
