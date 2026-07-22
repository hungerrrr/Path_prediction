# 双入口集成接口

## 总体设计

路径模型只消费骨架历史，不直接从视频像素提取人体姿态。项目提供两个彼此解耦、
共享模型核心的入口：

| 模式 | 入口 | 骨架来源 | 输出范围 |
|---|---|---|---|
| 单帧 | `run_path_prediction_frame.py` | 按 `video_id` 查询可持续追加的骨架文件 | 当前请求帧 |
| 批量 | `run_path_prediction_batch.py` | 显式传入完整骨架文件 | 文件中的全部帧 |

`FramePredictionService` 会复用已加载模型，但每次调用重新读取骨架文件。这样骨架模块
可以持续追加记录，路径模块无需重启。若以后改为数据库、Redis 或消息缓存，只需实现
新的历史提供器，不需要修改模型和输出层。

## 外部开关

外部 `config.json` 只承担合并系统的模块开关：

```json
{
  "modules": {
    "path_prediction": {
      "enabled": true
    }
  }
}
```

单帧服务每次调用都会重新读取该开关。关闭时不查询骨架、不加载模型，返回
`status=disabled`。逐帧输入不应通过反复覆盖配置文件传递。

## 内部参数

[`src/path_prediction/config.py`](../src/path_prediction/config.py) 中的
`PathPredictionSettings` 统一管理：

- `model_method`、`model_path`、`device`；
- `observation_seconds`、`prediction_seconds`；
- MC Dropout 和多模态采样参数；
- 骨架查询目录、默认输出目录和缓存限制。

当前检查点按 2 秒观察、3 秒预测训练。修改这两个时长时必须使用结构匹配的检查点或
重新训练；运行时会校验，不会静默使用错误形状。

## 单帧输入

必填公共字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `video_path` | string | 样例/可视化视频路径 |
| `video_id` | string | 跨模块一致的视频 ID，也是默认骨架查询键 |
| `frame_index` | integer | 当前调用帧 |
| `timestamp_sec` | number | 当前调用时间戳（秒） |

可额外指定：

- `skeleton_path`：明确指定骨架文件；
- `skeleton_dir`：未指定文件时，按 `video_id` 查询的目录。

查询顺序支持：`<video_id>`、`keypoints_<video_id>`、
`urfall_keypoints_<video_id>`，扩展名支持 xlsx/xls/csv/parquet。若没有精确名称，允许
唯一的模糊匹配；匹配多个文件时直接报错，避免使用错误人物数据。

单帧状态：

- `ok`：观察历史完整并产生预测；
- `insufficient_history`：当前帧存在，但历史不足观察时长；
- `skeleton_not_ready`：即时骨架文件尚未写入当前帧；
- `disabled`：外部开关关闭。

## 批量输入

批量入口显式接收 `video_path`、`video_id`、`skeleton_path` 和 `output_dir`。骨架表接受
`frame_index`，兼容旧字段 `frame_id`；可提供 `timestamp_sec`，缺失时才按源 FPS 推导。

## 统一记录字段

两种模式的 `records` 使用同一组字段：

| 字段 | 说明 |
|---|---|
| `video_path`、`video_id` | 公共视频标识 |
| `frame_index`、`timestamp_sec` | 预测基准帧与时间戳 |
| `person_id` | 骨架跟踪人物 ID |
| `has_prediction` | 是否已经满足观察窗口 |
| `current_position` | 当前人体中心坐标 |
| `predicted_activity_region` | 路径及不确定性覆盖的多边形区域 |
| `predicted_trajectory` | 未来 `[x,y]` 点序列 |
| `path_confidence` | `[0,1]` 路径可靠性分数 |
| `confidence_type/details` | 置信度算法和验证 ADE、采样离散度等组成 |
| `model_method` | 实际模型类型 |
| `candidate_paths` | 多模态模型候选轨迹；GRU 为空 |

GRU 使用 `mc_dropout_ade_v1`：当前样本 MC Dropout 离散度与验证 ADE 共同校准。
该分数表示路径预测可靠性，不是跌倒概率。
