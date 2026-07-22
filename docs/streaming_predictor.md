# StreamingPredictor 使用说明

有状态的流式路径预测器，支持多人物独立跟踪与逐帧推理。适用于将路径预测模块集成到更大的老人跌倒检测系统中。

## 数据流

```
上游（姿态估计）               StreamingPredictor              下游（跌倒检测）
      │                              │                              │
      │  PersonFrameData (每帧每人)   │                              │
      │ ──────────────────────────►  │                              │
      │                              │  PredictionResult (每帧每人) │
      │                              │ ──────────────────────────►  │
```


## 与大型系统集成前必须确认的边界

### 当前接口层级

StreamingPredictor 是进程内、有状态、同步的逐帧 Python API。它本身不是
WebSocket、gRPC、Kafka、Redis Stream 或 HTTP 服务。建议其他项目先依赖本文件
定义的数据契约，再由系统集成方根据整体技术栈增加传输适配器，避免路径预测核心
与某一种消息中间件绑定。

### 坐标与时间约定

- 输入关键点和 bbox 必须已经映射到模型画布坐标，当前正式模型使用 640×240。
- source 目前只作为元数据保留，不会自动完成分辨率或坐标转换。
- 同一 person_id 的 frame_id 必须严格递增。
- 默认要求逐帧连续；帧间隔超过 max_frame_gap 时，该人物历史会被清空并重新积累。
- 视频、摄像头或场景切换时必须调用 reset()，不能沿用上一场景状态。
- FrameData 中人物的帧号和时间戳必须与外层帧一致，同一帧不能出现重复 ID。

### 并发与吞吐

公开状态接口使用重入锁保护，同一实例可避免并发修改导致缓冲区损坏。但锁内包含
模型推理，因此同一实例不会并行执行多个请求。当前多人 update() 仍按人物逐个
推理；大规模多摄像头部署应采用“每路流独立状态 + 就绪人物微批处理”的服务层，
或者为每路流配置独立实例。

### 置信度语义

detection_confidence 目前只做范围校验，不参与轨迹融合或过滤。输出的
confidence 是模型候选分数：GRU 使用验证误差换算值，Transformer 使用 softmax，
扩散模型使用采样分数。它们尚未进行跨模型概率校准，不能直接解释为真实跌倒概率。

### 状态持久化

get_state() 返回带 schema_version 的 JSON 可序列化字典；
PredictionResult.to_dict() 也可以直接交给 json.dumps()。恢复状态时会校验模型
类型、观测长度、人物数量、坐标和边界框，防止损坏状态污染实时服务。

## 快速开始

### 1. 从已训练模型加载

```python
from streaming_predictor import StreamingPredictor

predictor = StreamingPredictor.from_checkpoint(
    model_path="models/trajectory_gru.pt",
    method="gru",                   # "gru" / "eigen_transformer" / "leapfrog_diffusion"
    device="auto",                  # "auto" / "cpu" / "cuda"
    num_samples=6,                  # 多模态模型的候选轨迹数，GRU 忽略
    max_persons=50,                 # 单实例最大跟踪人数
    stale_frame_threshold=30,       # 人物离场后的状态保留帧数
    max_frame_gap=1,                # 同一人物只接受连续帧
)
```

工厂方法自动完成：
- 读取检查点中的模型配置和权重
- 重建对应架构的模型（TrajectoryGRU / EigenTrajectoryTransformer / LeapfrogDiffusionPredictor）
- 恢复数据处理参数（obs_frames、pred_frames、归一化尺度等）
- GRU 模型还会根据验证集 ADE 自动计算置信度

### 2. 预热模型（可选，推荐）

```python
predictor.warmup(n_times=1)
```

消除首次推理的冷启动开销（PyTorch 的 JIT 编译、cuDNN 自动调优等）。

### 3. 在主循环中逐帧调用

```python
# 假设这是视频帧处理回调
def on_frame(frame_id: int, timestamp: float,
             pose_results: list[dict]) -> list[PredictionResult]:
    """被上游帧处理回调调用。"""
    persons = []
    for det in pose_results:
        persons.append(PersonFrameData(
            frame_id=frame_id,
            timestamp=timestamp,
            person_id=str(det["track_id"]),         # 跟踪器分配的 ID
            keypoints=det["keypoints"],         # {"nose": (x,y), "midhip": (x,y), ...}
            bbox=det["bbox"],                   # (xmin, ymin, xmax, ymax)
            source=det.get("source", "custom"),
            detection_confidence=det["score"],
        ))
    return predictor.update(FrameData(frame_id, timestamp, persons))


# 使用
for result in on_frame(100, 3.33, pose_data):
    if result.has_prediction and result.confidence > 0.3:
        # 交给下游模块
        fall_risk_score = compute_fall_risk(result)
```

## 输入数据契约

### PersonFrameData（单帧单人）

| 字段 | 类型 | 说明 |
|------|------|------|
| `frame_id` | `int` | 帧号，用于缓冲区管理和陈旧淘汰 |
| `timestamp` | `float` | 时间戳（秒） |
| `person_id` | `str` | 跟踪器分配的唯一 ID，如 `"person_1"` |
| `keypoints` | `dict[str, tuple[float\|None, float\|None]]` | BODY25 关键点字典，未检测到时为 `None`；至少需要 `nose`、`midhip`、`rankle`、`lankle` |
| `bbox` | `tuple[float, float, float, float]` | 边界框 `(xmin, ymin, xmax, ymax)`，不足 4 个有效关键点时作为人体中心回退 |
| `source` | `str` | 数据源标识（用于分辨率和帧率映射），如 `"har-up"`、`"urfall"` |
| `detection_confidence` | `float` | 检测器置信度 `[0, 1]` |

### FrameData（单帧多人）

| 字段 | 类型 | 说明 |
|------|------|------|
| `frame_id` | `int` | 帧号 |
| `timestamp` | `float` | 时间戳（秒） |
| `persons` | `list[PersonFrameData]` | 该帧所有检测到的人物 |

## 输出数据契约

### PredictionResult（单帧单人）

| 字段 | 类型 | 说明 |
|------|------|------|
| `frame_id` | `int` | 对应输入帧号 |
| `timestamp` | `float` | 对应输入时间戳 |
| `person_id` | `str` | 对应人物 ID |
| `has_prediction` | `bool` | `True`=有足够的观测帧并输出了轨迹；`False`=观测帧不足 |
| `current_position` | `tuple[float, float]` | 当前帧人体中心 `(x, y)` 像素坐标 |
| `predicted_path` | `list[tuple[float, float]]` | Top-1 预测轨迹，长度 = `pred_frames`（默认 90 帧 = 3 秒） |
| `candidate_paths` | `list[CandidatePath]` | 多模态候选轨迹列表（GRU 返回 1 条，多模态模型返回 K 条） |
| `path_corridor` | `list[tuple[float, float]]` | 不确定性走廊四角 `[左上, 右上, 右下, 左下]` |
| `confidence` | `float` | Top-1 轨迹置信度 `[0, 1]` |
| `model_method` | `str` | `"gru"` / `"eigen_transformer"` / `"leapfrog_diffusion"` |

### CandidatePath

| 字段 | 类型 | 说明 |
|------|------|------|
| `probability` | `float` | 该候选轨迹的概率 |
| `path` | `list[tuple[float, float]]` | 轨迹坐标序列，长度 = `pred_frames` |

## 关键行为说明

### 观测帧不足（has_prediction=False）

每个人物需要积累 `obs_frames`（默认 60 帧 = 2 秒）帧的历史数据后才会输出预测。在积累阶段，`has_prediction=False`，但内部缓冲区会持续累积。

### 多人同时跟踪

内部为每个 `person_id` 独立维护观测缓冲区和边界框。多人的推理在单帧内按顺序执行。

### 陈旧人物自动淘汰

若某 `person_id` 连续 `stale_frame_threshold` 帧（默认 30 帧）未在当前帧中出现，其缓冲区会被自动清理。通过 `update()` 的批量接口触发，`update_single()` 不会触发淘汰。

### 人体中心计算

优先使用 4 个中心关键点的加权平均作为人体中心：

```
nose(0.3) + midhip(0.5) + rankle(0.1) + lankle(0.1)
```

若有效关键点权重之和 < 0.5，回退到 `bbox` 的中心 `((xmin+xmax)/2, (ymin+ymax)/2)`。

## 状态管理

### 导出/恢复状态

```python
# 场景切换前保存
state = predictor.get_state()

# 新场景恢复
predictor.set_state(state)
```

适用于同一视频流实例的短暂重启和容灾恢复。状态包括每个人的历史观测中心、最新边界框和最后出现帧号。不同摄像头或不同模型之间不能直接复用状态。

### 手动管理

```python
# 人物离开场景
predictor.remove_person("person_3")

# 完全重置
predictor.reset()
```

## 模型切换

```python
# GRU：确定性单轨迹，推理最快
gru = StreamingPredictor.from_checkpoint("models/trajectory_gru.pt", "gru")

# EigenTransformer：K 条多模态轨迹 + softmax 概率
transformer = StreamingPredictor.from_checkpoint(
    "models/eigen_transformer.pt", "eigen_transformer",
)

# Leapfrog Diffusion：K 条采样轨迹 + 条件扩散概率
diffusion = StreamingPredictor.from_checkpoint(
    "models/leapfrog_diffusion.pt", "leapfrog_diffusion",
)
```

| 模型 | 输出轨迹数 | 推理速度 | 适用场景 |
|------|-----------|---------|----------|
| GRU | 1 条 | 最快 (~5ms) | 低延迟实时系统 |
| EigenTransformer | K 条 (默认 6) | 中等 | 需多模态评估的场景 |
| LeapfrogDiffusion | K 条 (默认 6) | 较慢 | 需分布采样的不确定性分析 |

## 完整示例

```python
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, "src")
from streaming_predictor import (
    StreamingPredictor, PersonFrameData, FrameData,
)

# 1. 加载模型
predictor = StreamingPredictor.from_checkpoint(
    "models/trajectory_gru.pt",
    method="gru",
    device="cpu",
)
predictor.warmup()
obs_frames = predictor.config.obs_frames

# 2. 模拟单人物逐帧输入
for frame_id in range(200):
    person = PersonFrameData(
        frame_id=frame_id,
        timestamp=frame_id / 30.0,
        person_id="person_1",
        keypoints={
            "nose": (320.0 + frame_id * 0.5, 120.0),
            "midhip": (310.0 + frame_id * 0.5, 150.0),
            "rankle": (315.0 + frame_id * 0.5, 220.0),
            "lankle": (305.0 + frame_id * 0.5, 218.0),
        },
        bbox=(280.0, 80.0, 360.0, 220.0),
        source="custom",
        detection_confidence=0.95,
    )
    result = predictor.update_single(person)

    if result.has_prediction:
        print(
            f"帧 {result.frame_id}: "
            f"位置 {result.current_position}, "
            f"置信度 {result.confidence:.3f}, "
            f"轨迹长度 {len(result.predicted_path)}"
        )
    else:
        # 观测帧不足，仍在积累
        print(
            f"帧 {person.frame_id}: 积累中 "
            f"({frame_id + 1}/{obs_frames})"
        )

# 3. 清理
predictor.reset()
```

## 运行测试

```powershell
$env:PYTHONPATH="src"
python -m pytest tests/test_streaming_predictor.py -v
```
