# 人体路径预测模块

本项目根据人体骨架历史预测未来活动区域与运动轨迹，面向跌倒预警系统中的路径预测模块。
模型不直接从视频像素提取人体姿态：骨架点由上游骨架模块提供，本模块负责历史查询、
轨迹推理、路径置信度计算以及 JSON、CSV 和可视化视频导出。

项目提供两个相互解耦的正式入口：

- **单帧调用**：根据 `video_id` 查询持续生成的骨架文件，读取当前帧之前的观察窗口，
  只返回本次请求帧的预测；适合系统合并和近实时调用。
- **批量调用**：显式传入完整骨架文件，对文件中的全部可用帧预测；适合离线验证、回放
  和整体结果导出。

两种入口共享模型加载、数据契约、置信度和导出代码，可以独立保留或删除，不会形成两套
重复模型实现。

## 1. 环境准备

项目支持 Python 3.8 或更高版本。建议在任意操作系统上使用独立虚拟环境，避免与系统
Python 的依赖发生冲突：

```bash
python -m venv .venv
```

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1` 激活；Linux 或 macOS 使用
`source .venv/bin/activate` 激活。随后安装依赖：

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

`requirements.txt` 是运行依赖；`requirements-dev.txt` 额外包含测试依赖。

## 2. 快速运行

以下多行命令使用 PowerShell 续行符。其他终端可使用相同参数，并按对应 shell 的语法
调整续行符。

### 2.1 单帧调用：按 video_id 查询骨架

```powershell
python run_path_prediction_frame.py `
  --config config.json `
  --video-path examples/inputs/adl-01-cam0.mp4 `
  --video-id adl-01-cam0 `
  --frame-index 100 `
  --timestamp-sec 3.333333 `
  --skeleton-dir examples/inputs `
  --output-dir examples/outputs/frame_prediction
```

该命令没有显式提供骨架文件。程序会使用 `video_id=adl-01-cam0` 在
`examples/inputs` 中查询到 `urfall_keypoints_adl-01-cam0.xlsx`，截取第 100 帧之前
最近 2 秒历史，并只输出第 100 帧的预测。

也可以通过 `--skeleton-path` 明确指定即时骨架文件。单帧服务每次调用都会重新读取该
文件，因此能看到上游刚写入的完整新行；模型在同一服务实例内只加载一次。

样例输出：

- `examples/outputs/frame_prediction/path_prediction_frame_output.json`
- `examples/outputs/frame_prediction/path_prediction_frame_output.csv`
- `examples/outputs/frame_prediction/frame_prediction_visual.mp4`

### 2.2 批量调用：完整骨架文件

```powershell
python run_path_prediction_batch.py `
  --config config.json `
  --video-path examples/inputs/adl-01-cam0.mp4 `
  --video-id adl-01-cam0 `
  --skeleton-path examples/inputs/urfall_keypoints_adl-01-cam0.xlsx `
  --output-dir examples/outputs/batch_prediction
```

样例输出：

- `examples/outputs/batch_prediction/path_prediction_batch_output.json`
- `examples/outputs/batch_prediction/path_prediction_batch_output.csv`
- `examples/outputs/batch_prediction/batch_prediction_visual.mp4`

`run_path_prediction.py` 是批量入口的兼容别名。新代码应明确使用
`run_path_prediction_frame.py` 或 `run_path_prediction_batch.py`。

## 3. 外部开关与内部参数

根目录 `config.json` 只负责合并系统的模块开关：

```json
{
  "modules": {
    "path_prediction": {
      "enabled": true
    }
  }
}
```

单帧调用会在每次请求时重新读取该文件。关闭后不查询骨架、不加载模型，并返回
`status=disabled`。`video_path`、`video_id`、`frame_index` 和 `timestamp_sec` 是请求
数据，不应通过持续覆盖配置文件传递。

模块内部参数位于 `src/path_prediction/config.py`，包括：

- 模型类型、检查点和计算设备；
- 观察时长和预测时长；
- MC Dropout、多模态候选数和置信度校准尺度；
- 骨架查询目录、默认输出目录和人物缓存限制。

当前正式模型按照 **2 秒观察、3 秒预测** 训练。修改时长必须同时换用结构匹配的检查点
或重新训练；运行时会主动校验，避免输入形状不匹配。

## 4. 输入与输出

公共请求字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `video_path` | string | 视频路径，用于标识和生成可视化 |
| `video_id` | string | 跨模块统一 ID，也是默认骨架查询键 |
| `frame_index` | integer | 当前预测基准帧 |
| `timestamp_sec` | number | 当前帧时间戳，单位秒 |

骨架表支持 xlsx、xls、csv 和 parquet，接受统一字段 `frame_index`，并兼容旧字段
`frame_id`。关键点至少需要 `nose`、`midhip`、左右脚踝的 x/y 坐标；现有样例采用
BODY25 展开列格式。

两种模式的 `records` 共享以下核心输出：

| 字段 | 说明 |
|---|---|
| `person_id` | 上游骨架跟踪人物 ID |
| `current_position` | 当前人体中心 |
| `predicted_activity_region` | 路径及不确定性覆盖的多边形区域 |
| `predicted_trajectory` | 按未来时间排列的 `[x,y]` 点序列 |
| `path_confidence` | `[0,1]` 路径可靠性分数 |
| `confidence_details` | 验证 ADE、当前采样离散度和校准参数 |
| `candidate_paths` | 多模态模型的候选轨迹；GRU 为空 |

GRU 的置信度算法为 `mc_dropout_ade_v1`：使用当前输入的 MC Dropout 离散度和验证集
ADE 联合估计路径可靠性。该数值不是跌倒概率。

完整状态、文件查询规则和 JSON schema 说明见
[双入口集成接口](docs/integration_interfaces.md)。

## 5. 项目结构

```text
config.json                    外部模块开关
pytest.ini                     跨平台测试路径配置
run_path_prediction_frame.py  单帧正式入口
run_path_prediction_batch.py  批量正式入口
run_path_prediction.py        批量入口兼容别名

src/path_prediction/           两种入口共享的生产代码
  config.py                    内部模型与窗口参数
  control.py                   动态读取外部开关
  skeleton_provider.py         按 video_id 查询和截取骨架历史
  frame_service.py             单帧服务编排与模型复用
  window_predictor.py          内部观察窗口推理核心
  export.py                    JSON、CSV 和视频导出

src/gru/                       GRU 模型、训练、底层批量预测
src/eigen_transformer/         低秩多模态 Transformer
src/leapfrog_diffusion/        Leapfrog 条件扩散模型
src/data_alignment.py          多数据集骨架字段与坐标对齐
src/build_aligned_dataset.py   生成统一训练数据
src/evaluate_models.py         三种模型统一评估

models/                        正式检查点、训练历史和评估报告
examples/inputs/               可直接运行的样例视频与骨架
examples/outputs/              单帧版和批量版正式样例结果
tests/                         接口、骨架追加、窗口推理和模型加载测试
docs/                          集成接口与模型训练补充文档
```

`data/processed/` 和 `outputs/` 是可再生成目录，已从仓库清理并由 `.gitignore` 排除。

## 6. 单帧调用状态

| 状态 | 含义 |
|---|---|
| `ok` | 当前帧存在、观察历史完整并已预测 |
| `insufficient_history` | 当前帧存在，但历史不足模型观察时长 |
| `skeleton_not_ready` | 骨架文件尚未写入当前帧 |
| `disabled` | 外部 `config.json` 已关闭模块 |

骨架文件模糊匹配到多个候选时会直接报错，不会猜测使用其中一个文件。

## 7. 模型训练与评估

生成训练数据：

```powershell
python src/gru/prepare_dataset.py `
  --data-root "<骨架数据目录>" `
  --split-file examples/inputs/sequence_split_recommendation.csv `
  --output-dir data/processed
```

训练默认 GRU：

```powershell
python src/gru/train.py `
  --dataset-dir data/processed `
  --model-out models/trajectory_gru.pt
```

Transformer、Diffusion 的定位、训练参数和底层预测命令见
[路径预测模型与训练](docs/model_architectures.md)。

## 8. 测试

```bash
python -m pytest tests -q
```

测试覆盖：

- 外部开关动态读取；
- 按 `video_id` 查询骨架文件；
- 骨架文件追加新帧后的重新读取；
- 当前帧历史窗口、多人缓存与间断帧处理；
- 三种模型检查点加载；
- JSON/CSV 契约和逐轨迹置信度。

底层模型脚本仍可独立训练和预测，但正式系统集成应只使用根目录的两个明确入口。

## 9. 打包提交

提交前建议在项目根目录执行测试，并确认样例输出可以重新生成。压缩包应包含源码、模型、
样例输入输出、配置、依赖清单和文档；不应包含 `.git`、`.venv`、`.idea`、缓存目录、
临时训练数据及其他可再生成文件。项目内输出的样例路径采用相对路径，可在不同机器和
操作系统间直接查看。
