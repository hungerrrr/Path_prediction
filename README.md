# 人体路径预测 Demo

项目现在分为三个互相独立的阶段：数据规范化、模型训练和实际路径预测。推荐的数据划分按完整序列执行，同一序列产生的滑动窗口不会被分到不同集合中。

## 项目结构

```text
src/gru/             原始 GRU（可独立训练和预测）
src/eigen_transformer/ 方案 A（可独立训练和预测）
src/leapfrog_diffusion/ 方案 B（可独立训练和预测）
models/              已训练模型、训练历史和测试报告
data/processed/      可直接用于训练的 train/val/test 数据
examples/inputs/     可直接运行的样例输入
examples/outputs/    使用正式模型生成的样例输出
```

## 1. 规范化完整数据集

数据目录可以包含任意层级的 `.xlsx`、`.xls`、`.csv` 或 `.parquet` 关键点表格。程序会递归发现文件，并从文件名或 `source_id` 中识别 `adl-xx` / `fall-xx` 序列。

```powershell
python src\gru\prepare_dataset.py `
  --data-root "D:\你的数据集目录" `
  --split-file "D:\推荐划分\sequence_split_recommendation.csv" `
  --output-dir data\processed `
  --stride 5 `
  --min-future-seconds 0.5 `
  --min-center-weight 1.0
```

输出包括：

- `aligned_keypoints.csv`：清洗并统一到视频画布坐标系的逐帧数据；
- `train.npz`、`val.npz`、`test.npz`：按序列隔离的训练窗口；
- `manifest.json`：处理参数、样本统计、缺失序列和跳过文件的原因。

数据尾部未来不足 3 秒时会使用有效掩码；补齐位置不参与损失和评估。`--min-future-seconds` 控制一个窗口至少必须拥有多少真实未来数据。

默认要求计算人体中心的四个关键点全部有效（`--min-center-weight 1.0`）。实验表明放宽该阈值虽然能增加窗口，但会因中心定义跳变明显降低验证和测试精度。

加入 `--strict` 后，只要推荐划分中的序列缺少数据，处理就会失败。默认模式会保留已有数据并把缺失项写进清单。

## 2. 单独训练

```powershell
python src\gru\train.py `
  --dataset-dir data\processed `
  --model-out models\trajectory_gru.pt `
  --epochs 100 `
  --batch-size 128
```

训练只读取 `train.npz`，模型选择只读取 `val.npz`。最佳权重、数据参数和归一化信息会一起写入模型文件；训练历史写入 `models/trajectory_gru.history.json`，最终独立测试结果写入 `models/trajectory_gru.report.json`。测试集不会参与训练或早停。

默认使用温和的数据源均衡采样，避免最大数据集完全主导训练；如需按原始窗口比例训练，可传入 `--sampling random`。

## 3. 单独预测

不提供视频时，只生成 JSON、CSV 和误差统计，不创建视频：

```powershell
python src\gru\predict.py `
  --keypoints examples\inputs\urfall_keypoints_adl-01-cam0.xlsx `
  --model models\trajectory_gru.pt `
  --output-dir outputs
```

如需叠加视频，额外传入 `--video`：

```powershell
python src\gru\predict.py `
  --keypoints examples\inputs\urfall_keypoints_adl-01-cam0.xlsx `
  --model models\trajectory_gru.pt `
  --output-dir outputs `
  --video examples\inputs\adl-01-cam0.mp4
```

预测始终生成：

- `module1_aligned.csv`
- `module1_frame_map.json`
- `module3_GRU_predict_result.json`
- `standard_pred_output.json`
- `standard_pred_output.csv`
- `pred_error_stat.csv`

只有传入 `--video` 时才额外生成 `prediction_visual.mp4`。训练与预测是两个独立入口，运行预测不会触发训练。

## 数据约定

输入至少需要 `frame_id`、`nosex/nosey`、`midhipx/midhipy`、`ranklex/rankley`、`lanklex/lankley`。可选的其他人体关键点会用于计算更准确的包围框。训练目标使用“相对最后一帧的位置 / 固定画布尺度”归一化，避免不同文件各自按最小最大值缩放造成不一致，也避免验证和测试数据参与归一化统计。

## 本次全量训练结果

已使用 `HAR-UP`、`LEI2FALL`、`mmFall`、`Pre-VFall` 和 `URFall` 的推荐分组划分完成训练：

- 训练集：1321 个序列组，112350 个窗口；
- 验证集：389 个序列组，19952 个窗口；
- 测试集：390 个序列组，18539 个窗口；
- 三套集合之间序列泄漏数：0；
- 最佳检查点：第 13 轮；
- 独立测试集：ADE 12.36 像素，FDE 17.99 像素。

正式模型位于 `models/trajectory_gru.pt`，完整指标位于 `models/trajectory_gru.report.json`，训练历史位于 `models/trajectory_gru.history.json`。已生成的演示输出位于 `examples/outputs`。
