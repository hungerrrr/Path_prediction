# 路径预测模型与训练

项目支持 GRU、低秩多模态 Transformer 和 Leapfrog Diffusion。正式双入口通过
`src/path_prediction/config.py` 选择模型；本页记录各模型的定位、训练和底层独立预测方式。

以下多行命令使用 PowerShell 续行符；在其他终端中执行时，只需替换续行符，参数与
相对路径保持不变。

## 文件对应关系

| 功能 | 方案 A：低秩 Transformer | 方案 B：Leapfrog Diffusion |
|---|---|---|
| 训练入口 | `src/eigen_transformer/train.py` | `src/leapfrog_diffusion/train.py` |
| 预测入口 | `src/eigen_transformer/predict.py` | `src/leapfrog_diffusion/predict.py` |
| 默认模型 | `models/eigen_transformer.pt` | `models/leapfrog_diffusion.pt` |
| 默认输出 | `outputs/transformer` | `outputs/diffusion` |

## GRU

GRU 是当前正式样例默认模型，确定性输出一条主轨迹，并通过 MC Dropout 与验证集 ADE
生成逐轨迹置信度。

```powershell
python src/gru/train.py `
  --dataset-dir data/processed `
  --model-out models/trajectory_gru.pt
```

## 增强版对齐数据

生成命令：

```powershell
python src/build_aligned_dataset.py `
  --data-root "<骨架数据目录>" `
  --split-file "<数据划分文件>" `
  --output data/processed/aligned_keypoints.csv
```

增强文件保留：

- 7 个原始元信息字段；
- 25 个 BODY25 骨架点的 50 个坐标字段；
- 32 个 gd/gm 派生向量；
- `video_person_id`、`global_person_id` 和统一的 `person_id`；
- 数据划分、序列组、帧率、人体中心、包围框及中心计算方式。

## 方案 A

```powershell
python src/eigen_transformer/train.py `
  --dataset-dir data/processed `
  --model-out models/eigen_transformer.pt `
  --num-modes 6 `
  --basis-rank 16
```

```powershell
python src/eigen_transformer/predict.py `
  --keypoints examples/inputs/urfall_keypoints_adl-01-cam0.xlsx `
  --model models/eigen_transformer.pt `
  --output-dir outputs/transformer
```

## 方案 B

```powershell
python src/leapfrog_diffusion/train.py `
  --dataset-dir data/processed `
  --model-out models/leapfrog_diffusion.pt `
  --diffusion-steps 50 `
  --sample-steps 8 `
  --num-samples 6
```

```powershell
python src/leapfrog_diffusion/predict.py `
  --keypoints examples/inputs/urfall_keypoints_adl-01-cam0.xlsx `
  --model models/leapfrog_diffusion.pt `
  --output-dir outputs/diffusion `
  --num-samples 6
```

两套预测输出都继续提供 `predicted_path` 作为 Top-1 结果，以兼容现有
视频可视化；同时新增 `candidate_paths`，保存候选轨迹和对应概率。
