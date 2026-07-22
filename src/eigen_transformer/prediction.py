"""方案 A/B 共享的逐帧多模态预测和结果导出流程。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models import EigenTrajectoryTransformer
from core import (
    DataConfig,
    read_table,
    split_contiguous_tracks,
    standardize_keypoints,
)


def add_common_arguments(
    parser: argparse.ArgumentParser,
    default_model: Path,
    default_output: Path,
) -> None:
    """为两套预测入口添加一致的参数。"""

    parser.add_argument("--keypoints", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=default_model)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--video-out", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="方案 B 的采样轨迹数；省略时使用检查点配置",
    )


def corridor(
    paths: np.ndarray,
    bbox: np.ndarray,
    width: int,
    height: int,
    margin: float = 8.0,
) -> list[list[int]]:
    """根据全部候选轨迹和当前人体框生成不确定性走廊。"""

    x1, y1, x2, y2 = bbox.astype(float)
    body_width = max(x2 - x1, 1.0)
    body_height = max(y2 - y1, 1.0)
    boxes = [[x1, y1, x2, y2]]
    for x, y in paths.reshape(-1, 2):
        boxes.append(
            [
                x - body_width / 2,
                y - body_height / 2,
                x + body_width / 2,
                y + body_height / 2,
            ]
        )
    values = np.asarray(boxes)
    left = max(values[:, 0].min() - margin, 0)
    top = max(values[:, 1].min() - margin, 0)
    right = min(values[:, 2].max() + margin, width - 1)
    bottom = min(values[:, 3].max() + margin, height - 1)
    return [
        [round(left), round(top)],
        [round(right), round(top)],
        [round(right), round(bottom)],
        [round(left), round(bottom)],
    ]


def load_advanced_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict, str]:
    """仅加载方案 A 的 EigenTrajectory Transformer 检查点。"""

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    schema = str(checkpoint.get("schema_version", ""))
    if not schema.startswith("eigen_transformer"):
        raise ValueError(f"方案 A 不支持该检查点: {schema}")
    model = EigenTrajectoryTransformer(
        basis_mean=checkpoint["basis_mean"],
        basis_components=checkpoint["basis_components"],
        **checkpoint["model_config"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint, "eigen_transformer"


def infer_candidates(
    model: torch.nn.Module,
    method: str,
    observations: np.ndarray,
    device: torch.device,
    num_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """执行方案 A 多模态推理。"""

    del method, num_samples
    model_input = torch.from_numpy(observations).unsqueeze(0).to(device)
    with torch.no_grad():
        paths, logits = model(model_input)
        probabilities = torch.softmax(logits, dim=1)
    return paths.squeeze(0).cpu().numpy(), probabilities.squeeze(0).cpu().numpy()


def run_prediction(
    args: argparse.Namespace,
    expected_method: str,
) -> None:
    """执行对齐、逐帧预测、标准结果导出和可选视频渲染。"""

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前环境不可用")
    use_cuda = args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    )
    device = torch.device("cuda" if use_cuda else "cpu")
    model, checkpoint, method = load_advanced_checkpoint(args.model, device)
    if method != expected_method:
        raise ValueError(
            f"入口要求 {expected_method}，检查点实际为 {method}"
        )

    torch.manual_seed(int(checkpoint.get("seed", 42)))
    config_data = checkpoint["data_config"]
    config = DataConfig(
        fps=float(config_data["fps"]),
        obs_seconds=float(config_data["obs_seconds"]),
        pred_seconds=float(config_data["pred_seconds"]),
        canvas_width=int(config_data["canvas_width"]),
        canvas_height=int(config_data["canvas_height"]),
        x_offset=float(config_data["x_offset"]),
        coordinate_scale=float(config_data["coordinate_scale"]),
    )
    scale = np.asarray(
        checkpoint["normalization"]["scale"],
        dtype=np.float32,
    )
    default_samples = int(
        checkpoint.get(
            "num_samples",
            checkpoint["model_config"].get("num_modes", 6),
        )
    )
    num_samples = args.num_samples or default_samples
    if num_samples < 1:
        raise ValueError("--num-samples 必须大于 0")

    aligned = standardize_keypoints(
        read_table(args.keypoints),
        args.keypoints,
        config,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(
        args.output_dir / "module1_aligned.csv",
        index=False,
        encoding="utf-8-sig",
    )

    samples = []
    standard_frames = []
    errors = []
    candidate_count = 0
    for track in split_contiguous_tracks(aligned):
        centers = track[["human_center_x", "human_center_y"]].to_numpy(
            np.float32
        )
        bboxes = track[
            ["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]
        ].to_numpy(np.float32)

        for index in range(len(track)):
            observations = centers[
                max(0, index - config.obs_frames + 1) : index + 1
            ]
            if len(observations) < config.obs_frames:
                observations = np.concatenate(
                    [
                        np.repeat(
                            observations[:1],
                            config.obs_frames - len(observations),
                            axis=0,
                        ),
                        observations,
                    ],
                    axis=0,
                )
            anchor = observations[-1:]
            normalized = (observations - anchor) / scale
            candidate_paths, probabilities = infer_candidates(
                model,
                method,
                normalized.astype(np.float32),
                device,
                num_samples,
            )
            candidate_count = int(len(candidate_paths))
            candidate_paths = anchor[None] + candidate_paths * scale
            candidate_paths[..., 0] = np.clip(
                candidate_paths[..., 0],
                0,
                config.canvas_width - 1,
            )
            candidate_paths[..., 1] = np.clip(
                candidate_paths[..., 1],
                0,
                config.canvas_height - 1,
            )
            order = np.argsort(probabilities)[::-1]
            candidate_paths = candidate_paths[order]
            probabilities = probabilities[order]
            best_path = candidate_paths[0]
            row = track.iloc[index]
            truth = centers[index + 1 : index + 1 + config.pred_frames]
            pixel_mse = None
            if len(truth) == config.pred_frames:
                pixel_mse = float(np.mean(np.square(best_path - truth)))
                errors.append(
                    {
                        "frame_index": int(row["frame_index"]),
                        "person_id": row["person_id"],
                        "mse_pixel": round(pixel_mse, 3),
                    }
                )

            path_corridor = corridor(
                candidate_paths,
                bboxes[index],
                config.canvas_width,
                config.canvas_height,
            )
            candidates = [
                {
                    "probability": round(float(probability), 6),
                    "path": path.tolist(),
                }
                for path, probability in zip(candidate_paths, probabilities)
            ]
            frame_record = {
                "frame_index": int(row["frame_index"]),
                "timestamp_sec": round(float(row["timestamp_sec"]), 4),
                "person_id": row["person_id"],
                "current_position": anchor[0].tolist(),
                "predicted_path": best_path.tolist(),
                "candidate_paths": candidates,
                "path_corridor": path_corridor,
                "confidence": round(float(probabilities[0]), 6),
            }
            standard_frames.append(frame_record)
            samples.append(
                {
                    **frame_record,
                    "true_future": truth.tolist(),
                    "pixel_mse": (
                        None if pixel_mse is None else round(pixel_mse, 3)
                    ),
                }
            )

    standard_frames.sort(
        key=lambda item: (item["frame_index"], item["person_id"])
    )
    standard = {
        "schema_version": "path_prediction.multimodal.v1",
        "model_method": method,
        "coordinate_system": "original_video_pixels",
        "video": args.video.name if args.video else args.keypoints.stem + ".mp4",
        "frames": standard_frames,
    }
    average_mse = (
        round(float(np.mean([item["mse_pixel"] for item in errors])), 3)
        if errors
        else None
    )
    raw = {
        "source_dataset": str(args.keypoints.resolve()),
        "predict_method": method,
        "fps": config.fps,
        "obs_sec": config.obs_seconds,
        "pred_sec": config.pred_seconds,
        "candidate_count": candidate_count,
        "avg_top1_pixel_mse": average_mse,
        "sample_count": len(samples),
        "predict_samples": samples,
    }

    with (args.output_dir / "advanced_predict_result.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(raw, handle, ensure_ascii=False, indent=2)
    with (args.output_dir / "standard_pred_output.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(standard, handle, ensure_ascii=False, indent=2)

    csv_rows = [
        {
            **item,
            "current_position": json.dumps(item["current_position"]),
            "predicted_path": json.dumps(item["predicted_path"]),
            "candidate_paths": json.dumps(item["candidate_paths"]),
            "path_corridor": json.dumps(item["path_corridor"]),
        }
        for item in standard_frames
    ]
    pd.DataFrame(csv_rows).to_csv(
        args.output_dir / "standard_pred_output.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        errors,
        columns=["frame_index", "person_id", "mse_pixel"],
    ).to_csv(
        args.output_dir / "pred_error_stat.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"预测结果: {args.output_dir.resolve()}")

    if args.video:
        if not args.video.is_file():
            raise FileNotFoundError(f"视频不存在: {args.video}")
        from video_visualizer import render_visual_video

        video_output = (
            args.video_out
            or args.output_dir / "prediction_visual.mp4"
        )
        render_visual_video(
            args.video,
            args.output_dir / "standard_pred_output.json",
            video_output,
        )
