"""读取关键点并执行路径预测；视频可视化为可选步骤。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from core import (
    DataConfig,
    load_checkpoint,
    read_table,
    split_contiguous_tracks,
    standardize_keypoints,
)


def parse_args() -> argparse.Namespace:
    """解析预测输入、模型、设备和可视化参数。"""

    parser = argparse.ArgumentParser(description="Predict paths with a trained model.")
    parser.add_argument(
        "--keypoints",
        type=Path,
        required=True,
        help="一份 xlsx/csv/parquet 关键点表格",
    )
    parser.add_argument("--model", type=Path, default=Path("models/trajectory_gru.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--video",
        type=Path,
        help="可选的原始视频；省略时不生成可视化视频",
    )
    parser.add_argument("--video-out", type=Path, help="Optional output video path")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def corridor(
    path: np.ndarray,
    bbox: np.ndarray,
    width: int,
    height: int,
    margin: float = 8.0,
) -> list[list[int]]:
    """根据当前人体框和预测轨迹生成覆盖未来运动范围的矩形走廊。"""

    x1, y1, x2, y2 = bbox.astype(float)
    body_w, body_h = max(x2 - x1, 1), max(y2 - y1, 1)
    boxes = [[x1, y1, x2, y2]]
    boxes.extend(
        [
            [x - body_w / 2, y - body_h / 2, x + body_w / 2, y + body_h / 2]
            for x, y in path
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


def infer_video_name(keypoints: Path) -> str:
    """根据关键点文件名推断原始视频文件名。"""

    stem = keypoints.stem
    for prefix in ("urfall_keypoints_", "keypoints_"):
        if stem.lower().startswith(prefix):
            stem = stem[len(prefix):]
            break
    return f"{stem}.mp4"


def main() -> None:
    """完成关键点对齐、逐帧推理、结果导出和可选视频渲染。"""

    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    use_cuda = args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    )
    device = torch.device("cuda" if use_cuda else "cpu")
    model, checkpoint = load_checkpoint(args.model, device)
    cfg = checkpoint["data_config"]
    config = DataConfig(
        fps=float(cfg["fps"]),
        obs_seconds=float(cfg["obs_seconds"]),
        pred_seconds=float(cfg["pred_seconds"]),
        canvas_width=int(cfg["canvas_width"]), canvas_height=int(cfg["canvas_height"]),
        x_offset=float(cfg["x_offset"]), coordinate_scale=float(cfg["coordinate_scale"]),
    )
    scale = np.asarray(checkpoint["normalization"]["scale"], dtype=np.float32)
    aligned = standardize_keypoints(read_table(args.keypoints), args.keypoints, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(args.output_dir / "module1_aligned.csv", index=False, encoding="utf-8-sig")
    frame_map = {
        str(int(frame)): round(int(frame) / config.fps, 6)
        for frame in sorted(aligned["frame_index"].unique())
    }
    with (args.output_dir / "module1_frame_map.json").open("w", encoding="utf-8") as handle:
        json.dump(frame_map, handle, ensure_ascii=False, indent=2)

    validation_ade = float(checkpoint.get("validation_metrics", {}).get("ade_pixel", 50.0))
    confidence = round(float(math.exp(-validation_ade / 100.0)), 3)
    raw_samples, standard_frames, errors = [], [], []

    # 每个连续人物轨迹独立推理，避免跨人物或跨缺帧片段拼接历史。
    for track in split_contiguous_tracks(aligned):
        centers = track[["human_center_x", "human_center_y"]].to_numpy(np.float32)
        bboxes = track[["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]].to_numpy(np.float32)
        for index in range(len(track)):
            obs = centers[max(0, index - config.obs_frames + 1):index + 1]
            # 观测不足时在序列头部复制首个位置，保持模型输入长度固定。
            if len(obs) == 1:
                padded = np.repeat(obs, config.obs_frames, axis=0)
            elif len(obs) < config.obs_frames:
                padded = np.concatenate(
                    [
                        np.repeat(obs[:1], config.obs_frames - len(obs), axis=0),
                        obs,
                    ],
                    axis=0,
                )
            else:
                padded = obs
            normalized = (padded - padded[-1:]) / scale
            # 模型输出的是相对位移，需要乘尺度后加回最后一个观测点。
            with torch.no_grad():
                model_input = torch.from_numpy(normalized).unsqueeze(0).to(device)
                output = model(model_input).squeeze(0).cpu().numpy()
            predicted = padded[-1:] + output * scale
            predicted[:, 0] = np.clip(predicted[:, 0], 0, config.canvas_width - 1)
            predicted[:, 1] = np.clip(predicted[:, 1], 0, config.canvas_height - 1)
            row = track.iloc[index]
            poly = corridor(predicted, bboxes[index], config.canvas_width, config.canvas_height)
            truth = centers[index + 1:index + 1 + config.pred_frames]
            mse = None
            if len(truth) == config.pred_frames:
                mse = float(np.mean(np.square(predicted - truth)))
                errors.append(
                    {
                        "frame_index": int(row["frame_index"]),
                        "person_id": row["person_id"],
                        "mse_pixel": round(mse, 3),
                    }
                )
            raw_samples.append({
                "window_start": int(
                    track.iloc[max(0, index - config.obs_frames + 1)]["frame_index"]
                ),
                "window_end": int(row["frame_index"]),
                "person_id": row["person_id"],
                "action_label": row["label_name"],
                "obs_raw": obs.tolist(),
                "obs_smooth": obs.tolist(),
                "predict_path": predicted.tolist(),
                "path_corridor": poly,
                "true_future": truth.tolist(), "pixel_mse": None if mse is None else round(mse, 3),
            })
            standard_frames.append({
                "frame_index": int(row["frame_index"]),
                "timestamp_sec": round(int(row["frame_index"]) / config.fps, 4),
                "person_id": row["person_id"], "current_position": padded[-1].tolist(),
                "predicted_path": predicted.tolist(),
                "path_corridor": poly,
                "confidence": confidence,
            })

    standard_frames.sort(key=lambda item: (item["frame_index"], item["person_id"]))
    video_name = args.video.name if args.video else infer_video_name(args.keypoints)
    standard = {
        "schema_version": "path_prediction.v1",
        "coordinate_system": "original_video_pixels",
        "video": video_name,
        "frames": standard_frames,
    }
    average_mse = (
        round(float(np.mean([item["mse_pixel"] for item in errors])), 3)
        if errors
        else None
    )
    raw = {
        "source_dataset": str(args.keypoints.resolve()),
        "predict_method": "GRU relative-displacement v2",
        "fps": config.fps, "obs_sec": config.obs_seconds, "pred_sec": config.pred_seconds,
        "avg_pixel_mse": average_mse,
        "sample_count": len(raw_samples),
        "predict_samples": raw_samples,
    }
    with (args.output_dir / "module3_GRU_predict_result.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(raw, handle, ensure_ascii=False, indent=2)
    with (args.output_dir / "standard_pred_output.json").open("w", encoding="utf-8") as handle:
        json.dump(standard, handle, ensure_ascii=False, indent=2)
    csv_rows = [
        {
            **item,
            "current_position": json.dumps(item["current_position"]),
            "predicted_path": json.dumps(item["predicted_path"]),
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
    print(f"Prediction outputs: {args.output_dir.resolve()}")

    if args.video:
        if not args.video.is_file():
            raise FileNotFoundError(f"video does not exist: {args.video}")
        from video_visualizer import render_visual_video
        video_out = args.video_out or args.output_dir / "prediction_visual.mp4"
        render_visual_video(args.video, args.output_dir / "standard_pred_output.json", video_out)
    else:
        print("No --video supplied; skipped visualization.")


if __name__ == "__main__":
    main()
