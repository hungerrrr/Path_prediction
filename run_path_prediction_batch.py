"""批量入口：对一个完整骨架文件中的全部可用帧执行预测。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from path_prediction.config import SETTINGS, portable_path  # noqa: E402
from path_prediction.control import path_prediction_enabled  # noqa: E402
from path_prediction.export import write_payload  # noqa: E402


METHODS = {
    "gru": ("src/gru/predict.py", "models/trajectory_gru.pt"),
    "eigen_transformer": (
        "src/eigen_transformer/predict.py", "models/eigen_transformer.pt"
    ),
    "leapfrog_diffusion": (
        "src/leapfrog_diffusion/predict.py", "models/leapfrog_diffusion.pt"
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="根据完整骨架文件执行批量路径预测。"
    )
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--video-path", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--skeleton-path", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=SETTINGS.batch_output_directory
    )
    return parser.parse_args(argv)


def _legacy_command(args: argparse.Namespace) -> list[str]:
    settings = SETTINGS
    settings.validate()
    script, default_model = METHODS[settings.model_method]
    model_path = settings.model_path or PROJECT_ROOT / default_model
    command = [
        sys.executable,
        str(PROJECT_ROOT / script),
        "--keypoints", str(args.skeleton_path.resolve()),
        "--model", str(model_path),
        "--output-dir", str(args.output_dir.resolve()),
        "--video", str(args.video_path.resolve()),
        "--video-out", str(args.output_dir.resolve() / "batch_prediction_visual.mp4"),
        "--device", settings.device,
    ]
    if settings.model_method == "gru":
        command.extend([
            "--confidence-samples", str(settings.confidence_samples),
            "--confidence-scale-px", str(settings.confidence_scale_px),
            "--confidence-seed", str(settings.confidence_seed),
        ])
    else:
        command.extend(["--num-samples", str(settings.multimodal_samples)])
    return command


def _normalize(args: argparse.Namespace) -> dict:
    standard_path = args.output_dir.resolve() / "standard_pred_output.json"
    with standard_path.open(encoding="utf-8") as handle:
        standard = json.load(handle)
    confidence_type = (
        "mc_dropout_ade_v1"
        if SETTINGS.model_method == "gru"
        else "model_candidate_score"
    )
    records = []
    for item in standard.get("frames", []):
        record = {
            "video_path": portable_path(args.video_path),
            "video_id": args.video_id,
            "frame_index": int(item["frame_index"]),
            "timestamp_sec": float(item["timestamp_sec"]),
            "person_id": str(item["person_id"]),
            "has_prediction": True,
            "current_position": item.get("current_position", []),
            "predicted_activity_region": {
                "type": "polygon",
                "coordinate_system": standard.get(
                    "coordinate_system", "model_canvas_pixels"
                ),
                "points": item.get("path_corridor", []),
            },
            "predicted_trajectory": item.get("predicted_path", []),
            "path_confidence": float(item.get("confidence", 0.0)),
            "confidence_type": confidence_type,
            "confidence_details": item.get("confidence_details", {}),
            "model_method": SETTINGS.model_method,
            "candidate_paths": item.get("candidate_paths", []),
        }
        records.append(record)
    return {
        "schema_version": "path_prediction.batch.v1",
        "module": "path_prediction",
        "mode": "batch",
        "enabled": True,
        "status": "ok",
        "video_path": portable_path(args.video_path),
        "video_id": args.video_id,
        "skeleton_path": portable_path(args.skeleton_path),
        "model_method": SETTINGS.model_method,
        "observation_seconds": SETTINGS.observation_seconds,
        "prediction_seconds": SETTINGS.prediction_seconds,
        "records": records,
    }


def _cleanup_legacy(output_dir: Path) -> None:
    if SETTINGS.keep_legacy_outputs:
        return
    for name in (
        "module1_aligned.csv", "module1_frame_map.json",
        "module3_GRU_predict_result.json", "advanced_predict_result.json",
        "pred_error_stat.csv", "standard_pred_output.json",
        "standard_pred_output.csv",
    ):
        path = output_dir / name
        if path.is_file():
            path.unlink()


def run(args: argparse.Namespace) -> tuple[Path, Path] | None:
    if not path_prediction_enabled(args.config):
        print("config.json 已关闭路径预测模块，本次任务已跳过。")
        return None
    if not args.video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在：{args.video_path}")
    if not args.skeleton_path.is_file():
        raise FileNotFoundError(f"骨架文件不存在：{args.skeleton_path}")
    if not args.video_id.strip():
        raise ValueError("video_id 不能为空")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(_legacy_command(args), cwd=PROJECT_ROOT, check=True)
    payload = _normalize(args)
    outputs = write_payload(payload, args.output_dir, "path_prediction_batch_output")
    _cleanup_legacy(args.output_dir.resolve())
    print(f"批量 JSON：{outputs[0]}")
    print(f"批量 CSV：{outputs[1]}")
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
