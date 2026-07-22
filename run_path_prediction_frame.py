"""单帧入口：按视频 ID 查询骨架历史并执行一次预测。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from path_prediction.config import SETTINGS  # noqa: E402
from path_prediction.export import render_records_video, write_payload  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取当前帧之前的骨架历史，并执行一次路径预测。"
    )
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--video-path", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--frame-index", type=int, required=True)
    parser.add_argument("--timestamp-sec", type=float, required=True)
    parser.add_argument("--skeleton-path", type=Path)
    parser.add_argument(
        "--skeleton-dir", type=Path, default=SETTINGS.skeleton_directory
    )
    parser.add_argument(
        "--output-dir", type=Path, default=SETTINGS.frame_output_directory
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    from path_prediction.frame_service import FramePredictionService
    from path_prediction.skeleton_provider import FileSkeletonHistoryProvider

    provider = FileSkeletonHistoryProvider(args.skeleton_dir, SETTINGS)
    service = FramePredictionService(SETTINGS, provider, args.config)
    payload = service.predict(
        video_path=args.video_path,
        video_id=args.video_id,
        frame_index=args.frame_index,
        timestamp_sec=args.timestamp_sec,
        skeleton_path=args.skeleton_path,
    )
    outputs = write_payload(
        payload, args.output_dir, "path_prediction_frame_output"
    )
    if payload["status"] == "ok":
        render_records_video(
            args.video_path,
            payload["records"],
            args.output_dir / "frame_prediction_visual.mp4",
        )
    print(f"单帧 JSON：{outputs[0]}")
    print(f"单帧 CSV：{outputs[1]}")
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
