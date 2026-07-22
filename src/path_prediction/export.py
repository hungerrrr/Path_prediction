"""共用的 JSON、CSV 导出与视频可视化适配器。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


RECORD_FIELDS = [
    "video_path",
    "video_id",
    "frame_index",
    "timestamp_sec",
    "person_id",
    "has_prediction",
    "current_position",
    "predicted_activity_region",
    "predicted_trajectory",
    "path_confidence",
    "confidence_type",
    "confidence_details",
    "model_method",
    "candidate_paths",
]


def write_payload(
    payload: dict[str, Any],
    output_dir: str | Path,
    stem: str,
) -> tuple[Path, Path]:
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{stem}.json"
    csv_path = directory / f"{stem}.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECORD_FIELDS)
        writer.writeheader()
        for record in payload.get("records", []):
            row = {field: record.get(field, "") for field in RECORD_FIELDS}
            for field in (
                "predicted_activity_region",
                "current_position",
                "predicted_trajectory",
                "confidence_details",
                "candidate_paths",
            ):
                row[field] = json.dumps(row[field] or [], ensure_ascii=False)
            writer.writerow(row)
    return json_path, csv_path


def render_records_video(
    video_path: str | Path,
    records: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """将统一记录转换为视频渲染器所需的数据格式。"""

    from gru.video_visualizer import render_visual_video

    output_path = Path(output_path).resolve()
    compatibility_path = output_path.with_suffix(".visualizer.json")
    frames = []
    for record in records:
        if not record.get("has_prediction", True):
            continue
        frames.append({
            "frame_index": record["frame_index"],
            "person_id": record["person_id"],
            "current_position": record.get("current_position", [0.0, 0.0]),
            "predicted_path": record.get("predicted_trajectory", []),
            "path_corridor": record.get(
                "predicted_activity_region", {}
            ).get("points", []),
            "confidence": record.get("path_confidence", 0.0),
        })
    with compatibility_path.open("w", encoding="utf-8") as handle:
        json.dump({"frames": frames}, handle, ensure_ascii=False)
    try:
        return render_visual_video(video_path, compatibility_path, output_path)
    finally:
        compatibility_path.unlink(missing_ok=True)
