"""将标准路径预测结果叠加到原始视频上。"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


# OpenCV 使用 BGR 颜色顺序。
COLOR_REAL = (255, 255, 255)
COLOR_PRED = (0, 0, 255)
COLOR_CORRIDOR = (255, 0, 0)


def _point(value: list[float], width: int, height: int) -> tuple[int, int]:
    """将浮点坐标限制在画面范围内，并转换为整数像素点。"""

    x = int(np.clip(round(value[0]), 0, width - 1))
    y = int(np.clip(round(value[1]), 0, height - 1))
    return x, y


def _draw_dashed_line(
    image,
    start,
    end,
    color,
    thickness=2,
    dash=5,
    gap=5,
) -> None:
    """按给定虚线长度和间隔绘制一条线段。"""

    delta = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    distance = float(np.linalg.norm(delta))
    if distance < 1e-6:
        return
    direction = delta / distance
    cursor = 0.0
    while cursor < distance:
        segment_end = min(cursor + dash, distance)
        p1 = tuple(np.rint(np.asarray(start) + cursor * direction).astype(int))
        p2 = tuple(np.rint(np.asarray(start) + segment_end * direction).astype(int))
        cv2.line(image, p1, p2, color, thickness)
        cursor += dash + gap


def render_visual_video(
    video_path: str | Path,
    prediction_path: str | Path,
    output_path: str | Path,
) -> Path:
    """逐帧绘制历史轨迹、预测路径和路径走廊，并输出 MP4 视频。"""

    video_path, prediction_path, output_path = map(
        Path,
        (video_path, prediction_path, output_path),
    )
    with prediction_path.open("r", encoding="utf-8") as handle:
        prediction = json.load(handle)
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for item in prediction.get("frames", []):
        by_frame[int(item["frame_index"])].append(item)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"无法打开视频：{video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"视频尺寸无效：{video_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"无法创建输出视频：{output_path}")

    history: dict[str, list[tuple[int, int]]] = defaultdict(list)
    frame_index = 0
    # 按视频帧号索引预测结果；每个人分别维护最近 300 个历史点。
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        for item in by_frame.get(frame_index, []):
            person_id = str(item.get("person_id", "person_1"))
            current = _point(item["current_position"], width, height)
            history[person_id].append(current)
            history[person_id] = history[person_id][-300:]
            if len(history[person_id]) > 1:
                history_points = np.asarray(history[person_id], np.int32)
                cv2.polylines(frame, [history_points], False, COLOR_REAL, 2)

            polygon = item.get("path_corridor", [])
            if len(polygon) >= 3:
                points = np.asarray([_point(value, width, height) for value in polygon], np.int32)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [points], COLOR_CORRIDOR)
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                cv2.polylines(frame, [points], True, COLOR_CORRIDOR, 2)

            predicted = [_point(value, width, height) for value in item.get("predicted_path", [])]
            for start, end in zip(predicted, predicted[1:]):
                _draw_dashed_line(frame, start, end, COLOR_PRED)
            cv2.circle(frame, current, 4, (0, 255, 255), -1)
        cv2.putText(
            frame,
            f"Frame: {frame_index}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            1,
        )
        writer.write(frame)
        frame_index += 1

    capture.release()
    writer.release()
    print(f"可视化视频：{output_path.resolve()}")
    return output_path


if __name__ == "__main__":
    raise SystemExit("请通过 predict.py --video <视频路径> 生成可选的可视化结果。")
