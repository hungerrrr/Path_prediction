"""面向单个视频帧的一次性路径预测服务。"""

from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Any

import torch

from .window_predictor import WindowPredictor

from .config import PathPredictionSettings, portable_path
from .control import path_prediction_enabled
from .skeleton_provider import FileSkeletonHistoryProvider


class FramePredictionService:
    """模型仅加载一次，但每次调用都查询最新的骨架文件。"""

    def __init__(
        self,
        settings: PathPredictionSettings,
        provider: FileSkeletonHistoryProvider,
        control_config_path: str | Path,
    ) -> None:
        settings.validate()
        self.settings = settings
        self.provider = provider
        self.control_config_path = Path(control_config_path).resolve()
        self._predictor: WindowPredictor | None = None
        self._lock = threading.RLock()

    def _get_predictor(self) -> WindowPredictor:
        if self._predictor is None:
            self._predictor = WindowPredictor.from_checkpoint(
                self.settings.model_path,
                self.settings.model_method,
                device=self.settings.device,
                num_samples=(
                    self.settings.confidence_samples
                    if self.settings.model_method == "gru"
                    else self.settings.multimodal_samples
                ),
                max_persons=self.settings.max_persons,
                stale_frame_threshold=self.settings.stale_frame_threshold,
                max_frame_gap=self.settings.max_frame_gap,
                confidence_scale_px=self.settings.confidence_scale_px,
            )
            actual_obs = self._predictor.config.obs_seconds
            actual_pred = self._predictor.config.pred_seconds
            if not math.isclose(actual_obs, self.settings.observation_seconds):
                raise ValueError(
                    f"模型 observation_seconds={actual_obs} 与内部配置 "
                    f"{self.settings.observation_seconds} 不一致；请重新训练或更换模型"
                )
            if not math.isclose(actual_pred, self.settings.prediction_seconds):
                raise ValueError(
                    f"模型 prediction_seconds={actual_pred} 与内部配置 "
                    f"{self.settings.prediction_seconds} 不一致；请重新训练或更换模型"
                )
        return self._predictor

    def predict(
        self,
        video_path: str | Path,
        video_id: str,
        frame_index: int,
        timestamp_sec: float,
        skeleton_path: str | Path | None = None,
    ) -> dict[str, Any]:
        video_path = Path(video_path).resolve()
        video_id = str(video_id).strip()
        if not video_path.is_file():
            raise FileNotFoundError(f"视频文件不存在：{video_path}")
        if not video_id:
            raise ValueError("video_id 不能为空")
        if not isinstance(frame_index, int) or frame_index < 0:
            raise ValueError("frame_index 必须是非负整数")
        if not math.isfinite(float(timestamp_sec)) or timestamp_sec < 0:
            raise ValueError("timestamp_sec 必须是非负有限数值")

        base = {
            "schema_version": "path_prediction.frame.v1",
            "module": "path_prediction",
            "video_path": portable_path(video_path),
            "video_id": video_id,
            "frame_index": frame_index,
            "timestamp_sec": float(timestamp_sec),
            "records": [],
        }
        if not path_prediction_enabled(self.control_config_path):
            return {**base, "enabled": False, "status": "disabled"}

        skeleton_file, history = self.provider.history_at(
            video_id,
            frame_index,
            float(timestamp_sec),
            skeleton_path,
        )
        if history.empty:
            return {
                **base,
                "enabled": True,
                "status": "skeleton_not_ready",
                "skeleton_path": portable_path(skeleton_file),
            }

        with self._lock:
            torch.manual_seed(self.settings.confidence_seed + frame_index)
            predictor = self._get_predictor()
            predictor.reset()
            final_records: list[dict[str, Any]] = []
            for current_frame, rows in history.groupby("frame_index", sort=True):
                row_timestamp = float(rows["timestamp_sec"].max())
                if int(current_frame) == frame_index:
                    row_timestamp = float(timestamp_sec)
                persons = []
                for _, row in rows.iterrows():
                    center = [float(row["human_center_x"]), float(row["human_center_y"])]
                    persons.append({
                        "person_id": str(row["person_id"]),
                        "keypoints": {
                            "nose": center,
                            "midhip": center,
                            "rankle": center,
                            "lankle": center,
                        },
                        "bbox": [
                            float(row["bbox_xmin"]),
                            float(row["bbox_ymin"]),
                            float(row["bbox_xmax"]),
                            float(row["bbox_ymax"]),
                        ],
                        "source": portable_path(skeleton_file),
                        "detection_confidence": 1.0,
                    })
                records = predictor.update_unified({
                    "video_path": portable_path(video_path),
                    "video_id": video_id,
                    "frame_index": int(current_frame),
                    "timestamp_sec": row_timestamp,
                    "persons": persons,
                })
                if int(current_frame) == frame_index:
                    final_records = records

        status = (
            "ok"
            if final_records and all(item["has_prediction"] for item in final_records)
            else "insufficient_history"
        )
        return {
            **base,
            "enabled": True,
            "status": status,
            "skeleton_path": portable_path(skeleton_file),
            "model_method": self.settings.model_method,
            "observation_seconds": self.settings.observation_seconds,
            "prediction_seconds": self.settings.prediction_seconds,
            "records": final_records,
        }
