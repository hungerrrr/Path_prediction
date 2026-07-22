"""供单帧推理服务使用的内部观察窗口预测器。"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn

CENTER_POINTS = ("nose", "midhip", "rankle", "lankle")
CENTER_WEIGHTS = np.asarray([0.3, 0.5, 0.1, 0.1], dtype=np.float32)
SUPPORTED_METHODS = (
    "gru",
    "eigen_transformer",
    "leapfrog_diffusion",
)
STATE_SCHEMA_VERSION = "window_predictor_state.v1"


# ─── 数据契约 ────────────────────────────────────────


@dataclass(frozen=True)
class WindowConfig:
    """从检查点恢复的最小流式数据配置。"""

    fps: float
    obs_seconds: float
    pred_seconds: float
    canvas_width: int
    canvas_height: int
    x_offset: float
    coordinate_scale: float

    @property
    def obs_frames(self) -> int:
        return int(round(self.obs_seconds * self.fps))

    @property
    def pred_frames(self) -> int:
        return int(round(self.pred_seconds * self.fps))


class JsonDataclassMixin:
    """为对外数据契约提供可直接交给 JSON 编码器的字典。"""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PersonFrameData(JsonDataclassMixin):
    """单帧中单人的姿态数据，来自上游姿态估计模块。"""
    frame_id: int
    timestamp: float
    person_id: str
    keypoints: dict[str, tuple[float | None, float | None]]
    bbox: tuple[float, float, float, float]
    source: str
    detection_confidence: float


@dataclass
class FrameData(JsonDataclassMixin):
    """单帧中所有检测到的人物。"""
    frame_id: int
    timestamp: float
    persons: list[PersonFrameData]


@dataclass
class CandidatePath(JsonDataclassMixin):
    """一条候选轨迹及其概率。"""
    probability: float
    path: list[tuple[float, float]]


@dataclass
class PredictionResult(JsonDataclassMixin):
    """单帧单人的路径预测结果。"""
    frame_id: int
    timestamp: float
    person_id: str
    has_prediction: bool
    current_position: tuple[float, float]
    predicted_path: list[tuple[float, float]]
    candidate_paths: list[CandidatePath]
    path_corridor: list[tuple[float, float]]
    confidence: float
    model_method: str
    confidence_details: dict[str, Any] | None = None


# ─── 流式预测器 ─────────────────────────────────────


class WindowPredictor:
    """有状态的流式路径预测器。

    对每个被跟踪人物维护独立的观测历史缓冲区，
    每输入一帧关键点数据，立即输出该帧的路径预测。

    Parameters
    ----------
    model : nn.Module
        已加载的 PyTorch 模型（evaluation mode）。
    method : str
        模型类型: "gru" / "eigen_transformer" / "leapfrog_diffusion"。
    config : DataConfig
        数据处理配置（obs_frames, pred_frames, 画布尺寸等）。
    normalization_scale : np.ndarray
        归一化尺度，通常为 [canvas_width/2, canvas_height]。
    device : torch.device
        推理设备。
    num_samples : int
        多模态模型的候选轨迹采样数（GRU 忽略此参数）。
    max_persons : int
        最大并行跟踪人数，超出时自动淘汰最旧人物。
    stale_frame_threshold : int
        人物连续多少帧未出现即视为离开并清理缓冲区。
    max_frame_gap : int
        同一人物允许的最大连续帧间隔；超过后清空该人物历史。
    gru_confidence : float
        GRU 的固定置信度（由 from_checkpoint 自动计算）。
    """

    def __init__(
        self,
        model: nn.Module,
        method: Literal["gru", "eigen_transformer", "leapfrog_diffusion"],
        config: WindowConfig,
        normalization_scale: np.ndarray,
        device: torch.device,
        num_samples: int = 6,
        max_persons: int = 50,
        stale_frame_threshold: int = 30,
        max_frame_gap: int = 1,
        gru_confidence: float = 0.6,
        gru_validation_ade: float | None = None,
        confidence_scale_px: float = 100.0,
    ) -> None:
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"不支持的模型方法: {method}")
        if max_persons < 1:
            raise ValueError("max_persons 必须大于 0")
        if stale_frame_threshold < 1:
            raise ValueError("stale_frame_threshold 必须大于 0")
        if max_frame_gap < 1:
            raise ValueError("max_frame_gap 必须大于 0")
        if num_samples < 1:
            raise ValueError("num_samples 必须大于 0")
        if confidence_scale_px <= 0:
            raise ValueError("confidence_scale_px 必须大于 0")
        scale = np.asarray(normalization_scale, dtype=np.float32)
        if scale.shape != (2,) or not np.all(np.isfinite(scale)):
            raise ValueError("normalization_scale 必须是两个有限数值")
        if np.any(scale <= 0):
            raise ValueError("normalization_scale 必须大于 0")

        self.model = model
        self.method = method
        self.config = config
        self.scale = scale
        self.device = device
        self.num_samples = num_samples
        self.max_persons = max_persons
        self.stale_frame_threshold = stale_frame_threshold
        self.max_frame_gap = max_frame_gap
        self.gru_confidence = gru_confidence
        self.gru_validation_ade = (
            float(gru_validation_ade)
            if gru_validation_ade is not None
            else float(-100.0 * math.log(max(gru_confidence, 1e-8)))
        )
        self.confidence_scale_px = float(confidence_scale_px)
        self._last_confidence_details: dict[str, Any] | None = None
        self._last_uncertainty_paths_norm: np.ndarray | None = None

        self._buffers: dict[str, deque] = {}
        self._bboxes: dict[str, np.ndarray] = {}
        self._last_seen: dict[str, int] = {}
        self._lock = threading.RLock()

    # ─── 公开接口 ─────────────────────────────────────

    def update(self, frame_data: FrameData) -> list[PredictionResult]:
        """接收一帧多人数据，并按输入顺序返回对应预测结果。"""

        with self._lock:
            self._validate_frame_data(frame_data)
            active_ids = {person.person_id for person in frame_data.persons}
            results = [
                self._update_single(person)
                for person in frame_data.persons
            ]
            self._evict_stale(active_ids, frame_data.frame_id)
            return results

    def update_unified(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """接收统一跨模块 JSON 字段，并返回统一路径预测字典。

        对外使用 video_path/video_id/frame_index/timestamp_sec；内部继续复用
        FrameData 和 PersonFrameData，避免破坏已有调用方。
        """

        if not isinstance(payload, dict):
            raise TypeError("unified payload 必须是字典")
        video_path = str(payload.get("video_path", "")).strip()
        video_id = str(payload.get("video_id", "")).strip()
        if not video_path:
            raise ValueError("video_path 不能为空")
        if not video_id:
            raise ValueError("video_id 不能为空")
        frame_index = payload.get("frame_index")
        if not isinstance(frame_index, int):
            raise TypeError("frame_index 必须是 int")
        timestamp_sec = float(payload.get("timestamp_sec", float("nan")))
        if not math.isfinite(timestamp_sec):
            raise ValueError("timestamp_sec 必须是有限数值")
        raw_persons = payload.get("persons", [])
        if not isinstance(raw_persons, list):
            raise TypeError("persons 必须是列表")

        persons = []
        for raw in raw_persons:
            if not isinstance(raw, dict):
                raise TypeError("persons 中的每一项必须是字典")
            raw_keypoints = raw.get("keypoints", {})
            if not isinstance(raw_keypoints, dict):
                raise TypeError("keypoints 必须是字典")
            persons.append(
                PersonFrameData(
                    frame_id=frame_index,
                    timestamp=timestamp_sec,
                    person_id=str(raw.get("person_id", "")),
                    keypoints={
                        str(name): tuple(value)
                        for name, value in raw_keypoints.items()
                    },
                    bbox=tuple(raw.get("bbox", ())),
                    source=str(raw.get("source", video_id)),
                    detection_confidence=float(
                        raw.get("detection_confidence", 1.0)
                    ),
                )
            )
        results = self.update(
            FrameData(
                frame_id=frame_index,
                timestamp=timestamp_sec,
                persons=persons,
            )
        )
        return [
            {
                "video_path": video_path,
                "video_id": video_id,
                "frame_index": result.frame_id,
                "timestamp_sec": result.timestamp,
                "person_id": result.person_id,
                "has_prediction": result.has_prediction,
                "current_position": result.current_position,
                "predicted_activity_region": {
                    "type": "polygon",
                    "coordinate_system": "original_video_pixels",
                    "points": result.path_corridor,
                },
                "predicted_trajectory": result.predicted_path,
                "candidate_paths": [item.to_dict() for item in result.candidate_paths],
                "path_confidence": result.confidence,
                "confidence_type": (
                    "mc_dropout_ade_v1"
                    if result.model_method == "gru"
                    else "model_candidate_score"
                ),
                "confidence_details": result.confidence_details or {},
                "model_method": result.model_method,
            }
            for result in results
        ]

    def update_single(self, person: PersonFrameData) -> PredictionResult:
        """接收单人帧；该入口不会淘汰其他未出现人物。"""

        with self._lock:
            self._validate_person(person)
            return self._update_single(person)

    def remove_person(self, person_id: str) -> None:
        """手动移除一个人物及其缓冲区。"""

        with self._lock:
            self._remove_person_unlocked(person_id)

    def reset(self) -> None:
        """重置所有内部状态，场景或视频切换时必须调用。"""

        with self._lock:
            self._buffers.clear()
            self._bboxes.clear()
            self._last_seen.clear()
            self._last_confidence_details = None
            self._last_uncertainty_paths_norm = None

    def warmup(self, n_times: int = 1) -> None:
        """预热模型，消除首次推理的冷启动开销。"""

        if n_times < 1:
            raise ValueError("n_times 必须大于 0")
        dummy = torch.zeros(
            1,
            self.config.obs_frames,
            2,
            device=self.device,
        )
        with self._lock, torch.inference_mode():
            self.model.eval()
            for _ in range(n_times):
                if self.method in ("gru", "eigen_transformer"):
                    self.model(dummy)
                else:
                    self.model.sample(
                        dummy,
                        num_samples=self.num_samples,
                    )

    def get_state(self) -> dict[str, Any]:
        """导出可直接由 json.dumps 编码的版本化状态。"""

        with self._lock:
            return {
                "schema_version": STATE_SCHEMA_VERSION,
                "model_method": self.method,
                "obs_frames": self.config.obs_frames,
                "buffers": {
                    pid: {
                        "centers": [
                            np.asarray(center, dtype=np.float32).tolist()
                            for center in buf
                        ],
                        "bbox": (
                            self._bboxes[pid].tolist()
                            if pid in self._bboxes
                            else None
                        ),
                        "last_seen": int(self._last_seen.get(pid, 0)),
                    }
                    for pid, buf in self._buffers.items()
                },
            }

    def set_state(self, state: dict[str, Any]) -> None:
        """校验并恢复由 get_state 导出的状态。"""

        if not isinstance(state, dict):
            raise TypeError("state 必须是字典")
        schema = state.get("schema_version")
        if schema not in (None, STATE_SCHEMA_VERSION):
            raise ValueError(f"不支持的状态版本: {schema}")
        state_method = state.get("model_method")
        if state_method not in (None, self.method):
            raise ValueError(
                f"状态属于 {state_method}，当前模型为 {self.method}"
            )
        state_obs_frames = state.get("obs_frames")
        if state_obs_frames not in (None, self.config.obs_frames):
            raise ValueError("状态的 obs_frames 与当前模型不一致")

        raw_buffers = state.get("buffers", {})
        if not isinstance(raw_buffers, dict):
            raise TypeError("state['buffers'] 必须是字典")
        if len(raw_buffers) > self.max_persons:
            raise ValueError("状态中的人物数量超过 max_persons")

        buffers: dict[str, deque] = {}
        bboxes: dict[str, np.ndarray] = {}
        last_seen: dict[str, int] = {}
        for raw_pid, data in raw_buffers.items():
            pid = str(raw_pid).strip()
            if not pid or not isinstance(data, dict):
                raise ValueError("状态包含无效人物记录")
            centers = np.asarray(
                data.get("centers", []),
                dtype=np.float32,
            )
            if centers.size == 0:
                centers = centers.reshape(0, 2)
            if (
                centers.ndim != 2
                or centers.shape[1] != 2
                or len(centers) > self.config.obs_frames
                or not np.all(np.isfinite(centers))
            ):
                raise ValueError(f"{pid} 的 centers 无效")
            bbox = np.asarray(data.get("bbox"), dtype=np.float32)
            if (
                bbox.shape != (4,)
                or not np.all(np.isfinite(bbox))
                or bbox[2] < bbox[0]
                or bbox[3] < bbox[1]
            ):
                raise ValueError(f"{pid} 的 bbox 无效")
            frame_id = int(data.get("last_seen", 0))
            buffers[pid] = deque(
                centers.tolist(),
                maxlen=self.config.obs_frames,
            )
            bboxes[pid] = bbox
            last_seen[pid] = frame_id

        with self._lock:
            self._buffers = buffers
            self._bboxes = bboxes
            self._last_seen = last_seen

    @staticmethod
    def from_checkpoint(
        model_path: str | Path,
        method: str,
        device: str = "auto",
        num_samples: int = 6,
        max_persons: int = 50,
        stale_frame_threshold: int = 30,
        max_frame_gap: int = 1,
        confidence_scale_px: float = 100.0,
    ) -> "WindowPredictor":
        """从受信任的项目检查点构建流式预测器。"""

        if method not in SUPPORTED_METHODS:
            raise ValueError(
                f"未知模型方法: {method}，可选: "
                + ", ".join(SUPPORTED_METHODS)
            )
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("device 必须是 auto、cpu 或 cuda")
        model_path = Path(model_path)
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("已指定 CUDA，但当前环境不可用")
        use_cuda = device == "cuda" or (
            device == "auto" and torch.cuda.is_available()
        )
        torch_device = torch.device("cuda" if use_cuda else "cpu")
        checkpoint = torch.load(
            model_path,
            map_location=torch_device,
            weights_only=False,
        )
        schema = str(checkpoint.get("schema_version", ""))
        expected_prefix = {
            "gru": "trajectory_gru",
            "eigen_transformer": "eigen_transformer",
            "leapfrog_diffusion": "leapfrog_diffusion",
        }[method]
        if not schema.startswith(expected_prefix):
            raise ValueError(
                f"检查点类型 {schema!r} 与方法 {method!r} 不匹配"
            )

        if method == "gru":
            from gru.core import TrajectoryGRU

            model = TrajectoryGRU(
                **checkpoint["model_config"]
            ).to(torch_device)
            model.load_state_dict(checkpoint["model_state"])
            validation_ade = float(
                checkpoint.get(
                    "validation_metrics",
                    {},
                ).get("ade_pixel", 50.0)
            )
            gru_confidence = round(
                float(
                    np.clip(
                        math.exp(-validation_ade / 100.0),
                        0.0,
                        1.0,
                    )
                ),
                3,
            )
            effective_samples = max(2, int(num_samples))
        elif method == "eigen_transformer":
            from eigen_transformer.models import EigenTrajectoryTransformer

            model = EigenTrajectoryTransformer(
                basis_mean=checkpoint["basis_mean"],
                basis_components=checkpoint["basis_components"],
                **checkpoint["model_config"],
            ).to(torch_device)
            model.load_state_dict(checkpoint["model_state"])
            gru_confidence = 0.6
            effective_samples = int(
                checkpoint["model_config"].get(
                    "num_modes",
                    num_samples,
                )
            )
        else:
            from leapfrog_diffusion.models import LeapfrogDiffusionPredictor

            model = LeapfrogDiffusionPredictor(
                **checkpoint["model_config"]
            ).to(torch_device)
            model.load_state_dict(checkpoint["model_state"])
            gru_confidence = 0.6
            effective_samples = int(
                checkpoint.get("num_samples", num_samples)
            )

        model.eval()
        config_data = checkpoint["data_config"]
        config = WindowConfig(
            fps=float(config_data["fps"]),
            obs_seconds=float(config_data["obs_seconds"]),
            pred_seconds=float(config_data["pred_seconds"]),
            canvas_width=int(config_data["canvas_width"]),
            canvas_height=int(config_data["canvas_height"]),
            x_offset=float(config_data["x_offset"]),
            coordinate_scale=float(config_data["coordinate_scale"]),
        )
        return WindowPredictor(
            model=model,
            method=method,
            config=config,
            normalization_scale=np.asarray(
                checkpoint["normalization"]["scale"],
                dtype=np.float32,
            ),
            device=torch_device,
            num_samples=effective_samples,
            max_persons=max_persons,
            stale_frame_threshold=stale_frame_threshold,
            max_frame_gap=max_frame_gap,
            gru_confidence=gru_confidence,
            gru_validation_ade=(validation_ade if method == "gru" else None),
            confidence_scale_px=confidence_scale_px,
        )

    # ─── 内部方法 ─────────────────────────────────────

    def _validate_frame_data(self, frame_data: FrameData) -> None:
        """在修改任何状态前校验一帧多人输入。"""

        if not isinstance(frame_data.frame_id, int):
            raise TypeError("FrameData.frame_id 必须是 int")
        if not math.isfinite(float(frame_data.timestamp)):
            raise ValueError("FrameData.timestamp 必须是有限数值")
        seen: set[str] = set()
        for person in frame_data.persons:
            self._validate_person(person)
            if person.frame_id != frame_data.frame_id:
                raise ValueError(
                    f"{person.person_id} 的 frame_id 与 FrameData 不一致"
                )
            if not math.isclose(
                float(person.timestamp),
                float(frame_data.timestamp),
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    f"{person.person_id} 的 timestamp 与 FrameData 不一致"
                )
            if person.person_id in seen:
                raise ValueError(
                    f"同一帧出现重复 person_id: {person.person_id}"
                )
            seen.add(person.person_id)

    @staticmethod
    def _validate_person(person: PersonFrameData) -> None:
        """校验单人输入的数据类型、范围和几何有效性。"""

        if not isinstance(person.frame_id, int):
            raise TypeError("PersonFrameData.frame_id 必须是 int")
        if not isinstance(person.person_id, str) or not person.person_id.strip():
            raise ValueError("person_id 必须是非空字符串")
        if not math.isfinite(float(person.timestamp)):
            raise ValueError("timestamp 必须是有限数值")
        confidence = float(person.detection_confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("detection_confidence 必须位于 [0, 1]")
        if not isinstance(person.keypoints, dict):
            raise TypeError("keypoints 必须是字典")
        bbox = np.asarray(person.bbox, dtype=np.float32)
        if (
            bbox.shape != (4,)
            or not np.all(np.isfinite(bbox))
            or bbox[2] < bbox[0]
            or bbox[3] < bbox[1]
        ):
            raise ValueError("bbox 必须是有效的 (xmin, ymin, xmax, ymax)")

    def _remove_person_unlocked(self, person_id: str) -> None:
        self._buffers.pop(person_id, None)
        self._bboxes.pop(person_id, None)
        self._last_seen.pop(person_id, None)

    def _ensure_capacity(self, incoming_person_id: str) -> None:
        """为新人物腾出容量，优先淘汰最久未出现的人物。"""

        if (
            incoming_person_id in self._buffers
            or len(self._buffers) < self.max_persons
        ):
            return
        oldest = min(
            self._buffers,
            key=lambda pid: self._last_seen.get(pid, -1),
        )
        self._remove_person_unlocked(oldest)

    def _update_single(self, person: PersonFrameData) -> PredictionResult:
        center = self._compute_center(person.keypoints, person.bbox)
        pid = person.person_id
        previous_frame = self._last_seen.get(pid)
        if previous_frame is not None:
            if person.frame_id <= previous_frame:
                raise ValueError(
                    f"{pid} 收到乱序或重复帧: "
                    f"{person.frame_id} <= {previous_frame}"
                )
            if person.frame_id - previous_frame > self.max_frame_gap:
                self._remove_person_unlocked(pid)

        self._ensure_capacity(pid)
        if pid not in self._buffers:
            self._buffers[pid] = deque(maxlen=self.config.obs_frames)
        self._buffers[pid].append(center)
        self._bboxes[pid] = np.asarray(person.bbox, dtype=np.float32)
        self._last_seen[pid] = person.frame_id

        buffer = np.asarray(self._buffers[pid], dtype=np.float32)
        if len(buffer) < self.config.obs_frames:
            return self._empty_result(person, center)

        observations = buffer[-self.config.obs_frames :]
        anchor = observations[-1:]
        normalized = (observations - anchor) / self.scale

        paths_norm, probabilities = self._infer(
            normalized.astype(np.float32)
        )
        paths_norm, probabilities = self._validate_inference_output(
            paths_norm,
            probabilities,
        )

        paths_pixel = anchor[None] + paths_norm * self.scale
        paths_pixel = self._clip_to_canvas(paths_pixel)

        order = np.argsort(probabilities)[::-1]
        paths_pixel = paths_pixel[order]
        probabilities = probabilities[order]
        best_path = paths_pixel[0]

        corridor_paths = paths_pixel
        if self._last_uncertainty_paths_norm is not None:
            uncertainty_paths = (
                anchor[None] + self._last_uncertainty_paths_norm * self.scale
            )
            uncertainty_paths = self._clip_to_canvas(uncertainty_paths)
            corridor_paths = np.concatenate(
                [paths_pixel, uncertainty_paths], axis=0
            )
        corridor = self._corridor(corridor_paths, self._bboxes[pid])
        corridor = self._clip_to_canvas(corridor)

        return PredictionResult(
            frame_id=person.frame_id,
            timestamp=person.timestamp,
            person_id=pid,
            has_prediction=True,
            current_position=(float(anchor[0, 0]), float(anchor[0, 1])),
            predicted_path=[
                (float(x), float(y)) for x, y in best_path
            ],
            candidate_paths=[
                CandidatePath(
                    probability=float(p),
                    path=[(float(x), float(y)) for x, y in path],
                )
                for path, p in zip(paths_pixel, probabilities)
            ],
            path_corridor=[
                (float(x), float(y)) for x, y in corridor
            ],
            confidence=float(probabilities[0]),
            model_method=self.method,
            confidence_details=self._last_confidence_details,
        )

    def _validate_inference_output(
        self,
        paths: np.ndarray,
        probabilities: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """阻止形状错误或 NaN 模型输出进入下游系统。"""

        paths = np.asarray(paths, dtype=np.float32)
        probabilities = np.asarray(probabilities, dtype=np.float32)
        if paths.ndim != 3 or paths.shape[1:] != (
            self.config.pred_frames,
            2,
        ):
            raise RuntimeError(
                f"模型轨迹形状无效: {paths.shape}"
            )
        if probabilities.shape != (len(paths),):
            raise RuntimeError(
                f"模型概率形状无效: {probabilities.shape}"
            )
        if (
            len(paths) == 0
            or not np.all(np.isfinite(paths))
            or not np.all(np.isfinite(probabilities))
        ):
            raise RuntimeError("模型输出包含空值、NaN 或无穷大")
        if self.method != "gru":
            probabilities = np.clip(probabilities, 0.0, None)
            total = float(probabilities.sum())
            if total <= 0.0:
                probabilities.fill(1.0 / len(probabilities))
            else:
                probabilities /= total
        return paths, probabilities

    def _infer(
        self,
        normalized_obs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """执行模型推理。

        Parameters
        ----------
        normalized_obs : np.ndarray
            形状 (obs_frames, 2) 的归一化观测序列。

        Returns
        -------
        paths : np.ndarray
            形状 (K, pred_frames, 2) 的候选轨迹（归一化空间）。
        probabilities : np.ndarray
            形状 (K,) 的候选轨迹概率或置信度。
        """
        model_input = (
            torch.from_numpy(normalized_obs).unsqueeze(0).to(self.device)
        )

        with torch.inference_mode():
            if self.method == "gru":
                self.model.eval()
                output = self.model(model_input)
                path = output.squeeze(0).cpu().numpy()
                self.model.train()
                samples = np.stack(
                    [
                        self.model(model_input).squeeze(0).cpu().numpy()
                        for _ in range(max(2, self.num_samples))
                    ],
                    axis=0,
                )
                self.model.eval()
                samples_px = samples * self.scale[None, None, :]
                mean_px = samples_px.mean(axis=0, keepdims=True)
                radial_variance = np.sum(
                    np.square(samples_px - mean_px), axis=-1
                )
                epistemic_std_px = float(np.sqrt(np.mean(radial_variance)))
                estimated_ade_px = float(
                    math.hypot(self.gru_validation_ade, epistemic_std_px)
                )
                confidence = float(
                    np.clip(
                        math.exp(-estimated_ade_px / self.confidence_scale_px),
                        0.0,
                        1.0,
                    )
                )
                self._last_confidence_details = {
                    "method": "mc_dropout_ade_v1",
                    "validation_ade_px": round(self.gru_validation_ade, 6),
                    "epistemic_std_px": round(epistemic_std_px, 6),
                    "estimated_ade_px": round(estimated_ade_px, 6),
                    "observation_coverage": 1.0,
                    "confidence_scale_px": round(self.confidence_scale_px, 6),
                    "mc_samples": max(2, self.num_samples),
                }
                self._last_uncertainty_paths_norm = samples
                return path[None], np.array([confidence], dtype=np.float32)

            if self.method == "eigen_transformer":
                self._last_confidence_details = None
                self._last_uncertainty_paths_norm = None
                paths, logits = self.model(model_input)
                paths_np = paths.squeeze(0).cpu().numpy()
                probs = (
                    torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                )
                return paths_np, probs

            if self.method == "leapfrog_diffusion":
                self._last_confidence_details = None
                self._last_uncertainty_paths_norm = None
                paths, probs_tensor = self.model.sample(
                    model_input, num_samples=self.num_samples,
                )
                return (
                    paths.squeeze(0).cpu().numpy(),
                    probs_tensor.squeeze(0).cpu().numpy(),
                )

            raise ValueError(f"未知模型方法: {self.method}")

    @staticmethod
    def _compute_center(
        keypoints: dict[str, tuple[float | None, float | None]],
        bbox: tuple[float, float, float, float],
    ) -> np.ndarray:
        """从关键点和边界框计算加权人体中心。

        优先使用 4 个中心关键点的加权平均；
        若有效权重不足则回退到边界框中心。
        """
        xs: list[float] = []
        ys: list[float] = []
        weights: list[float] = []

        for name, weight in zip(CENTER_POINTS, CENTER_WEIGHTS):
            point = keypoints.get(name)
            if point is None or len(point) != 2:
                continue
            x, y = point
            if x is None or y is None:
                continue
            x_value, y_value = float(x), float(y)
            if not math.isfinite(x_value) or not math.isfinite(y_value):
                continue
            xs.append(x_value)
            ys.append(y_value)
            weights.append(float(weight))

        weight_sum = sum(weights)
        if weight_sum >= 0.5:
            cx = sum(x * w for x, w in zip(xs, weights)) / weight_sum
            cy = sum(y * w for y, w in zip(ys, weights)) / weight_sum
        else:
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0

        return np.array([cx, cy], dtype=np.float32)

    @staticmethod
    def _corridor(
        paths: np.ndarray,
        bbox: np.ndarray,
        margin: float = 8.0,
    ) -> np.ndarray:
        """根据全部候选轨迹和当前人体框生成不确定性走廊矩形。

        Returns
        -------
        np.ndarray
            形状 (4, 2)，四个角点 [左上, 右上, 右下, 左下] 的坐标。
        """
        x1, y1, x2, y2 = bbox.astype(float)
        body_w = max(x2 - x1, 1.0)
        body_h = max(y2 - y1, 1.0)
        boxes = [[x1, y1, x2, y2]]
        for x, y in paths.reshape(-1, 2):
            boxes.append([
                x - body_w / 2, y - body_h / 2,
                x + body_w / 2, y + body_h / 2,
            ])
        values = np.asarray(boxes)
        left = values[:, 0].min() - margin
        top = values[:, 1].min() - margin
        right = values[:, 2].max() + margin
        bottom = values[:, 3].max() + margin
        return np.array([
            [left, top], [right, top], [right, bottom], [left, bottom],
        ], dtype=np.float32)

    def _clip_to_canvas(self, paths: np.ndarray) -> np.ndarray:
        paths[..., 0] = np.clip(paths[..., 0], 0, self.config.canvas_width - 1)
        paths[..., 1] = np.clip(paths[..., 1], 0, self.config.canvas_height - 1)
        return paths

    def _empty_result(
        self, person: PersonFrameData, center: np.ndarray,
    ) -> PredictionResult:
        return PredictionResult(
            frame_id=person.frame_id,
            timestamp=person.timestamp,
            person_id=person.person_id,
            has_prediction=False,
            current_position=(float(center[0]), float(center[1])),
            predicted_path=[],
            candidate_paths=[],
            path_corridor=[],
            confidence=0.0,
            model_method=self.method,
            confidence_details=None,
        )

    def _evict_stale(
        self, active_ids: set[str], current_frame: int,
    ) -> None:
        threshold = current_frame - self.stale_frame_threshold
        stale = [
            pid
            for pid in list(self._buffers.keys())
            if pid not in active_ids
            and self._last_seen.get(pid, 0) < threshold
        ]
        for pid in stale:
            self._remove_person_unlocked(pid)
