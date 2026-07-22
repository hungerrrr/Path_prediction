"""路径预测模块的内部参数。

外部 ``config.json`` 仅保留系统集成所需的模块开关；模型与算法参数集中在
本文件中，使单帧入口和批量入口共享同一份配置。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def portable_path(path: str | Path) -> str:
    """将项目内路径序列化为正斜杠相对路径，避免输出绑定到某台机器。"""

    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        # 项目外部的用户数据无法安全表示为项目相对路径，保留规范化绝对路径。
        return resolved.as_posix()


@dataclass(frozen=True)
class PathPredictionSettings:
    model_method: str = "gru"
    model_path: Path = PROJECT_ROOT / "models" / "trajectory_gru.pt"
    device: str = "auto"
    observation_seconds: float = 2.0
    prediction_seconds: float = 3.0
    confidence_samples: int = 16
    confidence_scale_px: float = 100.0
    confidence_seed: int = 42
    multimodal_samples: int = 6
    max_persons: int = 50
    max_frame_gap: int = 1
    stale_frame_threshold: int = 30
    skeleton_directory: Path = PROJECT_ROOT / "examples" / "inputs"
    frame_output_directory: Path = (
        PROJECT_ROOT / "examples" / "outputs" / "frame_prediction"
    )
    batch_output_directory: Path = (
        PROJECT_ROOT / "examples" / "outputs" / "batch_prediction"
    )
    keep_legacy_outputs: bool = False

    def validate(self) -> None:
        if self.model_method not in {
            "gru", "eigen_transformer", "leapfrog_diffusion"
        }:
            raise ValueError(f"不支持的 model_method：{self.model_method}")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device 必须是 auto、cpu 或 cuda")
        if self.observation_seconds <= 0 or self.prediction_seconds <= 0:
            raise ValueError("观察时长和预测时长必须大于 0")
        if self.confidence_samples < 2:
            raise ValueError("confidence_samples 不能小于 2")
        if self.confidence_scale_px <= 0:
            raise ValueError("confidence_scale_px 必须大于 0")


SETTINGS = PathPredictionSettings()
