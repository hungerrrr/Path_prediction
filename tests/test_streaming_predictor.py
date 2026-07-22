"""单元测试：流式路径预测器的数据契约与核心逻辑。"""

from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, sentinel

import numpy as np
import torch
import torch.nn as nn
import pytest

from streaming_predictor import (
    CENTER_POINTS,
    CENTER_WEIGHTS,
    CandidatePath,
    FrameData,
    PersonFrameData,
    PredictionResult,
    StreamingPredictor,
)
from gru.core import DataConfig


def _gru_state_dict(config=None):
    """生成 GRU 模型的 mock state_dict。"""
    if config is None:
        config = _default_config()
    from gru.core import TrajectoryGRU
    m = TrajectoryGRU(
        obs_frames=config.obs_frames, pred_frames=config.pred_frames,
    )
    return m.state_dict()


def _eigen_state_dict(config=None):
    """生成 EigenTransformer 模型的 mock state_dict。"""
    if config is None:
        config = _default_config()
    from eigen_transformer.models import EigenTrajectoryTransformer
    basis_mean = np.zeros(config.pred_frames * 2, dtype=np.float32)
    basis_components = np.zeros((16, config.pred_frames * 2), dtype=np.float32)
    m = EigenTrajectoryTransformer(
        obs_frames=config.obs_frames, pred_frames=config.pred_frames,
        basis_mean=basis_mean, basis_components=basis_components,
    )
    return m.state_dict()


def _diffusion_state_dict(config=None):
    """生成 LeapfrogDiffusion 模型的 mock state_dict。"""
    if config is None:
        config = _default_config()
    from leapfrog_diffusion.models import LeapfrogDiffusionPredictor
    m = LeapfrogDiffusionPredictor(
        obs_frames=config.obs_frames, pred_frames=config.pred_frames,
    )
    return m.state_dict()


# ─── 夹具 ────────────────────────────────────────────


def _make_config(
    obs_seconds: float = 2.0,
    pred_seconds: float = 3.0,
    fps: float = 30.0,
) -> DataConfig:
    return DataConfig(
        fps=fps,
        obs_seconds=obs_seconds,
        pred_seconds=pred_seconds,
        canvas_width=640,
        canvas_height=240,
        x_offset=320.0,
        coordinate_scale=0.5,
    )


def _default_config() -> DataConfig:
    return _make_config()


def _make_person(
    person_id: str = "person_1",
    frame_id: int = 0,
    keypoints: dict | None = None,
    bbox: tuple[float, float, float, float] = (100, 50, 200, 150),
) -> PersonFrameData:
    if keypoints is None:
        keypoints = {
            "nose": (150.0, 80.0),
            "midhip": (145.0, 120.0),
            "rankle": (148.0, 200.0),
            "lankle": (140.0, 198.0),
        }
    return PersonFrameData(
        frame_id=frame_id,
        timestamp=frame_id / 30.0,
        person_id=person_id,
        keypoints=keypoints,
        bbox=bbox,
        source="test",
        detection_confidence=0.95,
    )


def _make_frame(
    persons: list[PersonFrameData] | None = None,
    frame_id: int = 0,
) -> FrameData:
    if persons is None:
        persons = [_make_person(frame_id=frame_id)]
    return FrameData(
        frame_id=frame_id,
        timestamp=frame_id / 30.0,
        persons=persons,
    )


def _make_mock_model(method: str = "gru", pred_frames: int = 90, num_modes: int = 6):
    """根据方法类型创建 mock PyTorch 模型。"""
    model = MagicMock(spec=nn.Module)

    if method == "gru":
        def forward(obs):
            B = obs.shape[0]
            return torch.randn(B, pred_frames, 2)
        model.side_effect = forward
        model.forward = forward

    elif method == "eigen_transformer":
        def forward(obs):
            B = obs.shape[0]
            paths = torch.randn(B, num_modes, pred_frames, 2)
            logits = torch.randn(B, num_modes)
            return paths, logits
        model.forward = forward

    elif method == "leapfrog_diffusion":
        def sample(obs, num_samples=6):
            B = obs.shape[0]
            paths = torch.randn(B, num_samples, pred_frames, 2)
            logits = torch.randn(B, num_samples)
            probs = torch.softmax(logits, dim=1)
            return paths, probs
        model.sample = sample

    # 为 EigenTransformer 生成 mock state_dict 键
    return model


def _make_eigen_mock_model(pred_frames=90, num_modes=6):
    """EigenTransformer mock: __call__ 返回 (paths, logits) 元组。"""
    model = MagicMock(spec=nn.Module)
    def forward(obs):
        B = obs.shape[0]
        paths = torch.randn(B, num_modes, pred_frames, 2)
        logits = torch.randn(B, num_modes)
        return paths, logits
    model.side_effect = forward
    return model


def _make_predictor(
    method: str = "gru",
    config: DataConfig | None = None,
    num_samples: int = 6,
    stale_threshold: int = 30,
    max_persons: int = 50,
    max_frame_gap: int = 1,
) -> StreamingPredictor:
    if config is None:
        config = _default_config()
    if method == "eigen_transformer":
        model = _make_eigen_mock_model(
            pred_frames=config.pred_frames, num_modes=num_samples,
        )
    else:
        model = _make_mock_model(method=method, pred_frames=config.pred_frames)
    return StreamingPredictor(
        model=model,
        method=method,
        config=config,
        normalization_scale=np.asarray([320.0, 240.0], dtype=np.float32),
        device=torch.device("cpu"),
        num_samples=num_samples,
        max_persons=max_persons,
        stale_frame_threshold=stale_threshold,
        max_frame_gap=max_frame_gap,
    )


# ─── 数据类测试 ──────────────────────────────────────


class TestDataClasses:
    def test_person_frame_data_defaults(self):
        p = _make_person()
        assert p.frame_id == 0
        assert p.person_id == "person_1"
        assert p.detection_confidence == 0.95
        assert "nose" in p.keypoints

    def test_frame_data(self):
        p1 = _make_person(person_id="p1")
        p2 = _make_person(person_id="p2")
        fd = FrameData(frame_id=5, timestamp=5.0 / 30.0, persons=[p1, p2])
        assert fd.frame_id == 5
        assert len(fd.persons) == 2

    def test_candidate_path(self):
        cp = CandidatePath(probability=0.5, path=[(1.0, 2.0), (3.0, 4.0)])
        assert cp.probability == 0.5
        assert len(cp.path) == 2

    def test_prediction_result_defaults(self):
        pr = PredictionResult(
            frame_id=1,
            timestamp=1.0 / 30.0,
            person_id="p1",
            has_prediction=True,
            current_position=(150.0, 120.0),
            predicted_path=[(160.0, 130.0), (170.0, 140.0)],
            candidate_paths=[
                CandidatePath(probability=0.8, path=[(160.0, 130.0), (170.0, 140.0)]),
            ],
            path_corridor=[(0, 0), (100, 0), (100, 100), (0, 100)],
            confidence=0.8,
            model_method="gru",
        )
        assert pr.has_prediction
        assert pr.confidence == 0.8


# ─── 人体中心计算 ────────────────────────────────────


class TestComputeCenter:
    def test_weighted_keypoints(self):
        kp = {
            "nose": (150.0, 80.0),
            "midhip": (145.0, 120.0),
            "rankle": (148.0, 200.0),
            "lankle": (140.0, 198.0),
        }
        bbox = (100, 50, 200, 150)
        center = StreamingPredictor._compute_center(kp, bbox)
        # 加权平均: nose(0.3) + midhip(0.5) + rankle(0.1) + lankle(0.1)
        expected_x = (
            150.0 * 0.3 + 145.0 * 0.5 + 148.0 * 0.1 + 140.0 * 0.1
        ) / 1.0
        expected_y = (
            80.0 * 0.3 + 120.0 * 0.5 + 200.0 * 0.1 + 198.0 * 0.1
        ) / 1.0
        assert np.isclose(center[0], expected_x)
        assert np.isclose(center[1], expected_y)

    def test_fallback_to_bbox_center_when_no_keypoints(self):
        kp = {"nose": (None, None), "midhip": None, "rankle": None, "lankle": None}
        bbox = (100.0, 50.0, 200.0, 150.0)
        center = StreamingPredictor._compute_center(kp, bbox)
        assert np.isclose(center[0], 150.0)  # (100+200)/2
        assert np.isclose(center[1], 100.0)  # (50+150)/2

    def test_fallback_when_insufficient_weight(self):
        """仅一个关键点有效，权重 0.3 < 0.5 → 回退 bbox"""
        kp = {"nose": (150.0, 80.0), "midhip": None, "rankle": None, "lankle": None}
        bbox = (100.0, 50.0, 200.0, 150.0)
        center = StreamingPredictor._compute_center(kp, bbox)
        assert np.isclose(center[0], 150.0)
        assert np.isclose(center[1], 100.0)

    def test_partial_keypoints_sufficient_weight(self):
        """nose(0.3) + midhip(0.5) = 0.8 >= 0.5"""
        kp = {"nose": (150.0, 80.0), "midhip": (145.0, 120.0), "rankle": None, "lankle": None}
        bbox = (100, 50, 200, 150)
        center = StreamingPredictor._compute_center(kp, bbox)
        expected_x = (150.0 * 0.3 + 145.0 * 0.5) / 0.8
        expected_y = (80.0 * 0.3 + 120.0 * 0.5) / 0.8
        assert np.isclose(center[0], expected_x)
        assert np.isclose(center[1], expected_y)


# ─── 画布裁剪 ────────────────────────────────────────


class TestClipToCanvas:
    def test_clip_within_bounds(self):
        config = _default_config()
        predictor = _make_predictor(config=config)
        paths = np.array([[[100.0, 50.0], [200.0, 100.0]]])
        result = predictor._clip_to_canvas(paths)
        np.testing.assert_array_equal(result, paths)

    def test_clip_negative_values(self):
        config = _default_config()
        predictor = _make_predictor(config=config)
        paths = np.array([[[-10.0, -5.0], [300.0, 120.0]]])
        result = predictor._clip_to_canvas(paths)
        assert result[0, 0, 0] == 0
        assert result[0, 0, 1] == 0

    def test_clip_exceeding_bounds(self):
        config = _default_config()
        predictor = _make_predictor(config=config)
        paths = np.array([[[600.0, 200.0], [700.0, 300.0]]])
        result = predictor._clip_to_canvas(paths)
        assert result[0, 1, 0] == config.canvas_width - 1  # 639
        assert result[0, 1, 1] == config.canvas_height - 1  # 239


# ─── 走廊计算 ────────────────────────────────────────


class TestCorridor:
    def test_corridor_basic(self):
        bbox = np.array([100.0, 50.0, 200.0, 150.0])
        paths = np.array([[[120.0, 60.0], [250.0, 130.0]]])
        result = StreamingPredictor._corridor(paths, bbox, margin=0.0)
        # bbox 范围: x [100,200], y [50,150]; body_w=100, body_h=100
        # 轨迹点扩展框: [120-50,60-50,120+50,60+50] = [70,10,170,110]
        #              [250-50,130-50,250+50,130+50] = [200,80,300,180]
        # 总体范围: x [70,300], y [10,180]
        # margin=0, 所以结果应该是 [70,10], [300,10], [300,180], [70,180]
        np.testing.assert_array_almost_equal(
            result,
            [[70.0, 10.0], [300.0, 10.0], [300.0, 180.0], [70.0, 180.0]],
        )

    def test_corridor_with_margin(self):
        bbox = np.array([100.0, 50.0, 150.0, 100.0])  # body_w=50, body_h=50
        paths = np.array([[[120.0, 60.0]]])
        result = StreamingPredictor._corridor(paths, bbox, margin=10.0)
        # bbox: [100,50,150,100]
        # path 扩展: [120-25,60-25,120+25,60+25] = [95,35,145,85]
        # 总体: x [95,150], y [35,100]
        # margin=10: x [85,160], y [25,110]
        np.testing.assert_array_almost_equal(
            result,
            [[85.0, 25.0], [160.0, 25.0], [160.0, 110.0], [85.0, 110.0]],
        )


# ─── 缓冲区与状态管理 ────────────────────────────────


class TestBufferManagement:
    def test_buffer_initializes_on_first_update(self):
        predictor = _make_predictor()
        p = _make_person(frame_id=0)
        predictor._update_single(p)
        assert "person_1" in predictor._buffers
        assert len(predictor._buffers["person_1"]) == 1

    def test_buffer_grows_until_full(self):
        config = _make_config(obs_seconds=0.1, fps=10.0)  # obs_frames = 1
        predictor = _make_predictor(config=config)
        for i in range(5):
            p = _make_person(frame_id=i)
            predictor._update_single(p)
        assert len(predictor._buffers["person_1"]) == 1  # maxlen=1

    def test_buffer_maxlen(self):
        config = _make_config(obs_seconds=2.0, fps=10.0)  # obs_frames = 20
        predictor = _make_predictor(config=config)
        for i in range(50):
            p = _make_person(frame_id=i)
            predictor._update_single(p)
        assert len(predictor._buffers["person_1"]) == 20

    def test_multiple_persons(self):
        predictor = _make_predictor()
        p1 = _make_person(person_id="alice", frame_id=0)
        p2 = _make_person(person_id="bob", frame_id=0)
        predictor._update_single(p1)
        predictor._update_single(p2)
        assert "alice" in predictor._buffers
        assert "bob" in predictor._buffers
        assert len(predictor._buffers) == 2

    def test_remove_person(self):
        predictor = _make_predictor()
        predictor._update_single(_make_person(frame_id=0))
        assert "person_1" in predictor._buffers
        predictor.remove_person("person_1")
        assert "person_1" not in predictor._buffers
        assert "person_1" not in predictor._bboxes
        assert "person_1" not in predictor._last_seen

    def test_reset_clears_all(self):
        predictor = _make_predictor()
        predictor._update_single(_make_person(frame_id=0, person_id="a"))
        predictor._update_single(_make_person(frame_id=0, person_id="b"))
        predictor.reset()
        assert len(predictor._buffers) == 0
        assert len(predictor._bboxes) == 0
        assert len(predictor._last_seen) == 0

    def test_stale_eviction(self):
        predictor = _make_predictor(stale_threshold=5)
        predictor._update_single(_make_person(frame_id=0, person_id="alice"))
        predictor._update_single(_make_person(frame_id=0, person_id="bob"))

        # 第 10 帧只有 bob，alice 最后出现在第 0 帧，超过阈值
        frame = _make_frame(
            persons=[_make_person(person_id="bob", frame_id=10)],
            frame_id=10,
        )
        predictor.update(frame)
        assert "alice" not in predictor._buffers
        assert "bob" in predictor._buffers

    def test_no_eviction_within_threshold(self):
        predictor = _make_predictor(stale_threshold=10)
        predictor._update_single(_make_person(frame_id=0, person_id="alice"))
        predictor._update_single(_make_person(frame_id=0, person_id="bob"))

        frame = _make_frame(
            persons=[_make_person(person_id="bob", frame_id=8)],
            frame_id=8,
        )
        predictor.update(frame)
        assert "alice" in predictor._buffers  # 8-0=8 < 10


# ─── get_state / set_state ──────────────────────────


class TestStateSerialization:
    def test_get_state_contains_buffers(self):
        predictor = _make_predictor()
        predictor._update_single(_make_person(frame_id=0))
        state = predictor.get_state()
        assert "buffers" in state
        assert "person_1" in state["buffers"]

    def test_set_state_restores_buffers(self):
        predictor = _make_predictor()
        state = {
            "buffers": {
                "p1": {
                    "centers": [[100.0, 50.0], [110.0, 55.0]],
                    "bbox": [90.0, 30.0, 130.0, 80.0],
                    "last_seen": 5,
                },
            },
        }
        predictor.set_state(state)
        assert "p1" in predictor._buffers
        assert len(predictor._buffers["p1"]) == 2
        np.testing.assert_array_almost_equal(
            predictor._bboxes["p1"], [90.0, 30.0, 130.0, 80.0],
        )
        assert predictor._last_seen["p1"] == 5

    def test_set_state_resets_first(self):
        predictor = _make_predictor()
        predictor._update_single(_make_person(frame_id=0))
        predictor.set_state({"buffers": {}})
        assert len(predictor._buffers) == 0


# ─── update / update_single ─────────────────────────


class TestUpdate:
    def test_update_single_returns_empty_before_buffer_full(self):
        config = _make_config(obs_seconds=2.0, fps=30.0)  # obs_frames = 60
        predictor = _make_predictor(config=config)
        p = _make_person(frame_id=0)
        result = predictor.update_single(p)
        assert not result.has_prediction
        assert result.person_id == "person_1"
        assert len(result.predicted_path) == 0

    def test_update_single_returns_prediction_when_buffer_full(self):
        config = _make_config(obs_seconds=0.1, fps=10.0)  # obs_frames = 1
        predictor = _make_predictor(config=config)
        p = _make_person(frame_id=0)
        result = predictor.update_single(p)
        assert result.has_prediction
        assert len(result.predicted_path) > 0

    def test_update_handles_multiple_persons(self):
        config = _make_config(obs_seconds=0.1, fps=10.0)
        predictor = _make_predictor(config=config)
        p1 = _make_person(person_id="alice", frame_id=0)
        p2 = _make_person(person_id="bob", frame_id=0)
        frame = FrameData(
            frame_id=0, timestamp=0.0, persons=[p1, p2],
        )
        results = predictor.update(frame)
        assert len(results) == 2
        assert all(r.has_prediction for r in results)

    def test_update_mixed_buffer_status(self):
        """一人缓冲区已满，另一人刚出现"""
        config = _make_config(obs_seconds=0.1, fps=10.0)  # obs_frames = 1
        predictor = _make_predictor(config=config)
        predictor._update_single(_make_person(person_id="alice", frame_id=0))

        config2 = _make_config(obs_seconds=2.0, fps=30.0)  # obs_frames = 60
        predictor2 = _make_predictor(config=config2)
        predictor2._buffers["alice"] = deque(
            [[100.0, 50.0]] * 60, maxlen=60,
        )
        predictor2._last_seen["alice"] = 0
        frame = FrameData(
            frame_id=1, timestamp=1.0 / 30.0,
            persons=[
                _make_person(person_id="alice", frame_id=1),
                _make_person(person_id="bob", frame_id=1),
            ],
        )
        results = predictor2.update(frame)
        assert results[0].has_prediction  # alice 已满
        assert not results[1].has_prediction  # bob 刚出现

    def test_prediction_result_fields(self):
        config = _make_config(obs_seconds=0.1, fps=10.0)
        predictor = _make_predictor(config=config)
        p = _make_person(frame_id=0)
        result = predictor.update_single(p)
        assert isinstance(result.current_position, tuple)
        assert len(result.current_position) == 2
        assert isinstance(result.predicted_path, list)
        assert len(result.predicted_path) > 0
        assert isinstance(result.predicted_path[0], tuple)
        assert len(result.candidate_paths) > 0
        assert isinstance(result.candidate_paths[0], CandidatePath)
        assert len(result.path_corridor) == 4
        assert 0.0 <= result.confidence <= 1.0


# ─── _infer ─────────────────────────────────────────


class TestInfer:
    def test_gru_infer_returns_single_path(self):
        config = _make_config(obs_seconds=0.1, fps=10.0)
        predictor = _make_predictor(method="gru", config=config)
        obs = np.zeros((config.obs_frames, 2), dtype=np.float32)
        paths, probs = predictor._infer(obs)
        assert paths.shape == (1, config.pred_frames, 2)
        assert probs.shape == (1,)

    def test_eigen_transformer_infer_returns_multiple_paths(self):
        config = _make_config(obs_seconds=0.1, fps=10.0)
        predictor = _make_predictor(
            method="eigen_transformer", config=config, num_samples=6,
        )
        obs = np.zeros((config.obs_frames, 2), dtype=np.float32)
        paths, probs = predictor._infer(obs)
        assert paths.shape[0] == 6
        assert paths.shape[1] == config.pred_frames
        assert paths.shape[2] == 2
        assert probs.shape == (6,)
        assert np.isclose(probs.sum(), 1.0)

    def test_leapfrog_diffusion_infer_returns_multiple_paths(self):
        config = _make_config(obs_seconds=0.1, fps=10.0)
        predictor = _make_predictor(
            method="leapfrog_diffusion", config=config, num_samples=6,
        )
        obs = np.zeros((config.obs_frames, 2), dtype=np.float32)
        paths, probs = predictor._infer(obs)
        assert paths.shape[0] == 6
        assert paths.shape[1] == config.pred_frames
        assert paths.shape[2] == 2
        assert probs.shape == (6,)
        assert np.isclose(probs.sum(), 1.0)


# ─── warmup ─────────────────────────────────────────


class TestWarmup:
    def test_gru_warmup_runs_without_error(self):
        predictor = _make_predictor(method="gru")
        predictor.warmup(n_times=2)

    def test_eigen_transformer_warmup_runs_without_error(self):
        predictor = _make_predictor(method="eigen_transformer")
        predictor.warmup(n_times=2)

    def test_leapfrog_diffusion_warmup_runs_without_error(self):
        predictor = _make_predictor(method="leapfrog_diffusion")
        predictor.warmup(n_times=2)


# ─── from_checkpoint ────────────────────────────────


class TestFromCheckpoint:
    """通过 mock 验证检查点加载逻辑。"""

    @patch("streaming_predictor.torch.load")
    def test_load_gru(self, mock_torch_load):
        config = _default_config()
        mock_torch_load.return_value = {
            "schema_version": "trajectory_gru.v2",
            "model_config": {
                "obs_frames": config.obs_frames,
                "pred_frames": config.pred_frames,
                "hidden_dim": 128,
                "num_layers": 2,
                "dropout": 0.1,
            },
            "model_state": _gru_state_dict(config),
            "data_config": config.to_dict(),
            "normalization": {"type": "relative_to_last_observation", "scale": [320.0, 240.0]},
            "validation_metrics": {"ade_pixel": 15.0},
        }
        predictor = StreamingPredictor.from_checkpoint(
            "dummy.pt", "gru", device="cpu",
        )
        assert predictor.method == "gru"
        assert predictor.config.obs_frames == config.obs_frames
        assert predictor.gru_confidence == round(math.exp(-15.0 / 100.0), 3)

    @patch("streaming_predictor.torch.load")
    def test_load_eigen_transformer(self, mock_torch_load):
        config = _default_config()
        mock_torch_load.return_value = {
            "schema_version": "eigen_transformer.v1",
            "model_config": {
                "obs_frames": config.obs_frames,
                "pred_frames": config.pred_frames,
                "hidden_dim": 128,
                "num_layers": 3,
                "num_heads": 4,
                "num_modes": 6,
                "dropout": 0.1,
                "use_velocity": True,
            },
            "model_state": _eigen_state_dict(config),
            "basis_mean": np.zeros(config.pred_frames * 2, dtype=np.float32),
            "basis_components": np.zeros((16, config.pred_frames * 2), dtype=np.float32),
            "data_config": config.to_dict(),
            "normalization": {"scale": [320.0, 240.0]},
        }
        predictor = StreamingPredictor.from_checkpoint(
            "dummy.pt", "eigen_transformer", device="cpu",
        )
        assert predictor.method == "eigen_transformer"
        assert predictor.num_samples == 6

    @patch("streaming_predictor.torch.load")
    def test_load_leapfrog_diffusion(self, mock_torch_load):
        config = _default_config()
        mock_torch_load.return_value = {
            "schema_version": "leapfrog_diffusion.v1",
            "model_config": {
                "obs_frames": config.obs_frames,
                "pred_frames": config.pred_frames,
                "hidden_dim": 128,
                "num_layers": 3,
                "num_heads": 4,
                "diffusion_steps": 50,
                "sample_steps": 8,
                "dropout": 0.1,
            },
            "model_state": _diffusion_state_dict(config),
            "data_config": config.to_dict(),
            "normalization": {"scale": [320.0, 240.0]},
            "num_samples": 8,
        }
        predictor = StreamingPredictor.from_checkpoint(
            "dummy.pt", "leapfrog_diffusion", device="cpu", num_samples=6,
        )
        assert predictor.method == "leapfrog_diffusion"
        assert predictor.num_samples == 8  # checkpoint 值优先

    @patch("streaming_predictor.torch.load")
    def test_from_checkpoint_runs_on_gru(self, mock_torch_load):
        config = _default_config()
        mock_torch_load.return_value = {
            "schema_version": "trajectory_gru.v2",
            "model_config": {
                "obs_frames": config.obs_frames,
                "pred_frames": config.pred_frames,
                "hidden_dim": 128,
                "num_layers": 2,
                "dropout": 0.1,
            },
            "model_state": _gru_state_dict(config),
            "data_config": config.to_dict(),
            "normalization": {"scale": [320.0, 240.0]},
            "validation_metrics": {"ade_pixel": 12.36},
        }
        predictor = StreamingPredictor.from_checkpoint(
            "dummy.pt", "gru", device="cpu",
        )
        predictor.warmup()
        person = _make_person(frame_id=0)
        result = predictor.update_single(person)
        assert not result.has_prediction  # 缓冲区不够
        # 填满缓冲区
        for i in range(1, config.obs_frames):
            person = _make_person(frame_id=i)
            predictor._update_single(person)
        # 再预测一次
        result2 = predictor._update_single(
            _make_person(frame_id=config.obs_frames),
        )
        assert result2.has_prediction
        assert result2.model_method == "gru"

    @patch("streaming_predictor.torch.load")
    def test_from_checkpoint_invalid_method(self, mock_torch_load):
        mock_torch_load.return_value = {}
        try:
            StreamingPredictor.from_checkpoint("dummy.pt", "invalid", device="cpu")
            assert False, "应抛出 ValueError"
        except ValueError:
            pass

    def test_from_checkpoint_cuda_unavailable(self):
        try:
            StreamingPredictor.from_checkpoint("dummy.pt", "gru", device="cuda")
            assert False, "应抛出 RuntimeError"
        except RuntimeError:
            pass

# ─── 集成边界回归测试 ────────────────────────────────


class TestIntegrationBoundaries:
    def test_state_is_json_serializable(self):
        predictor = _make_predictor()
        predictor.update_single(_make_person(frame_id=0))
        encoded = json.dumps(predictor.get_state())
        restored = json.loads(encoded)
        assert restored["schema_version"] == "streaming_predictor_state.v1"

    def test_prediction_result_to_dict_is_json_serializable(self):
        config = _make_config(obs_seconds=0.1, fps=10.0)
        predictor = _make_predictor(config=config)
        result = predictor.update_single(_make_person(frame_id=0))
        assert json.loads(json.dumps(result.to_dict()))["has_prediction"]

    def test_max_persons_evicts_oldest_track(self):
        predictor = _make_predictor(max_persons=2)
        predictor.update_single(_make_person("oldest", 0))
        predictor.update_single(_make_person("newer", 1))
        predictor.update_single(_make_person("newest", 2))
        assert set(predictor._buffers) == {"newer", "newest"}

    def test_duplicate_person_in_frame_is_rejected_before_mutation(self):
        predictor = _make_predictor()
        person = _make_person("duplicate", 0)
        frame = _make_frame([person, person], frame_id=0)
        with pytest.raises(ValueError, match="重复 person_id"):
            predictor.update(frame)
        assert not predictor._buffers

    def test_out_of_order_frame_is_rejected(self):
        predictor = _make_predictor()
        predictor.update_single(_make_person(frame_id=2))
        with pytest.raises(ValueError, match="乱序或重复帧"):
            predictor.update_single(_make_person(frame_id=1))

    def test_frame_gap_resets_person_history(self):
        config = _make_config(obs_seconds=0.3, fps=10.0)
        predictor = _make_predictor(config=config, max_frame_gap=1)
        predictor.update_single(_make_person(frame_id=0))
        predictor.update_single(_make_person(frame_id=1))
        result = predictor.update_single(_make_person(frame_id=4))
        assert not result.has_prediction
        assert len(predictor._buffers["person_1"]) == 1

    def test_invalid_detection_confidence_is_rejected(self):
        predictor = _make_predictor()
        person = _make_person()
        person.detection_confidence = float("nan")
        with pytest.raises(ValueError, match="detection_confidence"):
            predictor.update_single(person)

    def test_state_method_mismatch_is_rejected(self):
        predictor = _make_predictor(method="gru")
        with pytest.raises(ValueError, match="状态属于"):
            predictor.set_state({
                "schema_version": "streaming_predictor_state.v1",
                "model_method": "eigen_transformer",
                "obs_frames": predictor.config.obs_frames,
                "buffers": {},
            })

    @patch("streaming_predictor.torch.load")
    def test_checkpoint_method_mismatch_is_rejected(self, mock_torch_load):
        mock_torch_load.return_value = {
            "schema_version": "eigen_transformer.v1",
        }
        with pytest.raises(ValueError, match="不匹配"):
            StreamingPredictor.from_checkpoint(
                "dummy.pt",
                "gru",
                device="cpu",
            )

