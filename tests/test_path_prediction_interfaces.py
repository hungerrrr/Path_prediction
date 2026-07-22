from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gru.predict import trajectory_confidence
from path_prediction.config import PathPredictionSettings
from path_prediction.control import path_prediction_enabled
from path_prediction.export import write_payload
from path_prediction.frame_service import FramePredictionService
from path_prediction.skeleton_provider import FileSkeletonHistoryProvider


def _switch(tmp_path: Path, enabled: bool) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "modules": {"path_prediction": {"enabled": enabled}}
    }), encoding="utf-8")
    return path


def test_external_config_only_controls_switch(tmp_path):
    assert path_prediction_enabled(_switch(tmp_path, True)) is True
    assert path_prediction_enabled(_switch(tmp_path, False)) is False


def test_provider_resolves_skeleton_by_video_id(tmp_path):
    expected = tmp_path / "keypoints_camera-01.csv"
    expected.write_text("frame_id\n1\n", encoding="utf-8")
    provider = FileSkeletonHistoryProvider(tmp_path, PathPredictionSettings())
    assert provider.resolve("camera-01") == expected.resolve()


def test_provider_rereads_file_and_sees_new_current_frame(tmp_path):
    path = tmp_path / "keypoints_adl-01.csv"

    def write_frames(frame_ids):
        rows = []
        for frame_id in frame_ids:
            rows.append({
                "frame_id": frame_id,
                "source_id": "adl-01",
                "dataset_source": "URFall",
                "video_person_id": 1,
                "nosex": 100 + frame_id,
                "nosey": 100,
                "midhipx": 100 + frame_id,
                "midhipy": 120,
                "ranklex": 95 + frame_id,
                "rankley": 200,
                "lanklex": 105 + frame_id,
                "lankley": 200,
            })
        pd.DataFrame(rows).to_csv(path, index=False)

    settings = PathPredictionSettings(observation_seconds=0.1)
    provider = FileSkeletonHistoryProvider(tmp_path, settings)
    write_frames([0, 1])
    _, before = provider.history_at("adl-01", 2, 2 / 30)
    assert before.empty
    write_frames([0, 1, 2])
    _, after = provider.history_at("adl-01", 2, 2 / 30)
    assert int(after["frame_index"].max()) == 2


def test_disabled_frame_call_does_not_load_model_or_skeleton(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"test")
    settings = PathPredictionSettings(model_path=tmp_path / "missing.pt")
    provider = FileSkeletonHistoryProvider(tmp_path / "missing", settings)
    service = FramePredictionService(settings, provider, _switch(tmp_path, False))
    result = service.predict(video, "video-1", 10, 1.0)
    assert result["status"] == "disabled"
    assert result["records"] == []


def test_shared_export_writes_json_and_csv(tmp_path):
    payload = {
        "schema_version": "path_prediction.frame.v1",
        "records": [{
            "video_path": "video.mp4",
            "video_id": "video-1",
            "frame_index": 10,
            "timestamp_sec": 1.0,
            "person_id": "person_1",
            "has_prediction": True,
            "current_position": [1.0, 2.0],
            "predicted_activity_region": {"type": "polygon", "points": []},
            "predicted_trajectory": [[2.0, 3.0]],
            "path_confidence": 0.8,
            "confidence_type": "mc_dropout_ade_v1",
            "confidence_details": {},
            "model_method": "gru",
            "candidate_paths": [],
        }],
    }
    json_path, csv_path = write_payload(payload, tmp_path, "result")
    assert json.loads(json_path.read_text(encoding="utf-8"))["records"]
    assert csv_path.is_file()


def test_trajectory_confidence_reflects_context_and_uncertainty():
    short_history, _ = trajectory_confidence(12.0, 2.0, 10, 60, 100.0)
    complete_history, details = trajectory_confidence(
        12.0, 2.0, 60, 60, 100.0
    )
    uncertain, _ = trajectory_confidence(12.0, 40.0, 60, 60, 100.0)
    assert short_history < complete_history
    assert uncertain < complete_history
    assert details["method"] == "mc_dropout_ade_v1"
