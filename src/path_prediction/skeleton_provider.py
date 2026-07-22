"""根据视频 ID 和当前帧查询可持续追加的骨架文件。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gru.core import DataConfig, read_table, standardize_keypoints

from .config import PathPredictionSettings


SUPPORTED_SUFFIXES = (".xlsx", ".xls", ".csv", ".parquet")


class FileSkeletonHistoryProvider:
    """为每次单帧请求定位并重新读取骨架文件。"""

    def __init__(
        self,
        skeleton_directory: str | Path,
        settings: PathPredictionSettings,
    ) -> None:
        self.skeleton_directory = Path(skeleton_directory).resolve()
        self.settings = settings

    def resolve(self, video_id: str, explicit_path: str | Path | None = None) -> Path:
        if explicit_path is not None:
            path = Path(explicit_path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"骨架文件不存在：{path}")
            return path
        if not self.skeleton_directory.is_dir():
            raise FileNotFoundError(
                f"骨架目录不存在：{self.skeleton_directory}"
            )
        names = (
            video_id,
            f"keypoints_{video_id}",
            f"urfall_keypoints_{video_id}",
        )
        for name in names:
            for suffix in SUPPORTED_SUFFIXES:
                path = self.skeleton_directory / f"{name}{suffix}"
                if path.is_file():
                    return path.resolve()
        matches = sorted(
            path.resolve()
            for path in self.skeleton_directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_SUFFIXES
            and video_id.casefold() in path.stem.casefold()
        )
        if not matches:
            raise FileNotFoundError(
                f"未在以下目录找到 video_id={video_id!r} 对应的骨架文件："
                f"{self.skeleton_directory}"
            )
        if len(matches) > 1:
            raise RuntimeError(
                f"video_id={video_id!r} 匹配到多个骨架文件："
                + ", ".join(str(path) for path in matches)
            )
        return matches[0]

    def load_aligned(
        self,
        video_id: str,
        explicit_path: str | Path | None = None,
    ) -> tuple[Path, pd.DataFrame]:
        path = self.resolve(video_id, explicit_path)
        raw = read_table(path)
        # 上游骨架模块可能直接使用统一字段名 frame_index。
        if "frame_index" in raw.columns and "frame_id" not in raw.columns:
            raw = raw.copy()
            raw["frame_id"] = raw["frame_index"]
        if "source_id" not in raw.columns:
            raw = raw.copy()
            raw["source_id"] = video_id
        cfg = DataConfig(
            fps=30.0,
            obs_seconds=self.settings.observation_seconds,
            pred_seconds=self.settings.prediction_seconds,
        )
        aligned = standardize_keypoints(raw, path, cfg)
        return path, aligned

    def history_at(
        self,
        video_id: str,
        frame_index: int,
        timestamp_sec: float,
        explicit_path: str | Path | None = None,
    ) -> tuple[Path, pd.DataFrame]:
        path, aligned = self.load_aligned(video_id, explicit_path)
        available = aligned[
            (aligned["frame_index"] <= frame_index)
            & (aligned["timestamp_sec"] <= timestamp_sec + 1e-3)
        ].copy()
        if available.empty:
            return path, available
        current_ids = set(
            available.loc[
                available["frame_index"] == frame_index, "person_id"
            ].astype(str)
        )
        if not current_ids:
            return path, available.iloc[0:0].copy()
        available = available[available["person_id"].astype(str).isin(current_ids)]
        histories = []
        for _, track in available.groupby("person_id", sort=False):
            source_fps = float(track["source_fps"].iloc[-1])
            histories.append(
                track.sort_values("frame_index").tail(
                    int(round(self.settings.observation_seconds * source_fps))
                )
            )
        return path, pd.concat(histories, ignore_index=True).sort_values(
            ["frame_index", "person_id"]
        )
