"""路径预测项目共享的数据处理、模型定义与序列切分工具。"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# 不同数据集使用的序列命名规则。
SEQUENCE_RE = re.compile(r"(?P<sequence>(?:adl|fall)-\d+)", re.IGNORECASE)
PREVFALL_RE = re.compile(r"(.+?)_(\d+)_keypoints$", re.IGNORECASE)
# 计算人体中心所需的最小字段集合。
REQUIRED_COLUMNS = {
    "nosex",
    "nosey",
    "midhipx",
    "midhipy",
    "ranklex",
    "rankley",
    "lanklex",
    "lankley",
}
CENTER_POINTS = ("nose", "midhip", "rankle", "lankle")
CENTER_WEIGHTS = np.asarray([0.3, 0.5, 0.1, 0.1], dtype=np.float32)
# 各数据源的原始分辨率和采样帧率，用于统一坐标系与时间尺度。
SOURCE_GEOMETRY = {
    "har-up": (640.0, 480.0),
    "lei2fall": (320.0, 240.0),
    "mmfall": (320.0, 180.0),
    "pre-vfall": (2560.0, 1440.0),
    "urfall": (640.0, 480.0),
}
SOURCE_FPS = {"har-up": 18.0, "lei2fall": 25.0, "mmfall": 30.0, "pre-vfall": 30.0, "urfall": 30.0}


@dataclass(frozen=True)
class DataConfig:
    """统一管理观测窗口、预测窗口和目标画布参数。"""

    fps: float = 30.0
    obs_seconds: float = 2.0
    pred_seconds: float = 3.0
    canvas_width: int = 640
    canvas_height: int = 240
    x_offset: float = 320.0
    coordinate_scale: float = 0.5

    @property
    def obs_frames(self) -> int:
        return int(round(self.obs_seconds * self.fps))

    @property
    def pred_frames(self) -> int:
        return int(round(self.pred_seconds * self.fps))

    @property
    def norm_scale(self) -> np.ndarray:
        return np.asarray([self.canvas_width / 2, self.canvas_height], dtype=np.float32)

    def to_dict(self) -> dict:
        result = asdict(self)
        result.update(obs_frames=self.obs_frames, pred_frames=self.pred_frames)
        return result


class TrajectoryGRU(nn.Module):
    """根据一段二维观测轨迹，一次性回归未来全部轨迹点。"""

    def __init__(
        self,
        obs_frames: int,
        pred_frames: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.obs_frames = obs_frames
        self.pred_frames = pred_frames
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pred_frames * 2),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """编码观测序列，输出 [批次, 预测帧, 2] 形状的坐标。"""

        encoded, _ = self.gru(observations)
        return self.head(encoded[:, -1]).reshape(-1, self.pred_frames, 2)


def extract_sequence_id(*values: object) -> str | None:
    for value in values:
        if value is None:
            continue
        match = SEQUENCE_RE.search(str(value))
        if match:
            return match.group("sequence").lower()
    return None


def extract_group_id(source_id: object) -> str | None:
    if source_id is None or pd.isna(source_id):
        return None
    value = str(source_id)
    if "_frame" in value:
        return value.rsplit("_frame", 1)[0]
    match = PREVFALL_RE.match(value)
    if match:
        return match.group(1)
    return extract_sequence_id(value) or value


def canonical_source(value: object) -> str:
    return str(value).strip().casefold()


def discover_keypoint_files(data_root: Path) -> list[Path]:
    """递归查找关键点表格，同时排除本项目生成的中间文件。"""

    candidates: list[Path] = []
    for pattern in ("*.xlsx", "*.xls", "*.csv", "*.parquet"):
        candidates.extend(data_root.rglob(pattern))
    ignored_names = {
        "sequence_split_recommendation.csv",
        "recommended_group_split.csv",
        "group_stats.csv",
        "standard_pred_output.csv",
        "aligned_keypoints.csv",
        "module1_aligned.csv",
    }
    return sorted(
        path
        for path in set(candidates)
        if path.name.lower() not in ignored_names and not path.name.startswith("~$")
    )


def read_table(path: Path) -> pd.DataFrame:
    """根据文件扩展名读取一份完整关键点表格。"""

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def read_table_chunks(path: Path, chunk_size: int = 100_000) -> Iterable[pd.DataFrame]:
    if path.suffix.lower() == ".csv":
        yield from pd.read_csv(path, encoding="utf-8-sig", chunksize=chunk_size)
    else:
        yield read_table(path)


def _valid_pose_columns(df: pd.DataFrame) -> list[tuple[str, str]]:
    result = []
    for col in df.columns:
        if col.endswith("x") and f"{col[:-1]}y" in df.columns:
            result.append((col, f"{col[:-1]}y"))
    return result


def standardize_keypoints(
    raw: pd.DataFrame,
    source_path: Path,
    config: DataConfig,
    min_center_weight: float = 1.0,
) -> pd.DataFrame:
    """清洗关键点，并统一序列标识、坐标范围与人体中心定义。"""

    missing = sorted(REQUIRED_COLUMNS.difference(raw.columns))
    if missing:
        raise ValueError(f"缺少必要字段：{', '.join(missing)}")
    if "frame_index" not in raw.columns and "frame_id" not in raw.columns:
        raise ValueError("缺少必要字段 frame_index（或兼容字段 frame_id）")

    df = raw.copy()
    sequence_from_file = extract_sequence_id(source_path.name)
    if "source_id" in df.columns:
        sequence_values = df["source_id"].map(extract_group_id)
    else:
        sequence_values = pd.Series([None] * len(df), index=df.index)
    if sequence_from_file is not None:
        sequence_values = sequence_values.fillna(sequence_from_file)
    df["sequence_id"] = sequence_values
    if df["sequence_id"].isna().any():
        raise ValueError("无法从 source_id 或文件名确定序列 ID")
    if "dataset_source" not in df:
        df["dataset_source"] = source_path.parent.name.replace("_89", "")
    df["dataset_source"] = df["dataset_source"].astype(str).str.strip()

    frame_column = "frame_index" if "frame_index" in df.columns else "frame_id"
    df["frame_index"] = pd.to_numeric(df[frame_column], errors="coerce")
    if "video_person_id" not in df:
        df["video_person_id"] = 1
    df["person_id"] = df["video_person_id"].fillna(1).astype(str).map(lambda x: f"person_{x}")
    if "label_name" not in df:
        df["label_name"] = "Unknown"

    pose_pairs = _valid_pose_columns(df)
    required_numeric = {name for pair in pose_pairs for name in pair}
    required_numeric.add("frame_index")
    for col in required_numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 常见姿态导出器会用零或负坐标表示关键点缺失，先统一转换为 NaN。
    for x_col, y_col in pose_pairs:
        missing_joint = (df[x_col] <= 0) | (df[y_col] <= 0)
        df.loc[missing_joint, [x_col, y_col]] = np.nan

    center_x_cols = [f"{name}x" for name in CENTER_POINTS]
    center_y_cols = [f"{name}y" for name in CENTER_POINTS]
    df = df.dropna(subset=["frame_index"]).copy()

    x_cols = [x for x, _ in pose_pairs]
    y_cols = [y for _, y in pose_pairs]
    rgb_width = config.canvas_width - config.x_offset
    df["source_fps"] = config.fps

    # 按各数据源的原始分辨率，将坐标映射到统一画布的 RGB 区域。
    for source_name, indices in df.groupby("dataset_source", sort=False).groups.items():
        source_key = canonical_source(source_name)
        if source_key in SOURCE_GEOMETRY:
            source_width, source_height = SOURCE_GEOMETRY[source_key]
        else:
            source_width = max(float(np.nanmax(df.loc[indices, x_cols].to_numpy())), 1.0)
            source_height = max(float(np.nanmax(df.loc[indices, y_cols].to_numpy())), 1.0)
        df.loc[indices, x_cols] = (
            df.loc[indices, x_cols] / source_width * rgb_width + config.x_offset
        )
        df.loc[indices, y_cols] = df.loc[indices, y_cols] / source_height * config.canvas_height
        df.loc[indices, "source_fps"] = SOURCE_FPS.get(source_key, config.fps)
    if "timestamp_sec" in df.columns:
        df["timestamp_sec"] = pd.to_numeric(df["timestamp_sec"], errors="coerce")
    else:
        df["timestamp_sec"] = np.nan
    derived_timestamp = df["frame_index"] / df["source_fps"]
    df["timestamp_sec"] = df["timestamp_sec"].fillna(derived_timestamp)
    df[x_cols] = df[x_cols].clip(0, config.canvas_width - 1)
    df[y_cols] = df[y_cols].clip(0, config.canvas_height - 1)

    center_x = df[center_x_cols].to_numpy(np.float32)
    center_y = df[center_y_cols].to_numpy(np.float32)

    # 仅让有效关键点参与加权，避免缺失关节造成中心位置跳变。
    valid_center = np.isfinite(center_x) & np.isfinite(center_y)
    effective_weights = valid_center * CENTER_WEIGHTS[None, :]
    weight_sum = effective_weights.sum(axis=1)
    keep_center = weight_sum >= min_center_weight
    df = df.loc[keep_center].copy()
    effective_weights = effective_weights[keep_center]
    weight_sum = weight_sum[keep_center]
    center_x = center_x[keep_center]
    center_y = center_y[keep_center]
    df["human_center_x"] = np.nansum(center_x * effective_weights, axis=1) / weight_sum
    df["human_center_y"] = np.nansum(center_y * effective_weights, axis=1) / weight_sum
    df["pose_center_weight"] = weight_sum
    xs = df[x_cols].to_numpy(np.float32)
    ys = df[y_cols].to_numpy(np.float32)
    df["bbox_xmin"] = np.nanmin(xs, axis=1).clip(0, config.canvas_width - 1)
    df["bbox_xmax"] = np.nanmax(xs, axis=1).clip(0, config.canvas_width - 1)
    df["bbox_ymin"] = np.nanmin(ys, axis=1).clip(0, config.canvas_height - 1)
    df["bbox_ymax"] = np.nanmax(ys, axis=1).clip(0, config.canvas_height - 1)
    df["frame_index"] = df["frame_index"].astype(int)
    df["source_file"] = str(source_path.resolve())
    keep = [
        "dataset_source", "sequence_id", "person_id", "frame_index", "timestamp_sec", "source_fps", "label_name",
        "pose_center_weight", "human_center_x", "human_center_y",
        "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax", "source_file",
    ]
    return df[keep].sort_values(["sequence_id", "person_id", "frame_index"]).drop_duplicates(
        ["sequence_id", "person_id", "frame_index"], keep="last"
    )


def split_contiguous_tracks(df: pd.DataFrame) -> Iterable[pd.DataFrame]:
    """按人物轨迹分组，并在帧号不连续处切成独立片段。"""

    group_columns = ["dataset_source", "sequence_id", "person_id", "source_file"]
    for _, track in df.groupby(group_columns, sort=False):
        track = track.sort_values("frame_index").reset_index(drop=True)
        segment_ids = track["frame_index"].diff().fillna(1).ne(1).cumsum()
        for _, segment in track.groupby(segment_ids, sort=False):
            if len(segment):
                yield segment.reset_index(drop=True)


def resample_track(track: pd.DataFrame, target_fps: float) -> pd.DataFrame:
    """将连续轨迹重采样到目标帧率，并线性插值坐标。"""

    source_fps = float(track["source_fps"].iloc[0])
    if len(track) < 2 or abs(source_fps - target_fps) < 1e-6:
        return track
    old_t = np.arange(len(track), dtype=np.float64) / source_fps
    new_t = np.arange(0.0, old_t[-1] + 0.5 / target_fps, 1.0 / target_fps)
    nearest = np.minimum(np.rint(new_t * source_fps).astype(int), len(track) - 1)
    result = track.iloc[nearest].reset_index(drop=True).copy()
    columns = (
        "human_center_x",
        "human_center_y",
        "bbox_xmin",
        "bbox_ymin",
        "bbox_xmax",
        "bbox_ymax",
    )
    for column in columns:
        result[column] = np.interp(new_t, old_t, track[column].to_numpy(np.float64))
    result["frame_index"] = np.rint(np.interp(new_t, old_t, track["frame_index"])).astype(int)
    result["source_fps"] = target_fps
    return result


def make_windows(
    df: pd.DataFrame,
    config: DataConfig,
    stride: int = 1,
    min_future_frames: int = 1,
) -> dict[str, np.ndarray]:
    """把连续轨迹切成定长窗口，并保留未来帧的有效掩码。"""

    obs_values, pred_values, mask_values, bbox_values = [], [], [], []
    datasets, sequences, persons, end_frames, labels, sources = [], [], [], [], [], []
    for track in split_contiguous_tracks(df):
        track = resample_track(track, config.fps)
        if len(track) < config.obs_frames + min_future_frames:
            continue
        centers = track[["human_center_x", "human_center_y"]].to_numpy(np.float32)
        bboxes = track[["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]].to_numpy(np.float32)
        last_start = len(track) - config.obs_frames - min_future_frames
        for start in range(0, last_start + 1, stride):
            split_at = start + config.obs_frames
            obs_values.append(centers[start:split_at])
            future = centers[split_at:split_at + config.pred_frames]
            valid_length = len(future)
            if valid_length < config.pred_frames:
                # 尾部用最后一个真实点补齐；掩码保证补齐部分不参与损失。
                future = np.concatenate(
                    [
                        future,
                        np.repeat(future[-1:], config.pred_frames - valid_length, axis=0),
                    ],
                    axis=0,
                )
            pred_values.append(future)
            mask = np.zeros(config.pred_frames, dtype=np.float32)
            mask[:valid_length] = 1.0
            mask_values.append(mask)
            bbox_values.append(bboxes[split_at - 1])
            row = track.iloc[split_at - 1]
            datasets.append(row["dataset_source"])
            sequences.append(row["sequence_id"])
            persons.append(row["person_id"])
            end_frames.append(row["frame_index"])
            labels.append(row["label_name"])
            sources.append(row["source_file"])
    return {
        "obs": np.asarray(obs_values, dtype=np.float32).reshape(-1, config.obs_frames, 2),
        "pred": np.asarray(pred_values, dtype=np.float32).reshape(-1, config.pred_frames, 2),
        "pred_mask": np.asarray(mask_values, dtype=np.float32).reshape(-1, config.pred_frames),
        "bbox": np.asarray(bbox_values, dtype=np.float32).reshape(-1, 4),
        "dataset_source": np.asarray(datasets, dtype=str),
        "sequence": np.asarray(sequences, dtype=str),
        "person": np.asarray(persons, dtype=str),
        "end_frame": np.asarray(end_frames, dtype=np.int64),
        "label": np.asarray(labels, dtype=str),
        "source": np.asarray(sources, dtype=str),
    }


def relative_normalize(
    obs: np.ndarray,
    pred: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """以最后一个观测点为原点，对观测和未来坐标做相对归一化。"""

    anchor = obs[:, -1:, :]
    return (obs - anchor) / scale, (pred - anchor) / scale


def load_checkpoint(path: Path, device: torch.device) -> tuple[TrajectoryGRU, dict]:
    """恢复模型参数，并切换到评估模式。"""

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_cfg = checkpoint["model_config"]
    model = TrajectoryGRU(**model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def save_json(data: dict, path: Path) -> None:
    """以 UTF-8 和便于阅读的缩进格式写入 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
