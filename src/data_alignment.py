"""骨架 CSV 的流式对齐与字段保留工具。

本模块刻意不依赖 PyTorch，便于在只安装 NumPy/Pandas 的环境中整理大型数据集。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


GROUP_SUFFIX_RE = re.compile(r"(.+?)_(\d+)_keypoints$", re.IGNORECASE)
BODY25_POINTS = (
    "nose",
    "neck",
    "rshoulder",
    "relbow",
    "rwrist",
    "lshoulder",
    "lelbow",
    "lwrist",
    "midhip",
    "rhip",
    "rknee",
    "rankle",
    "lhip",
    "lknee",
    "lankle",
    "reye",
    "leye",
    "rear",
    "lear",
    "lbigtoe",
    "lsmalltoe",
    "lheel",
    "rbigtoe",
    "rsmalltoe",
    "rheel",
)
POSE_COLUMNS = tuple(
    coordinate
    for point in BODY25_POINTS
    for coordinate in (f"{point}x", f"{point}y")
)
VECTOR_COLUMNS = tuple(
    [f"gd_vector{index}" for index in range(16)]
    + [f"gm_vector{index}" for index in range(16)]
)
META_COLUMNS = (
    "source_id",
    "dataset_source",
    "label_name",
    "label_id",
    "frame_id",
    "video_person_id",
    "global_person_id",
)
CENTER_POINTS = ("nose", "midhip", "rankle", "lankle")
CENTER_WEIGHTS = np.asarray([0.3, 0.5, 0.1, 0.1], dtype=np.float32)

# 原始坐标分辨率与采样帧率，名称统一使用 casefold 后的形式。
SOURCE_GEOMETRY = {
    "har-up": (640.0, 480.0),
    "lei2fall": (320.0, 240.0),
    "mmfall": (320.0, 180.0),
    "pre-vfall": (2560.0, 1440.0),
    "urfall": (640.0, 480.0),
}
SOURCE_FPS = {
    "har-up": 18.0,
    "lei2fall": 25.0,
    "mmfall": 30.0,
    "pre-vfall": 30.0,
    "urfall": 30.0,
}


@dataclass(frozen=True)
class AlignmentConfig:
    """统一画布和人体中心计算参数。"""

    canvas_width: int = 640
    canvas_height: int = 240
    x_offset: float = 320.0
    default_fps: float = 30.0
    min_center_weight: float = 1.0


def canonical_source(value: object) -> str:
    """统一数据源名称的空白与大小写。"""

    return str(value).strip().casefold()


def extract_group_id(source_id: object) -> str:
    """从逐帧 source_id 中恢复视频或连续序列级 group_id。"""

    value = str(source_id)
    if "_frame" in value:
        return value.rsplit("_frame", 1)[0]

    match = GROUP_SUFFIX_RE.match(value)
    if match:
        return match.group(1)
    return value


def discover_source_csv_files(data_root: Path) -> list[Path]:
    """发现骨架 CSV，排除 splits 目录和项目生成结果。"""

    ignored = {
        "recommended_group_split.csv",
        "group_stats.csv",
        "aligned_keypoints.csv",
        "standard_pred_output.csv",
    }
    return sorted(
        path
        for path in data_root.rglob("*.csv")
        if "splits" not in {part.casefold() for part in path.parts}
        and path.name.casefold() not in ignored
    )


def _validate_columns(frame: pd.DataFrame, source_path: Path) -> None:
    required = set(META_COLUMNS) | set(POSE_COLUMNS)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"{source_path} 缺少必要字段: {', '.join(missing)}"
        )


def _numeric_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def align_keypoint_chunk(
    raw: pd.DataFrame,
    source_path: Path,
    config: AlignmentConfig,
) -> pd.DataFrame:
    """对齐一块骨架数据，同时保留完整 BODY25、身份和派生向量字段。"""

    _validate_columns(raw, source_path)
    frame = raw.copy()
    _numeric_columns(
        frame,
        ("frame_id", "video_person_id", "global_person_id", "label_id")
        + POSE_COLUMNS
        + VECTOR_COLUMNS,
    )
    frame["frame_index"] = frame["frame_id"]
    frame = frame.dropna(subset=["frame_index", "source_id", "dataset_source"]).copy()
    frame["frame_index"] = frame["frame_index"].astype(np.int64)
    frame["group_id"] = frame["source_id"].map(extract_group_id)
    frame["sequence_id"] = frame["group_id"]
    frame["person_id"] = (
        "person_" + frame["video_person_id"].fillna(0).astype("Int64").astype(str)
    )

    # 零或负坐标代表缺失关节；转为 NaN 后不会参与中心及包围框计算。
    for point in BODY25_POINTS:
        x_column, y_column = f"{point}x", f"{point}y"
        missing_joint = (frame[x_column] <= 0) | (frame[y_column] <= 0)
        frame.loc[missing_joint, [x_column, y_column]] = np.nan

    x_columns = [f"{point}x" for point in BODY25_POINTS]
    y_columns = [f"{point}y" for point in BODY25_POINTS]
    rgb_width = config.canvas_width - config.x_offset
    frame["source_fps"] = config.default_fps

    # 每个数据源独立映射，避免混合分辨率造成坐标尺度错误。
    for source_name, indices in frame.groupby("dataset_source", sort=False).groups.items():
        source_key = canonical_source(source_name)
        if source_key in SOURCE_GEOMETRY:
            source_width, source_height = SOURCE_GEOMETRY[source_key]
        else:
            source_width = max(
                float(np.nanmax(frame.loc[indices, x_columns].to_numpy())),
                1.0,
            )
            source_height = max(
                float(np.nanmax(frame.loc[indices, y_columns].to_numpy())),
                1.0,
            )

        frame.loc[indices, x_columns] = (
            frame.loc[indices, x_columns] / source_width * rgb_width
            + config.x_offset
        )
        frame.loc[indices, y_columns] = (
            frame.loc[indices, y_columns]
            / source_height
            * config.canvas_height
        )
        frame.loc[indices, "source_fps"] = SOURCE_FPS.get(
            source_key,
            config.default_fps,
        )

    frame[x_columns] = frame[x_columns].clip(0, config.canvas_width - 1)
    frame[y_columns] = frame[y_columns].clip(0, config.canvas_height - 1)

    x_values = frame[x_columns].to_numpy(np.float32)
    y_values = frame[y_columns].to_numpy(np.float32)
    valid_pose = np.isfinite(x_values) & np.isfinite(y_values)
    has_pose = valid_pose.any(axis=1)
    frame = frame.loc[has_pose].copy()
    x_values = x_values[has_pose]
    y_values = y_values[has_pose]

    frame["bbox_xmin"] = np.nanmin(x_values, axis=1)
    frame["bbox_xmax"] = np.nanmax(x_values, axis=1)
    frame["bbox_ymin"] = np.nanmin(y_values, axis=1)
    frame["bbox_ymax"] = np.nanmax(y_values, axis=1)

    center_x = frame[[f"{point}x" for point in CENTER_POINTS]].to_numpy(
        np.float32
    )
    center_y = frame[[f"{point}y" for point in CENTER_POINTS]].to_numpy(
        np.float32
    )
    valid_center = np.isfinite(center_x) & np.isfinite(center_y)
    effective_weights = valid_center * CENTER_WEIGHTS[None, :]
    weight_sum = effective_weights.sum(axis=1)
    safe_weight_sum = np.maximum(weight_sum, np.finfo(np.float32).eps)

    weighted_x = np.nansum(center_x * effective_weights, axis=1) / safe_weight_sum
    weighted_y = np.nansum(center_y * effective_weights, axis=1) / safe_weight_sum
    fallback_x = (frame["bbox_xmin"].to_numpy() + frame["bbox_xmax"].to_numpy()) / 2
    fallback_y = (frame["bbox_ymin"].to_numpy() + frame["bbox_ymax"].to_numpy()) / 2
    use_weighted = weight_sum >= config.min_center_weight

    frame["pose_center_weight"] = weight_sum
    frame["center_method"] = np.where(
        use_weighted,
        "weighted_keypoints",
        "bbox_fallback",
    )
    frame["human_center_x"] = np.where(use_weighted, weighted_x, fallback_x)
    frame["human_center_y"] = np.where(use_weighted, weighted_y, fallback_y)
    frame["source_file"] = str(source_path.resolve())

    derived_columns = [
        "group_id",
        "sequence_id",
        "person_id",
        "frame_index",
        "source_fps",
        "pose_center_weight",
        "center_method",
        "human_center_x",
        "human_center_y",
        "bbox_xmin",
        "bbox_ymin",
        "bbox_xmax",
        "bbox_ymax",
    ]
    preserved_columns = [
        column
        for column in META_COLUMNS + POSE_COLUMNS + VECTOR_COLUMNS
        if column in frame
    ]
    output_columns = preserved_columns + derived_columns + ["source_file"]
    return frame[output_columns].drop_duplicates(
        ["dataset_source", "group_id", "video_person_id", "frame_index"],
        keep="last",
    )
