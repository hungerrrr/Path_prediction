"""规范化关键点目录，并生成可避免序列泄漏的训练、验证和测试数据。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from core import (
    SOURCE_FPS,
    SOURCE_GEOMETRY,
    DataConfig,
    canonical_source,
    discover_keypoint_files,
    extract_sequence_id,
    make_windows,
    read_table_chunks,
    save_json,
    standardize_keypoints,
)


def parse_args() -> argparse.Namespace:
    """解析数据准备阶段的命令行参数。"""

    parser = argparse.ArgumentParser(
        description="从关键点表格目录生成轨迹训练数据。"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="递归包含 xlsx/csv/parquet 关键点文件的目录",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        required=True,
        help="推荐的序列或分组划分 CSV",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--obs-seconds", type=float, default=2.0)
    parser.add_argument("--pred-seconds", type=float, default=3.0)
    parser.add_argument("--stride", type=int, default=5, help="滑动窗口步长（帧）")
    parser.add_argument(
        "--min-future-seconds",
        type=float,
        default=0.5,
        help="每个窗口至少需要的真实未来时长",
    )
    parser.add_argument(
        "--min-center-weight",
        type=float,
        default=1.0,
        help="人体中心关键点的最小有效权重（1.0 表示四点齐全）",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="划分中的序列缺少匹配表格时直接失败",
    )
    return parser.parse_args()


def main() -> None:
    """执行文件发现、关键点对齐、数据划分和窗口生成。"""

    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride 不能小于 1")
    config = DataConfig(fps=args.fps, obs_seconds=args.obs_seconds, pred_seconds=args.pred_seconds)
    # 划分表优先使用“数据源 + 组编号”，兼容仅提供序列号的旧格式。
    split_df = pd.read_csv(args.split_file, encoding="utf-8-sig")
    split_df = split_df.rename(
        columns={column: column.strip().lower() for column in split_df.columns}
    )
    if not {"split"}.issubset(split_df.columns):
        raise ValueError("数据划分文件必须包含 split 字段")
    invalid = sorted(set(split_df["split"]) - {"train", "val", "test"})
    if invalid:
        raise ValueError(f"不支持的数据划分值：{invalid}")
    group_split = {"dataset_source", "group_id"}.issubset(split_df.columns)
    sequence_split = "sequence" in split_df.columns
    if not group_split and not sequence_split:
        raise ValueError("数据划分文件必须包含 dataset_source 与 group_id，或包含 Sequence")
    if group_split:
        split_map = {
            (
                canonical_source(row.dataset_source),
                str(row.group_id).strip().casefold(),
            ): str(row.split).strip().lower()
            for row in split_df.itertuples()
        }
    else:
        split_map = {
            str(row.sequence).strip().casefold(): str(row.split).strip().lower()
            for row in split_df.itertuples()
        }

    # 大型 CSV 按块处理，避免一次性加载全部原始数据。
    files = discover_keypoint_files(args.data_root)
    if not files:
        raise FileNotFoundError(f"未在 {args.data_root} 下找到骨架表")
    frames, skipped, unknown_keys = [], [], set()
    for path in files:
        try:
            file_rows = 0
            for chunk in read_table_chunks(path):
                prepared = standardize_keypoints(
                    chunk,
                    path,
                    config,
                    min_center_weight=args.min_center_weight,
                )
                if group_split:
                    keys = list(
                        zip(
                            prepared["dataset_source"].map(canonical_source),
                            prepared["sequence_id"].str.casefold(),
                        )
                    )
                else:
                    keys = [
                        (extract_sequence_id(value) or str(value)).casefold()
                        for value in prepared["sequence_id"]
                    ]
                prepared["split"] = [split_map.get(key) for key in keys]
                unknown_keys.update(
                    key
                    for key, value in zip(keys, prepared["split"])
                    if value is None
                )
                prepared = prepared.dropna(subset=["split"])
                if len(prepared):
                    frames.append(prepared)
                    file_rows += len(prepared)
            if not file_rows:
                skipped.append(
                    {
                        "file": str(path),
                        "reason": "no rows matched the recommended split",
                    }
                )
        except Exception as exc:
            skipped.append({"file": str(path), "reason": str(exc)})

    if not frames:
        raise RuntimeError("没有可用骨架记录匹配推荐的数据划分")
    # 合并后再次去重，保证同一人物同一帧只保留最后一条记录。
    all_frames = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["dataset_source", "sequence_id", "person_id", "frame_index"], keep="last"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_frames.to_csv(args.output_dir / "aligned_keypoints.csv", index=False, encoding="utf-8-sig")

    stats = {}
    if group_split:
        found_sequences = set(
            zip(
                all_frames["dataset_source"].map(canonical_source),
                all_frames["sequence_id"].str.casefold(),
            )
        )
    else:
        found_sequences = {
            (extract_sequence_id(value) or str(value)).casefold()
            for value in all_frames["sequence_id"]
        }
    missing_keys = sorted(set(split_map) - found_sequences)
    missing_sequences = ["::".join(key) if isinstance(key, tuple) else key for key in missing_keys]
    # 三个集合分别生成窗口，杜绝同一序列跨集合泄漏。
    for split in ("train", "val", "test"):
        split_frames = all_frames[all_frames["split"] == split].copy()
        min_future_frames = max(1, int(round(args.min_future_seconds * config.fps)))
        arrays = make_windows(
            split_frames,
            config,
            stride=args.stride,
            min_future_frames=min_future_frames,
        )
        np.savez_compressed(args.output_dir / f"{split}.npz", **arrays)
        stats[split] = {
            "sequences": int(
                split_frames[["dataset_source", "sequence_id"]]
                .drop_duplicates()
                .shape[0]
            ),
            "frames": int(len(split_frames)),
            "windows": int(len(arrays["obs"])),
        }

    manifest = {
        "schema_version": "trajectory_dataset.v2",
        "data_config": config.to_dict(),
        "normalization": {
            "type": "relative_to_last_observation",
            "scale": config.norm_scale.tolist(),
        },
        "source_geometry": {key: list(value) for key, value in SOURCE_GEOMETRY.items()},
        "source_fps": SOURCE_FPS,
        "target_fps": config.fps,
        "stride": args.stride,
        "min_future_seconds": args.min_future_seconds,
        "min_center_weight": args.min_center_weight,
        "source_files": len(files),
        "matched_sequences": len(found_sequences),
        "missing_sequences": missing_sequences,
        "unknown_data_groups": [
            "::".join(key) if isinstance(key, tuple) else key
            for key in sorted(unknown_keys)
        ],
        "skipped_files": skipped,
        "splits": stats,
    }
    save_json(manifest, args.output_dir / "manifest.json")
    print(f"Prepared dataset: {args.output_dir.resolve()}")
    for split, values in stats.items():
        print(
            f"  {split}: {values['sequences']} sequences, "
            f"{values['frames']} frames, {values['windows']} windows"
        )
    if skipped:
        print(f"  skipped files: {len(skipped)} (details in manifest.json)")
    if missing_sequences:
        print(f"  split sequences without data: {len(missing_sequences)}")
        if args.strict:
            raise RuntimeError("严格模式：推荐划分中的部分序列没有数据")


if __name__ == "__main__":
    main()
