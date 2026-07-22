"""从统一 89 列骨架数据流式生成增强版 aligned_keypoints.csv。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from data_alignment import (
    AlignmentConfig,
    align_keypoint_chunk,
    canonical_source,
    discover_source_csv_files,
)


def parse_args() -> argparse.Namespace:
    """解析增强对齐数据集的生成参数。"""

    parser = argparse.ArgumentParser(
        description="流式生成保留完整 BODY25 和身份字段的对齐数据集。"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/aligned_keypoints.csv"),
    )
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--canvas-width", type=int, default=640)
    parser.add_argument("--canvas-height", type=int, default=240)
    parser.add_argument("--x-offset", type=float, default=320.0)
    parser.add_argument("--default-fps", type=float, default=30.0)
    parser.add_argument("--min-center-weight", type=float, default=1.0)
    return parser.parse_args()


def load_split_map(path: Path) -> dict[tuple[str, str], str]:
    """读取数据源和 group_id 到 train/val/test 的映射。"""

    split_frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"dataset_source", "group_id", "split"}
    missing = required.difference(split_frame.columns)
    if missing:
        raise ValueError(f"划分文件缺少字段: {', '.join(sorted(missing))}")

    return {
        (
            canonical_source(row.dataset_source),
            str(row.group_id).strip().casefold(),
        ): str(row.split).strip().lower()
        for row in split_frame.itertuples()
    }


def main() -> None:
    """按文件和数据块完成对齐，避免一次性占用大量内存。"""

    args = parse_args()
    if args.chunk_size < 1:
        raise ValueError("--chunk-size 必须大于 0")

    files = discover_source_csv_files(args.data_root)
    if not files:
        raise FileNotFoundError(f"未在 {args.data_root} 找到骨架 CSV")

    config = AlignmentConfig(
        canvas_width=args.canvas_width,
        canvas_height=args.canvas_height,
        x_offset=args.x_offset,
        default_fps=args.default_fps,
        min_center_weight=args.min_center_weight,
    )
    split_map = load_split_map(args.split_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    if temporary_output.exists():
        temporary_output.unlink()

    input_rows = Counter()
    output_rows = Counter()
    split_rows = Counter()
    center_methods = Counter()
    header_written = False

    for source_path in files:
        for raw_chunk in pd.read_csv(
            source_path,
            encoding="utf-8-sig",
            chunksize=args.chunk_size,
        ):
            input_rows[source_path.name] += len(raw_chunk)
            aligned = align_keypoint_chunk(raw_chunk, source_path, config)
            keys = zip(
                aligned["dataset_source"].map(canonical_source),
                aligned["group_id"].astype(str).str.strip().str.casefold(),
            )
            aligned["split"] = [
                split_map.get(key, "unassigned")
                for key in keys
            ]

            # split 放在派生标识附近，便于人工查看和后续筛选。
            split_column = aligned.pop("split")
            insert_at = aligned.columns.get_loc("group_id") + 1
            aligned.insert(insert_at, "split", split_column)

            aligned.to_csv(
                temporary_output,
                mode="a",
                header=not header_written,
                index=False,
                encoding="utf-8-sig" if not header_written else "utf-8",
                float_format="%.6f",
            )
            header_written = True
            output_rows[source_path.name] += len(aligned)
            split_rows.update(aligned["split"].value_counts().to_dict())
            center_methods.update(
                aligned["center_method"].value_counts().to_dict()
            )

    temporary_output.replace(args.output)
    manifest = {
        "schema_version": "aligned_keypoints.v3",
        "source_root": str(args.data_root.resolve()),
        "split_file": str(args.split_file.resolve()),
        "output_file": str(args.output.resolve()),
        "config": {
            "canvas_width": config.canvas_width,
            "canvas_height": config.canvas_height,
            "x_offset": config.x_offset,
            "default_fps": config.default_fps,
            "min_center_weight": config.min_center_weight,
        },
        "input_rows_by_file": dict(input_rows),
        "output_rows_by_file": dict(output_rows),
        "split_rows": dict(split_rows),
        "center_methods": dict(center_methods),
        "preserved_fields": {
            "metadata": 7,
            "body25_coordinates": 50,
            "derived_vectors": 32,
            "identity_fields": [
                "video_person_id",
                "global_person_id",
                "person_id",
            ],
        },
    }
    manifest_path = args.output.with_name("aligned_keypoints_manifest.json")
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(f"增强对齐文件: {args.output.resolve()}")
    print(f"输出总行数: {sum(output_rows.values())}")
    print(f"数据划分统计: {dict(split_rows)}")
    print(f"人体中心来源: {dict(center_methods)}")


if __name__ == "__main__":
    main()
