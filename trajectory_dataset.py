import json
import pandas as pd
import numpy as np
from config import *


def normalize_coord(coord_arr, x_min, x_max, y_min, y_max):
    x = (coord_arr[:, 0] - x_min) / (x_max - x_min)
    y = (coord_arr[:, 1] - y_min) / (y_max - y_min)
    return np.stack([x, y], axis=-1)


def filter_static_traj(traj, threshold=0.1):  # 降低静态阈值（几乎不过滤）
    diffs = np.diff(traj, axis=0)
    if len(diffs) == 0:
        return True  # 单帧轨迹保留
    avg_move = np.mean(np.sqrt(np.sum(diffs ** 2, axis=-1)))
    return avg_move > threshold


def build_trajectory_dataset(df_aligned):
    print("【步骤2】构建滑动窗口轨迹样本数据集")
    df = df_aligned.sort_values("frame_index").reset_index(drop=True)
    all_x = df["human_center_x"].values
    all_y = df["human_center_y"].values
    X_MIN, X_MAX = all_x.min(), all_x.max()
    Y_MIN, Y_MAX = all_y.min(), all_y.max()
    print(f"人体中心坐标全局范围 X:[{X_MIN},{X_MAX}] Y:[{Y_MIN},{Y_MAX}]")

    # 修改：支持任意长度窗口（不再强制WINDOW_TOTAL_LEN）
    total_frames = len(df)
    samples = []
    valid_cnt = 0

    for current_frame in range(total_frames):
        # 取当前帧及之前的所有帧作为观测（最多OBS_FRAMES）
        obs_start = max(0, current_frame - OBS_FRAMES + 1)
        obs_df = df.iloc[obs_start:current_frame + 1].copy()

        # 取当前帧之后的帧作为预测（最多PRED_FRAMES，不足则补当前位置）
        pred_end = min(total_frames, current_frame + 1 + PRED_FRAMES)
        pred_df = df.iloc[current_frame + 1:pred_end].copy()

        # 过滤规则1：仅Normal行走
        action_label = obs_df["label_name"].unique()[0]
        if action_label != "Normal":
            continue

        # 过滤规则2：放宽静态轨迹过滤
        obs_traj_raw = np.stack([obs_df["human_center_x"], obs_df["human_center_y"]], axis=1)
        if not filter_static_traj(obs_traj_raw):
            # 静态轨迹：预测路径填充为当前位置
            pred_traj_raw = np.array([obs_traj_raw[-1]] * PRED_FRAMES)
        else:
            # 动态轨迹：取真实预测帧，不足则补最后位置
            if len(pred_df) < PRED_FRAMES:
                pred_pad = np.array([[obs_traj_raw[-1][0], obs_traj_raw[-1][1]]] * (PRED_FRAMES - len(pred_df)))
                pred_traj_raw = np.vstack([
                    np.stack([pred_df["human_center_x"], pred_df["human_center_y"]], axis=1),
                    pred_pad
                ])
            else:
                pred_traj_raw = np.stack([pred_df["human_center_x"], pred_df["human_center_y"]], axis=1)

        # 归一化
        obs_norm = normalize_coord(obs_traj_raw, X_MIN, X_MAX, Y_MIN, Y_MAX)
        pred_norm = normalize_coord(pred_traj_raw, X_MIN, X_MAX, Y_MIN, Y_MAX)
        env_scene = obs_df.iloc[-1]["env_objects"]
        obs_end_frame = int(obs_df.iloc[-1]["frame_index"])

        # 新增：提取当前帧的人体真实BBox
        current_bbox = {
            "xmin": int(obs_df.iloc[-1]["human_bbox_xmin"]),
            "ymin": int(obs_df.iloc[-1]["human_bbox_ymin"]),
            "xmax": int(obs_df.iloc[-1]["human_bbox_xmax"]),
            "ymax": int(obs_df.iloc[-1]["human_bbox_ymax"])
        }

        sample = {
            "window_start_frame": int(obs_df.iloc[0]["frame_index"]),
            "window_end_frame": obs_end_frame,
            "obs_seconds": len(obs_df) / FPS,  # 实际观测时长（不再固定1秒）
            "pred_seconds": PRED_SECOND,
            "fps": FPS,
            "obs_trajectory_pixel": obs_traj_raw.tolist(),
            "pred_trajectory_pixel": pred_traj_raw.tolist(),
            "obs_trajectory_norm": obs_norm.tolist(),
            "pred_trajectory_norm": pred_norm.tolist(),
            "scene_env_objects": json.loads(env_scene),
            "action_type": action_label,
            "person_id": "person_001",
            "human_bbox": current_bbox  # 新增：加入人体真实BBox
        }
        samples.append(sample)
        valid_cnt += 1

    print(f"有效轨迹样本数量：{valid_cnt}")
    dataset_out = {
        "dataset_info": {
            "obs_second": OBS_SECOND,
            "pred_second": PRED_SECOND,
            "fps": FPS,
            "coord_x_range": [float(X_MIN), float(X_MAX)],
            "coord_y_range": [float(Y_MIN), float(Y_MAX)],
            "sample_num": valid_cnt
        },
        "samples": samples
    }
    with open(OUT_TRAJ_DATASET, "w", encoding="utf-8") as f:
        json.dump(dataset_out, f, ensure_ascii=False, indent=2)
    print(f"轨迹数据集已输出：{OUT_TRAJ_DATASET}")
    return dataset_out


if __name__ == "__main__":
    from data_preprocess import preprocess_keypoint_data

    df, _ = preprocess_keypoint_data()
    build_trajectory_dataset(df)