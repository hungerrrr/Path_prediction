import pandas as pd
import json
import numpy as np
from config import *

def calc_human_center(row, kp_list, weights):
    """多关键点加权计算人体中心坐标"""
    total_x = 0.0
    total_y = 0.0
    for idx, (kp_name, x_col, y_col) in enumerate(kp_list):
        x = row[x_col]
        y = row[y_col]
        w = weights[idx]
        total_x += x * w
        total_y += y * w
    return total_x, total_y

# 新增：计算单帧人体真实BBox（基于关键点）
def calc_human_bbox(row, kp_list, margin=HUMAN_BBOX_MARGIN):
    """
    基于关键点计算人体真实包围盒（适配蹲下/行走等动作）
    :param row: 单帧数据行
    :param kp_list: 关键点列表（KEYPOINTS_FOR_CENTER）
    :param margin: 极小扩展margin（贴合人体）
    :return: xmin, ymin, xmax, ymax
    """
    # 提取所有关键点的x/y坐标
    kp_xs = [row[x_col] for (_, x_col, y_col) in kp_list]
    kp_ys = [row[y_col] for (_, x_col, y_col) in kp_list]

    # 计算人体真实边界（无margin）
    xmin_raw = min(kp_xs)
    xmax_raw = max(kp_xs)
    ymin_raw = min(kp_ys)  # 头部（nose）是最高点
    ymax_raw = max(kp_ys)  # 脚踝（ankle）是最低点

    # 加极小margin（贴合人体）
    xmin = max(xmin_raw - margin, CLIP_X_MIN)
    xmax = min(xmax_raw + margin, CLIP_X_MAX)
    ymin = max(ymin_raw - margin, CLIP_Y_MIN)
    ymax = min(ymax_raw + margin, CLIP_Y_MAX)

    return xmin, ymin, xmax, ymax

def preprocess_keypoint_data():
    print("【步骤1】读取并清洗全身骨架关键点数据")
    df_raw = pd.read_excel(INPUT_KEYPOINT_XLSX)
    # 保留全部需要的关键点列
    keep_cols = [
        "source_id", "dataset_source", "label_name", "label_id", "frame_id", "video_person_id", "global_person_id",
        "nosex", "nosey", "midhipx", "midhipy", "ranklex", "rankley", "lanklex", "lankley"
    ]
    df = df_raw[keep_cols].copy()
    df = df.dropna(subset=["nosex", "nosey", "midhipx", "midhipy", "ranklex", "rankley", "lanklex", "lankley"])
    df.rename(columns={"frame_id": "frame_index"}, inplace=True)
    df["frame_index"] = df["frame_index"].astype(int)

    # 坐标校正：原始坐标/2 + RGB偏移
    for x_col in ["nosex", "midhipx", "ranklex", "lanklex"]:
        df[x_col] = (df[x_col] / 2 + RGB_X_OFFSET).round().astype(int)
    for y_col in ["nosey", "midhipy", "rankley", "lankley"]:
        df[y_col] = (df[y_col] / 2).round().astype(int)
    # 坐标裁剪到RGB有效区域
    for x_col in ["nosex", "midhipx", "ranklex", "lanklex"]:
        df[x_col] = df[x_col].clip(CLIP_X_MIN, CLIP_X_MAX)
    for y_col in ["nosey", "midhipy", "rankley", "lankley"]:
        df[y_col] = df[y_col].clip(CLIP_Y_MIN, CLIP_Y_MAX)

    # 计算多关键点加权人体中心
    center_x_list = []
    center_y_list = []
    for _, row in df.iterrows():
        cx, cy = calc_human_center(row, KEYPOINTS_FOR_CENTER, KP_WEIGHTS)
        center_x_list.append(round(cx))
        center_y_list.append(round(cy))
    df["human_center_x"] = center_x_list
    df["human_center_y"] = center_y_list

    # 新增：计算每帧人体真实BBox
    bbox_xmin = []
    bbox_ymin = []
    bbox_xmax = []
    bbox_ymax = []
    for _, row in df.iterrows():
        xmin, ymin, xmax, ymax = calc_human_bbox(row, KEYPOINTS_FOR_CENTER)
        bbox_xmin.append(xmin)
        bbox_ymin.append(ymin)
        bbox_xmax.append(xmax)
        bbox_ymax.append(ymax)
    df["human_bbox_xmin"] = bbox_xmin
    df["human_bbox_ymin"] = bbox_ymin
    df["human_bbox_xmax"] = bbox_xmax
    df["human_bbox_ymax"] = bbox_ymax

    df["timestamp_sec"] = df["frame_index"] / FPS
    df["env_objects"] = json.dumps([])

    # 读取序列划分信息
    df_seq = pd.read_csv(INPUT_SEQ_CSV, encoding="utf-8-sig")
    adl01_split = df_seq[df_seq["Sequence"] == "adl-01"]["Split"].values[0]
    print(f"数据集划分类型：{adl01_split}")

    # 输出对齐CSV
    df.to_csv(OUT_ALIGNED_CSV, index=False, encoding="utf-8-sig")
    # 帧-时间映射
    frame2time = dict(zip(df["frame_index"], df["timestamp_sec"]))
    with open(OUT_FRAME_MAP_JSON, "w", encoding="utf-8") as f:
        json.dump(frame2time, f, ensure_ascii=False, indent=2)

    print(f"预处理完成，有效帧数：{len(df)}，输出：{OUT_ALIGNED_CSV}")
    return df, frame2time

if __name__ == "__main__":
    preprocess_keypoint_data()