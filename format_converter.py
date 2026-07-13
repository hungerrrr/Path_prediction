import json
import os
import pandas as pd  # 新增：导入pandas处理CSV
from config import *
from kalman_predictor import generate_corridor_bbox_polygon


def convert_to_standard_schema(kalman_input_path, out_path):
    with open(kalman_input_path, "r", encoding="utf-8") as f:
        kalman_data = json.load(f)
    standard_out = {
        "schema_version": "path_prediction.v1",
        "coordinate_system": COORD_SYSTEM,
        "video": VIDEO_NAME,
        "frames": []
    }
    pred_samples = kalman_data["predict_samples"]

    # 构建帧到预测数据的映射（确保全帧覆盖）
    frame_pred_map = {}
    for samp in pred_samples:
        frame_pred_map[samp["window_end"]] = samp

    # 补全所有帧（无预测时基于最近帧生成）
    all_frame_indices = sorted(frame_pred_map.keys())
    for frame_idx in range(all_frame_indices[0], all_frame_indices[-1] + 1):
        # 找最近的预测样本
        if frame_idx in frame_pred_map:
            samp = frame_pred_map[frame_idx]
        else:
            # 取最近的前一帧样本
            nearest_idx = max([f for f in all_frame_indices if f <= frame_idx], default=all_frame_indices[0])
            samp = frame_pred_map[nearest_idx]

        ts_sec = round(frame_idx / FPS, 2)
        current_pos = samp["obs_raw"][-1]
        pred_path = samp["predict_path"]
        corridor_poly = samp["path_corridor"]
        confidence = round(1.0 - min(samp["pixel_mse"] / 100, 0.3), 2)

        frame_item = {
            "frame_index": frame_idx,
            "timestamp_sec": ts_sec,
            "person_id": samp["person_id"],
            "current_position": current_pos,
            "predicted_path": pred_path,
            "path_corridor": corridor_poly,
            "confidence": confidence
        }
        standard_out["frames"].append(frame_item)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(standard_out, f, ensure_ascii=False, indent=2)
    print(f"标准格式预测JSON已生成：{out_path}")

    # ===================== 新增：生成CSV版本 =====================
    # 1. 提取frames列表并转换为DataFrame（处理嵌套列表为字符串）
    frames_data = standard_out["frames"]
    for item in frames_data:
        # 嵌套列表转字符串（CSV不支持直接存储列表）
        item["current_position"] = str(item["current_position"])
        item["predicted_path"] = str(item["predicted_path"])
        item["path_corridor"] = str(item["path_corridor"])

    # 2. 生成CSV文件路径（复用JSON路径，替换后缀为.csv）
    csv_out_path = out_path.replace(".json", ".csv")
    # 3. 写入CSV
    df = pd.DataFrame(frames_data)
    df.to_csv(csv_out_path, index=False, encoding="utf-8-sig")
    print(f"标准格式预测CSV已生成：{csv_out_path}")
    # ============================================================

    return standard_out


if __name__ == "__main__":
    if os.path.exists(OUT_GRU_PRED):
        convert_to_standard_schema(OUT_GRU_PRED, OUT_STANDARD_JSON)
    else:
        print("错误：缺少卡尔曼预测输出文件，请先运行预测流程")