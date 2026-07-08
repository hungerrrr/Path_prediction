import cv2
import json
import numpy as np
from config import *


def convert_original_to_video_coord(orig_x, orig_y):
    """原始640*480 RGB坐标 → 拼接视频右半区RGB坐标"""
    new_x = np.clip(orig_x, RGB_X_OFFSET, TARGET_VIDEO_W)
    new_y = np.clip(orig_y, 0, TARGET_VIDEO_H)
    return int(new_x), int(new_y)


def load_pred_json(json_path):
    frame_map = {}
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data["frames"]:
        fid = item["frame_index"]
        frame_map[fid] = item
    return frame_map


def draw_dashed_line(img, p1, p2, color, thickness, dash_len=5, gap=5):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    if dist < 1e-6:
        return
    ux, uy = dx / dist, dy / dist
    cur = 0
    while cur < dist:
        seg_end = min(cur + dash_len, dist)
        x1 = int(p1[0] + cur * ux)
        y1 = int(p1[1] + cur * uy)
        x2 = int(p1[0] + seg_end * ux)
        y2 = int(p1[1] + seg_end * uy)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
        cur += dash_len + gap


def draw_polygon_corridor(img, poly, color, alpha):
    """增强鲁棒性：处理空/无效多边形"""
    if not poly or len(poly) < 3:
        return
    try:
        pts = np.array([convert_original_to_video_coord(p[0], p[1]) for p in poly], np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        cv2.polylines(img, [pts], True, color, 2)
    except Exception as e:
        print(f"绘制走廊失败：{e}")
        return


def render_visual_video():
    frame_data_map = load_pred_json(OUT_STANDARD_JSON)
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频 {INPUT_VIDEO}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUT_VIS_VIDEO, fourcc, FPS, (TARGET_VIDEO_W, TARGET_VIDEO_H))
    real_track_cache = []
    current_fid = 0
    print("【步骤5】开始渲染可视化视频...")

    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or current_fid >= total_frames:
            break

        frame = cv2.resize(frame, (TARGET_VIDEO_W, TARGET_VIDEO_H))
        frame_info = frame_data_map.get(current_fid)

        # 兜底：无帧信息时取最近帧
        if not frame_info:
            nearest_fid = max([f for f in frame_data_map.keys() if f <= current_fid], default=0)
            frame_info = frame_data_map.get(nearest_fid, {})

        if frame_info:
            curr_pos = frame_info.get("current_position", [0, 0])
            real_track_cache.append(curr_pos)

            # 绘制真实历史轨迹 白色实线
            if len(real_track_cache) >= 2:
                pts = [convert_original_to_video_coord(p[0], p[1]) for p in real_track_cache]
                for i in range(1, len(pts)):
                    cv2.line(frame, pts[i - 1], pts[i], COLOR_REAL, 2)

            # 绘制预测路径 红色虚线
            pred_path = frame_info.get("predicted_path", [])
            if len(pred_path) >= 2:
                pred_pts = [convert_original_to_video_coord(p[0], p[1]) for p in pred_path]
                for i in range(1, len(pred_pts)):
                    draw_dashed_line(frame, pred_pts[i - 1], pred_pts[i], COLOR_PRED, 2)

            # 绘制走廊多边形（核心修复）
            corridor_poly = frame_info.get("path_corridor", [])
            draw_polygon_corridor(frame, corridor_poly, COLOR_CORRIDOR, ALPHA_CORRIDOR)

        # 文本信息
        text_x = RGB_X_OFFSET + 10
        cv2.putText(frame, f"Frame:{current_fid}", (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        ts = frame_info.get("timestamp_sec", round(current_fid / FPS, 2))
        cv2.putText(frame, f"Time:{ts:.2f}s", (text_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f"Pred:{PRED_SECOND}s", (text_x, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        out.write(frame)
        current_fid += 1

    cap.release()
    out.release()
    print(f"可视化视频渲染完成：{OUT_VIS_VIDEO}")


if __name__ == "__main__":
    render_visual_video()