import json
import numpy as np
import pandas as pd
from config import *


class MultiKP2DKalman:
    """多关键点联合2D卡尔曼，平滑人体中心轨迹"""

    def __init__(self):
        self.state = np.zeros(4)  # [x,y,vx,vy]
        self.P = np.eye(4)
        self.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = np.eye(4) * PROCESS_NOISE
        self.R = np.eye(2) * MEASURE_NOISE

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, x, y):
        z = np.array([x, y])
        y_res = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state += K @ y_res
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.state[0], self.state[1]


def smooth_trajectory(raw_traj):
    smoother = MultiKP2DKalman()
    smooth_points = []
    for x, y in raw_traj:
        smoother.predict()
        sx, sy = smoother.update(x, y)
        smooth_points.append([sx, sy])
    return np.array(smooth_points)


def constant_velocity_predict(smooth_obs, pred_frames):
    obs_len = smooth_obs.shape[0]
    if obs_len < 2:  # 观测帧不足（如前1秒），预测路径停在当前位置
        return np.array([smooth_obs[-1]] * pred_frames)

    # 计算速度（如果速度接近0，判定为静止）
    start = smooth_obs[0]
    end = smooth_obs[-1]
    vx = (end[0] - start[0]) / (obs_len - 1)
    vy = (end[1] - start[1]) / (obs_len - 1)

    # 静止判定（速度阈值）
    if abs(vx) < 0.1 and abs(vy) < 0.1:
        return np.array([end] * pred_frames)

    # 正常匀速预测
    pred_path = []
    cx, cy = end[0], end[1]
    for _ in range(pred_frames):
        cx += vx
        cy += vy
        cx = np.clip(cx, CLIP_X_MIN, CLIP_X_MAX)
        cy = np.clip(cy, CLIP_Y_MIN, CLIP_Y_MAX)
        pred_path.append([cx, cy])
    return np.array(pred_path)


# def generate_corridor_polygon(center_point, pred_path=None, padding_x=CORRIDOR_PADDING_X, padding_y=CORRIDOR_PADDING_Y):
#     """
#     重构走廊生成逻辑：
#     - 基于人体中心+人体尺寸生成走廊，确保覆盖整个人体
#     - 有预测路径时：融合路径范围+人体尺寸
#     - 无预测路径时：仅基于人体中心生成（停在人身上）
#     """
#     # 基础人体包围盒（确保覆盖整个人体）
#     base_x_min = center_point[0] - HUMAN_BBOX_W - padding_x
#     base_x_max = center_point[0] + HUMAN_BBOX_W + padding_x
#     base_y_min = center_point[1] - HUMAN_BBOX_H - padding_y
#     base_y_max = center_point[1] + HUMAN_BBOX_H + padding_y
#
#     # 有预测路径时，扩展走廊到预测范围
#     if pred_path is not None and len(pred_path) > 0:
#         pts = np.array(pred_path)
#         pred_x_min, pred_x_max = pts[:, 0].min(), pts[:, 0].max()
#         pred_y_min, pred_y_max = pts[:, 1].min(), pts[:, 1].max()
#         xmin = min(base_x_min, pred_x_min - padding_x)
#         xmax = max(base_x_max, pred_x_max + padding_x)
#         ymin = min(base_y_min, pred_y_min - padding_y)
#         ymax = max(base_y_max, pred_y_max + padding_y)
#     else:
#         xmin, xmax = base_x_min, base_x_max
#         ymin, ymax = base_y_min, base_y_max
#
#     # 裁剪到有效坐标范围
#     xmin = max(xmin, CLIP_X_MIN)
#     xmax = min(xmax, CLIP_X_MAX)
#     ymin = max(ymin, CLIP_Y_MIN)
#     ymax = min(ymax, CLIP_Y_MAX)
#
#     # 顺时针4顶点多边形（确保覆盖整个人体）
#     polygon = [
#         [xmin, ymin],
#         [xmax, ymin],
#         [xmax, ymax],
#         [xmin, ymax]
#     ]
#     return polygon

# def batch_predict(traj_dataset):
#     info = traj_dataset["dataset_info"]
#     pred_frames = int(info["pred_second"] * info["fps"])
#     samples = traj_dataset["samples"]
#     batch_res = []
#     error_records = []
#     total_mse = 0.0
#
#     print(f"【步骤3】批量卡尔曼预测，总样本：{len(samples)}")
#     for idx, samp in enumerate(samples):
#         obs_raw = np.array(samp["obs_trajectory_pixel"])
#         obs_smooth = smooth_trajectory(obs_raw)
#         pred_pixel = constant_velocity_predict(obs_smooth, pred_frames)
#
#         # 生成走廊（基于人体中心+预测路径）
#         center_point = obs_smooth[-1]  # 最后一帧人体中心
#         corridor_poly = generate_corridor_polygon(center_point, pred_pixel)
#
#         # 计算MSE（无真实值时设为0）
#         truth = np.array(samp["pred_trajectory_pixel"])
#         if len(truth) == len(pred_pixel):
#             mse = np.mean(np.square(pred_pixel - truth))
#         else:
#             mse = 0.0
#         total_mse += mse
#
#         res_item = {
#             "window_start": samp["window_start_frame"],
#             "window_end": samp["window_end_frame"],
#             "person_id": samp["person_id"],
#             "action_label": samp["action_type"],
#             "obs_raw": obs_raw.tolist(),
#             "obs_smooth": obs_smooth.tolist(),
#             "predict_path": pred_pixel.tolist(),
#             "path_corridor": corridor_poly,
#             "true_future": truth.tolist(),
#             "pixel_mse": round(mse, 2)
#         }
#         batch_res.append(res_item)
#         error_records.append({
#             "window_end_frame": samp["window_end_frame"],
#             "person_id": samp["person_id"],
#             "mse_pixel": round(mse, 2)
#         })
#         if (idx + 1) % 5 == 0:
#             print(f"已处理 {idx + 1}/{len(samples)}")
#
#     avg_mse = round(total_mse / len(batch_res), 2) if batch_res else 0.0
#     # 输出误差CSV
#     pd.DataFrame(error_records).to_csv(OUT_ERROR_CSV, index=False, encoding="utf-8-sig")
#     print(f"全部样本平均MSE误差：{avg_mse}，误差CSV输出：{OUT_ERROR_CSV}")
#
#     output_data = {
#         "source_dataset": OUT_TRAJ_DATASET,
#         "predict_method": "多关键点加权中心+卡尔曼平滑+匀速外推（兜底版）",
#         "fps": FPS,
#         "obs_sec": OBS_SECOND,
#         "pred_sec": PRED_SECOND,
#         "avg_pixel_mse": avg_mse,
#         "sample_count": len(batch_res),
#         "predict_samples": batch_res
#     }
#     with open(OUT_KALMAN_PRED, "w", encoding="utf-8") as f:
#         json.dump(output_data, f, ensure_ascii=False, indent=2)
#     print(f"卡尔曼预测结果输出：{OUT_KALMAN_PRED}")
#     return output_data

def generate_corridor_bbox_polygon(pred_path, human_bbox, padding=HUMAN_BBOX_MARGIN):
    """
    修复坐标索引错误：基于人体真实BBox + 预测路径生成贴合人体的corridor
    :param pred_path: 预测路径坐标列表 [[x1,y1], [x2,y2], ...]
    :param human_bbox: 人体真实BBox {"xmin":x1, "ymin":y1, "xmax":x2, "ymax":y2}
    :param padding: 极小margin（贴合人体）
    :return: 4顶点顺时针多边形（BBox）
    """
    # 兼容空路径：直接用人体真实BBox
    if len(pred_path) == 0 or np.array(pred_path).size == 0:
        return [
            [human_bbox["xmin"], human_bbox["ymin"]],
            [human_bbox["xmax"], human_bbox["ymin"]],
            [human_bbox["xmax"], human_bbox["ymax"]],
            [human_bbox["xmin"], human_bbox["ymax"]]
        ]

    # 1. 提取人体真实BBox的尺寸和中心（基准）
    human_xmin = human_bbox["xmin"]
    human_ymin = human_bbox["ymin"]
    human_xmax = human_bbox["xmax"]
    human_ymax = human_bbox["ymax"]
    human_w = human_xmax - human_xmin  # 人体宽度
    human_h = human_ymax - human_ymin  # 人体高度
    human_cx = (human_xmin + human_xmax) / 2  # 人体中心x
    human_cy = (human_ymin + human_ymax) / 2  # 人体中心y

    # 2. 处理预测路径（确保是二维数组）
    pred_pts = np.array(pred_path)
    if pred_pts.ndim != 2:
        pred_pts = pred_pts.reshape(-1, 2)

    # 3. 为预测路径的每个中心生成「和人体等大」的BBox
    pred_bboxes = []
    for (cx, cy) in pred_pts:
        # 基于人体真实尺寸，以预测点为中心生成BBox（核心修正：索引不再错位）
        xmin = cx - (human_w / 2)
        ymin = cy - (human_h / 2)
        xmax = cx + (human_w / 2)
        ymax = cy + (human_h / 2)
        pred_bboxes.append([xmin, ymin, xmax, ymax])

    # 4. 计算所有预测BBox的整体包围盒（核心修复：坐标列索引修正）
    pred_bboxes = np.array(pred_bboxes)
    # 正确索引：[:,0]=xmin  | [:,2]=xmax  | [:,1]=ymin  | [:,3]=ymax
    global_xmin = max(pred_bboxes[:, 0].min() - padding, CLIP_X_MIN)
    global_xmax = min(pred_bboxes[:, 2].max() + padding, CLIP_X_MAX)
    global_ymin = max(pred_bboxes[:, 1].min() - padding, CLIP_Y_MIN)
    global_ymax = min(pred_bboxes[:, 3].max() + padding, CLIP_Y_MAX)

    # 5. 兜底：如果预测路径异常，直接用当前人体BBox
    if global_xmax <= global_xmin or global_ymax <= global_ymin:
        global_xmin = human_xmin - padding
        global_xmax = human_xmax + padding
        global_ymin = human_ymin - padding
        global_ymax = human_ymax + padding

    # 6. 生成顺时针4顶点多边形（确保在有效坐标范围内）
    bbox_polygon = [
        [round(global_xmin), round(global_ymin)],
        [round(global_xmax), round(global_ymin)],
        [round(global_xmax), round(global_ymax)],
        [round(global_xmin), round(global_ymax)]
    ]
    return bbox_polygon


def batch_predict(traj_dataset):
    info = traj_dataset["dataset_info"]
    pred_frames = int(info["pred_second"] * info["fps"])
    samples = traj_dataset["samples"]
    batch_res = []
    error_records = []
    total_mse = 0.0

    print(f"【步骤3】批量卡尔曼预测，总样本：{len(samples)}")
    for idx, samp in enumerate(samples):
        obs_raw = np.array(samp["obs_trajectory_pixel"])
        obs_smooth = smooth_trajectory(obs_raw)
        pred_pixel = constant_velocity_predict(obs_smooth, pred_frames)

        # 新增：获取当前样本的人体真实BBox
        human_bbox = samp["human_bbox"]
        # 重构：调用新的corridor生成函数（传入人体BBox）
        corridor_poly = generate_corridor_bbox_polygon(pred_pixel, human_bbox)

        truth = np.array(samp["pred_trajectory_pixel"])
        mse = np.mean(np.square(pred_pixel - truth))
        total_mse += mse

        res_item = {
            "window_start": samp["window_start_frame"],
            "window_end": samp["window_end_frame"],
            "person_id": samp["person_id"],
            "action_label": samp["action_type"],
            "obs_raw": obs_raw.tolist(),
            "obs_smooth": obs_smooth.tolist(),
            "predict_path": pred_pixel.tolist(),
            "path_corridor": corridor_poly,  # 仍保持字段名兼容
            "true_future": truth.tolist(),
            "pixel_mse": round(mse, 2)
        }
        batch_res.append(res_item)
        error_records.append({
            "window_end_frame": samp["window_end_frame"],
            "person_id": samp["person_id"],
            "mse_pixel": round(mse, 2)
        })
        if (idx + 1) % 5 == 0:
            print(f"已处理 {idx + 1}/{len(samples)}")

    # （后续逻辑不变）
    avg_mse = round(total_mse / len(batch_res), 2)
    pd.DataFrame(error_records).to_csv(OUT_ERROR_CSV, index=False, encoding="utf-8-sig")
    print(f"全部样本平均MSE误差：{avg_mse}，误差CSV输出：{OUT_ERROR_CSV}")

    output_data = {
        "source_dataset": OUT_TRAJ_DATASET,
        "predict_method": "多关键点加权中心+卡尔曼平滑+匀速外推+人体适配BBox",
        "fps": FPS,
        "obs_sec": OBS_SECOND,
        "pred_sec": PRED_SECOND,
        "avg_pixel_mse": avg_mse,
        "sample_count": len(batch_res),
        "predict_samples": batch_res
    }
    with open(OUT_KALMAN_PRED, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"卡尔曼预测结果输出：{OUT_KALMAN_PRED}")
    return output_data


if __name__ == "__main__":
    with open(OUT_TRAJ_DATASET, "r", encoding="utf-8") as f:
        ds = json.load(f)
    batch_predict(ds)