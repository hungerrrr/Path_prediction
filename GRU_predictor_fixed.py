import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from config import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_WEIGHTS_PATH = OUT_DIR + "traj_gru_model.pth"

# 不改变模块输入/输出，只在模块内部增加坐标归一化上下文与模型可用标志
_COORD_CTX = {
    "x_min": CLIP_X_MIN,
    "x_max": CLIP_X_MAX,
    "y_min": CLIP_Y_MIN,
    "y_max": CLIP_Y_MAX,
}
MODEL_READY = False


def _set_coord_context_from_dataset(traj_dataset):
    """从trajectory_dataset的dataset_info读取人体中心坐标范围；失败时回退到可视范围。"""
    global _COORD_CTX
    info = traj_dataset.get("dataset_info", {}) if isinstance(traj_dataset, dict) else {}
    x_rng = info.get("coord_x_range", [CLIP_X_MIN, CLIP_X_MAX])
    y_rng = info.get("coord_y_range", [CLIP_Y_MIN, CLIP_Y_MAX])
    x_min, x_max = float(x_rng[0]), float(x_rng[1])
    y_min, y_max = float(y_rng[0]), float(y_rng[1])
    if not np.isfinite([x_min, x_max, y_min, y_max]).all() or x_max <= x_min or y_max <= y_min:
        x_min, x_max, y_min, y_max = CLIP_X_MIN, CLIP_X_MAX, CLIP_Y_MIN, CLIP_Y_MAX
    _COORD_CTX = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}


def _coord_scale():
    sx = max(_COORD_CTX["x_max"] - _COORD_CTX["x_min"], 1.0)
    sy = max(_COORD_CTX["y_max"] - _COORD_CTX["y_min"], 1.0)
    return np.array([sx, sy], dtype=np.float32)


class TrajectoryGRUPredictor(nn.Module):
    """GRU轨迹预测模型：输入观测相对位移序列，输出未来相对位移序列。"""

    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, pred_len=PRED_FRAMES):
        super().__init__()
        self.pred_len = pred_len
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * pred_len),
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_hidden = gru_out[:, -1, :]
        pred_flat = self.fc(last_hidden)
        return pred_flat.reshape(-1, self.pred_len, 2)


traj_model = TrajectoryGRUPredictor().to(DEVICE)
if os.path.exists(MODEL_WEIGHTS_PATH):
    try:
        traj_model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
        MODEL_READY = True
    except Exception as e:
        print(f"GRU权重加载失败，使用匀速兜底：{e}")
        MODEL_READY = False
traj_model.eval()


def smooth_trajectory(raw_traj):
    """轻量平滑：只做时域降噪，不使用未训练GRU隐藏态缩放原坐标。"""
    raw_traj = np.asarray(raw_traj, dtype=np.float32)
    obs_len = len(raw_traj)
    if obs_len < 3:
        return raw_traj.copy()

    smooth = raw_traj.copy()
    for i in range(obs_len):
        start = max(0, i - 1)
        end = min(obs_len, i + 2)
        smooth[i] = np.mean(raw_traj[start:end], axis=0)

    # 保留最后一帧当前位置，避免平滑把当前bbox/轨迹起点拉偏
    smooth[-1] = raw_traj[-1]
    return smooth


def _constant_velocity_fallback(smooth_obs, pred_frames):
    """原接口内部兜底：输出像素坐标[pred_frames,2]。"""
    obs_len = smooth_obs.shape[0]
    end = smooth_obs[-1].astype(np.float32)
    if obs_len < 2:
        return np.repeat(end[None, :], pred_frames, axis=0)

    # 用最近若干帧的中位速度，比只取首尾更稳健
    recent = smooth_obs[-min(obs_len, max(5, FPS // 3)) :]
    diffs = np.diff(recent, axis=0)
    v = np.median(diffs, axis=0) if len(diffs) else np.zeros(2, dtype=np.float32)
    if np.linalg.norm(v) < 0.1:
        return np.repeat(end[None, :], pred_frames, axis=0)

    steps = np.arange(1, pred_frames + 1, dtype=np.float32)[:, None]
    pred = end[None, :] + steps * v[None, :]
    pred[:, 0] = np.clip(pred[:, 0], CLIP_X_MIN, CLIP_X_MAX)
    pred[:, 1] = np.clip(pred[:, 1], CLIP_Y_MIN, CLIP_Y_MAX)
    return pred


def _pad_obs(obs_rel):
    if len(obs_rel) >= OBS_FRAMES:
        return obs_rel[-OBS_FRAMES:]
    pad_len = OBS_FRAMES - len(obs_rel)
    return np.pad(obs_rel, ((pad_len, 0), (0, 0)), mode="edge")


def _prediction_is_sane(pred_path, smooth_obs):
    if pred_path.shape != (PRED_FRAMES, 2):
        return False
    if not np.isfinite(pred_path).all():
        return False
    last = smooth_obs[-1]
    first_jump = np.linalg.norm(pred_path[0] - last)
    obs_diffs = np.diff(smooth_obs[-min(len(smooth_obs), OBS_FRAMES):], axis=0)
    median_step = float(np.median(np.linalg.norm(obs_diffs, axis=1))) if len(obs_diffs) else 0.0
    # 第一个预测点不应突然跳到RGB左边界；阈值随观测速度自适应
    if first_jump > max(25.0, 8.0 * median_step + 5.0):
        return False
    # 不能整段贴在左边界/上下边界，这通常是未训练网络输出被clip造成的
    if np.mean(pred_path[:, 0] <= CLIP_X_MIN + 1) > 0.8:
        return False
    return True


def constant_velocity_predict(smooth_obs, pred_frames):
    """GRU预测，接口保持不变；无权重或输出异常时自动退回稳定匀速预测。"""
    smooth_obs = np.asarray(smooth_obs, dtype=np.float32)
    obs_len = smooth_obs.shape[0]
    if obs_len < 2:
        return np.repeat(smooth_obs[-1][None, :], pred_frames, axis=0)

    fallback = _constant_velocity_fallback(smooth_obs, pred_frames)
    if not MODEL_READY:
        return fallback

    last = smooth_obs[-1].astype(np.float32)
    scale = _coord_scale()
    obs_rel = (smooth_obs.astype(np.float32) - last[None, :]) / scale[None, :]
    obs_rel = _pad_obs(obs_rel)

    with torch.no_grad():
        x = torch.tensor(obs_rel, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        pred_rel = traj_model(x).squeeze(0).cpu().numpy().astype(np.float32)
        # 模型输出长度固定为PRED_FRAMES；外部pred_frames通常一致，这里仍兼容裁剪/补齐
        if len(pred_rel) < pred_frames:
            pred_rel = np.vstack([pred_rel, np.repeat(pred_rel[-1][None, :], pred_frames - len(pred_rel), axis=0)])
        pred_rel = pred_rel[:pred_frames]
        pred_path = last[None, :] + pred_rel * scale[None, :]

    pred_path[:, 0] = np.clip(pred_path[:, 0], CLIP_X_MIN, CLIP_X_MAX)
    pred_path[:, 1] = np.clip(pred_path[:, 1], CLIP_Y_MIN, CLIP_Y_MAX)

    if not _prediction_is_sane(pred_path, smooth_obs):
        return fallback

    # 与物理模型轻度融合，避免小数据集GRU抖动；仍保持输出格式不变
    alpha = 0.65
    pred_path = alpha * pred_path + (1.0 - alpha) * fallback
    pred_path[:, 0] = np.clip(pred_path[:, 0], CLIP_X_MIN, CLIP_X_MAX)
    pred_path[:, 1] = np.clip(pred_path[:, 1], CLIP_Y_MIN, CLIP_Y_MAX)
    return pred_path


def generate_corridor_bbox_polygon(pred_path, human_bbox, padding=HUMAN_BBOX_MARGIN):
    """基于当前人体BBox + 预测路径生成走廊，保证bbox不会脱离当前人体。"""
    human_xmin = float(human_bbox["xmin"])
    human_ymin = float(human_bbox["ymin"])
    human_xmax = float(human_bbox["xmax"])
    human_ymax = float(human_bbox["ymax"])
    human_w = max(human_xmax - human_xmin, 1.0)
    human_h = max(human_ymax - human_ymin, 1.0)

    boxes = [[human_xmin, human_ymin, human_xmax, human_ymax]]
    pred_pts = np.asarray(pred_path, dtype=np.float32)
    if pred_pts.size > 0:
        pred_pts = pred_pts.reshape(-1, 2)
        for cx, cy in pred_pts:
            boxes.append([cx - human_w / 2, cy - human_h / 2, cx + human_w / 2, cy + human_h / 2])

    boxes = np.asarray(boxes, dtype=np.float32)
    global_xmin = max(float(boxes[:, 0].min()) - padding, CLIP_X_MIN)
    global_xmax = min(float(boxes[:, 2].max()) + padding, CLIP_X_MAX)
    global_ymin = max(float(boxes[:, 1].min()) - padding, CLIP_Y_MIN)
    global_ymax = min(float(boxes[:, 3].max()) + padding, CLIP_Y_MAX)

    if global_xmax <= global_xmin or global_ymax <= global_ymin:
        global_xmin = np.clip(human_xmin - padding, CLIP_X_MIN, CLIP_X_MAX)
        global_xmax = np.clip(human_xmax + padding, CLIP_X_MIN, CLIP_X_MAX)
        global_ymin = np.clip(human_ymin - padding, CLIP_Y_MIN, CLIP_Y_MAX)
        global_ymax = np.clip(human_ymax + padding, CLIP_Y_MIN, CLIP_Y_MAX)

    return [
        [round(global_xmin), round(global_ymin)],
        [round(global_xmax), round(global_ymin)],
        [round(global_xmax), round(global_ymax)],
        [round(global_xmin), round(global_ymax)],
    ]


def batch_predict(traj_dataset):
    """输入/输出字段保持原GRU_predictor.batch_predict完全兼容。"""
    _set_coord_context_from_dataset(traj_dataset)
    info = traj_dataset["dataset_info"]
    pred_frames = int(info["pred_second"] * info["fps"])
    samples = traj_dataset["samples"]
    batch_res = []
    error_records = []
    total_mse = 0.0

    print(f"【步骤3】GRU轨迹预测，总样本：{len(samples)}，MODEL_READY={MODEL_READY}")
    for idx, samp in enumerate(samples):
        obs_raw = np.array(samp["obs_trajectory_pixel"], dtype=np.float32)
        obs_smooth = smooth_trajectory(obs_raw)
        pred_pixel = constant_velocity_predict(obs_smooth, pred_frames)

        human_bbox = samp["human_bbox"]
        corridor_poly = generate_corridor_bbox_polygon(pred_pixel, human_bbox)

        truth = np.array(samp["pred_trajectory_pixel"], dtype=np.float32)
        mse = float(np.mean(np.square(pred_pixel - truth))) if len(truth) == len(pred_pixel) else 0.0
        total_mse += mse

        res_item = {
            "window_start": samp["window_start_frame"],
            "window_end": samp["window_end_frame"],
            "person_id": samp["person_id"],
            "action_label": samp["action_type"],
            "obs_raw": obs_raw.tolist(),
            "obs_smooth": obs_smooth.tolist(),
            "predict_path": pred_pixel.tolist(),
            "path_corridor": corridor_poly,
            "true_future": truth.tolist(),
            "pixel_mse": round(mse, 2),
        }
        batch_res.append(res_item)
        error_records.append({
            "window_end_frame": samp["window_end_frame"],
            "person_id": samp["person_id"],
            "mse_pixel": round(mse, 2),
        })
        if (idx + 1) % 5 == 0:
            print(f"已处理 {idx + 1}/{len(samples)}")

    avg_mse = round(total_mse / len(batch_res), 2) if batch_res else 0.0
    pd.DataFrame(error_records).to_csv(OUT_ERROR_CSV, index=False, encoding="utf-8-sig")
    print(f"全部样本平均MSE误差：{avg_mse}，误差CSV输出：{OUT_ERROR_CSV}")

    output_data = {
        "source_dataset": OUT_TRAJ_DATASET,
        "predict_method": "多关键点加权中心+GRU相对位移预测+人体适配BBox",
        "fps": FPS,
        "obs_sec": OBS_SECOND,
        "pred_sec": PRED_SECOND,
        "avg_pixel_mse": avg_mse,
        "sample_count": len(batch_res),
        "predict_samples": batch_res,
    }
    with open(OUT_GRU_PRED, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"GRU预测结果输出：{OUT_GRU_PRED}")
    return output_data


class TrajectoryDataset(Dataset):
    """训练用：使用相对位移/尺度归一化，避免模型学习绝对画面位置。"""

    def __init__(self, traj_dataset_path):
        with open(traj_dataset_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        self.data = obj["samples"]
        info = obj.get("dataset_info", {})
        x_rng = info.get("coord_x_range", [CLIP_X_MIN, CLIP_X_MAX])
        y_rng = info.get("coord_y_range", [CLIP_Y_MIN, CLIP_Y_MAX])
        self.scale = np.array([max(float(x_rng[1]) - float(x_rng[0]), 1.0), max(float(y_rng[1]) - float(y_rng[0]), 1.0)], dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        samp = self.data[idx]
        obs = np.array(samp["obs_trajectory_pixel"], dtype=np.float32)
        pred = np.array(samp["pred_trajectory_pixel"], dtype=np.float32)
        last = obs[-1].copy()
        obs_rel = (obs - last[None, :]) / self.scale[None, :]
        pred_rel = (pred - last[None, :]) / self.scale[None, :]
        obs_rel = _pad_obs(obs_rel)
        return torch.tensor(obs_rel, dtype=torch.float32), torch.tensor(pred_rel, dtype=torch.float32)


def train_traj_model():
    dataset = TrajectoryDataset(OUT_TRAJ_DATASET)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = TrajectoryGRUPredictor().to(DEVICE)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    epochs = 80
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for obs, pred in dataloader:
            obs = obs.to(DEVICE)
            pred = pred.to(DEVICE)
            optimizer.zero_grad()
            pred_out = model(obs)
            loss = criterion(pred_out, pred)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(len(dataloader), 1)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
    torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
    print(f"模型已保存到：{MODEL_WEIGHTS_PATH}")


if __name__ == "__main__":
    train_traj_model()
    with open(OUT_TRAJ_DATASET, "r", encoding="utf-8") as f:
        ds = json.load(f)
    MODEL_READY = True
    traj_model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
    traj_model.eval()
    batch_predict(ds)
