import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from config import *


# ===================== 新增：深度学习模型定义 =====================
class TrajectoryGRUPredictor(nn.Module):
    """
    GRU-based轨迹预测模型
    输入：观测轨迹序列 (seq_len, 2) → 输出：预测轨迹序列 (pred_len, 2)
    """

    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, pred_len=PRED_FRAMES):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len

        # GRU时序特征提取
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        # 预测头：将时序特征映射为预测轨迹
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2 * pred_len)  # 输出：pred_len个(x,y) → 展平为2*pred_len
        )

    def forward(self, x):
        # x: (batch_size, obs_len, 2)
        gru_out, _ = self.gru(x)  # (batch_size, obs_len, hidden_dim)
        # 取最后一步的隐藏状态（包含完整时序信息）
        last_hidden = gru_out[:, -1, :]  # (batch_size, hidden_dim)
        # 预测轨迹
        pred_flat = self.fc(last_hidden)  # (batch_size, 2*pred_len)
        pred_traj = pred_flat.reshape(-1, self.pred_len, 2)  # (batch_size, pred_len, 2)
        return pred_traj


# 全局模型实例 + 加载预训练权重（后续需训练）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
traj_model = TrajectoryGRUPredictor().to(DEVICE)
# 加载预训练权重（示例路径，需先训练模型并保存）
MODEL_WEIGHTS_PATH = OUT_DIR + "traj_gru_model.pth"
if os.path.exists(MODEL_WEIGHTS_PATH):
    traj_model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE, weights_only=True))
traj_model.eval()  # 推理模式


# ===================== 替换原卡尔曼平滑函数 =====================
def smooth_trajectory(raw_traj):
    """
    深度学习版轨迹平滑（替代卡尔曼）
    输入：raw_traj (np.array, [obs_len, 2]) → 输出：smooth_obs (np.array, [obs_len, 2])
    逻辑：轻量滑动平均 + 模型特征对齐，保持接口不变
    """
    # 步骤1：滑动平均预处理（降噪，替代卡尔曼的平滑）
    raw_traj = np.array(raw_traj)
    obs_len = len(raw_traj)
    if obs_len < 2:
        return raw_traj

    # 滑动平均（窗口大小=3，边缘补全）
    smooth_traj = np.zeros_like(raw_traj)
    for i in range(obs_len):
        start = max(0, i - 1)
        end = min(obs_len, i + 2)
        smooth_traj[i] = np.mean(raw_traj[start:end], axis=0)

    # 步骤2：模型特征对齐（可选，提升后续预测精度）
    with torch.no_grad():
        # 适配模型输入格式
        x = torch.tensor(smooth_traj, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, obs_len, 2)
        # 仅用GRU提取特征，不预测，对齐轨迹分布
        gru_out, _ = traj_model.gru(x)
        # 特征映射回原坐标空间（轻量对齐）
        align_factor = torch.mean(x) / (torch.mean(gru_out[:, :, :2]) + 1e-6)
        smooth_traj = smooth_traj * align_factor.cpu().numpy()

    return smooth_traj


# ===================== 替换原匀速预测函数 =====================
def constant_velocity_predict(smooth_obs, pred_frames):
    """
    深度学习版轨迹预测（替代匀速外推）
    输入：smooth_obs (np.array, [obs_len, 2])、pred_frames (int) → 输出：pred_path (np.array, [pred_frames, 2])
    逻辑：GRU模型预测，兼容原接口的边界裁剪/静止判定
    """
    obs_len = smooth_obs.shape[0]
    if obs_len < 2:  # 观测帧不足，兜底逻辑不变
        return np.array([smooth_obs[-1]] * pred_frames)

    # 步骤1：静止判定（保留原逻辑，兼容老人静止场景）
    start = smooth_obs[0]
    end = smooth_obs[-1]
    vx = (end[0] - start[0]) / (obs_len - 1)
    vy = (end[1] - start[1]) / (obs_len - 1)
    if abs(vx) < 0.1 and abs(vy) < 0.1:
        return np.array([end] * pred_frames)

    # 步骤2：深度学习模型预测
    with torch.no_grad():
        # 输入适配：补零到固定观测长度（OBS_FRAMES），兼容不同长度输入
        pad_len = max(0, OBS_FRAMES - obs_len)
        obs_padded = np.pad(smooth_obs, ((pad_len, 0), (0, 0)), mode="edge")  # 边缘补全
        # 转换为tensor
        x = torch.tensor(obs_padded, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, OBS_FRAMES, 2)
        # 模型预测
        pred_traj = traj_model(x)  # (1, pred_frames, 2)
        pred_path = pred_traj.squeeze(0).cpu().numpy()  # (pred_frames, 2)

    # 步骤3：边界裁剪（保留原逻辑，适配视频坐标范围）
    pred_path[:, 0] = np.clip(pred_path[:, 0], CLIP_X_MIN, CLIP_X_MAX)
    pred_path[:, 1] = np.clip(pred_path[:, 1], CLIP_Y_MIN, CLIP_Y_MAX)

    return pred_path


# ===================== 保留原走廊生成/批量预测逻辑（完全不变） =====================
def generate_corridor_bbox_polygon(pred_path, human_bbox, padding=HUMAN_BBOX_MARGIN):
    """原逻辑完全保留，不改动"""
    if len(pred_path) == 0 or np.array(pred_path).size == 0:
        return [
            [human_bbox["xmin"], human_bbox["ymin"]],
            [human_bbox["xmax"], human_bbox["ymin"]],
            [human_bbox["xmax"], human_bbox["ymax"]],
            [human_bbox["xmin"], human_bbox["ymax"]]
        ]

    human_xmin = human_bbox["xmin"]
    human_ymin = human_bbox["ymin"]
    human_xmax = human_bbox["xmax"]
    human_ymax = human_bbox["ymax"]
    human_w = human_xmax - human_xmin
    human_h = human_ymax - human_ymin
    human_cx = (human_xmin + human_xmax) / 2
    human_cy = (human_ymin + human_ymax) / 2

    pred_pts = np.array(pred_path)
    if pred_pts.ndim != 2:
        pred_pts = pred_pts.reshape(-1, 2)

    pred_bboxes = []
    for (cx, cy) in pred_pts:
        xmin = cx - (human_w / 2)
        ymin = cy - (human_h / 2)
        xmax = cx + (human_w / 2)
        ymax = cy + (human_h / 2)
        pred_bboxes.append([xmin, ymin, xmax, ymax])

    pred_bboxes = np.array(pred_bboxes)
    global_xmin = max(pred_bboxes[:, 0].min() - padding, CLIP_X_MIN)
    global_xmax = min(pred_bboxes[:, 2].max() + padding, CLIP_X_MAX)
    global_ymin = max(pred_bboxes[:, 1].min() - padding, CLIP_Y_MIN)
    global_ymax = min(pred_bboxes[:, 3].max() + padding, CLIP_Y_MAX)

    if global_xmax <= global_xmin or global_ymax <= global_ymin:
        global_xmin = human_xmin - padding
        global_xmax = human_xmax + padding
        global_ymin = human_ymin - padding
        global_ymax = human_ymax + padding

    bbox_polygon = [
        [round(global_xmin), round(global_ymin)],
        [round(global_xmax), round(global_ymin)],
        [round(global_xmax), round(global_ymax)],
        [round(global_xmin), round(global_ymax)]
    ]
    return bbox_polygon


def batch_predict(traj_dataset):
    """原逻辑完全保留，不改动"""
    info = traj_dataset["dataset_info"]
    pred_frames = int(info["pred_second"] * info["fps"])
    samples = traj_dataset["samples"]
    batch_res = []
    error_records = []
    total_mse = 0.0

    print(f"【步骤3】深度学习轨迹预测，总样本：{len(samples)}")
    for idx, samp in enumerate(samples):
        obs_raw = np.array(samp["obs_trajectory_pixel"])
        obs_smooth = smooth_trajectory(obs_raw)  # 调用新的平滑函数
        pred_pixel = constant_velocity_predict(obs_smooth, pred_frames)  # 调用新的预测函数

        human_bbox = samp["human_bbox"]
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
            "path_corridor": corridor_poly,
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

    avg_mse = round(total_mse / len(batch_res), 2)
    pd.DataFrame(error_records).to_csv(OUT_ERROR_CSV, index=False, encoding="utf-8-sig")
    print(f"全部样本平均MSE误差：{avg_mse}，误差CSV输出：{OUT_ERROR_CSV}")

    output_data = {
        "source_dataset": OUT_TRAJ_DATASET,
        "predict_method": "多关键点加权中心+GRU时序预测+人体适配BBox",  # 仅修改方法名
        "fps": FPS,
        "obs_sec": OBS_SECOND,
        "pred_sec": PRED_SECOND,
        "avg_pixel_mse": avg_mse,
        "sample_count": len(batch_res),
        "predict_samples": batch_res
    }
    with open(OUT_GRU_PRED, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"深度学习预测结果输出：{OUT_GRU_PRED}")
    return output_data


# ===================== 模型训练辅助函数（可选，离线训练用） =====================
class TrajectoryDataset(Dataset):
    """轨迹数据集类，用于训练GRU模型"""

    def __init__(self, traj_dataset_path):
        with open(traj_dataset_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)["samples"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        samp = self.data[idx]
        obs = np.array(samp["obs_trajectory_pixel"], dtype=np.float32)
        pred = np.array(samp["pred_trajectory_pixel"], dtype=np.float32)
        # 补零到固定长度
        obs = np.pad(obs, ((max(0, OBS_FRAMES - len(obs)), 0), (0, 0)), mode="edge")
        return torch.tensor(obs), torch.tensor(pred)


def train_traj_model():
    """训练GRU轨迹预测模型（离线执行）"""
    dataset = TrajectoryDataset(OUT_TRAJ_DATASET)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = TrajectoryGRUPredictor().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练循环
    epochs = 50
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
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    # 保存模型权重
    torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
    print(f"模型已保存到：{MODEL_WEIGHTS_PATH}")


if __name__ == "__main__":
    # 第一步：先训练模型（首次运行时执行）
    train_traj_model()

    # 第二步：执行批量预测（和原逻辑一致）
    with open(OUT_TRAJ_DATASET, "r", encoding="utf-8") as f:
        ds = json.load(f)
    batch_predict(ds)