"""两套高级轨迹模型共享的数据集、损失与评估工具。"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class NpzTrajectoryDataset(Dataset):
    """读取旧流程生成的 NPZ 窗口，保持与 GRU 数据完全兼容。"""

    def __init__(self, path: Path, scale: np.ndarray) -> None:
        data = np.load(path, allow_pickle=False)
        observations = data["obs"].astype(np.float32)
        predictions = data["pred"].astype(np.float32)
        anchor = observations[:, -1:, :]
        self.observations = (observations - anchor) / scale
        self.predictions = (predictions - anchor) / scale
        self.mask = data["pred_mask"].astype(np.float32)
        self.sources = data["dataset_source"].astype(str)
        self.sequences = data["sequence"].astype(str)

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.observations[index]),
            torch.from_numpy(self.predictions[index]),
            torch.from_numpy(self.mask[index]),
        )


def seed_everything(seed: int) -> None:
    """固定训练相关随机种子。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def multimodal_transformer_loss(
    paths: torch.Tensor,
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    classification_weight: float = 0.2,
    diversity_weight: float = 0.05,
    diversity_margin: float = 0.1,
) -> dict[str, torch.Tensor]:
    """计算 winner-takes-all 回归、模式分类和终点多样性损失。"""

    distances = torch.linalg.vector_norm(paths - target[:, None], dim=-1)
    masked_ade = (
        distances * mask[:, None]
    ).sum(dim=-1) / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    best_mode = masked_ade.detach().argmin(dim=1)
    batch_indices = torch.arange(len(paths), device=paths.device)
    best_paths = paths[batch_indices, best_mode]
    denominator = mask.sum().clamp_min(1.0) * target.shape[-1]
    regression = (
        F.smooth_l1_loss(best_paths, target, reduction="none")
        * mask[..., None]
    ).sum() / denominator
    classification = F.cross_entropy(logits, best_mode)

    endpoints = paths[:, :, -1]
    endpoint_distances = torch.cdist(endpoints, endpoints)
    identity = torch.eye(
        paths.shape[1],
        device=paths.device,
        dtype=torch.bool,
    )[None]
    diversity = F.relu(
        diversity_margin
        - endpoint_distances.masked_fill(identity, diversity_margin)
    ).mean()
    total = (
        regression
        + classification_weight * classification
        + diversity_weight * diversity
    )
    return {
        "loss": total,
        "regression_loss": regression,
        "classification_loss": classification,
        "diversity_loss": diversity,
    }


@torch.no_grad()
def multimodal_metrics(
    paths: torch.Tensor,
    probabilities: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    scale: torch.Tensor,
) -> dict[str, float]:
    """计算 Top-1 与 best-of-K 的 ADE/FDE 像素误差总量。"""

    pixel_delta = (paths - target[:, None]) * scale[None, None, None]
    distances = torch.linalg.vector_norm(pixel_delta, dim=-1)
    valid_steps = mask.sum(dim=1).clamp_min(1.0)
    ade_per_mode = (
        distances * mask[:, None]
    ).sum(dim=-1) / valid_steps[:, None]
    last_indices = mask.sum(dim=1).long().clamp_min(1) - 1
    gather_indices = last_indices[:, None, None].expand(
        -1, paths.shape[1], 1
    )
    fde_per_mode = distances.gather(2, gather_indices).squeeze(-1)
    top_mode = probabilities.argmax(dim=1)
    batch_indices = torch.arange(len(paths), device=paths.device)
    return {
        "top1_ade_sum": ade_per_mode[batch_indices, top_mode].sum().item(),
        "top1_fde_sum": fde_per_mode[batch_indices, top_mode].sum().item(),
        "min_ade_sum": ade_per_mode.min(dim=1).values.sum().item(),
        "min_fde_sum": fde_per_mode.min(dim=1).values.sum().item(),
        "count": float(len(paths)),
    }


def merge_metric_totals(
    total: dict[str, float],
    batch: dict[str, float],
) -> None:
    """原位累加一批评估指标。"""

    for key, value in batch.items():
        total[key] = total.get(key, 0.0) + value


def finalize_metric_totals(total: dict[str, float]) -> dict[str, float]:
    """把累计误差转换为按样本平均的最终指标。"""

    count = max(total.get("count", 0.0), 1.0)
    return {
        "top1_ade_pixel": total.get("top1_ade_sum", 0.0) / count,
        "top1_fde_pixel": total.get("top1_fde_sum", 0.0) / count,
        "min_ade_pixel": total.get("min_ade_sum", 0.0) / count,
        "min_fde_pixel": total.get("min_fde_sum", 0.0) / count,
    }
