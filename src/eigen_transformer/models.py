"""方案 A：低秩 EigenTrajectory 表示与多模态 Transformer。"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def fit_trajectory_basis(
    trajectories: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    """通过协方差特征分解拟合低秩未来轨迹基。"""

    flat = np.asarray(trajectories, dtype=np.float32).reshape(len(trajectories), -1)
    if not 1 <= rank <= flat.shape[1]:
        raise ValueError(f"rank 必须位于 [1, {flat.shape[1]}]")
    mean = flat.mean(axis=0, dtype=np.float64)
    centered = flat.astype(np.float64) - mean
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:rank]
    return mean.astype(np.float32), eigenvectors[:, order].T.astype(np.float32)


class MotionTransformerEncoder(nn.Module):
    """使用位置与速度特征编码历史运动。"""

    def __init__(
        self,
        obs_frames: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        use_velocity: bool = True,
    ) -> None:
        super().__init__()
        self.obs_frames = obs_frames
        self.use_velocity = use_velocity
        input_dimension = 4 if use_velocity else 2
        self.input_projection = nn.Linear(input_dimension, hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, obs_frames, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回逐帧记忆和全局运动表示。"""

        if observations.shape[1] != self.obs_frames:
            raise ValueError(
                f"期望 {self.obs_frames} 个观测帧，实际为 {observations.shape[1]}"
            )
        if self.use_velocity:
            velocity = torch.diff(
                observations,
                dim=1,
                prepend=observations[:, :1],
            )
            features = torch.cat([observations, velocity], dim=-1)
        else:
            features = observations
        memory = self.input_projection(features) + self.position_embedding
        memory = self.encoder(memory)
        return memory, memory.mean(dim=1)


class EigenTrajectoryTransformer(nn.Module):
    """低秩轨迹空间中的多模态 Transformer 预测器。"""

    def __init__(
        self,
        obs_frames: int,
        pred_frames: int,
        basis_mean: np.ndarray | torch.Tensor,
        basis_components: np.ndarray | torch.Tensor,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        num_modes: int = 6,
        dropout: float = 0.1,
        use_velocity: bool = True,
    ) -> None:
        super().__init__()
        components = torch.as_tensor(basis_components, dtype=torch.float32)
        mean = torch.as_tensor(basis_mean, dtype=torch.float32)
        expected_dimension = pred_frames * 2
        if components.ndim != 2 or components.shape[1] != expected_dimension:
            raise ValueError("低秩轨迹基的维度与 pred_frames 不匹配")
        if mean.shape != (expected_dimension,):
            raise ValueError("低秩轨迹均值的维度与 pred_frames 不匹配")

        self.obs_frames = obs_frames
        self.pred_frames = pred_frames
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_modes = num_modes
        self.rank = components.shape[0]
        self.dropout = dropout
        self.use_velocity = use_velocity
        self.history_encoder = MotionTransformerEncoder(
            obs_frames,
            hidden_dim,
            num_layers,
            num_heads,
            dropout,
            use_velocity,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.mode_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=max(1, num_layers - 1),
            norm=nn.LayerNorm(hidden_dim),
        )
        self.mode_queries = nn.Parameter(torch.empty(1, num_modes, hidden_dim))
        self.coefficient_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.rank),
        )
        self.probability_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("basis_mean", mean)
        self.register_buffer("basis_components", components)
        nn.init.trunc_normal_(self.mode_queries, std=0.02)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """并行返回 K 条未来轨迹及其未归一化概率。"""

        memory, _ = self.history_encoder(observations)
        queries = self.mode_queries.expand(len(observations), -1, -1)
        mode_features = self.mode_decoder(queries, memory)
        coefficients = self.coefficient_head(mode_features)
        flat_paths = self.basis_mean[None, None, :] + torch.einsum(
            "bkr,rd->bkd",
            coefficients,
            self.basis_components,
        )
        paths = flat_paths.reshape(-1, self.num_modes, self.pred_frames, 2)
        logits = self.probability_head(mode_features).squeeze(-1)
        return paths, logits

    def model_config(self) -> dict:
        """返回重建模型所需配置。"""

        return {
            "obs_frames": self.obs_frames,
            "pred_frames": self.pred_frames,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_modes": self.num_modes,
            "dropout": self.dropout,
            "use_velocity": self.use_velocity,
        }
