"""方案 B：Leapfrog 初始化器与少步条件扩散模型。"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def sinusoidal_time_embedding(
    timesteps: torch.Tensor,
    dimension: int,
) -> torch.Tensor:
    """生成扩散时间步的正弦位置编码。"""

    half = dimension // 2
    scale = math.log(10_000) / max(half - 1, 1)
    frequencies = torch.exp(
        -scale * torch.arange(half, device=timesteps.device, dtype=torch.float32)
    )
    angles = timesteps.float()[:, None] * frequencies[None, :]
    embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
    return F.pad(embedding, (0, dimension - embedding.shape[-1]))


class ConditionalTrajectoryDenoiser(nn.Module):
    """在历史运动条件下预测未来轨迹中的扩散噪声。"""

    def __init__(
        self,
        pred_frames: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pred_frames = pred_frames
        self.hidden_dim = hidden_dim
        self.coordinate_projection = nn.Linear(2, hidden_dim)
        self.context_projection = nn.Linear(hidden_dim, hidden_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.future_position = nn.Parameter(torch.zeros(1, pred_frames, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.network = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.output = nn.Linear(hidden_dim, 2)
        nn.init.trunc_normal_(self.future_position, std=0.02)

    def forward(
        self,
        noisy_future: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """预测与 noisy_future 同形状的噪声。"""

        time_features = self.time_projection(
            sinusoidal_time_embedding(timesteps, self.hidden_dim)
        )
        features = (
            self.coordinate_projection(noisy_future)
            + self.context_projection(context)[:, None, :]
            + time_features[:, None, :]
            + self.future_position
        )
        return self.output(self.network(features))


class LeapfrogDiffusionPredictor(nn.Module):
    """通过可学习初始化器跳过高噪声阶段的少步轨迹扩散模型。"""

    def __init__(
        self,
        obs_frames: int,
        pred_frames: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        diffusion_steps: int = 50,
        sample_steps: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not 1 <= sample_steps <= diffusion_steps:
            raise ValueError("sample_steps 必须位于 [1, diffusion_steps]")
        self.obs_frames = obs_frames
        self.pred_frames = pred_frames
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.diffusion_steps = diffusion_steps
        self.sample_steps = sample_steps
        self.dropout = dropout
        self.history_encoder = MotionTransformerEncoder(
            obs_frames, hidden_dim, num_layers, num_heads, dropout
        )
        self.initial_mean = nn.Linear(hidden_dim, pred_frames * 2)
        self.initial_log_scale = nn.Linear(hidden_dim, pred_frames * 2)
        self.denoiser = ConditionalTrajectoryDenoiser(
            pred_frames,
            hidden_dim,
            max(1, num_layers - 1),
            num_heads,
            dropout,
        )

        betas = torch.linspace(1e-4, 0.02, diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        previous_alpha_bars = F.pad(alpha_bars[:-1], (1, 0), value=1.0)
        posterior_variance = (
            betas * (1.0 - previous_alpha_bars) / (1.0 - alpha_bars)
        ).clamp_min(1e-12)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("posterior_variance", posterior_variance)

    def encode(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """编码历史并输出 Leapfrog 未来均值与尺度。"""

        _, context = self.history_encoder(observations)
        mean = self.initial_mean(context).reshape(-1, self.pred_frames, 2)
        log_scale = self.initial_log_scale(context).reshape(-1, self.pred_frames, 2)
        return context, mean, log_scale.clamp(-3.0, 2.0).exp()

    def training_losses(
        self,
        observations: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        initializer_weight: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """计算噪声预测损失与 Leapfrog 初始化损失。"""

        context, initial_mean, initial_scale = self.encode(observations)
        timesteps = torch.randint(
            0,
            self.diffusion_steps,
            (len(observations),),
            device=observations.device,
        )
        noise = torch.randn_like(target)
        alpha_bar = self.alpha_bars[timesteps, None, None]
        noisy_future = alpha_bar.sqrt() * target + (1.0 - alpha_bar).sqrt() * noise
        predicted_noise = self.denoiser(noisy_future, timesteps, context)
        expanded_mask = mask[..., None]
        denominator = expanded_mask.sum().clamp_min(1.0) * target.shape[-1]
        noise_loss = (
            (predicted_noise - noise).square() * expanded_mask
        ).sum() / denominator
        # Gaussian NLL 同时校准初始化均值和尺度，避免采样尺度成为未训练参数。
        standardized_error = (target - initial_mean) / initial_scale
        initializer_nll = (
            0.5 * standardized_error.square() + initial_scale.log()
        )
        initializer_loss = (
            initializer_nll * expanded_mask
        ).sum() / denominator
        return {
            "loss": noise_loss + initializer_weight * initializer_loss,
            "noise_loss": noise_loss,
            "initializer_loss": initializer_loss,
        }

    @torch.no_grad()
    def sample(
        self,
        observations: torch.Tensor,
        num_samples: int = 6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """从中间扩散步开始，生成多条未来轨迹及相对概率。"""

        context, initial_mean, initial_scale = self.encode(observations)
        batch_size = len(observations)
        context = context[:, None].expand(-1, num_samples, -1)
        context = context.reshape(batch_size * num_samples, -1)
        mean = initial_mean[:, None].expand(-1, num_samples, -1, -1)
        mean = mean.reshape(batch_size * num_samples, self.pred_frames, 2)
        scale = initial_scale[:, None].expand(-1, num_samples, -1, -1)
        scale = scale.reshape(batch_size * num_samples, self.pred_frames, 2)

        start_step = self.sample_steps - 1
        alpha_bar = self.alpha_bars[start_step]
        trajectory = (
            alpha_bar.sqrt() * mean
            + (1.0 - alpha_bar).sqrt() * torch.randn_like(mean) * scale
        )
        for step in range(start_step, -1, -1):
            timesteps = torch.full(
                (len(trajectory),),
                step,
                device=trajectory.device,
                dtype=torch.long,
            )
            predicted_noise = self.denoiser(trajectory, timesteps, context)
            alpha = self.alphas[step]
            step_alpha_bar = self.alpha_bars[step]
            model_mean = (
                trajectory
                - (1.0 - alpha)
                / (1.0 - step_alpha_bar).sqrt()
                * predicted_noise
            ) / alpha.sqrt()
            if step:
                trajectory = (
                    model_mean
                    + self.posterior_variance[step].sqrt()
                    * torch.randn_like(trajectory)
                )
            else:
                trajectory = model_mean

        paths = trajectory.reshape(batch_size, num_samples, self.pred_frames, 2)
        scores = -((paths - initial_mean[:, None]).square().mean(dim=(-1, -2)))
        return paths, torch.softmax(scores, dim=1)

    def model_config(self) -> dict:
        """返回重建模型所需配置。"""

        return {
            "obs_frames": self.obs_frames,
            "pred_frames": self.pred_frames,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "diffusion_steps": self.diffusion_steps,
            "sample_steps": self.sample_steps,
            "dropout": self.dropout,
        }
