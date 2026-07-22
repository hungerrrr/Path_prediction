"""以统一口径评估 GRU、方案 A 和方案 B。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from eigen_transformer.models import EigenTrajectoryTransformer
from leapfrog_diffusion.models import LeapfrogDiffusionPredictor
from gru.core import load_checkpoint


class EvaluationDataset(Dataset):
    """加载轨迹、掩码、数据源和标签，用于分组评估。"""

    def __init__(self, path: Path, scale: np.ndarray) -> None:
        data = np.load(path, allow_pickle=False)
        observations = data["obs"].astype(np.float32)
        predictions = data["pred"].astype(np.float32)
        anchor = observations[:, -1:, :]
        self.observations = (observations - anchor) / scale
        self.predictions = (predictions - anchor) / scale
        self.mask = data["pred_mask"].astype(np.float32)
        self.sources = data["dataset_source"].astype(str)
        self.labels = data["label"].astype(str)

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(self.observations[index]),
            torch.from_numpy(self.predictions[index]),
            torch.from_numpy(self.mask[index]),
            self.sources[index],
            self.labels[index],
        )


def load_candidate_model(
    path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict, str]:
    """加载任意一种项目检查点。"""

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    schema = str(checkpoint.get("schema_version", ""))
    if schema.startswith("eigen_transformer"):
        model = EigenTrajectoryTransformer(
            basis_mean=checkpoint["basis_mean"],
            basis_components=checkpoint["basis_components"],
            **checkpoint["model_config"],
        )
        method = "eigen_transformer"
    elif schema.startswith("leapfrog_diffusion"):
        model = LeapfrogDiffusionPredictor(**checkpoint["model_config"])
        method = "leapfrog_diffusion"
    else:
        model, checkpoint = load_checkpoint(path, device)
        return model, checkpoint, "gru"

    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint, method


def empty_totals() -> dict[str, float]:
    """创建一组可累加的指标。"""

    return {
        "count": 0.0,
        "top1_ade": 0.0,
        "top1_fde": 0.0,
        "top1_mse": 0.0,
        "min_ade": 0.0,
        "min_fde": 0.0,
    }


def update_totals(
    totals: dict[str, float],
    values: dict[str, float],
) -> None:
    """累加单个样本指标。"""

    totals["count"] += 1
    for key in ("top1_ade", "top1_fde", "top1_mse", "min_ade", "min_fde"):
        totals[key] += values[key]


def finalize(totals: dict[str, float]) -> dict[str, float]:
    """计算平均指标。"""

    count = max(totals["count"], 1.0)
    return {
        "samples": int(totals["count"]),
        "top1_ade_pixel": totals["top1_ade"] / count,
        "top1_fde_pixel": totals["top1_fde"] / count,
        "top1_mse_pixel": totals["top1_mse"] / count,
        "min_ade_pixel": totals["min_ade"] / count,
        "min_fde_pixel": totals["min_fde"] / count,
    }


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    method: str,
    loader: DataLoader,
    device: torch.device,
    scale: torch.Tensor,
    diffusion_samples: int,
) -> dict:
    """评估整体、数据源和标签分组指标。"""

    overall = empty_totals()
    by_source = defaultdict(empty_totals)
    by_label = defaultdict(empty_totals)
    for observations, target, mask, sources, labels in loader:
        observations = observations.to(device)
        target = target.to(device)
        mask = mask.to(device)
        if method == "gru":
            prediction = model(observations)
            paths = prediction[:, None]
            probabilities = torch.ones(
                len(observations), 1, device=device
            )
        elif method == "eigen_transformer":
            paths, logits = model(observations)
            probabilities = torch.softmax(logits, dim=1)
        else:
            paths, probabilities = model.sample(
                observations,
                num_samples=diffusion_samples,
            )

        pixel_error = (paths - target[:, None]) * scale
        distances = torch.linalg.vector_norm(pixel_error, dim=-1)
        valid_steps = mask.sum(dim=1).clamp_min(1.0)
        ade = (
            distances * mask[:, None]
        ).sum(dim=-1) / valid_steps[:, None]
        last_indices = mask.sum(dim=1).long().clamp_min(1) - 1
        gather_indices = last_indices[:, None, None].expand(
            -1, paths.shape[1], 1
        )
        fde = distances.gather(2, gather_indices).squeeze(-1)
        squared = pixel_error.square().mean(dim=-1)
        mse = (
            squared * mask[:, None]
        ).sum(dim=-1) / valid_steps[:, None]
        top_modes = probabilities.argmax(dim=1)

        for index in range(len(observations)):
            top_mode = int(top_modes[index])
            values = {
                "top1_ade": float(ade[index, top_mode]),
                "top1_fde": float(fde[index, top_mode]),
                "top1_mse": float(mse[index, top_mode]),
                "min_ade": float(ade[index].min()),
                "min_fde": float(fde[index].min()),
            }
            update_totals(overall, values)
            update_totals(by_source[str(sources[index])], values)
            update_totals(by_label[str(labels[index])], values)

    return {
        "overall": finalize(overall),
        "by_source": {
            key: finalize(value)
            for key, value in sorted(by_source.items())
        },
        "by_label": {
            key: finalize(value)
            for key, value in sorted(by_label.items())
        },
    }


def parse_args() -> argparse.Namespace:
    """解析统一评估参数。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--gru", type=Path, default=Path("models/trajectory_gru.pt"))
    parser.add_argument(
        "--transformer",
        type=Path,
        default=Path("models/eigen_transformer.pt"),
    )
    parser.add_argument(
        "--diffusion",
        type=Path,
        default=Path("models/leapfrog_diffusion.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/experiments/05_model_comparison.json"),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--diffusion-samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """依次评估三套模型并写出统一报告。"""

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    reference = torch.load(
        args.gru,
        map_location="cpu",
        weights_only=False,
    )
    scale_array = np.asarray(
        reference["normalization"]["scale"],
        dtype=np.float32,
    )
    dataset = EvaluationDataset(args.dataset_dir / "test.npz", scale_array)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    scale = torch.tensor(scale_array, device=device)

    report = {
        "device": str(device),
        "test_windows": len(dataset),
        "models": {},
    }
    for name, path in (
        ("gru", args.gru),
        ("transformer", args.transformer),
        ("diffusion", args.diffusion),
    ):
        model, checkpoint, method = load_candidate_model(path, device)
        torch.manual_seed(args.seed)
        report["models"][name] = {
            "path": str(path.resolve()),
            "schema_version": checkpoint.get("schema_version", "legacy"),
            "method": method,
            "metrics": evaluate_model(
                model,
                method,
                loader,
                device,
                scale,
                args.diffusion_samples,
            ),
        }
        print(name, report["models"][name]["metrics"]["overall"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
