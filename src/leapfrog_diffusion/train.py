"""训练方案 B：Leapfrog 少步扩散路径预测模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from common import (
    NpzTrajectoryDataset,
    finalize_metric_totals,
    merge_metric_totals,
    multimodal_metrics,
    seed_everything,
)
from models import LeapfrogDiffusionPredictor


def parse_args() -> argparse.Namespace:
    """解析方案 B 的训练参数。"""

    parser = argparse.ArgumentParser(
        description="训练 Leapfrog 少步扩散路径预测器。"
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models/leapfrog_diffusion.pt"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--initializer-weight", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def make_train_loader(
    dataset: NpzTrajectoryDataset,
    args: argparse.Namespace,
) -> DataLoader:
    """默认使用温和的数据源均衡采样。"""

    sources, counts = np.unique(dataset.sources, return_counts=True)
    largest = float(counts.max())
    weights_by_source = {
        source: min(np.sqrt(largest / count), 10.0)
        for source, count in zip(sources.tolist(), counts.tolist())
    }
    weights = torch.tensor(
        [weights_by_source[source] for source in dataset.sources],
        dtype=torch.double,
    )
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def initializer_validation_loss(
    model: LeapfrogDiffusionPredictor,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """使用确定性的初始化轨迹误差进行早停，避免采样噪声干扰模型选择。"""

    model.eval()
    loss_sum = 0.0
    valid_values = 0.0
    for observations, target, mask in loader:
        observations = observations.to(device)
        target = target.to(device)
        mask = mask.to(device)
        _, mean, _ = model.encode(observations)
        loss = F.smooth_l1_loss(mean, target, reduction="none")
        loss_sum += (loss * mask[..., None]).sum().item()
        valid_values += mask.sum().item() * target.shape[-1]
    return loss_sum / max(valid_values, 1.0)


@torch.no_grad()
def evaluate_samples(
    model: LeapfrogDiffusionPredictor,
    loader: DataLoader,
    device: torch.device,
    scale: torch.Tensor,
    num_samples: int,
) -> dict[str, float]:
    """评估扩散模型生成轨迹的 Top-1 和 best-of-K 指标。"""

    model.eval()
    totals: dict[str, float] = {}
    for observations, target, mask in loader:
        observations = observations.to(device)
        target = target.to(device)
        mask = mask.to(device)
        paths, probabilities = model.sample(
            observations,
            num_samples=num_samples,
        )
        merge_metric_totals(
            totals,
            multimodal_metrics(
                paths,
                probabilities,
                target,
                mask,
                scale,
            ),
        )
    return finalize_metric_totals(totals)


def main() -> None:
    """训练少步扩散模型并保存最佳初始化器验证检查点。"""

    args = parse_args()
    seed_everything(args.seed)
    with (args.dataset_dir / "manifest.json").open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(handle)

    scale_array = np.asarray(
        manifest["normalization"]["scale"],
        dtype=np.float32,
    )
    train_data = NpzTrajectoryDataset(args.dataset_dir / "train.npz", scale_array)
    val_data = NpzTrajectoryDataset(args.dataset_dir / "val.npz", scale_array)
    test_data = NpzTrajectoryDataset(args.dataset_dir / "test.npz", scale_array)
    if not len(train_data) or not len(val_data):
        raise RuntimeError("训练集和验证集都必须包含轨迹窗口")

    data_config = manifest["data_config"]
    model = LeapfrogDiffusionPredictor(
        obs_frames=int(data_config["obs_frames"]),
        pred_frames=int(data_config["pred_frames"]),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        diffusion_steps=args.diffusion_steps,
        sample_steps=args.sample_steps,
        dropout=args.dropout,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    scale = torch.tensor(scale_array, device=device)
    train_loader = make_train_loader(train_data, args)
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=4,
    )

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_totals = {
            "loss": 0.0,
            "noise_loss": 0.0,
            "initializer_loss": 0.0,
        }
        sample_count = 0
        for observations, target, mask in train_loader:
            observations = observations.to(device)
            target = target.to(device)
            mask = mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = model.training_losses(
                observations,
                target,
                mask,
                initializer_weight=args.initializer_weight,
            )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for key in loss_totals:
                loss_totals[key] += losses[key].item() * len(observations)
            sample_count += len(observations)

        validation_loss = initializer_validation_loss(
            model,
            val_loader,
            device,
        )
        scheduler.step(validation_loss)
        row = {
            "epoch": epoch,
            "train_loss": loss_totals["loss"] / max(sample_count, 1),
            "noise_loss": loss_totals["noise_loss"] / max(sample_count, 1),
            "initializer_loss": (
                loss_totals["initializer_loss"] / max(sample_count, 1)
            ),
            "validation_initializer_loss": validation_loss,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} | train {row['train_loss']:.6f} "
            f"| noise {row['noise_loss']:.6f} "
            f"| val-init {validation_loss:.6f}"
        )

        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_epoch = epoch
            torch.save(
                {
                    "schema_version": "leapfrog_diffusion.v1",
                    "model_state": model.state_dict(),
                    "model_config": model.model_config(),
                    "data_config": data_config,
                    "normalization": manifest["normalization"],
                    "validation_initializer_loss": validation_loss,
                    "num_samples": args.num_samples,
                    "epoch": epoch,
                    "seed": args.seed,
                },
                args.model_out,
            )
        if epoch - best_epoch >= args.patience:
            break

    checkpoint = torch.load(
        args.model_out,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    report = {
        "best_epoch": best_epoch,
        "validation_initializer_loss": best_loss,
        "validation": evaluate_samples(
            model,
            val_loader,
            device,
            scale,
            args.num_samples,
        ),
        "test": None,
    }
    if len(test_data):
        test_loader = DataLoader(
            test_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        report["test"] = evaluate_samples(
            model,
            test_loader,
            device,
            scale,
            args.num_samples,
        )

    with args.model_out.with_suffix(".history.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2)
    with args.model_out.with_suffix(".report.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
