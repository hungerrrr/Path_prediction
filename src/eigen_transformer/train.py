"""训练方案 A：低秩多模态 Transformer 路径预测模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from common import (
    NpzTrajectoryDataset,
    finalize_metric_totals,
    merge_metric_totals,
    multimodal_metrics,
    multimodal_transformer_loss,
    seed_everything,
)
from models import EigenTrajectoryTransformer, fit_trajectory_basis


def parse_args() -> argparse.Namespace:
    """解析方案 A 的训练参数。"""

    parser = argparse.ArgumentParser(
        description="训练低秩多模态 Transformer 路径预测器。"
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models/eigen_transformer.pt"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-modes", type=int, default=6)
    parser.add_argument("--basis-rank", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--classification-weight", type=float, default=0.2)
    parser.add_argument("--diversity-weight", type=float, default=0.05)
    parser.add_argument("--no-velocity", action="store_true")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--sampling",
        choices=("dataset-balanced", "random"),
        default="dataset-balanced",
    )
    return parser.parse_args()


def make_train_loader(
    dataset: NpzTrajectoryDataset,
    args: argparse.Namespace,
) -> DataLoader:
    """构建随机或数据源均衡的训练加载器。"""

    generator = torch.Generator().manual_seed(args.seed)
    sampler = None
    shuffle = args.sampling == "random"
    if args.sampling == "dataset-balanced":
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
            generator=generator,
        )

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator if sampler is None else None,
    )


@torch.no_grad()
def evaluate(
    model: EigenTrajectoryTransformer,
    loader: DataLoader,
    device: torch.device,
    scale: torch.Tensor,
    classification_weight: float,
    diversity_weight: float,
) -> dict[str, float]:
    """评估 Top-1 与 best-of-K 轨迹误差。"""

    model.eval()
    totals: dict[str, float] = {}
    loss_sum = 0.0
    sample_count = 0
    for observations, target, mask in loader:
        observations = observations.to(device)
        target = target.to(device)
        mask = mask.to(device)
        paths, logits = model(observations)
        losses = multimodal_transformer_loss(
            paths,
            logits,
            target,
            mask,
            classification_weight=classification_weight,
            diversity_weight=diversity_weight,
        )
        probabilities = torch.softmax(logits, dim=1)
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
        loss_sum += losses["loss"].item() * len(observations)
        sample_count += len(observations)

    return {
        "loss": loss_sum / max(sample_count, 1),
        **finalize_metric_totals(totals),
    }


def main() -> None:
    """拟合低秩轨迹基，训练模型并保存最佳验证检查点。"""

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

    basis_mean, basis_components = fit_trajectory_basis(
        train_data.predictions,
        args.basis_rank,
    )
    data_config = manifest["data_config"]
    model = EigenTrajectoryTransformer(
        obs_frames=int(data_config["obs_frames"]),
        pred_frames=int(data_config["pred_frames"]),
        basis_mean=basis_mean,
        basis_components=basis_components,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_modes=args.num_modes,
        dropout=args.dropout,
        use_velocity=not args.no_velocity,
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
    best_score = float("inf")
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for observations, target, mask in train_loader:
            observations = observations.to(device)
            target = target.to(device)
            mask = mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            paths, logits = model(observations)
            losses = multimodal_transformer_loss(
                paths,
                logits,
                target,
                mask,
                classification_weight=args.classification_weight,
                diversity_weight=args.diversity_weight,
            )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += losses["loss"].item() * len(observations)
            sample_count += len(observations)

        validation = evaluate(
            model,
            val_loader,
            device,
            scale,
            args.classification_weight,
            args.diversity_weight,
        )
        scheduler.step(validation["loss"])
        selection_score = (
            validation["top1_ade_pixel"] + validation["min_ade_pixel"]
        )
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(sample_count, 1),
            "selection_score": selection_score,
            **validation,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} | train {row['train_loss']:.6f} "
            f"| val {row['loss']:.6f} | Top1 ADE {row['top1_ade_pixel']:.2f}px "
            f"| minADE@{args.num_modes} {row['min_ade_pixel']:.2f}px"
        )

        if selection_score < best_score - 1e-7:
            best_score = selection_score
            best_epoch = epoch
            torch.save(
                {
                    "schema_version": "eigen_transformer.v1",
                    "model_state": model.state_dict(),
                    "model_config": model.model_config(),
                    "basis_mean": basis_mean,
                    "basis_components": basis_components,
                    "data_config": data_config,
                    "normalization": manifest["normalization"],
                    "validation_metrics": validation,
                    "selection_score": selection_score,
                    "epoch": epoch,
                    "seed": args.seed,
                },
                args.model_out,
            )
        if epoch - best_epoch >= args.patience:
            break

    report = {
        "best_epoch": best_epoch,
        "validation": history[best_epoch - 1] if best_epoch else None,
        "test": None,
    }
    if len(test_data):
        checkpoint = torch.load(
            args.model_out,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state"])
        test_loader = DataLoader(
            test_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        report["test"] = evaluate(
            model,
            test_loader,
            device,
            scale,
            args.classification_weight,
            args.diversity_weight,
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
