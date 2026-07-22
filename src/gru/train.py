"""使用已准备的数据集训练 GRU 路径预测模型。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from core import TrajectoryGRU, relative_normalize


class NpzTrajectoryDataset(Dataset):
    """读取 NPZ 轨迹窗口，并在取样前完成相对坐标归一化。"""

    def __init__(self, path: Path, scale: np.ndarray) -> None:
        data = np.load(path, allow_pickle=False)
        self.obs, self.pred = relative_normalize(data["obs"], data["pred"], scale)
        self.mask = data["pred_mask"].astype(np.float32)
        self.sources = data["dataset_source"].astype(str)
        self.sequences = data["sequence"].astype(str)

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.obs[index]),
            torch.from_numpy(self.pred[index]),
            torch.from_numpy(self.mask[index]),
        )


def parse_args() -> argparse.Namespace:
    """解析模型训练参数。"""

    parser = argparse.ArgumentParser(
        description="独立训练用于轨迹预测的 GRU 模型。"
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-out", type=Path, default=Path("models/trajectory_gru.pt"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--sampling",
        choices=("dataset-balanced", "random"),
        default="dataset-balanced",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    """固定 Python、NumPy 与 PyTorch 随机种子。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model: TrajectoryGRU,
    loader: DataLoader,
    device: torch.device,
    scale: np.ndarray,
) -> dict:
    """计算掩码损失、平均位移误差（ADE）和最终位移误差（FDE）。"""

    loss_sum = distance_sum = fde_sum = 0.0
    valid_steps = 0.0
    count = 0
    scale_tensor = torch.tensor(scale, device=device)
    model.eval()
    with torch.no_grad():
        for obs, target, mask in loader:
            obs, target, mask = obs.to(device), target.to(device), mask.to(device)
            output = model(obs)
            per_step_loss = torch.nn.functional.smooth_l1_loss(
                output,
                target,
                reduction="none",
            ).mean(dim=-1)
            loss_sum += (per_step_loss * mask).sum().item()
            pixel_distance = torch.linalg.vector_norm((output - target) * scale_tensor, dim=-1)
            distance_sum += (pixel_distance * mask).sum().item()
            last_indices = mask.sum(dim=1).long().clamp_min(1) - 1
            fde_sum += pixel_distance.gather(1, last_indices[:, None]).sum().item()
            valid_steps += mask.sum().item()
            count += len(obs)
    return {
        "loss": loss_sum / max(valid_steps, 1),
        "ade_pixel": distance_sum / max(valid_steps, 1),
        "fde_pixel": fde_sum / max(count, 1),
    }


def main() -> None:
    """训练模型，按验证损失早停，并对独立测试集进行最终评估。"""

    args = parse_args()
    seed_everything(args.seed)
    with (args.dataset_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    cfg = manifest["data_config"]
    scale = np.asarray(manifest["normalization"]["scale"], dtype=np.float32)
    train_data = NpzTrajectoryDataset(args.dataset_dir / "train.npz", scale)
    val_data = NpzTrajectoryDataset(args.dataset_dir / "val.npz", scale)
    test_data = NpzTrajectoryDataset(args.dataset_dir / "test.npz", scale)
    if not len(train_data):
        raise RuntimeError(
            "training split contains no windows; check data coverage and window lengths"
        )
    if not len(val_data):
        raise RuntimeError(
            "validation split contains no windows; do not select a model on training data"
        )

    generator = torch.Generator().manual_seed(args.seed)
    sampler = None
    shuffle = args.sampling == "random"
    # 按数据源设置温和的反频率权重，防止最大数据集主导训练。
    if args.sampling == "dataset-balanced":
        sources, counts = np.unique(train_data.sources, return_counts=True)
        count_map = dict(zip(sources.tolist(), counts.tolist()))
        largest = float(max(counts))
        source_weights = {
            source: min(np.sqrt(largest / count), 10.0)
            for source, count in count_map.items()
        }
        weights = torch.tensor(
            [source_weights[source] for source in train_data.sources],
            dtype=torch.double,
        )
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )
        print(f"Dataset sampling weights: {source_weights}")
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator if sampler is None else None,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = {
        "obs_frames": int(cfg["obs_frames"]),
        "pred_frames": int(cfg["pred_frames"]),
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": 0.1,
    }
    model = TrajectoryGRU(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_epoch = 0
    history = []
    # 仅在验证损失改善时保存检查点，测试集不参与模型选择。
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        valid_steps_seen = 0.0
        for obs, target, mask in train_loader:
            obs, target, mask = obs.to(device), target.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(obs)
            per_step_loss = torch.nn.functional.smooth_l1_loss(
                output,
                target,
                reduction="none",
            ).mean(dim=-1)
            loss = (per_step_loss * mask).sum() / mask.sum().clamp_min(1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_valid_steps = mask.sum().item()
            train_loss += loss.item() * batch_valid_steps
            valid_steps_seen += batch_valid_steps
        metrics = evaluate(model, val_loader, device, scale)
        scheduler.step(metrics["loss"])
        row = {"epoch": epoch, "train_loss": train_loss / max(valid_steps_seen, 1), **metrics}
        history.append(row)
        print(
            f"Epoch {epoch:03d} | train {row['train_loss']:.6f} | val {metrics['loss']:.6f} "
            f"| ADE {metrics['ade_pixel']:.2f}px | FDE {metrics['fde_pixel']:.2f}px"
        )
        if metrics["loss"] < best_loss - 1e-7:
            best_loss, best_epoch = metrics["loss"], epoch
            torch.save(
                {
                    "schema_version": "trajectory_gru.v2",
                    "model_state": model.state_dict(),
                    "model_config": model_config,
                    "data_config": cfg,
                    "normalization": manifest["normalization"],
                    "validation_metrics": metrics,
                    "epoch": epoch,
                    "seed": args.seed,
                },
                args.model_out,
            )
        if epoch - best_epoch >= args.patience:
            print(f"Early stopping after {args.patience} epochs without improvement.")
            break

    history_path = args.model_out.with_suffix(".history.json")
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"best_epoch": best_epoch, "epochs": history},
            handle,
            ensure_ascii=False,
            indent=2,
        )
    report = {
        "best_epoch": best_epoch,
        "validation": history[best_epoch - 1] if best_epoch else None,
        "test": None,
        "test_by_dataset": {},
    }
    if len(test_data):
        best_checkpoint = torch.load(args.model_out, map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint["model_state"])
        test_loader = DataLoader(
            test_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        report["test"] = evaluate(model, test_loader, device, scale)
        for source in sorted(set(test_data.sources.tolist())):
            indices = np.flatnonzero(test_data.sources == source).tolist()
            source_loader = DataLoader(
                Subset(test_data, indices),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
            report["test_by_dataset"][source] = evaluate(model, source_loader, device, scale)
        print(
            f"Held-out test | loss {report['test']['loss']:.6f} | "
            f"ADE {report['test']['ade_pixel']:.2f}px | FDE {report['test']['fde_pixel']:.2f}px"
        )
    report_path = args.model_out.with_suffix(".report.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"Best model: {args.model_out.resolve()} (epoch {best_epoch})")


if __name__ == "__main__":
    main()
