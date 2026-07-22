"""在固定测试子集上评估三类模型对历史缺失、丢点和噪声的鲁棒性。"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_models import load_candidate_model  # noqa: E402


def perturb(observations: torch.Tensor, name: str) -> torch.Tensor:
    """生成一种历史轨迹扰动，保持输入张量尺寸不变。"""

    result = observations.clone()
    if name.startswith("history_"):
        kept = int(name.split("_")[1])
        result[:, :-kept] = result[:, -kept : -kept + 1]
    elif name == "dropout_20_percent":
        generator = torch.Generator().manual_seed(20260716)
        missing = torch.rand(result.shape[:2], generator=generator) < 0.2
        missing[:, -1] = False
        result[missing] = 0.0
    elif name == "noise_0.01_normalized":
        generator = torch.Generator().manual_seed(20260716)
        noise = torch.randn(result.shape, generator=generator) * 0.01
        result += noise
    return result


@torch.no_grad()
def metrics(
    model: torch.nn.Module,
    method: str,
    observations: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    scale: torch.Tensor,
) -> dict[str, float]:
    """计算 Top-1 与 best-of-K 的 ADE/FDE。"""

    if method == "gru":
        paths = model(observations)[:, None]
        probabilities = torch.ones(len(observations), 1)
    elif method == "eigen_transformer":
        paths, logits = model(observations)
        probabilities = torch.softmax(logits, dim=1)
    else:
        paths, probabilities = model.sample(observations, num_samples=6)

    distance = torch.linalg.vector_norm(
        (paths - target[:, None]) * scale[None, None, None], dim=-1
    )
    valid = mask.sum(dim=1).clamp_min(1.0)
    ade = (distance * mask[:, None]).sum(dim=-1) / valid[:, None]
    last = mask.sum(dim=1).long().clamp_min(1) - 1
    fde = distance.gather(
        2, last[:, None, None].expand(-1, paths.shape[1], 1)
    ).squeeze(-1)
    top = probabilities.argmax(dim=1)
    row = torch.arange(len(observations))
    return {
        "top1_ade_pixel": float(ade[row, top].mean()),
        "top1_fde_pixel": float(fde[row, top].mean()),
        "min_ade_pixel": float(ade.min(dim=1).values.mean()),
        "min_fde_pixel": float(fde.min(dim=1).values.mean()),
    }


def main() -> None:
    """加载固定样本，依次评估所有模型与扰动场景。"""

    reference = torch.load(
        ROOT / "models" / "trajectory_gru.pt",
        map_location="cpu",
        weights_only=False,
    )
    scale_array = np.asarray(reference["normalization"]["scale"], np.float32)
    data = np.load(ROOT / "data" / "processed" / "test.npz")
    rng = np.random.default_rng(42)
    indices = np.sort(rng.choice(len(data["obs"]), size=2048, replace=False))
    raw_obs = data["obs"][indices].astype(np.float32)
    anchor = raw_obs[:, -1:, :]
    observations = torch.from_numpy((raw_obs - anchor) / scale_array)
    target = torch.from_numpy(
        (data["pred"][indices].astype(np.float32) - anchor) / scale_array
    )
    mask = torch.from_numpy(data["pred_mask"][indices].astype(np.float32))
    scale = torch.from_numpy(scale_array)

    report = {
        "seed": 42,
        "samples": int(len(indices)),
        "source_counts": {
            str(key): int(value)
            for key, value in sorted(Counter(data["dataset_source"][indices].astype(str)).items())
        },
        "perturbations": {},
    }
    variants = (
        "history_60",
        "history_15",
        "history_30",
        "dropout_20_percent",
        "noise_0.01_normalized",
    )
    model_paths = {
        "gru": ROOT / "models" / "trajectory_gru.pt",
        "transformer": ROOT / "models" / "eigen_transformer.pt",
        "diffusion": ROOT / "models" / "leapfrog_diffusion.pt",
    }
    for model_name, path in model_paths.items():
        model, _, method = load_candidate_model(path, torch.device("cpu"))
        report["perturbations"][model_name] = {}
        for variant in variants:
            torch.manual_seed(42)
            values = metrics(
                model,
                method,
                perturb(observations, variant),
                target,
                mask,
                scale,
            )
            report["perturbations"][model_name][variant] = values
            print(model_name, variant, values)

    output = ROOT / "outputs" / "experiments" / "08_robustness_metrics.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
