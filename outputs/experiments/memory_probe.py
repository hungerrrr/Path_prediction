"""独立进程内存探针，仅供效率实验使用。"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path("src").resolve()))
from evaluate_models import load_candidate_model

name = sys.argv[1]
paths = {
    "gru": Path("models/trajectory_gru.pt"),
    "transformer": Path("models/eigen_transformer.pt"),
    "diffusion": Path("models/leapfrog_diffusion.pt"),
}
model, _, method = load_candidate_model(paths[name], torch.device("cpu"))
observations = torch.randn(64, 60, 2)
with torch.no_grad():
    for _ in range(10):
        if method == "gru":
            model(observations)
        elif method == "eigen_transformer":
            model(observations)
        else:
            model.sample(observations, num_samples=6)
        time.sleep(0.02)
