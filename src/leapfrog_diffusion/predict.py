"""方案 B：Leapfrog 少步扩散模型独立预测入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from prediction import add_common_arguments, run_prediction


def main() -> None:
    """解析参数并执行方案 B 预测。"""

    parser = argparse.ArgumentParser(
        description="使用 Leapfrog 少步扩散模型预测路径。"
    )
    add_common_arguments(
        parser,
        default_model=Path("models/leapfrog_diffusion.pt"),
        default_output=Path("outputs/diffusion"),
    )
    run_prediction(parser.parse_args(), expected_method="leapfrog_diffusion")


if __name__ == "__main__":
    main()
