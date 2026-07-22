"""方案 A：低秩多模态 Transformer 独立预测入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from prediction import add_common_arguments, run_prediction


def main() -> None:
    """解析参数并执行方案 A 预测。"""

    parser = argparse.ArgumentParser(
        description="使用低秩多模态 Transformer 预测路径。"
    )
    add_common_arguments(
        parser,
        default_model=Path("models/eigen_transformer.pt"),
        default_output=Path("outputs/transformer"),
    )
    run_prediction(parser.parse_args(), expected_method="eigen_transformer")


if __name__ == "__main__":
    main()
