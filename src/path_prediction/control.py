"""读取由外部系统管理的模块开关。"""

from __future__ import annotations

import json
from pathlib import Path


def path_prediction_enabled(config_path: str | Path) -> bool:
    """每次调用均读取 config.json，以便上层系统动态切换模块。"""

    path = Path(config_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    try:
        enabled = config["modules"]["path_prediction"]["enabled"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "config.json 必须定义 modules.path_prediction.enabled"
        ) from exc
    if not isinstance(enabled, bool):
        raise ValueError("modules.path_prediction.enabled 必须是 true 或 false")
    return enabled
