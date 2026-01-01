# src/data/adapters/factory.py
from __future__ import annotations

import importlib
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Tuple

import yaml

from src.data.splits import SplitSpec


def _import_class(class_path: str):
    """
    class_path 形如: "src.data.adapters.avazu.AvazuAdapter"
    """
    module_name, cls_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_adapter_from_config(cfg: Dict[str, Any]):
    """
    根据 cfg 构建 adapter 实例：
    - import adapter class
    - SplitSpec(**cfg["split"])
    - adapter_class(..., split=split_spec, **params)
    """
    adapter_cfg = cfg["adapter"]
    class_path = adapter_cfg["class_path"]
    params = adapter_cfg.get("params", {}) or {}

    split_cfg = cfg.get("split", {}) or {}
    split_spec = SplitSpec(**split_cfg)

    cls = _import_class(class_path)

    # 兼容不同 Adapter 的签名：我们统一要求它至少接受 split=SplitSpec
    # AvazuAdapter 目前接受 train_path/test_path/usecols/split
    adapter = cls(split=split_spec, **params)
    return adapter


def dump_config_from_adapter(adapter, config_path: str, extra: Dict[str, Any] | None = None):
    """
    把一个 adapter 实例的关键信息导出成 YAML（用于“保存当前配置”）。
    这个函数尽量不侵入 Adapter 类本身，靠约定字段抽取。
    """
    # 1) adapter class path
    cls = adapter.__class__
    class_path = f"{cls.__module__}.{cls.__name__}"

    # 2) 尽量抽取 params（按你 AvazuAdapter 的字段）
    params = {}
    # 这些字段在你当前 AvazuAdapter 里是存在的
    for k in ["train_path", "test_path", "usecols", "dtypes"]:
        if hasattr(adapter, k):
            params[k] = getattr(adapter, k)

    # 3) split
    split = adapter.split
    if is_dataclass(split):
        split_dict = asdict(split)
    else:
        # 兜底：SplitSpec 本来就是 dataclass，不太会走到这里
        split_dict = dict(split)

    cfg = {
        "dataset": getattr(adapter, "name", "unknown"),
        "adapter": {
            "class_path": class_path,
            "params": params,
        },
        "split": split_dict,
    }
    if extra:
        cfg.update(extra)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    return cfg
