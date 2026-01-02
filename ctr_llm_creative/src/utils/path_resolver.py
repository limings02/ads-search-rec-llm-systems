# src/utils/path_resolver.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, MutableMapping
import os
import re


def get_project_root() -> Path:
    """
    获取项目根目录（ctr_llm_creative/）。

    约定：该文件位于 ctr_llm_creative/src/utils/ 下，
    因此 root = .../ctr_llm_creative（向上 3 层：utils -> src -> ctr_llm_creative）。
    """
    return Path(__file__).resolve().parents[2]


_WIN_ABS_RE = re.compile(r"^[a-zA-Z]:[\\/]")


def _is_abs_path_str(s: str) -> bool:
    """
    判断字符串是否是绝对路径（兼容 Windows 盘符 / Unix / UNC）。
    """
    if not s:
        return False
    # Windows: "C:\xxx" or "C:/xxx"
    if _WIN_ABS_RE.match(s):
        return True
    # UNC: "\\server\share"
    if s.startswith("\\\\"):
        return True
    # Unix absolute: "/home/xxx"
    return s.startswith("/")


def _should_resolve_key(key: str) -> bool:
    k = key.lower()
    if k in ("class_path", "module_path"):   # 关键：排除
        return False
    return k.endswith("_path") or k.endswith("_dir") or k.endswith("_root")



def _resolve_one_path(value: str, base: Path) -> str:
    """
    将 value 解析成绝对路径：
    - 支持环境变量：${VAR} 或 %VAR%
    - 已是绝对路径：原样返回
    - 相对路径：base / value
    """
    # 先展开环境变量（两种风格都尽量兼容）
    expanded = os.path.expandvars(value)

    # 处理类似 "~" 的 home 缩写（Unix/Windows 都可）
    expanded = str(Path(expanded).expanduser())

    if _is_abs_path_str(expanded):
        return str(Path(expanded))

    # 相对路径：挂到 base
    return str((base / expanded).resolve())


def resolve_paths_in_config(cfg: Any, base: Path | None = None) -> Any:
    """
    递归遍历配置对象，把路径字段统一解析为绝对路径。

    设计原则：
    1) 不要求你用某个特定的 config 类，只要是 dict/list 结构都能处理
    2) 只处理“看起来像路径”的 key（_path/_dir/_root），避免误伤其他字段
    3) 绝对路径不动，相对路径拼到 project_root 上
    """
    if base is None:
        base = get_project_root()

    # dict
    if isinstance(cfg, Mapping):
        out: MutableMapping[str, Any] = dict(cfg)  # 浅拷贝
        for k, v in cfg.items():
            if isinstance(v, str) and _should_resolve_key(str(k)):
                out[k] = _resolve_one_path(v, base)
            else:
                out[k] = resolve_paths_in_config(v, base)
        return out

    # list / tuple
    if isinstance(cfg, list):
        return [resolve_paths_in_config(x, base) for x in cfg]

    # 其他类型原样返回
    return cfg
