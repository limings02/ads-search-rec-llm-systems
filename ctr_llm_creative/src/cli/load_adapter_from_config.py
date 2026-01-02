# src/cli/load_adapter_from_config.py
from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from src.utils.path_resolver import resolve_paths_in_config, get_project_root
from src.data.adapters.factory import load_yaml, build_adapter_from_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to yaml config")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    cfg = resolve_paths_in_config(cfg)        # 新增：解析相对路径 → 绝对路径
    adp = build_adapter_from_config(cfg)

    # 证明你真的复现成功
    print("Adapter class:", adp.__class__.__name__)
    print("Adapter file :", inspect.getfile(adp.__class__))
    print("Split spec   :", adp.split)

    # 冒烟：取一批数据
    for k, df in adp.iter_splits(chunksize=200_000):
        print("FOUND", k, "rows=", len(df), "day=", df["_day"].unique()[:5])
        break


if __name__ == "__main__":
    main()
