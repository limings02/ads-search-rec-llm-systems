# src/cli/materialize_from_config.py
from __future__ import annotations

import argparse
import os
import time

import pandas as pd
from src.utils.path_resolver import resolve_paths_in_config
from src.data.adapters.factory import load_yaml, build_adapter_from_config


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="dataset adapter config yaml")
    ap.add_argument("--out_root", default=None, help="override output root (default use cfg.io.out_root if exists)")
    ap.add_argument("--chunksize", type=int, default=None, help="override chunksize")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    cfg = resolve_paths_in_config(cfg)
    adp = build_adapter_from_config(cfg)

    # out_root：CLI > cfg.io.out_root > 默认 interim/avazu
    out_root = args.out_root or (cfg.get("io", {}) or {}).get(
        "out_root",
        r"E:\沉淀项目\ads-search-rec-llm-systems\ctr_llm_creative\data\interim\avazu",
    )
    chunksize = args.chunksize or (cfg.get("io", {}) or {}).get("chunksize", 1_000_000)

    canonical_root = os.path.join(out_root, "canonical")
    meta_root = os.path.join(out_root, "meta")
    index_root = os.path.join(out_root, "index")
    ensure_dir(canonical_root)
    ensure_dir(meta_root)
    ensure_dir(index_root)

    file_counter = {}
    index_rows = []
    total_rows = {"train": 0, "valid": 0, "test": 0}

    t0 = time.time()
    for split_name, df in adp.iter_splits(chunksize=chunksize):
        if "_day" not in df.columns:
            raise ValueError("canonical df missing _day, cannot partition by day")

        for day, g in df.groupby("_day", sort=False):
            day_int = int(day)
            part_dir = os.path.join(canonical_root, f"split={split_name}", f"day={day_int}")
            ensure_dir(part_dir)

            key = f"{split_name}|{day_int}"
            idx = file_counter.get(key, 0)
            file_counter[key] = idx + 1

            part_path = os.path.join(part_dir, f"part-{idx:06d}.parquet")
            g.to_parquet(part_path, index=False, engine="pyarrow", compression="snappy")

            rows = int(len(g))
            total_rows[split_name] += rows
            index_rows.append({"split": split_name, "day": day_int, "file": part_path, "rows": rows})

        if sum(total_rows.values()) % 5_000_000 < chunksize:
            print("progress rows:", sum(total_rows.values()), "by_split:", total_rows)

    pd.DataFrame(index_rows).to_csv(os.path.join(index_root, "files_index.csv"), index=False, encoding="utf-8-sig")

    t1 = time.time()
    with open(os.path.join(meta_root, "materialize_log.txt"), "w", encoding="utf-8") as f:
        f.write(f"config={args.config}\n")
        f.write(f"chunksize={chunksize}\n")
        f.write(f"elapsed_sec={t1 - t0:.2f}\n")
        f.write(f"total_rows={total_rows}\n")

    print("DONE. total_rows:", total_rows)
    print("written to:", canonical_root)


if __name__ == "__main__":
    main()
