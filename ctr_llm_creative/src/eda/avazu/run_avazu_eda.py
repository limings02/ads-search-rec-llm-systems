# src/eda/avazu/run_avazu_eda.py
from __future__ import annotations

import argparse
import os

from src.data.adapters.factory import load_yaml, build_adapter_from_config
from src.eda.avazu.avazu_eda_from_interim import AvazuEdaConfig, run_avazu_eda
from src.utils.path_resolver import resolve_paths_in_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/datasets/avazu_parquet_v1.yaml")
    ap.add_argument("--interim_root", default=None, help=".../data/interim/avazu")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--vocab_k", type=int, default=2000)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    cfg = resolve_paths_in_config(cfg)
    adp = build_adapter_from_config(cfg)

    # 你要读的是 interim/canonical（不是 raw）
    interim_root = args.interim_root or (cfg.get("io", {}) or {}).get(
        "out_root",
        os.path.join(os.getcwd(), "data", "interim", "avazu"),
    )
    canonical_root = os.path.join(interim_root, "canonical")
    out_root = os.path.join(interim_root, "eda_v1_repro", "eda")

    batch_size = args.batch_size or (cfg.get("io", {}) or {}).get("chunksize", 500_000)

    eda_cfg = AvazuEdaConfig(
        canonical_root=canonical_root,
        out_root=out_root,
        batch_size=batch_size,
        topk=args.topk,
        vocab_k=args.vocab_k,
    )

    feature_cols = adp.get_features()  # 用 adapter 的“稳定特征列表”，确保口径一致
    run_avazu_eda(eda_cfg, feature_cols=feature_cols)

    print("EDA DONE. outputs at:", out_root)


if __name__ == "__main__":
    main()
