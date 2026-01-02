# src/eda/avazu/run_avazu_eda_extra.py
from __future__ import annotations

import argparse
import os

from src.data.adapters.factory import load_yaml, build_adapter_from_config
from src.eda.avazu.avazu_eda_extra import AvazuEdaExtraConfig, run_all_eda_extra
from src.utils.path_resolver import resolve_paths_in_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/datasets/avazu_parquet_v1.yaml")
    ap.add_argument("--interim_root", default=None, help="override data/interim/avazu")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--psi_topk_bins", type=int, default=50)
    ap.add_argument("--ctr_value_topn", type=int, default=2000)
    ap.add_argument("--pair_topk_each", type=int, default=30)
    ap.add_argument("--pair_min_imps", type=int, default=20000)
    ap.add_argument("--bloom_bits", type=int, default=(1 << 26))
    args = ap.parse_args()

    cfg_yaml = load_yaml(args.config)
    cfg_yaml = resolve_paths_in_config(cfg_yaml)
    adp = build_adapter_from_config(cfg_yaml)

    interim_root = args.interim_root or (cfg_yaml.get("io", {}) or {}).get(
        "out_root",
        os.path.join(os.getcwd(), "data", "interim", "avazu"),
    )
    canonical_root = os.path.join(interim_root, "canonical")
    out_root = os.path.join(interim_root, "eda_v1_repro", "eda_extra")

    batch_size = args.batch_size or (cfg_yaml.get("io", {}) or {}).get("chunksize", 500_000)

    cfg = AvazuEdaExtraConfig(
        canonical_root=canonical_root,
        out_root=out_root,
        batch_size=batch_size,
        psi_topk_bins=args.psi_topk_bins,
        ctr_value_topn=args.ctr_value_topn,
        pair_topk_each=args.pair_topk_each,
        pair_min_imps=args.pair_min_imps,
        bloom_bits=args.bloom_bits,
    )

    # 用 adapter 提供的“稳定特征列表”，避免你手工写漏/写错
    feature_cols = adp.get_features()

    run_all_eda_extra(cfg, feature_cols)

    print("EDA EXTRA DONE. outputs at:", out_root)
    print("  - psi_by_field_day.parquet")
    print("  - psi_components_day.parquet")
    print("  - new_value_rate_by_day.parquet")
    print("  - ctr_by_value_train.parquet")
    print("  - pair_interaction_summary.parquet")
    print("  - pair_interaction_combos.parquet")


if __name__ == "__main__":
    main()
