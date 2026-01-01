#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full-run EDA on Avazu train.csv (COMPLETE PRODUCTION ANALYSIS on full dataset).

This mode is designed for:
- Complete analysis: All stages 1-9 on entire dataset (~40M rows)
- Production results: Detailed evidence for FeatureMap and model selection
- Output directory: data/interim/avazu/ (separate from dry-run in avazu_dry/)

Usage:
    python src/eda/avazu/run_full.py

Parameters:
    - input_csv: Path to Avazu train.csv
    - out_root: Output root directory (data/interim/avazu) ← PRODUCTION (NOT _dry)
    - chunksize: 1M for balanced streaming
    - sample_rows: 200K rows for MI/OOV/PSI analysis
    - topk: 20 top-K values per column
    - top_values: 200 top values for detailed label analysis
    - min_support: 100 impressions minimum for CTR stability
    - train_days/test_days: 7/2 days (full temporal split)

Outputs:
    - Complete EDA artifacts in data/interim/avazu/eda/*, featuremap/, model_plan/, reports/
    - All 9 pipeline stages executed end-to-end on full data
    - Full evidence chain for FeatureMap design and model structure selection

Execution time:
    - Depends on dataset size (train.csv typically 40M+ rows)
    - Expect 30-60 minutes with streaming chunksize=1M
    - Memory footprint: ~500MB-1GB (streaming-safe)
"""
import sys
import os
from types import SimpleNamespace

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.eda import avazu_evidence_eda as m

def main():
    # Full-run parameters: COMPLETE PRODUCTION ANALYSIS
    # This mode processes ENTIRE DATASET with streaming chunks (~40M rows)
    # Expected runtime: 30-60 minutes depending on hardware
    args = SimpleNamespace(
        input_csv=r'E:\沉淀项目\ads-search-rec-llm-systems\ctr_llm_creative\data\raw\avazu\train.csv',
        out_root=r'E:\沉淀项目\ads-search-rec-llm-systems\ctr_llm_creative\data\interim\avazu',  # Production output (no _dry suffix)
        chunksize=1000000,           # 1M chunks for balanced speed/memory
        topk=20,                     # top-K values per feature
        top_values=200,              # detailed top values for CTR analysis
        min_support=100,             # min impressions for CTR stability
        sample_rows=200000,          # sample 200K rows for MI/OOV/PSI analysis
        seed=42,                     # random seed
        train_days=7,                # full train/test split (days)
        test_days=2,
        verbose=False                # quiet mode (set True for detailed logging)
    )

    print("=" * 80)
    print("[FULL-RUN] Avazu EDA - Complete Production Analysis")
    print("="* 80)
    print(f"Input:       {args.input_csv}")
    print(f"Output:      {args.out_root} ← PRODUCTION (separate from dry-run)")
    print(f"Chunksize:   {args.chunksize:,}")
    print(f"Sample rows: {args.sample_rows:,}")
    print(f"Expected time: 30-60 minutes")
    print()
    print("Processing stages 1-9...")
    print("  1. Schema & Overview")
    print("  2. Columns Profile")
    print("  3. Temporal CTR")
    print("  4. CTR by Group & Top Values")
    print("  5. OOV Analysis")
    print("  6. PSI Drift Detection")
    print("  7. Hash Collision Simulation")
    print("  8. Pairwise MI Analysis")
    print("  9. Report Generation")
    print()

    try:
        m.main(args)
        print()
        print("=" * 80)
        print("[FULL-RUN] ✓ Completed Successfully")
        print("=" * 80)
        print(f"Artifacts saved to: {args.out_root}")
        print()
        print("Key outputs:")
        print(f"  ✓ Overview:      {args.out_root}/eda/overview.json")
        print(f"  ✓ Schema:        {args.out_root}/eda/schema.json")
        print(f"  ✓ Profile:       {args.out_root}/eda/columns_profile.csv")
        print(f"  ✓ Temporal:      {args.out_root}/eda/time/time_ctr_day.csv")
        print(f"  ✓ CTR Analysis:  {args.out_root}/eda/label/ctr_by_*_top.csv")
        print(f"  ✓ OOV Rates:     {args.out_root}/eda/split/oov_rate_train_test.csv")
        print(f"  ✓ PSI Drift:     {args.out_root}/eda/drift/psi_train_test_summary.csv")
        print(f"  ✓ Hash Collision:{args.out_root}/eda/hash/hash_collision_sim.csv")
        print(f"  ✓ Pairwise MI:   {args.out_root}/eda/interactions/pair_mi_topbins.csv")
        print(f"  ✓ FeatureMap:    {args.out_root}/featuremap/featuremap_spec.yml")
        print(f"  ✓ Model Plan:    {args.out_root}/model_plan/model_plan.yml")
        print(f"  ✓ Reports:       {args.out_root}/reports/")
        print()
        print("Next steps:")
        print("  1. Review reports/model_structure_evidence.md for MI-based model selection")
        print("  2. Review reports/featuremap_evidence.md for feature engineering decisions")
        print("  3. Load featuremap/featuremap_spec.yml into FeatureMap for model training")
        return 0
    except Exception as e:
        print()
        print("=" * 80)
        print(f"[FULL-RUN] ✗ Failed with error: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        print()
        print("Troubleshooting:")
        print("  - Check that input_csv exists and is readable")
        print("  - Ensure out_root directory has write permissions")
        print("  - Monitor disk space (EDA outputs can be 100MB+)")
        print("  - Check logs in data/interim/avazu/logs/ for details")
        return 1

if __name__ == "__main__":
    sys.exit(main())
