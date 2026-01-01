#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dry-run EDA on Avazu train.csv (SMALL SAMPLE for quick validation).

This mode is designed for:
- Pipeline validation: Test all 9 stages end-to-end without waiting
- Quick debugging: < 1 minute execution on first 1K rows
- Output directory: data/interim/avazu_dry/ (separate from full-run)

Usage:
    python src/eda/avazu/run_dry.py

Parameters:
    - input_csv: Path to Avazu train.csv
    - out_root: Output root directory (data/interim/avazu_DRY) ← SEPARATE from full-run
    - chunksize: 500K for standard processing
    - sample_rows: 1K rows for validation-only MI/OOV/PSI
    - topk: 10 top values (abbreviated for speed)
    - top_values: 10 top values for label analysis
    - train_days/test_days: 1 day each (short window for validation)

Outputs:
    - All EDA artifacts in data/interim/avazu_dry/eda/*, featuremap/, reports/
    - Quick validation of all 9 pipeline stages
    - Note: Results are APPROXIMATE due to small sample size; use full-run for production
"""
import sys
import os
from types import SimpleNamespace

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from ctr_llm_creative.src.eda.avazu import avazu_evidence_eda_patched as m

def main():
    # Dry-run parameters: SMALL SAMPLE FOR QUICK VALIDATION
    # This mode processes only first 1K rows for rapid pipeline validation (< 1 minute)
    args = SimpleNamespace(
        input_csv=r'E:\沉淀项目\ads-search-rec-llm-systems\ctr_llm_creative\data\raw\avazu\train.csv',
        out_root=r'E:\沉淀项目\ads-search-rec-llm-systems\ctr_llm_creative\data\interim\avazu_dry',  # Separate output: _dry suffix
        chunksize=500000,            # 500K chunks (standard)
        topk=10,                     # fewer top-K values (validation only)
        top_values=10,               # fewer top values (validation only)
        min_support=10,              # lower min support for dry run
        sample_rows=1000,            # VERY SMALL: 1K rows for MI/OOV/PSI (validation only)
        seed=42,                     # random seed
        train_days=1,                # shorter time split (1 day) for quick validation
        test_days=1,
        verbose=True                 # verbose=True to see progress in dry run
    )

    print("=" * 80)
    print("[DRY-RUN] Avazu EDA - Quick Validation (1K Sample)")
    print("="* 80)
    print(f"Input:  {args.input_csv}")
    print(f"Output: {args.out_root} ← SEPARATE from full-run")
    print(f"Sample size: {args.sample_rows} rows (fast validation)")
    print(f"Chunksize: {args.chunksize:,}")
    print(f"Expected time: < 1 minute")
    print()

    try:
        m.main(args)
        print()
        print("=" * 80)
        print("[DRY-RUN] ✓ Completed Successfully")
        print("=" * 80)
        print(f"Artifacts saved to: {args.out_root}")
        print()
        print("Key outputs (SMALL SAMPLE - for validation only):")
        print(f"  - Overview: {args.out_root}/eda/overview.json")
        print(f"  - Schema:   {args.out_root}/eda/schema.json")
        print(f"  - Profile:  {args.out_root}/eda/columns_profile.csv")
        print(f"  - Time CTR: {args.out_root}/eda/time/time_ctr_day.csv")
        print(f"  - Hash:     {args.out_root}/eda/hash/hash_collision_sim.csv")
        print(f"  - MI:       {args.out_root}/eda/interactions/pair_mi_topbins.csv")
        print(f"  - Reports:  {args.out_root}/reports/model_structure_evidence.md")
        print()
        print("⚠ NEXT: Run full-run for production analysis:")
        print(f"   python src/eda/avazu/run_full.py")
        print(f"   Results will be saved to: data/interim/avazu/")
        return 0
    except Exception as e:
        print()
        print("=" * 80)
        print(f"[DRY-RUN] ✗ Failed with error: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
