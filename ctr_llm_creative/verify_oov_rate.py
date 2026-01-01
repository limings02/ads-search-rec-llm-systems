#!/usr/bin/env python3
"""
Stage 6: OOV Rate Analysis & Interpretation

Provides detailed analysis of OOV results and recommendations for feature engineering.
"""

import pandas as pd
import sys
from pathlib import Path


def analyze_oov_report(oov_csv: str, columns_profile_csv: str = None) -> None:
    """
    Analyze OOV report and provide insights.
    
    Args:
        oov_csv: Path to oov_rate_train_test.csv
        columns_profile_csv: Path to columns_profile.csv (for cardinality info)
    """
    
    # Read OOV report
    df_oov = pd.read_csv(oov_csv)
    
    print("\n" + "="*80)
    print("STAGE 6: OOV RATE ANALYSIS")
    print("="*80)
    
    # Summary statistics
    print("\n[SUMMARY]")
    print(f"  Total features: {len(df_oov)}")
    print(f"  Perfect overlap (OOV=0%): {(df_oov['test_oov_rate'] == 0.0).sum()} features")
    print(f"  High OOV (>5%): {(df_oov['test_oov_rate'] > 0.05).sum()} features")
    print(f"  Vocab truncated: {df_oov['truncated'].sum()} features")
    
    # OOV rate distribution
    print("\n[OOV RATE DISTRIBUTION]")
    for threshold in [0.0, 0.01, 0.05, 0.1, 0.2]:
        count = (df_oov['test_oov_rate'] > threshold).sum()
        print(f"  > {threshold*100:5.1f}%: {count:2d} features")
    
    # Features sorted by OOV rate
    print("\n[TOP 10 FEATURES BY OOV RATE]")
    df_sorted = df_oov.sort_values('test_oov_rate', ascending=False)
    for idx, row in df_sorted.head(10).iterrows():
        truncated_mark = " [TRUNC]" if row['truncated'] else ""
        print(f"  {row['column']:15s}: {row['test_oov_rate']*100:6.2f}%  "
              f"({row['test_oov_count']:6.0f}/{row['test_total_count']:6.0f}){truncated_mark}")
    
    # Interpretation & recommendations
    print("\n[FEATURE ENGINEERING RECOMMENDATIONS]")
    
    for idx, row in df_sorted.head(5).iterrows():
        col = row['column']
        oov_rate = row['test_oov_rate']
        truncated = row['truncated']
        
        print(f"\n  • {col}:")
        
        if oov_rate == 0.0:
            print(f"    ✓ Perfect train/test overlap (OOV=0%)")
            print(f"      → Safe to use embedding layer; no out-of-vocab handling needed")
        elif oov_rate < 0.01:
            print(f"    ✓ Excellent coverage ({(1-oov_rate)*100:.2f}%)")
            print(f"      → Use embedding + <unk> token for rare OOV values")
        elif oov_rate < 0.05:
            print(f"    ⚠ Good coverage ({(1-oov_rate)*100:.2f}%)")
            print(f"      → Consider hash embedding or larger vocabulary")
        elif oov_rate < 0.1:
            print(f"    ⚠ Moderate coverage ({(1-oov_rate)*100:.2f}%)")
            print(f"      → Recommend hash embedding with collision handling")
        else:
            print(f"    ✗ Poor coverage ({(1-oov_rate)*100:.2f}%)")
            print(f"      → Strong distribution shift; consider feature engineering")
        
        if truncated:
            print(f"    ⚠ Vocabulary truncated; OOV rate is lower bound")
    
    # Correlation analysis (if profile available)
    if columns_profile_csv and Path(columns_profile_csv).exists():
        print("\n[CORRELATION WITH CARDINALITY]")
        df_profile = pd.read_csv(columns_profile_csv)
        df_merged = df_oov.merge(df_profile, left_on='column', right_on='column_name', how='left')
        
        # Show high-cardinality features
        df_high_card = df_merged[df_merged['nunique'] > 100].sort_values('nunique', ascending=False)
        print(f"  High-cardinality features (nunique > 100): {len(df_high_card)}")
        for idx, row in df_high_card.head(5).iterrows():
            print(f"    • {row['column']:15s}: {row['nunique']:6.0f} unique, "
                  f"OOV={row['test_oov_rate']*100:6.2f}%")
    
    # Distribution shift flag
    print("\n[DISTRIBUTION SHIFT ASSESSMENT]")
    avg_oov = df_oov['test_oov_rate'].mean()
    max_oov = df_oov['test_oov_rate'].max()
    
    if max_oov < 0.01 and avg_oov < 0.001:
        print("  ✓ LOW SHIFT: Train and test distributions are very similar")
        print("    → Standard embedding + simple <unk> handling sufficient")
    elif max_oov < 0.1 and avg_oov < 0.02:
        print("  ⚠ MODERATE SHIFT: Some vocabulary novelty in test")
        print("    → Use hash embedding or feature normalization")
    else:
        print("  ✗ HIGH SHIFT: Significant distribution mismatch")
        print("    → Strong feature engineering needed (bucketing, interaction, etc.)")
    
    print("\n" + "="*80)
    

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python verify_oov_rate.py <oov_rate_csv> [columns_profile_csv]")
        print("\nExample:")
        print("  python verify_oov_rate.py data/interim/avazu/eda/split/oov_rate_train_test.csv \\")
        print("         data/interim/avazu/eda/overview/columns_profile.csv")
        sys.exit(1)
    
    oov_csv = sys.argv[1]
    profile_csv = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(oov_csv).exists():
        print(f"ERROR: File not found: {oov_csv}", file=sys.stderr)
        sys.exit(1)
    
    analyze_oov_report(oov_csv, profile_csv)
