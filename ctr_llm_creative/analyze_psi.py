#!/usr/bin/env python
"""
Stage 7 PSI Analysis Tool - Utilities for analyzing and visualizing PSI drift.

Usage:
    python analyze_psi.py --psi_csv path/to/psi_train_test_summary.csv
    python analyze_psi.py --psi_csv ... --plot --save_plot drift_analysis.html
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys


def load_psi_summary(csv_path: str) -> pd.DataFrame:
    """Load PSI summary CSV."""
    df = pd.read_csv(csv_path)
    required_cols = {'column', 'psi', 'drift_level'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    return df.sort_values('psi', ascending=False)


def load_psi_detail(csv_path: str) -> pd.DataFrame:
    """Load PSI detail CSV for a specific feature."""
    df = pd.read_csv(csv_path)
    required_cols = {'bin', 'p_train', 'p_test', 'psi_term'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    return df.sort_values('psi_term', ascending=False)


def classify_drift(psi: float) -> Tuple[str, str]:
    """Classify drift level and return emoji/icon."""
    if psi >= 0.25:
        return "HIGH", "🔴"
    elif psi >= 0.1:
        return "MODERATE", "🟡"
    elif psi >= 0.01:
        return "LOW", "🟢"
    else:
        return "NEGLIGIBLE", "✓"


def print_summary_report(summary: pd.DataFrame) -> None:
    """Print formatted summary report."""
    print("\n" + "="*80)
    print("PSI DRIFT DETECTION SUMMARY REPORT")
    print("="*80)
    
    # Count drift levels
    high = (summary['psi'] >= 0.25).sum()
    moderate = ((summary['psi'] >= 0.1) & (summary['psi'] < 0.25)).sum()
    low = ((summary['psi'] >= 0.01) & (summary['psi'] < 0.1)).sum()
    negligible = (summary['psi'] < 0.01).sum()
    
    print(f"\n📊 DRIFT DISTRIBUTION (n={len(summary)}):")
    print(f"  🔴 HIGH       (PSI ≥ 0.25):  {high:3d} features")
    print(f"  🟡 MODERATE   (0.1 ≤ PSI < 0.25):  {moderate:3d} features")
    print(f"  🟢 LOW        (0.01 ≤ PSI < 0.1):  {low:3d} features")
    print(f"  ✓  NEGLIGIBLE (PSI < 0.01):  {negligible:3d} features")
    
    # Statistics
    print(f"\n📈 PSI STATISTICS:")
    print(f"  Mean:     {summary['psi'].mean():.4f}")
    print(f"  Median:   {summary['psi'].median():.4f}")
    print(f"  Std Dev:  {summary['psi'].std():.4f}")
    print(f"  Min:      {summary['psi'].min():.6f}")
    print(f"  Max:      {summary['psi'].max():.4f}")
    
    # Top drifters
    print(f"\n⚠️  TOP 10 DRIFTING FEATURES:")
    print(f"{'Rank':<4} {'Feature':<15} {'PSI':<10} {'Level':<10} {'Unique Bins':<15}")
    print("-" * 60)
    for idx, (_, row) in enumerate(summary.head(10).iterrows(), 1):
        icon = classify_drift(row['psi'])[1]
        bins = f"{row.get('train_unique_bins', '?')} → {row.get('test_unique_bins', '?')}"
        print(f"{idx:<4} {row['column']:<15} {row['psi']:<10.4f} {row['drift_level']:<10} {bins:<15} {icon}")
    
    # Most stable
    print(f"\n✅ TOP 10 MOST STABLE FEATURES:")
    print(f"{'Rank':<4} {'Feature':<15} {'PSI':<10} {'Level':<10}")
    print("-" * 40)
    for idx, (_, row) in enumerate(summary.tail(10).iterrows(), 1):
        icon = classify_drift(row['psi'])[1]
        print(f"{idx:<4} {row['column']:<15} {row['psi']:<10.6f} {row['drift_level']:<10} {icon}")
    
    print("\n" + "="*80)


def print_detailed_report(feature: str, detail_csv: str, summary: pd.DataFrame) -> None:
    """Print detailed bin-level analysis for a feature."""
    try:
        detail = load_psi_detail(detail_csv)
    except FileNotFoundError:
        print(f"❌ Detail CSV not found: {detail_csv}")
        return
    
    psi_value = summary[summary['column'] == feature]['psi'].values
    if len(psi_value) == 0:
        print(f"❌ Feature {feature} not found in summary")
        return
    
    psi = psi_value[0]
    drift_level, icon = classify_drift(psi)
    
    print("\n" + "="*80)
    print(f"DETAILED ANALYSIS: {feature} {icon}")
    print("="*80)
    print(f"Total PSI: {psi:.4f} | Level: {drift_level}")
    print()
    
    # Normalize contributions
    total_abs_contrib = detail['psi_term'].abs().sum()
    detail['contribution_%'] = (detail['psi_term'].abs() / total_abs_contrib * 100).round(2)
    
    print(f"{'Bin':<20} {'Train %':<10} {'Test %':<10} {'PSI Term':<12} {'Contrib %':<10}")
    print("-" * 65)
    
    for _, row in detail.iterrows():
        train_pct = row['p_train'] * 100
        test_pct = row['p_test'] * 100
        print(f"{str(row['bin']):<20} {train_pct:>8.2f}% {test_pct:>8.2f}% {row['psi_term']:>11.4f} {row['contribution_%']:>9.1f}%")
    
    # Key insights
    print()
    print("🔍 KEY INSIGHTS:")
    
    # Biggest shifts
    biggest_shift = detail.iloc[0]
    print(f"  • Largest shift: bin '{biggest_shift['bin']}'")
    print(f"    Train: {biggest_shift['p_train']*100:.1f}% → Test: {biggest_shift['p_test']*100:.1f}%")
    print(f"    Contribution: {biggest_shift['contribution_%']:.1f}% of total PSI")
    
    # Bins appearing/disappearing
    gone = detail[detail['p_train'] > 0.01][detail['p_test'] < 0.001]
    if len(gone) > 0:
        print(f"  • Bins disappearing ({len(gone)}): {', '.join(gone['bin'].astype(str).tolist()[:3])}")
    
    new = detail[detail['p_train'] < 0.001][detail['p_test'] > 0.01]
    if len(new) > 0:
        print(f"  • Bins appearing ({len(new)}): {', '.join(new['bin'].astype(str).tolist()[:3])}")
    
    print()
    print("="*80)


def export_drift_report(summary: pd.DataFrame, output_file: str) -> None:
    """Export report as formatted text file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("PSI DRIFT DETECTION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("SUMMARY BY DRIFT LEVEL\n")
        f.write("-"*80 + "\n")
        for level in ['HIGH', 'MODERATE', 'LOW', 'NEGLIGIBLE']:
            level_data = summary[summary['drift_level'] == level]
            if len(level_data) > 0:
                f.write(f"\n{level} DRIFT ({len(level_data)} features):\n")
                for _, row in level_data.iterrows():
                    f.write(f"  {row['column']:<20} PSI={row['psi']:<10.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("FULL TABLE (sorted by PSI)\n")
        f.write("="*80 + "\n")
        f.write(summary.to_string(index=False))
        f.write("\n")
    
    print(f"✅ Report exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze PSI drift detection results"
    )
    parser.add_argument(
        "--psi_csv",
        required=True,
        help="Path to psi_train_test_summary.csv"
    )
    parser.add_argument(
        "--detail_dir",
        default=None,
        help="Directory containing psi_train_test_*.csv detail files"
    )
    parser.add_argument(
        "--feature",
        default=None,
        help="Analyze specific feature in detail"
    )
    parser.add_argument(
        "--export",
        default=None,
        help="Export report to file"
    )
    parser.add_argument(
        "--threshold_high",
        type=float,
        default=0.25,
        help="PSI threshold for HIGH drift (default: 0.25)"
    )
    parser.add_argument(
        "--threshold_moderate",
        type=float,
        default=0.1,
        help="PSI threshold for MODERATE drift (default: 0.1)"
    )
    
    args = parser.parse_args()
    
    # Load summary
    try:
        summary = load_psi_summary(args.psi_csv)
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        sys.exit(1)
    
    # Print summary report
    print_summary_report(summary)
    
    # Detailed analysis if requested
    if args.feature:
        if args.detail_dir:
            detail_dir = Path(args.detail_dir)
        else:
            detail_dir = Path(args.psi_csv).parent
        
        detail_csv = detail_dir / f"psi_train_test_{args.feature}.csv"
        print_detailed_report(args.feature, str(detail_csv), summary)
    
    # Export if requested
    if args.export:
        export_drift_report(summary, args.export)


if __name__ == "__main__":
    main()
