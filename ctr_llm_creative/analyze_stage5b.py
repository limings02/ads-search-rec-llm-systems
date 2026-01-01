#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 5B Analysis Tool: Analyze CTR by Top Values results

Usage:
  python analyze_stage5b.py --input_dir output/eda/label/
  python analyze_stage5b.py --input_dir output/eda/label/ --export report.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys


def analyze_ctr_file(csv_path: str) -> dict:
    """Analyze a single ctr_by_{col}_top.csv file."""
    df = pd.read_csv(csv_path)
    
    # Convert numeric columns
    df['ctr'] = pd.to_numeric(df['ctr'], errors='coerce')
    df['ci_lower'] = pd.to_numeric(df['ci_lower'], errors='coerce')
    df['ci_upper'] = pd.to_numeric(df['ci_upper'], errors='coerce')
    df['lift_vs_global'] = pd.to_numeric(df['lift_vs_global'], errors='coerce')
    df['z_score'] = pd.to_numeric(df['z_score'], errors='coerce')
    
    return {
        'total_values': len(df),
        'high_risk_count': df['sample_size_risk'].sum(),
        'significant_count': df['is_significant'].sum(),
        'avg_ctr': df['ctr'].mean(),
        'min_ctr': df['ctr'].min(),
        'max_ctr': df['ctr'].max(),
        'avg_lift': df['lift_vs_global'].mean(),
        'high_lift_values': len(df[df['lift_vs_global'] > 1.5]),
        'low_lift_values': len(df[df['lift_vs_global'] < 0.5]),
        'min_impr': df['impressions'].min(),
        'max_impr': df['impressions'].max(),
        'avg_impr': df['impressions'].mean(),
        'dataframe': df
    }


def format_output(results: dict) -> str:
    """Format analysis results as text."""
    output = []
    output.append('=' * 80)
    output.append('STAGE 5B ANALYSIS REPORT: CTR by Top Values')
    output.append('=' * 80)
    
    # Overall summary
    output.append('\n[SUMMARY]')
    output.append(f'Features analyzed: {len(results)}')
    
    total_values = sum(r['total_values'] for r in results.values())
    total_high_risk = sum(r['high_risk_count'] for r in results.values())
    total_significant = sum(r['significant_count'] for r in results.values())
    
    output.append(f'Total values: {total_values}')
    output.append(f'High-risk values: {total_high_risk} ({total_high_risk/total_values*100:.1f}%)')
    output.append(f'Significant values: {total_significant} ({total_significant/total_values*100:.1f}%)')
    
    # Per-feature breakdown
    output.append('\n[FEATURES BREAKDOWN]')
    output.append(f'{"Feature":<20} {"Values":<8} {"High-Risk":<12} {"Significant":<12} {"Avg CTR":<10} {"Avg Lift":<10}')
    output.append('-' * 82)
    
    for col in sorted(results.keys()):
        r = results[col]
        high_risk_pct = r['high_risk_count'] / r['total_values'] * 100 if r['total_values'] > 0 else 0
        sig_pct = r['significant_count'] / r['total_values'] * 100 if r['total_values'] > 0 else 0
        
        output.append(f'{col:<20} {r["total_values"]:<8} '
                     f'{r["high_risk_count"]:>2} ({high_risk_pct:>5.1f}%)  '
                     f'{r["significant_count"]:>2} ({sig_pct:>5.1f}%)  '
                     f'{r["avg_ctr"]:<10.4f} {r["avg_lift"]:<10.4f}')
    
    # Risk analysis
    output.append('\n[RISK ANALYSIS]')
    output.append(f'{"Feature":<20} {"High-Risk %":<15} {"Risk Level":<15}')
    output.append('-' * 50)
    
    for col in sorted(results.keys()):
        r = results[col]
        risk_pct = r['high_risk_count'] / r['total_values'] * 100 if r['total_values'] > 0 else 0
        
        if risk_pct > 50:
            risk_level = 'HIGH'
        elif risk_pct > 30:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        output.append(f'{col:<20} {risk_pct:>6.1f}%          {risk_level:<15}')
    
    # High-value features
    output.append('\n[HIGH-VALUE FEATURES (Lift > 1.5)]')
    
    for col in sorted(results.keys()):
        r = results[col]
        if r['high_lift_values'] > 0:
            df = r['dataframe']
            high_val = df[df['lift_vs_global'] > 1.5].nlargest(3, 'lift_vs_global')
            
            output.append(f'\n{col}: {r["high_lift_values"]} values with lift > 1.5')
            for _, row in high_val.iterrows():
                sig = '✓' if row['is_significant'] else 'x'
                risk = '[RISK]' if row['sample_size_risk'] else ''
                output.append(f'  {row["value"]:<20} lift={row["lift_vs_global"]:>5.3f} '
                            f'impr={int(row["impressions"]):>5} {sig} {risk}')
    
    # Recommendations
    output.append('\n[RECOMMENDATIONS]')
    
    if total_high_risk / total_values > 0.5:
        output.append('⚠ High proportion of unreliable estimates (>50%)')
        output.append('  → Recommendation: Increase --min_support parameter')
        output.append('  → Current high-risk values should not be used in production')
    
    high_risk_features = [(col, r['high_risk_count'] / r['total_values']) 
                          for col, r in results.items() 
                          if r['high_risk_count'] / r['total_values'] > 0.3]
    
    if high_risk_features:
        output.append('\n⚠ Features with >30% high-risk values:')
        for col, ratio in sorted(high_risk_features, key=lambda x: x[1], reverse=True):
            output.append(f'  - {col}: {ratio*100:.1f}%')
        output.append('  → Consider grouping rare values or using embeddings')
    
    output.append('\n' + '=' * 80)
    return '\n'.join(output)


def main():
    parser = argparse.ArgumentParser(description='Analyze Stage 5B CTR results')
    parser.add_argument('--input_dir', required=True, help='Directory containing ctr_by_*.csv files')
    parser.add_argument('--export', help='Export report to file (optional)')
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f'ERROR: Directory not found: {input_dir}')
        sys.exit(1)
    
    # Find all ctr_by_*_top.csv files
    csv_files = sorted(input_dir.glob('ctr_by_*_top.csv'))
    
    if not csv_files:
        print(f'ERROR: No ctr_by_*_top.csv files found in {input_dir}')
        sys.exit(1)
    
    print(f'[INFO] Found {len(csv_files)} CTR result files')
    
    # Analyze each file
    results = {}
    for csv_file in csv_files:
        col_name = csv_file.stem.replace('ctr_by_', '').replace('_top', '')
        print(f'[OK] Analyzing {col_name}...')
        results[col_name] = analyze_ctr_file(str(csv_file))
    
    # Generate report
    report = format_output(results)
    print('\n' + report)
    
    # Export if requested
    if args.export:
        with open(args.export, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f'\n[OK] Report exported to {args.export}')


if __name__ == '__main__':
    main()
