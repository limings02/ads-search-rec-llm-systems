#!/usr/bin/env python3
"""
Example OOV Rate Report Visualization

Demonstrates Stage 6 output and interpretation for documentation.
"""

import pandas as pd
import io

# Sample output from test_stage6_oov.py
sample_oov_data = """column,test_oov_rate,test_oov_count,test_total_count,truncated,notes
C1,0.0,0,16779,False,
C14,0.0,0,16779,False,
C15,0.0,0,16779,False,
C16,0.0,0,16779,False,
C17,0.0,0,16779,False,
C18,0.0,0,16779,False,
C19,0.0,0,16779,False,
C20,0.0,0,16779,False,
C21,0.0,0,16779,False,
site_id,5.959830740806961e-05,1,16779,False,
app_id,0.0,0,16779,False,
device_type,0.0,0,16779,False,"""

# Parse sample data
df_oov = pd.read_csv(io.StringIO(sample_oov_data))

print("\n" + "="*80)
print("STAGE 6: OOV RATE ANALYSIS - EXAMPLE OUTPUT")
print("="*80)

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
print("\n[ALL FEATURES (sorted by OOV rate)]")
print("  " + "-"*75)
print(f"  {'Column':<15} {'OOV Rate':>10} {'OOV/Total':>12} {'Coverage':>10} {'Status':<20}")
print("  " + "-"*75)

df_sorted = df_oov.sort_values('test_oov_rate', ascending=False)
for idx, row in df_sorted.iterrows():
    col = row['column']
    oov_rate = row['test_oov_rate']
    oov_count = int(row['test_oov_count'])
    total = int(row['test_total_count'])
    truncated = "🚨 TRUNC" if row['truncated'] else "OK"
    coverage = (1 - oov_rate) * 100
    
    print(f"  {col:<15} {oov_rate*100:>9.3f}% {oov_count:>5}/{total:<6} {coverage:>9.2f}% {truncated:<20}")

print("  " + "-"*75)

# Interpretation
print("\n[FEATURE ENGINEERING RECOMMENDATIONS]")

perfect_overlap = df_oov[df_oov['test_oov_rate'] == 0.0]
low_oov = df_oov[(df_oov['test_oov_rate'] > 0.0) & (df_oov['test_oov_rate'] <= 0.01)]
medium_oov = df_oov[(df_oov['test_oov_rate'] > 0.01) & (df_oov['test_oov_rate'] <= 0.05)]
high_oov = df_oov[df_oov['test_oov_rate'] > 0.05]

print(f"\n  [TIER 1] Perfect Overlap (OOV=0%, {len(perfect_overlap)} features)")
for col in perfect_overlap['column'].head(5):
    print(f"    ✓ {col}: Safe for standard embedding, no special OOV handling needed")

print(f"\n  [TIER 2] Low OOV (<1%, {len(low_oov)} features)")
for col in low_oov['column'].head(5):
    rate = df_oov[df_oov['column']==col]['test_oov_rate'].values[0]
    print(f"    ✓ {col}: Excellent coverage ({(1-rate)*100:.1f}%), add <unk> token")

print(f"\n  [TIER 3] Medium OOV (1-5%, {len(medium_oov)} features)")
for col in medium_oov['column'].head(5):
    rate = df_oov[df_oov['column']==col]['test_oov_rate'].values[0]
    print(f"    ⚠ {col}: Good coverage ({(1-rate)*100:.1f}%), consider hash embedding")

print(f"\n  [TIER 4] High OOV (>5%, {len(high_oov)} features)")
for col in high_oov['column'].head(5):
    rate = df_oov[df_oov['column']==col]['test_oov_rate'].values[0]
    print(f"    ✗ {col}: Poor coverage ({(1-rate)*100:.1f}%), strong feature engineering needed")

# Overall assessment
print("\n[DISTRIBUTION SHIFT ASSESSMENT]")
avg_oov = df_oov['test_oov_rate'].mean()
max_oov = df_oov['test_oov_rate'].max()

print(f"  Average OOV rate: {avg_oov*100:.4f}%")
print(f"  Maximum OOV rate: {max_oov*100:.4f}%")

if max_oov < 0.01 and avg_oov < 0.001:
    print("\n  ✓ LOW SHIFT: Train and test distributions are very similar")
    print("    → Standard embedding + simple <unk> handling sufficient")
elif max_oov < 0.1 and avg_oov < 0.02:
    print("\n  ⚠ MODERATE SHIFT: Some vocabulary novelty in test")
    print("    → Use hash embedding or feature normalization")
else:
    print("\n  ✗ HIGH SHIFT: Significant distribution mismatch")
    print("    → Strong feature engineering needed (bucketing, interaction, etc.)")

print("\n" + "="*80)
print("\n[NEXT STEPS]")
print("  1. Stage 7: PSI Drift Detection - Complement OOV with distribution shape analysis")
print("  2. Use OOV rates to inform FeatureMap vocabulary sizes and embedding strategies")
print("  3. Recommend bucketing for high-OOV features (C1, C14, etc.)")
print("  4. Consider interaction features between low-OOV and high-CTR features")
print("\n" + "="*80 + "\n")
