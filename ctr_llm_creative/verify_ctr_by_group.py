import pandas as pd
import glob

label_files = glob.glob(r'E:\沉淀项目\data\interim\avazu_test\eda\label\ctr_by_*.csv')
print('='*70)
print('STAGE 5: CTR_BY_GROUP RESULTS')
print('='*70)
print()
print(f'Total features analyzed: {len(label_files)}')
print()

# 分析显著性
for f in sorted(label_files)[:5]:  # Show first 5
    df = pd.read_csv(f)
    feat_name = f.split('\\')[-1].replace('ctr_by_', '').replace('.csv', '')
    n_sig = df['is_significant'].sum()
    ctr_range = (df['ctr'].min(), df['ctr'].max())
    print(f'{feat_name}:')
    print(f'  - Groups: {len(df)}, Significant: {n_sig}')
    print(f'  - CTR range: {ctr_range[0]:.4f} ~ {ctr_range[1]:.4f}')
    print(f'  - Z-score range: {df["zscore"].min():.3f} ~ {df["zscore"].max():.3f}')
    print()
