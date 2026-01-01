import pandas as pd

# Read the generated files
hour_df = pd.read_csv(r'E:\沉淀项目\data\interim\avazu_test\eda\time\time_ctr_hour.csv')
day_df = pd.read_csv(r'E:\沉淀项目\data\interim\avazu_test\eda\time\time_ctr_day.csv')

print('='*70)
print('TEMPORAL_CTR VERIFICATION')
print('='*70)
print()
print('Hourly Statistics:')
print(f'  - Total unique hours: {len(hour_df)}')
print(f'  - Total impressions: {hour_df["impressions"].sum():,}')
print(f'  - Total clicks: {hour_df["clicks"].sum():,}')
print(f'  - Global CTR: {hour_df["clicks"].sum() / hour_df["impressions"].sum():.4f}')
print(f'  - CTR range: {hour_df["ctr"].min():.4f} ~ {hour_df["ctr"].max():.4f}')
print(f'  - CTR std dev: {hour_df["ctr"].std():.4f}')
print()
print('Top 5 Peak Hours:')
top_hours = hour_df.nlargest(5, 'ctr')[['hour', 'impressions', 'clicks', 'ctr']]
for idx, row in top_hours.iterrows():
    print(f'  {row["hour"]}: ctr={row["ctr"]:.4f} (clicks={row["clicks"]}, impressions={row["impressions"]})')
print()
print('Daily Statistics:')
print(f'  - Total unique days: {len(day_df)}')
print(f'  - Daily CTR range: {day_df["ctr"].min():.4f} ~ {day_df["ctr"].max():.4f}')
print(f'  - Daily CTR std dev: {day_df["ctr"].std():.4f}')
print()
print('Top 3 Peak Days:')
top_days = day_df.nlargest(3, 'ctr')[['day', 'impressions', 'clicks', 'ctr']]
for idx, row in top_days.iterrows():
    print(f'  {row["day"]}: ctr={row["ctr"]:.4f} (clicks={row["clicks"]}, impressions={row["impressions"]})')
print()
print('='*70)
