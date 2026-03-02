#!/usr/bin/env python3
"""
Advanced dataset - AI өөрөө сурах, бодит хорогдол дээр суурилсан
"""

import pandas as pd
import numpy as np

print("="*60)
print("Advanced Dzud AI Dataset Creation")
print("="*60)

# 1. Weather monthly уншиx
print("\n1. Loading weather data...")
weather = pd.read_csv('weather_omnogovi_monthly_clean.csv')
print(f"   Weather: {len(weather)} rows")

# 2. Livestock Өмнөговь уншиx (бодит өгөгдөл)
print("\n2. Loading livestock data...")
livestock = pd.read_csv('livestock_omnogovi.csv')

# Өмнөговь аймгийн нийт малыг авах
livestock_total = livestock[
    (livestock['Бүс'] == '               Өмнөговь') & 
    (livestock['Малын төрөл'] == 'Бүгд')
][['Он', 'Утга']].copy()

livestock_total.columns = ['year', 'total_livestock']
livestock_total = livestock_total.sort_values('year')
print(f"   Livestock: {len(livestock_total)} years")
print(livestock_total)

# 3. Жилийн өөрчлөлт тооцоолох (зудын шинж тэмдэг)
livestock_total['livestock_change'] = livestock_total['total_livestock'].diff()
livestock_total['livestock_change_pct'] = livestock_total['total_livestock'].pct_change() * 100

# Зудын жил: эхлээд data-driven (мал 10% буурсан), дараа нь албан ёсны зудын жил байвал баяжуулна
livestock_total['dzud_year'] = (livestock_total['livestock_change_pct'] < -10).astype(int)

# Label баяжуулах: official_dzud_years.csv байвал унших (year, dzud_official 0/1)
# ХХААХҮЯ, онцгой байдал гэх мэт эх сурвалжаас бөглөнө
official_dzud_path = 'official_dzud_years.csv'
try:
    official = pd.read_csv(official_dzud_path)
    if 'year' in official.columns and 'dzud_official' in official.columns:
        official = official[['year', 'dzud_official']].drop_duplicates('year')
        livestock_total = livestock_total.merge(official, on='year', how='left')
        livestock_total['dzud_official'] = livestock_total['dzud_official'].fillna(-1).astype(int)
        # Албан ёсны 1 байвал зуд гэж тэмдэглэх; бусад тохиолдолд өмнөх 10% rule хадгалах
        livestock_total.loc[livestock_total['dzud_official'] == 1, 'dzud_year'] = 1
        livestock_total = livestock_total.drop(columns=['dzud_official'], errors='ignore')
        print("   Label enriched with official_dzud_years.csv")
except FileNotFoundError:
    pass
except Exception as e:
    print(f"   Note: could not load {official_dzud_path}: {e}")

print("\n3. Dzud years identified:")
print(livestock_total[['year', 'total_livestock', 'livestock_change_pct', 'dzud_year']])

# 4. Weather-д livestock нэмэх
print("\n4. Merging weather and livestock...")
df = weather.merge(
    livestock_total[['year', 'total_livestock', 'livestock_change_pct', 'dzud_year']],
    on='year',
    how='left'
)

# 5. Advanced features үүсгэх
print("\n5. Creating advanced features...")

# Sort by soum, year, month
df = df.sort_values(['soum', 'year', 'month']).reset_index(drop=True)

# Өмнөх сарын өгөгдөл (lag features)
for col in ['avg_temp', 'min_temp', 'wind_speed', 'snowfall_sum', 'precip_sum']:
    df[f'{col}_lag1'] = df.groupby('soum')[col].shift(1)
    df[f'{col}_lag2'] = df.groupby('soum')[col].shift(2)

# Өвлийн сарууд (11, 12, 1, 2, 3) - зудын гол үе
df['is_winter'] = df['month'].isin([11, 12, 1, 2, 3]).astype(int)

# Хүйтний индекс (wind chill effect)
df['cold_index'] = df['min_temp'] - (df['wind_speed'] * 0.5)

# Цасан бүрхэвч (snowfall cumulative)
df['snow_cumulative'] = df.groupby(['soum', 'year'])['snowfall_sum'].cumsum()

# Хур тунадасны дутагдал (drought indicator)
df['precip_deficit'] = 20 - df['precip_sum']  # 20mm-ээс бага бол дутагдалтай

# Extreme weather events
df['extreme_cold'] = (df['min_temp'] < -25).astype(int)
df['extreme_wind'] = (df['wind_speed'] > 18).astype(int)
df['heavy_snow'] = (df['snowfall_sum'] > 10).astype(int)

# 6. Target variable: Дараа жилийн зуд (өвлийн сарууд дараа жилд нөлөөлнө)
# Өвөл (11-3 сар) -> дараа жилийн зуд
df['target_dzud'] = df.groupby('soum')['dzud_year'].shift(-6)  # 6 сарын дараа

# 7. NaN утгуудыг устгах (lag features-ийн эхний мөрүүд)
df_clean = df.dropna().copy()

print(f"\n6. Dataset after feature engineering: {len(df_clean)} rows")
print(f"   Features: {len(df_clean.columns)} columns")

# 8. Target distribution
print(f"\n7. Target distribution:")
print(df_clean['target_dzud'].value_counts())

# 9. Feature list
feature_cols = [
    'avg_temp', 'min_temp', 'wind_speed', 'snowfall_sum', 'precip_sum',
    'avg_temp_lag1', 'min_temp_lag1', 'wind_speed_lag1', 'snowfall_sum_lag1', 'precip_sum_lag1',
    'avg_temp_lag2', 'min_temp_lag2', 'wind_speed_lag2', 'snowfall_sum_lag2', 'precip_sum_lag2',
    'is_winter', 'cold_index', 'snow_cumulative', 'precip_deficit',
    'extreme_cold', 'extreme_wind', 'heavy_snow',
    'total_livestock', 'livestock_change_pct'
]

# 10. Final dataset
final_cols = ['aimag', 'soum', 'lat', 'lon', 'year', 'month'] + feature_cols + ['target_dzud']
df_final = df_clean[final_cols].copy()

# 11. Save
output_file = 'dzud_ai_dataset_advanced.csv'
df_final.to_csv(output_file, index=False)

print(f"\n✅ Advanced dataset saved: {output_file}")
print(f"   Total rows: {len(df_final)}")
print(f"   Features: {len(feature_cols)}")
print(f"\nFeature list:")
for i, feat in enumerate(feature_cols, 1):
    print(f"   {i:2d}. {feat}")

print(f"\nSample data:")
print(df_final.head(10))

print("\n" + "="*60)
print("Dataset creation complete!")
print("="*60)
