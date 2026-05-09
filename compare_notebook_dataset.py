#!/usr/bin/env python3
"""
Notebook vs zud_ai dataset харьцуулалт
"""
import pandas as pd

# Notebook-д ашигласан хувьсагчид
nb_inputs = {
    'snow_depth_cm':    ('snowfall_sum', 'мм — гэхдээ нэгж өөр: cm vs mm'),
    'min_temp_c':       ('min_temp', 'байна'),
    'wind_speed_ms':    ('wind_speed', 'байна'),
    'precipitation_mm': ('precip_sum', 'байна'),
    'pasture_index':    (None, 'БАЙХГҮЙ — NDVI/pasture index'),
    'grass_quality':    (None, 'БАЙХГҮЙ — өвсний чанар'),
    'water_access':     (None, 'БАЙХГҮЙ — усны хүртээмж'),
    'livestock_count':  ('total_livestock', 'байна'),
    'pasture_capacity': (None, 'БАЙХГҮЙ — бэлчээрийн даац K')
}

df = pd.read_csv('dzud_ai_dataset_advanced.csv')

print('='*80)
print('NOTEBOOK vs ZUD_AI DATASET ХАРЬЦУУЛАЛТ')
print('='*80)

print('\n1. NOTEBOOK ОРОЛТ vs DATASET БАГАНУУД')
print('-'*80)
print(f'{"Notebook хувьсагч":<25} {"Dataset багана":<25} {"Тайлбар"}')
print('-'*80)

for nb, (ds, note) in nb_inputs.items():
    mark = '✅' if ds else '❌'
    ds_str = ds if ds else '—'
    print(f'{mark} {nb:<25} {ds_str:<25} {note}')

print('\n2. DATASET БАГАНУУД (31 багана)')
print('-'*80)
for i, col in enumerate(df.columns, 1):
    print(f'{i:2d}. {col}')

print('\n3. НОРМЧИЛОЛЫН ХЯЗГААР ХАРЬЦУУЛАЛТ')
print('-'*80)
print(f'{"Хувьсагч":<20} {"Dataset min":<15} {"Dataset max":<15} {"Notebook хязгаар"}')
print('-'*80)

comparisons = [
    ('snowfall_sum', 'snowfall_sum', '5–35 cm'),
    ('min_temp', 'min_temp', 'abs(15–40)°C'),
    ('wind_speed', 'wind_speed', '2–18 m/s'),
    ('precip_sum', 'precip_sum', '0–30 mm'),
    ('total_livestock', 'total_livestock', '—')
]

for label, col, nb_range in comparisons:
    mn, mx = df[col].min(), df[col].max()
    print(f'{label:<20} {mn:<15.2f} {mx:<15.2f} {nb_range}')

print('\n4. ДҮГНЭЛТ')
print('-'*80)
print('✅ БАЙГАА өгөгдөл:')
print('   • Цаг уурын өгөгдөл (H): snowfall_sum, min_temp, wind_speed, precip_sum')
print('   • Малын өгөгдөл (D): total_livestock')
print()
print('❌ БАЙХГҮЙ өгөгдөл (V бүрэлдэхүүн):')
print('   • pasture_index (NDVI)')
print('   • grass_quality (өвсний чанар)')
print('   • water_access (усны хүртээмж)')
print('   • pasture_capacity (бэлчээрийн даац K)')
print()
print('⚠️  АНХААРУУЛГА:')
print('   • snowfall_sum нэгж: мм (notebook: cm) — хөрвүүлэлт хэрэгтэй')
print('   • Notebook нь FastAPI/Pydantic ашигладаг — zud_ai нь Flask')
print('   • Notebook нь математик загвар (R=αH+βV+γD) — zud_ai нь ML+Rule hybrid')
print()
print('📊 ХЭРЭГЖҮҮЛЭХ БОЛОМЖ:')
print('   • H (цаг уур): БҮРЭН хэрэгжүүлэх боломжтой')
print('   • D (малын дарамт): L/L_mean proxy ашиглаж болно')
print('   • V (бэлчээр): Proxy хувьсагчдаар орлуулах хэрэгтэй')
print('='*80)
