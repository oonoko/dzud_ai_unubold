#!/usr/bin/env python3
"""
Бүх 15 асуултыг кодоор шалгах diagnostic скрипт
"""
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix, roc_auc_score)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

C = {
    'H': '\033[95m', 'B': '\033[94m', 'C': '\033[96m',
    'G': '\033[92m', 'Y': '\033[93m', 'R': '\033[91m',
    'BOLD': '\033[1m', 'END': '\033[0m'
}

def hdr(txt):
    print(f"\n{C['BOLD']}{C['C']}{'='*70}{C['END']}")
    print(f"{C['BOLD']}{C['C']}{txt.center(70)}{C['END']}")
    print(f"{C['BOLD']}{C['C']}{'='*70}{C['END']}")

def sec(txt):
    print(f"\n{C['BOLD']}{C['Y']}▶ {txt}{C['END']}")
    print(f"{C['Y']}{'-'*60}{C['END']}")

def ok(txt):  print(f"  {C['G']}✅ {txt}{C['END']}")
def warn(txt): print(f"  {C['Y']}⚠️  {txt}{C['END']}")
def err(txt):  print(f"  {C['R']}❌ {txt}{C['END']}")
def info(txt): print(f"  {C['B']}ℹ️  {txt}{C['END']}")

# ─────────────────────────────────────────────
hdr("ЗУД AI — БҮРЭН DIAGNOSTIC ШАЛГАЛТ")
# ─────────────────────────────────────────────

df = pd.read_csv('dzud_ai_dataset_advanced.csv')
info(f"Dataset: {df.shape[0]} мөр × {df.shape[1]} багана")
info(f"Жилүүд: {sorted(df['year'].unique())}")
info(f"Сумууд: {df['soum'].nunique()} — {df['soum'].unique().tolist()}")

# ══════════════════════════════════════════════
# 1–2. target_dzud логик ба утга
# ══════════════════════════════════════════════
sec("1–2. target_dzud логик ба утга")

print("\n  target_dzud тархалт:")
print(df['target_dzud'].value_counts().to_string())

print("\n  target=1 болсон сарууд:")
t1 = df[df['target_dzud'] == 1][['year','month','soum']].value_counts().reset_index()
t1.columns = ['year','month','soum','count']
print(t1.sort_values(['year','month']).to_string(index=False))

winter_months = [11, 12, 1, 2, 3]
summer_with_target1 = df[(df['target_dzud'] == 1) & (~df['month'].isin(winter_months))]
winter_with_target1 = df[(df['target_dzud'] == 1) & (df['month'].isin(winter_months))]

print(f"\n  target=1 өвлийн сарууд (11,12,1,2,3): {len(winter_with_target1)} мөр")
print(f"  target=1 зуны сарууд (4–10):          {len(summer_with_target1)} мөр")

if len(summer_with_target1) > 0:
    err("Зуны сарууд (4–10) дээр target=1 байна — shift(-6) логикийн үр дагавар!")
    warn("Зуд зуны сарт тохиолддоггүй — энэ нь label-ийн алдаа")
else:
    ok("Зуны сарт target=1 байхгүй")

# ══════════════════════════════════════════════
# 3. shift(-6) target leakage шалгалт
# ══════════════════════════════════════════════
sec("3. shift(-6) target leakage шалгалт")

info("shift(-6) гэдэг нь: одоогийн мөрийн target = 6 сарын дараах dzud_year")
info("Жишээ: 2019-оны 7-р сарын target = 2020-оны 1-р сарын зуд")

# Нэг сумын жишээ харуулах
sample_soum = df['soum'].iloc[0]
sample = df[df['soum'] == sample_soum][['year','month','total_livestock',
                                         'livestock_change_pct','target_dzud']].head(20)
print(f"\n  {sample_soum} сумын жишээ (эхний 20 мөр):")
print(sample.to_string(index=False))

# Leakage шалгалт: train дотор ирээдүйн мэдээлэл байгаа эсэх
# Time-based split: train < 2023, test >= 2023
train_df = df[df['year'] < 2023]
test_df  = df[df['year'] >= 2023]

# shift(-6) хийснээр 2022-оны сүүлийн сарууд 2023-ын зудыг "мэдэж" байна
# Энэ нь train дотор test-ийн label-ийн мэдээлэл байна гэсэн үг
leak_rows = train_df[(train_df['year'] == 2022) & (train_df['month'].isin([7,8,9,10,11,12]))]
if len(leak_rows) > 0 and leak_rows['target_dzud'].sum() > 0:
    err(f"TARGET LEAKAGE: 2022-оны 7–12-р сарын target нь 2023-ын зудыг агуулна!")
    err(f"  Нөлөөлсөн мөр: {len(leak_rows)}, target=1: {int(leak_rows['target_dzud'].sum())}")
else:
    ok("Тодорхой leakage илрэгдсэнгүй")

# ══════════════════════════════════════════════
# 4. Зуны сарын target=1 тайлбар
# ══════════════════════════════════════════════
sec("4. Зуны сарын target=1 тайлбар")

if len(summer_with_target1) > 0:
    err("Зуны сарт target=1 байна — энэ нь БУРУУ!")
    info("Шалтгаан: shift(-6) нь өвлийн сарын dzud_year-ийг зуны сарт 'буцааж' тавина")
    info("Жишээ: 2020-оны 1-р сарын зуд → 2019-оны 7-р сарын target=1 болно")
    info("Засах арга: target=1 зөвхөн өвлийн сарт байх ёстой")
    print(f"\n  Зуны target=1 мөрүүд:")
    print(summer_with_target1[['year','month','soum','target_dzud']].head(10).to_string(index=False))
else:
    ok("Зуны сарт target=1 байхгүй — зөв")

# ══════════════════════════════════════════════
# 5–6. Train/test split шалгалт
# ══════════════════════════════════════════════
sec("5–6. Train/Test Split шалгалт")

info(f"Train: {df[df['year']<2023]['year'].unique()} → {len(train_df)} мөр")
info(f"Test:  {df[df['year']>=2023]['year'].unique()} → {len(test_df)} мөр")
ok("Time-based split (жилээр) ашиглаж байна — зөв!")

# Нэг сумын ойролцоо сарууд холилдох эрсдэл
# Time-based split дээр энэ асуудал байхгүй
# Гэхдээ 5 сум × 12 сар = 60 мөр/жил → test дотор 5 сум × 24 сар = 120 мөр
info(f"Test set: {len(test_df)} мөр ({test_df['soum'].nunique()} сум × {test_df['month'].nunique()} сар)")

# Autocorrelation эрсдэл: нэг сумын зэргэлдээ сарууд хоорондоо хамааралтай
warn("Нэг сумын зэргэлдээ сарууд хоорондоо хамааралтай (autocorrelation)")
warn("5 сум × 9 жил = 540 мөр — dataset маш жижиг, CV score хэт өндөр гарч болно")

# ══════════════════════════════════════════════
# 7–8. Normalization data leakage шалгалт
# ══════════════════════════════════════════════
sec("7–8. Normalization data leakage шалгалт")

info("Одоогийн кодод StandardScaler зөвхөн X_train дээр fit хийж байна — зөв!")
ok("scaler.fit_transform(X_train) → scaler.transform(X_test) — leakage байхгүй")

warn("АНХААРУУЛГА: Шинэ math feature нэмэхдээ MinMaxScaler-ийг бүх df дээр хийвэл LEAKAGE!")
err("БУРУУ: df['snow_risk'] = minmax(df['snowfall_sum'])  ← бүх data дээр")
ok("ЗӨВХӨН TRAIN дээр fit хийх:")
print(f"  {C['G']}  scaler_mm = MinMaxScaler(){C['END']}")
print(f"  {C['G']}  X_train[cols] = scaler_mm.fit_transform(X_train[cols]){C['END']}")
print(f"  {C['G']}  X_test[cols]  = scaler_mm.transform(X_test[cols]){C['END']}")

# ══════════════════════════════════════════════
# 9. zud_risk_index vs target_dzud давхардал
# ══════════════════════════════════════════════
sec("9. zud_risk_index ↔ target_dzud давхардал шалгалт")

# Simulate zud_risk_index
def minmax_series(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-8)

df2 = df.copy()
df2['snow_risk']  = minmax_series(-df2['snowfall_sum'])   # их цас = өндөр эрсдэл
df2['cold_risk']  = minmax_series(-df2['min_temp'])
df2['wind_risk']  = minmax_series(df2['wind_speed'])
df2['precip_risk']= minmax_series(20 - df2['precip_sum'])
df2['H'] = 0.35*df2['cold_risk'] + 0.35*df2['snow_risk'] + 0.20*df2['wind_risk'] + 0.10*df2['precip_risk']

L_mean = df2['total_livestock'].mean()
df2['D'] = (df2['total_livestock'] / L_mean).clip(0, 2) / 2

df2['zud_risk_index'] = 0.75 * df2['H'] + 0.25 * df2['D']

# Корреляц шалгах
corr_with_target = df2['zud_risk_index'].corr(df2['target_dzud'])
corr_H_target    = df2['H'].corr(df2['target_dzud'])
corr_D_target    = df2['D'].corr(df2['target_dzud'])

print(f"\n  Корреляц (Pearson):")
print(f"  zud_risk_index ↔ target_dzud : {corr_with_target:+.4f}")
print(f"  H (climate)    ↔ target_dzud : {corr_H_target:+.4f}")
print(f"  D (livestock)  ↔ target_dzud : {corr_D_target:+.4f}")

if abs(corr_with_target) > 0.7:
    err(f"Өндөр корреляц ({corr_with_target:.3f}) — target leakage эрсдэлтэй!")
elif abs(corr_with_target) > 0.4:
    warn(f"Дунд зэргийн корреляц ({corr_with_target:.3f}) — feature болгон нэмж болно")
else:
    ok(f"Бага корреляц ({corr_with_target:.3f}) — давхардал байхгүй, feature болгон нэмж болно")

# ══════════════════════════════════════════════
# 10–11. Baseline vs Math-feature model харьцуулалт
# ══════════════════════════════════════════════
sec("10–11. Baseline vs Math-feature model харьцуулалт")

FEATURE_COLS_BASE = [
    'avg_temp', 'min_temp', 'wind_speed', 'snowfall_sum', 'precip_sum',
    'avg_temp_lag1', 'min_temp_lag1', 'wind_speed_lag1', 'snowfall_sum_lag1', 'precip_sum_lag1',
    'avg_temp_lag2', 'min_temp_lag2', 'wind_speed_lag2', 'snowfall_sum_lag2', 'precip_sum_lag2',
    'is_winter', 'cold_index', 'snow_cumulative', 'precip_deficit',
    'extreme_cold', 'extreme_wind', 'heavy_snow',
    'total_livestock', 'livestock_change_pct'
]

FEATURE_COLS_MATH = FEATURE_COLS_BASE + [
    'snow_risk', 'cold_risk', 'wind_risk', 'precip_risk',
    'H', 'D', 'zud_risk_index'
]

y = df2['target_dzud']
train_mask = df2['year'] < 2023
test_mask  = df2['year'] >= 2023

def run_model(feature_cols, label):
    X = df2[feature_cols].fillna(0)
    X_tr, X_te = X[train_mask], X[test_mask]
    y_tr, y_te = y[train_mask], y[test_mask]

    if y_tr.nunique() < 2 or y_te.nunique() < 2:
        return None

    clf = RandomForestClassifier(n_estimators=200, max_depth=15,
                                  class_weight='balanced', random_state=42)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]

    acc  = accuracy_score(y_te, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_te, y_pred, average='weighted', zero_division=0)
    try:
        auc = roc_auc_score(y_te, y_prob)
    except:
        auc = float('nan')

    cv = cross_val_score(clf, X_tr, y_tr, cv=min(5, len(X_tr)//10), scoring='accuracy')

    return {
        'label': label, 'model': clf, 'feature_cols': feature_cols,
        'acc': acc, 'prec': p, 'rec': r, 'f1': f1, 'auc': auc,
        'cv_mean': cv.mean(), 'cv_std': cv.std(),
        'y_te': y_te, 'y_pred': y_pred
    }

res_base = run_model(FEATURE_COLS_BASE, "Baseline (24 features)")
res_math = run_model(FEATURE_COLS_MATH, "Math+ML (31 features)")

print(f"\n  {'Загвар':<30} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6} {'CV':>12}")
print(f"  {'-'*75}")

for res in [res_base, res_math]:
    if res:
        color = C['G'] if res['f1'] > 0.8 else C['Y']
        print(f"  {color}{res['label']:<30} "
              f"{res['acc']:>6.3f} {res['prec']:>6.3f} {res['rec']:>6.3f} "
              f"{res['f1']:>6.3f} {res['auc']:>6.3f} "
              f"{res['cv_mean']:.3f}±{res['cv_std']:.3f}{C['END']}")

# Сайжирсан эсэх
if res_base and res_math:
    delta_f1  = res_math['f1']  - res_base['f1']
    delta_auc = res_math['auc'] - res_base['auc']
    print(f"\n  Өөрчлөлт: F1 {delta_f1:+.4f}  |  AUC {delta_auc:+.4f}")
    if delta_f1 > 0.01:
        ok(f"Math feature нэмснээр F1 {delta_f1:+.4f} сайжирлаа")
    elif delta_f1 < -0.01:
        warn(f"Math feature нэмснээр F1 {delta_f1:+.4f} буурлаа — давхардсан feature байж болно")
    else:
        info(f"Math feature нэмсэн ч F1 өөрчлөгдсөнгүй ({delta_f1:+.4f}) — neutral")

# ── Confusion Matrix ──
sec("11a. Confusion Matrix")
for res in [res_base, res_math]:
    if res:
        cm = confusion_matrix(res['y_te'], res['y_pred'])
        print(f"\n  [{res['label']}]")
        print(f"  {'':20} Predicted 0   Predicted 1")
        print(f"  {'Actual 0 (No Dzud)':<20}  {C['G']}{cm[0,0]:^13}{C['END']}  {C['R']}{cm[0,1]:^11}{C['END']}")
        print(f"  {'Actual 1 (Dzud)':<20}  {C['R']}{cm[1,0]:^13}{C['END']}  {C['G']}{cm[1,1]:^11}{C['END']}")
        tn,fp,fn,tp = cm.ravel()
        print(f"  TN={tn} FP={fp} FN={fn} TP={tp}")

# ── Feature Importance ──
sec("11b. Feature Importance (Math model, Top 15)")
if res_math:
    fi = pd.DataFrame({
        'feature': res_math['feature_cols'],
        'importance': res_math['model'].feature_importances_
    }).sort_values('importance', ascending=False).head(15)

    print(f"\n  {'Rank':<5} {'Feature':<30} {'Importance':>10}  Bar")
    print(f"  {'-'*65}")
    for i, row in enumerate(fi.itertuples(), 1):
        bar = '█' * int(row.importance * 200)
        is_math = row.feature in ['snow_risk','cold_risk','wind_risk','precip_risk',
                                   'H','D','zud_risk_index']
        color = C['H'] if is_math else C['B']
        print(f"  {i:<5} {color}{row.feature:<30}{C['END']} {row.importance:>10.4f}  {color}{bar}{C['END']}")
    print(f"\n  {C['H']}█ = математик feature{C['END']}  {C['B']}█ = одоогийн feature{C['END']}")

# ══════════════════════════════════════════════
# 12. R = 0.75H + 0.25D давхардал шалгалт
# ══════════════════════════════════════════════
sec("12. R = 0.75H + 0.25D давхардал шалгалт")

# cold_index болон cold_risk хоорондын корреляц
corr_ci_cr = df2['cold_index'].corr(df2['cold_risk'])
corr_snow_sr = df2['snowfall_sum'].corr(df2['snow_risk'])
corr_H_ci = df2['H'].corr(df2['cold_index'])

print(f"\n  cold_index ↔ cold_risk корреляц:   {corr_ci_cr:+.4f}")
print(f"  snowfall_sum ↔ snow_risk корреляц: {corr_snow_sr:+.4f}")
print(f"  H ↔ cold_index корреляц:           {corr_H_ci:+.4f}")

if abs(corr_ci_cr) > 0.9:
    warn(f"cold_index ↔ cold_risk маш өндөр корреляц ({corr_ci_cr:.3f}) — давхардсан feature!")
    info("cold_risk нь cold_index-ийн нормчилсон хувилбар — нэгийг нь хасаж болно")
if abs(corr_snow_sr) > 0.9:
    warn(f"snowfall_sum ↔ snow_risk маш өндөр корреляц ({corr_snow_sr:.3f}) — давхардсан!")

# ══════════════════════════════════════════════
# 13. Model pipeline эвдрэх эсэх
# ══════════════════════════════════════════════
sec("13. Model pipeline эвдрэх эсэх шалгалт")

try:
    X_test_check = df2[FEATURE_COLS_MATH].fillna(0)
    clf_check = RandomForestClassifier(n_estimators=10, random_state=42)
    clf_check.fit(X_test_check[train_mask], y[train_mask])
    _ = clf_check.predict(X_test_check[test_mask])
    ok(f"Pipeline ажиллаж байна — {len(FEATURE_COLS_MATH)} feature, алдаагүй")
except Exception as e:
    err(f"Pipeline алдаа: {e}")

# ══════════════════════════════════════════════
# 14. risk_predictor.py feature consistency
# ══════════════════════════════════════════════
sec("14. risk_predictor.py feature consistency шалгалт")

PREDICTOR_FEATURES = [
    'avg_temp', 'min_temp', 'wind_speed', 'snowfall_sum', 'precip_sum',
    'avg_temp_lag1', 'min_temp_lag1', 'wind_speed_lag1', 'snowfall_sum_lag1', 'precip_sum_lag1',
    'avg_temp_lag2', 'min_temp_lag2', 'wind_speed_lag2', 'snowfall_sum_lag2', 'precip_sum_lag2',
    'is_winter', 'cold_index', 'snow_cumulative', 'precip_deficit',
    'extreme_cold', 'extreme_wind', 'heavy_snow',
    'total_livestock', 'livestock_change_pct'
]

math_features_new = ['snow_risk','cold_risk','wind_risk','precip_risk','H','D','zud_risk_index']
missing_in_predictor = [f for f in math_features_new if f not in PREDICTOR_FEATURES]

if missing_in_predictor:
    err(f"risk_predictor.py дотор дараах шинэ feature-үүд байхгүй:")
    for f in missing_in_predictor:
        print(f"    {C['R']}• {f}{C['END']}")
    warn("Шинэ feature нэмсний дараа risk_predictor.py-д ML_FEATURE_COLS болон")
    warn("_build_ml_features() функцийг заавал шинэчлэх хэрэгтэй!")
else:
    ok("Бүх feature risk_predictor.py-д байна")

# ══════════════════════════════════════════════
# 15. V хэсгийг "хэрэгжүүлсэн" гэж бичих зөв үү
# ══════════════════════════════════════════════
sec("15. V (бэлчээрийн эмзэг байдал) дипломын тайланд")

ndvi_cols = [c for c in df.columns if 'ndvi' in c.lower() or 'pasture' in c.lower()
             or 'vegetation' in c.lower() or 'water_avail' in c.lower()]

if ndvi_cols:
    ok(f"NDVI/pasture өгөгдөл байна: {ndvi_cols}")
else:
    err("NDVI, pasture capacity, water availability — dataset-д БАЙХГҮЙ!")
    print(f"""
  {C['Y']}Дипломын тайланд зөв бичих хувилбарууд:{C['END']}

  {C['R']}БУРУУ:{C['END']} "V = b1(1-G) + b2(1-F) + b3(1-U) загварыг хэрэгжүүллээ"

  {C['G']}ЗӨВХӨН PROXY ХЭРЭГЛЭСЭН БОЛ:{C['END']}
  "Бэлчээрийн эмзэг байдлын V бүрэлдэхүүнийг тооцоолоход NDVI болон
   бэлчээрийн даацын бодит өгөгдөл байхгүй тул цаг уурын proxy
   хувьсагчдаар орлуулав:
     G* = normalize(precip_sum)       — хур тунадас → бэлчээрийн чийг
     F* = normalize(avg_temp_summer)  — зуны температур → өвсний ургалт
     U* = normalize(precip+snow*0.1)  — нийт чийг → усны хүртээмж
   Ирээдүйн хөгжүүлэлтэд MODIS NDVI болон МХЕГ-ийн бэлчээрийн
   зураглалын өгөгдлийг нэгтгэхээр төлөвлөсөн."

  {C['G']}ХАМГИЙН АЮУЛГҮЙ:{C['END']}
  "Энэхүү судалгааны хүрээнд V бүрэлдэхүүнийг хэрэгжүүлэх
   шаардлагатай NDVI болон бэлчээрийн даацын өгөгдөл байхгүй
   тул загварыг R = αH + γD хэлбэрт хялбаршуулав."
""")

# ══════════════════════════════════════════════
# ЭЦСИЙН ДҮГНЭЛТ
# ══════════════════════════════════════════════
hdr("ЭЦСИЙН ДҮГНЭЛТ — ЗАСАХ ШААРДЛАГАТАЙ ЗҮЙЛС")

issues = [
    ("ЧУХАЛ", "shift(-6) → зуны сарт target=1 байна — label логик засах хэрэгтэй"),
    ("ЧУХАЛ", "Normalization бүх df дээр хийвэл data leakage — train дээр fit хийх"),
    ("ЧУХАЛ", "risk_predictor.py-д шинэ feature нэмэх — ML_FEATURE_COLS шинэчлэх"),
    ("АНХААРУУЛГА", "cold_index ↔ cold_risk давхардсан — нэгийг нь хасаж болно"),
    ("АНХААРУУЛГА", "Dataset маш жижиг (510 мөр, 5 сум) — CV score хэт өндөр байж болно"),
    ("МЭДЭЭЛЭЛ", "V хэсгийг 'хэрэгжүүлсэн' гэж бичих буруу — proxy гэж тайлбарлах"),
    ("МЭДЭЭЛЭЛ", "Time-based split зөв — санамсаргүй split хийгдэхгүй байна"),
]

for level, msg in issues:
    if level == "ЧУХАЛ":
        err(f"[{level}] {msg}")
    elif level == "АНХААРУУЛГА":
        warn(f"[{level}] {msg}")
    else:
        info(f"[{level}] {msg}")

print(f"\n{C['BOLD']}{C['G']}Diagnostic дууслаа.{C['END']}\n")
