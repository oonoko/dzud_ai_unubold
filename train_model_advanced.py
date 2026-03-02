#!/usr/bin/env python3
"""
Advanced AI model training - бодит зудын таамаглал
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib
import json

print("="*60)
print("Advanced Dzud AI Model Training")
print("="*60)

# 1. Dataset уншиx
print("\n1. Loading dataset...")
df = pd.read_csv('dzud_ai_dataset_advanced.csv')
print(f"   Dataset: {len(df)} rows, {len(df.columns)} columns")

# 2. Features and target
feature_cols = [
    'avg_temp', 'min_temp', 'wind_speed', 'snowfall_sum', 'precip_sum',
    'avg_temp_lag1', 'min_temp_lag1', 'wind_speed_lag1', 'snowfall_sum_lag1', 'precip_sum_lag1',
    'avg_temp_lag2', 'min_temp_lag2', 'wind_speed_lag2', 'snowfall_sum_lag2', 'precip_sum_lag2',
    'is_winter', 'cold_index', 'snow_cumulative', 'precip_deficit',
    'extreme_cold', 'extreme_wind', 'heavy_snow',
    'total_livestock', 'livestock_change_pct'
]

X = df[feature_cols].copy()
y = df['target_dzud'].copy()

print(f"\n2. Features: {len(feature_cols)}")
print(f"   Target distribution:")
print(y.value_counts())

# Check class imbalance
class_counts = y.value_counts()
if len(class_counts) > 1:
    imbalance_ratio = class_counts.max() / class_counts.min()
    print(f"   Class imbalance ratio: {imbalance_ratio:.2f}:1")

# 3. Handle missing values
X = X.fillna(X.mean())

# 4. Train/test split — цагийн дарааллаар (жилээр салгах): өмнөх жилүүд train, сүүлийн 2 жил test
TEST_YEAR_START = 2023  # 2023 талаас эхлэн test set
years = sorted(df['year'].unique())
test_years = [y for y in years if y >= TEST_YEAR_START]
train_years = [y for y in years if y < TEST_YEAR_START]

if len(test_years) >= 1 and len(train_years) >= 1:
    train_mask = df['year'].isin(train_years)
    test_mask = df['year'].isin(test_years)
    X_train = X.loc[train_mask].copy()
    X_test = X.loc[test_mask].copy()
    y_train = y.loc[train_mask].copy()
    y_test = y.loc[test_mask].copy()
    print(f"\n3. Train/test split (time-based by year):")
    print(f"   Train years: {train_years} -> {len(X_train)} rows")
    print(f"   Test years:  {test_years} -> {len(X_test)} rows")
else:
    # Fallback: not enough years for time split
    if len(class_counts) > 1 and class_counts.min() >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    print(f"\n3. Train/test split (random fallback):")
    print(f"   Train: {len(X_train)} rows")
    print(f"   Test: {len(X_test)} rows")

# 5. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train models
print("\n" + "="*60)
print("4. Training models...")
print("="*60)

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=2000,
        random_state=42,
        class_weight='balanced'  # Handle imbalance
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.05,
        random_state=42
    )
}

results = {}

for name, model in models.items():
    print(f"\n{name}:")
    
    # Train
    if 'Logistic' in name:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {accuracy:.3f}")
    
    # AUC score (if binary classification)
    if len(np.unique(y)) == 2:
        auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        print(f"  AUC-ROC: {auc:.3f}")
    
    # Cross-validation
    try:
        if 'Logistic' in name:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=min(5, len(X_train)//2))
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=min(5, len(X_train)//2))
        print(f"  CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    except:
        print(f"  CV Score: N/A (insufficient data)")
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

# 7. Select best model
best_model_name = max(results, key=lambda k: results[k]['accuracy'])
best_model = results[best_model_name]['model']
y_pred_best = results[best_model_name]['predictions']

print("\n" + "="*60)
print(f"5. Best Model: {best_model_name}")
print("="*60)

# 8. Detailed evaluation
print("\nClassification Report:")
target_names = ['No Dzud', 'Dzud'] if len(np.unique(y)) == 2 else [f'Class {i}' for i in np.unique(y)]
print(classification_report(y_test, y_pred_best, target_names=target_names, zero_division=0))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_best)
print(cm)

# 9. Feature importance
if hasattr(best_model, 'feature_importances_'):
    print("\n6. Feature Importance (Top 15):")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(15).to_string(index=False))
elif hasattr(best_model, 'coef_'):
    print("\n6. Feature Coefficients (Top 15):")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': np.abs(best_model.coef_[0])
    }).sort_values('coefficient', ascending=False)
    
    print(feature_importance.head(15).to_string(index=False))

# 10. Save model
model_file = 'dzud_risk_model_advanced.pkl'
joblib.dump(best_model, model_file)
print(f"\n✅ Model saved: {model_file}")

# 11. Save scaler
scaler_file = 'scaler_advanced.pkl'
joblib.dump(scaler, scaler_file)
print(f"✅ Scaler saved: {scaler_file}")

# 12. Save metadata
metadata = {
    'model_type': best_model_name,
    'features': feature_cols,
    'num_features': len(feature_cols),
    'accuracy': float(results[best_model_name]['accuracy']),
    'target_classes': target_names,
    'note': 'Advanced model using real livestock loss data and engineered features'
}

if hasattr(best_model, 'feature_importances_'):
    metadata['feature_importance'] = feature_importance.head(15).to_dict('records')

metadata_file = 'model_metadata_advanced.json'
with open(metadata_file, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"✅ Metadata saved: {metadata_file}")

# 13. Example prediction
if len(X_test) > 0:
    print("\n" + "="*60)
    print("7. Example Prediction:")
    print("="*60)
    
    sample_idx = 0
    sample = X_test.iloc[sample_idx:sample_idx+1]
    
    if 'Logistic' in best_model_name:
        sample_scaled = scaler.transform(sample)
        prediction = best_model.predict(sample_scaled)
        proba = best_model.predict_proba(sample_scaled)
    else:
        prediction = best_model.predict(sample)
        proba = best_model.predict_proba(sample)
    
    print(f"\nInput features (selected):")
    print(f"  min_temp: {sample['min_temp'].values[0]:.1f}°C")
    print(f"  wind_speed: {sample['wind_speed'].values[0]:.1f} m/s")
    print(f"  snowfall_sum: {sample['snowfall_sum'].values[0]:.1f} mm")
    print(f"  cold_index: {sample['cold_index'].values[0]:.1f}")
    print(f"  is_winter: {sample['is_winter'].values[0]}")
    
    print(f"\nPrediction: {target_names[int(prediction[0])]}")
    print(f"Probabilities: {proba[0]}")
    print(f"Actual: {target_names[int(y_test.iloc[sample_idx])]}")

print("\n" + "="*60)
print("Training complete!")
print("="*60)
