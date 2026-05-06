#!/usr/bin/env python3
"""
Модель статистикийг terminal дээр харуулах
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

# ANSI өнгө кодууд
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_header(text):
    """Гарчиг хэвлэх"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(70)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}\n")

def print_section(text):
    """Хэсэг гарчиг"""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}{text}{Colors.END}")
    print(f"{Colors.YELLOW}{'-'*70}{Colors.END}")

def print_metric(name, value, color=Colors.GREEN):
    """Метрик хэвлэх"""
    print(f"{Colors.BOLD}{name:.<50}{color}{value:>15}{Colors.END}")

def print_table_header(headers):
    """Хүснэгтийн толгой"""
    header_line = " | ".join([f"{h:^15}" for h in headers])
    print(f"{Colors.BOLD}{Colors.BLUE}{header_line}{Colors.END}")
    print(f"{Colors.BLUE}{'-'*70}{Colors.END}")

def print_table_row(values, color=Colors.END):
    """Хүснэгтийн мөр"""
    row_line = " | ".join([f"{str(v):^15}" for v in values])
    print(f"{color}{row_line}{Colors.END}")

def load_model_and_data():
    """Модель болон өгөгдөл ачаалах"""
    print_section("📂 Өгөгдөл ачаалж байна...")
    
    try:
        # Dataset
        df = pd.read_csv('dzud_ai_dataset_advanced.csv')
        print(f"✅ Dataset: {len(df)} мөр")
        
        # Model
        model = joblib.load('dzud_risk_model_advanced.pkl')
        print(f"✅ Model: {type(model).__name__}")
        
        # Scaler
        scaler = joblib.load('scaler_advanced.pkl')
        print(f"✅ Scaler: {type(scaler).__name__}")
        
        # Metadata
        with open('model_metadata_advanced.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"✅ Metadata: {len(metadata)} keys")
        
        return df, model, scaler, metadata
    except Exception as e:
        print(f"{Colors.RED}❌ Алдаа: {e}{Colors.END}")
        return None, None, None, None

def show_dataset_info(df):
    """Dataset мэдээлэл"""
    print_section("📊 DATASET МЭДЭЭЛЭЛ")
    
    print_metric("Нийт мөр", f"{len(df):,}")
    print_metric("Нийт багана", f"{len(df.columns):,}")
    print_metric("Огноо", f"{df['year'].min()}-{df['year'].max()}")
    print_metric("Сумдын тоо", f"{df['soum'].nunique():,}")
    
    print("\n📈 Эрсдэлийн түвшин:")
    target_dist = df['target_dzud'].value_counts().sort_index()
    for val, count in target_dist.items():
        label = "Зуд байхгүй" if val == 0 else "Зуд"
        pct = (count / len(df)) * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:.<20} {count:>5} ({pct:>5.1f}%) {Colors.GREEN}{bar}{Colors.END}")

def show_training_log(df, model, scaler):
    """Сургалтын log"""
    print_section("🎓 МОДЕЛЬ СУРГАЛТЫН LOG")
    
    # Feature columns
    feature_cols = [
        'avg_temp', 'min_temp', 'wind_speed', 'snowfall_sum', 'precip_sum',
        'avg_temp_lag1', 'min_temp_lag1', 'wind_speed_lag1', 'snowfall_sum_lag1', 'precip_sum_lag1',
        'avg_temp_lag2', 'min_temp_lag2', 'wind_speed_lag2', 'snowfall_sum_lag2', 'precip_sum_lag2',
        'is_winter', 'cold_index', 'snow_cumulative', 'precip_deficit',
        'extreme_cold', 'extreme_wind', 'heavy_snow',
        'total_livestock', 'livestock_change_pct'
    ]
    
    X = df[feature_cols].fillna(0)
    y = df['target_dzud']
    
    # Train/Test split
    train_mask = df['year'] < 2023
    test_mask = df['year'] >= 2023
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    print(f"📦 Features: {len(feature_cols)}")
    print(f"🔢 Train size: {len(X_train):,} ({len(X_train)/len(df)*100:.1f}%)")
    print(f"🔢 Test size: {len(X_test):,} ({len(X_test)/len(df)*100:.1f}%)")
    print(f"📅 Train years: {df[train_mask]['year'].min()}-{df[train_mask]['year'].max()}")
    print(f"📅 Test years: {df[test_mask]['year'].min()}-{df[test_mask]['year'].max()}")
    
    # Model parameters
    print(f"\n🌲 Random Forest Parameters:")
    print(f"  n_estimators: {model.n_estimators}")
    print(f"  max_depth: {model.max_depth}")
    print(f"  min_samples_split: {model.min_samples_split}")
    print(f"  min_samples_leaf: {model.min_samples_leaf}")

def show_accuracy(df, model, scaler):
    """Accuracy харуулах"""
    print_section("🎯 ACCURACY")
    
    feature_cols = [
        'avg_temp', 'min_temp', 'wind_speed', 'snowfall_sum', 'precip_sum',
        'avg_temp_lag1', 'min_temp_lag1', 'wind_speed_lag1', 'snowfall_sum_lag1', 'precip_sum_lag1',
        'avg_temp_lag2', 'min_temp_lag2', 'wind_speed_lag2', 'snowfall_sum_lag2', 'precip_sum_lag2',
        'is_winter', 'cold_index', 'snow_cumulative', 'precip_deficit',
        'extreme_cold', 'extreme_wind', 'heavy_snow',
        'total_livestock', 'livestock_change_pct'
    ]
    
    X = df[feature_cols].fillna(0)
    y = df['target_dzud']
    
    train_mask = df['year'] < 2023
    test_mask = df['year'] >= 2023
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Accuracy
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print_metric("Train Accuracy", f"{train_acc:.4f} ({train_acc*100:.2f}%)", Colors.GREEN)
    print_metric("Test Accuracy", f"{test_acc:.4f} ({test_acc*100:.2f}%)", Colors.GREEN)
    
    # Cross-validation
    print(f"\n🔄 5-Fold Cross-Validation:")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    for i, score in enumerate(cv_scores, 1):
        bar = "█" * int(score * 50)
        print(f"  Fold {i}: {score:.4f} {Colors.CYAN}{bar}{Colors.END}")
    print_metric("CV Mean", f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}", Colors.YELLOW)

def show_precision_recall_f1(df, model):
    """Precision, Recall, F1 харуулах"""
    print_section("📊 PRECISION / RECALL / F1-SCORE")
    
    feature_cols = [
        'avg_temp', 'min_temp', 'wind_speed', 'snowfall_sum', 'precip_sum',
        'avg_temp_lag1', 'min_temp_lag1', 'wind_speed_lag1', 'snowfall_sum_lag1', 'precip_sum_lag1',
        'avg_temp_lag2', 'min_temp_lag2', 'wind_speed_lag2', 'snowfall_sum_lag2', 'precip_sum_lag2',
        'is_winter', 'cold_index', 'snow_cumulative', 'precip_deficit',
        'extreme_cold', 'extreme_wind', 'heavy_snow',
        'total_livestock', 'livestock_change_pct'
    ]
    
    X = df[feature_cols].fillna(0)
    y = df['target_dzud']
    
    test_mask = df['year'] >= 2023
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    y_pred = model.predict(X_test)
    
    # Metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
    
    # Table
    print_table_header(['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
    
    class_names = ['Зуд байхгүй', 'Зуд']
    for i, name in enumerate(class_names):
        color = Colors.GREEN if f1[i] > 0.85 else Colors.YELLOW if f1[i] > 0.75 else Colors.RED
        print_table_row([
            name,
            f"{precision[i]:.4f}",
            f"{recall[i]:.4f}",
            f"{f1[i]:.4f}",
            f"{support[i]}"
        ], color)
    
    # Weighted average
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    print(f"\n{Colors.BOLD}Weighted Average:{Colors.END}")
    print_metric("Precision", f"{precision_avg:.4f}", Colors.CYAN)
    print_metric("Recall", f"{recall_avg:.4f}", Colors.CYAN)
    print_metric("F1-Score", f"{f1_avg:.4f}", Colors.CYAN)

def show_confusion_matrix(df, model):
    """Confusion Matrix"""
    print_section("🔢 CONFUSION MATRIX")
    
    feature_cols = [
        'avg_temp', 'min_temp', 'wind_speed', 'snowfall_sum', 'precip_sum',
        'avg_temp_lag1', 'min_temp_lag1', 'wind_speed_lag1', 'snowfall_sum_lag1', 'precip_sum_lag1',
        'avg_temp_lag2', 'min_temp_lag2', 'wind_speed_lag2', 'snowfall_sum_lag2', 'precip_sum_lag2',
        'is_winter', 'cold_index', 'snow_cumulative', 'precip_deficit',
        'extreme_cold', 'extreme_wind', 'heavy_snow',
        'total_livestock', 'livestock_change_pct'
    ]
    
    X = df[feature_cols].fillna(0)
    y = df['target_dzud']
    
    test_mask = df['year'] >= 2023
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n                 Predicted")
    print("              Зуд байхгүй    Зуд")
    print(f"Actual  Зуд байхгүй    {Colors.GREEN}{cm[0,0]:^8}{Colors.END}    {Colors.RED}{cm[0,1]:^8}{Colors.END}")
    print(f"        Зуд            {Colors.RED}{cm[1,0]:^8}{Colors.END}    {Colors.GREEN}{cm[1,1]:^8}{Colors.END}")
    
    # Metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    print(f"\n📊 Дэлгэрэнгүй:")
    print_metric("True Negatives (TN)", f"{tn}", Colors.GREEN)
    print_metric("False Positives (FP)", f"{fp}", Colors.RED)
    print_metric("False Negatives (FN)", f"{fn}", Colors.RED)
    print_metric("True Positives (TP)", f"{tp}", Colors.GREEN)

def show_feature_importance(model, metadata):
    """Feature Importance"""
    print_section("⭐ FEATURE IMPORTANCE")
    
    feature_importance = pd.DataFrame(metadata['feature_importance'])
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    print_table_header(['Rank', 'Feature', 'Importance', 'Bar'])
    
    for i, row in enumerate(feature_importance.head(15).itertuples(), 1):
        importance = row.importance
        bar_length = int(importance * 100)
        bar = "█" * bar_length
        
        color = Colors.RED if i <= 3 else Colors.YELLOW if i <= 7 else Colors.GREEN
        
        print(f"{i:^5} | {row.feature:<25} | {importance:>10.4f} | {color}{bar}{Colors.END}")
    
    print(f"\n{Colors.BOLD}Нийт features: {len(feature_importance)}{Colors.END}")

def main():
    """Үндсэн функц"""
    print_header("🤖 ЗУД AI - МОДЕЛЬ СТАТИСТИК")
    
    # Load
    df, model, scaler, metadata = load_model_and_data()
    
    if df is None:
        return
    
    # Show stats
    show_dataset_info(df)
    show_training_log(df, model, scaler)
    show_accuracy(df, model, scaler)
    show_precision_recall_f1(df, model)
    show_confusion_matrix(df, model)
    show_feature_importance(model, metadata)
    
    print_header("✅ ДУУССАН")
    print(f"\n{Colors.CYAN}💡 Модель: dzud_risk_model_advanced.pkl{Colors.END}")
    print(f"{Colors.CYAN}💡 Scaler: scaler_advanced.pkl{Colors.END}")
    print(f"{Colors.CYAN}💡 Metadata: model_metadata_advanced.json{Colors.END}\n")

if __name__ == "__main__":
    main()
