#!/usr/bin/env python3
"""
Daily Dzud Risk Forecast - 7-30 хоногийн таамаглал
Open-Meteo API ашиглана (үнэгүй, бүртгэлгүй)
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib

class DailyDzudForecast:
    def __init__(self):
        """Initialize with trained model"""
        try:
            self.model = joblib.load('dzud_risk_model_monthly.pkl')
            self.scaler = joblib.load('scaler_monthly.pkl')
            self.has_model = True
        except:
            self.has_model = False
            print("⚠️  Model not found")
        
        self.api_base = "https://api.open-meteo.com/v1/forecast"
        
    def get_weather_forecast(self, lat: float, lon: float, days: int = 14):
        """
        Get weather forecast from Open-Meteo API
        Free, no API key needed
        """
        params = {
            'latitude': lat,
            'longitude': lon,
            'daily': [
                'temperature_2m_max',
                'temperature_2m_min',
                'temperature_2m_mean',
                'precipitation_sum',
                'snowfall_sum',
                'windspeed_10m_max'
            ],
            'timezone': 'Asia/Ulaanbaatar',
            'forecast_days': min(days, 16)  # API limit
        }
        
        try:
            response = requests.get(self.api_base, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            daily = data['daily']
            df = pd.DataFrame({
                'date': pd.to_datetime(daily['time']),
                'temp_max': daily['temperature_2m_max'],
                'temp_min': daily['temperature_2m_min'],
                'temp_mean': daily['temperature_2m_mean'],
                'precip': daily['precipitation_sum'],
                'snowfall': daily['snowfall_sum'],
                'wind_max': daily['windspeed_10m_max']
            })
            
            return df
            
        except Exception as e:
            print(f"❌ Weather API error: {e}")
            return None
    
    def calculate_daily_risk(self, row, livestock_count: float):
        """
        Calculate dzud risk for a single day
        """
        # Check if winter month
        month = row['date'].month
        if month not in [11, 12, 1, 2, 3]:
            return 0, "Зун - зудын эрсдэл байхгүй"
        
        score = 0
        reasons = []
        
        # Temperature risk
        if row['temp_min'] < -25:
            score += 35
            reasons.append(f"Маш хүйтэн ({row['temp_min']:.1f}°C)")
        elif row['temp_min'] < -20:
            score += 25
            reasons.append(f"Хүйтэн ({row['temp_min']:.1f}°C)")
        elif row['temp_min'] < -15:
            score += 15
            reasons.append(f"Сэрүүн ({row['temp_min']:.1f}°C)")
        
        # Wind risk
        if row['wind_max'] > 18:
            score += 25
            reasons.append(f"Хүчтэй салхи ({row['wind_max']:.1f} м/с)")
        elif row['wind_max'] > 15:
            score += 15
            reasons.append(f"Салхитай ({row['wind_max']:.1f} м/с)")
        
        # Snowfall risk
        if row['snowfall'] > 10:
            score += 20
            reasons.append(f"Их цас ({row['snowfall']:.1f} мм)")
        elif row['snowfall'] > 5:
            score += 10
            reasons.append(f"Цастай ({row['snowfall']:.1f} мм)")
        
        # Precipitation deficit
        if row['precip'] < 2:
            score += 15
            reasons.append("Хуурай")
        
        # Livestock exposure
        if livestock_count > 100:
            score += 15
        elif livestock_count > 50:
            score += 10
        
        # Wind chill effect
        wind_chill = row['temp_min'] - (row['wind_max'] * 0.5)
        if wind_chill < -30:
            score += 10
            reasons.append(f"Хүйтний индекс өндөр ({wind_chill:.1f})")
        
        return min(score, 100), reasons
    
    def get_risk_level(self, score):
        """Convert score to risk level"""
        if score < 25:
            return 0, "Бага", "green"
        elif score < 50:
            return 1, "Дунд", "yellow"
        elif score < 75:
            return 2, "Өндөр", "orange"
        else:
            return 3, "Маш өндөр", "red"
    
    def forecast(self, lat: float, lon: float, livestock_count: float, days: int = 14, verbose: bool = True):
        """
        Main forecast function
        Returns daily risk forecast for next N days
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"ЗУДЫН ЭРСДЭЛИЙН ТААМАГЛАЛ")
            print(f"{'='*60}")
            print(f"📍 Байршил: {lat:.2f}°N, {lon:.2f}°E")
            print(f"� Маалын тоо: {livestock_count:.0f}")
            print(f"📅 Хугацаа: {days} хоног")
        
        # Get weather forecast
        if verbose:
            print(f"\n🌤️  Цаг агаарын мэдээ татаж байна...")
        weather_df = self.get_weather_forecast(lat, lon, days)
        
        if weather_df is None:
            return None
        
        if verbose:
            print(f"✅ {len(weather_df)} өдрийн мэдээ авлаа")
        
        # Calculate risk for each day
        results = []
        for idx, row in weather_df.iterrows():
            score, reasons = self.calculate_daily_risk(row, livestock_count)
            level, label, color = self.get_risk_level(score)
            
            results.append({
                'date': row['date'],
                'day_name': row['date'].strftime('%A'),
                'temp_min': row['temp_min'],
                'temp_max': row['temp_max'],
                'temp_mean': row['temp_mean'],
                'wind_max': row['wind_max'],
                'snowfall': row['snowfall'],
                'precip': row['precip'],
                'risk_score': score,
                'risk_level': level,
                'risk_label': label,
                'risk_color': color,
                'reasons': reasons[:3]  # Top 3
            })
        
        results_df = pd.DataFrame(results)
        
        # Summary statistics
        if verbose:
            print(f"\n{'='*60}")
            print(f"ТААМАГЛАЛЫН ДҮГНЭЛТ")
            print(f"{'='*60}")
        
        # Risk distribution
        if verbose:
            print(f"\nЭрсдэлийн түвшин:")
            for label in ['Бага', 'Дунд', 'Өндөр', 'Маш өндөр']:
                count = len(results_df[results_df['risk_label'] == label])
                if count > 0:
                    pct = count / len(results_df) * 100
                    print(f"  {label}: {count} өдөр ({pct:.1f}%)")
        
        # High risk days
        high_risk = results_df[results_df['risk_level'] >= 2]
        if verbose and len(high_risk) > 0:
            print(f"\n⚠️  АНХААРУУЛГА: {len(high_risk)} өдөр өндөр эрсдэлтэй!")
            print(f"\nАюултай өдрүүд:")
            for _, day in high_risk.iterrows():
                print(f"  📅 {day['date'].strftime('%Y-%m-%d (%a)')}: {day['risk_label']} ({day['risk_score']:.0f}/100)")
                if day['reasons']:
                    print(f"     Шалтгаан: {', '.join(day['reasons'])}")
        elif verbose:
            print(f"\n✅ Өндөр эрсдэлтэй өдөр байхгүй")
        
        # Temperature extremes
        if verbose:
            coldest = results_df.loc[results_df['temp_min'].idxmin()]
            print(f"\n🥶 Хамгийн хүйтэн өдөр:")
            print(f"   {coldest['date'].strftime('%Y-%m-%d')}: {coldest['temp_min']:.1f}°C")
        
        # Windiest
        if verbose:
            windiest = results_df.loc[results_df['wind_max'].idxmax()]
            print(f"\n💨 Хамгийн салхитай өдөр:")
            print(f"   {windiest['date'].strftime('%Y-%m-%d')}: {windiest['wind_max']:.1f} м/с")
        
        # Snowiest
        if verbose and results_df['snowfall'].max() > 0:
            snowiest = results_df.loc[results_df['snowfall'].idxmax()]
            print(f"\n❄️  Хамгийн их цастай өдөр:")
            print(f"   {snowiest['date'].strftime('%Y-%m-%d')}: {snowiest['snowfall']:.1f} мм")
        
        if verbose:
            print(f"\n{'='*60}")
        
        return results_df
    
    def print_daily_forecast(self, results_df):
        """Print detailed daily forecast"""
        print(f"\n{'='*60}")
        print(f"ӨДӨР БҮРИЙН ДЭЛГЭРЭНГҮЙ")
        print(f"{'='*60}\n")
        
        for _, day in results_df.iterrows():
            icon = "🟢" if day['risk_level'] == 0 else "🟡" if day['risk_level'] == 1 else "🟠" if day['risk_level'] == 2 else "🔴"
            
            print(f"{icon} {day['date'].strftime('%Y-%m-%d (%A)')}")
            print(f"   Температур: {day['temp_min']:.1f}°C ~ {day['temp_max']:.1f}°C")
            print(f"   Салхи: {day['wind_max']:.1f} м/с")
            if day['snowfall'] > 0:
                print(f"   Цас: {day['snowfall']:.1f} мм")
            if day['precip'] > 0:
                print(f"   Хур тунадас: {day['precip']:.1f} мм")
            print(f"   Эрсдэл: {day['risk_label']} ({day['risk_score']:.0f}/100)")
            if day['reasons']:
                print(f"   Шалтгаан: {', '.join(day['reasons'])}")
            print()


# Example usage
if __name__ == "__main__":
    forecaster = DailyDzudForecast()
    
    # Dalanzadgad coordinates
    lat = 43.57
    lon = 104.43
    livestock = 200
    
    # Get 14-day forecast
    results = forecaster.forecast(lat, lon, livestock, days=14)
    
    if results is not None:
        forecaster.print_daily_forecast(results)
