#!/usr/bin/env python3
"""
Dzud Risk Predictor - MVP + ML Hybrid
Хэрэглэгч координат + малын тоо оруулахад эрсдэл тооцоолно.
ML модель байвал rule-based + ML магадлалыг 50/50 нэгтгэн hybrid эрсдэл гаргана.
"""

import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, date
from calendar import monthrange
from typing import Dict, List, Tuple, Optional

# ML загварын feature-үүд (train_model_advanced.py-тай ижил дараалал)
ML_FEATURE_COLS = [
    'avg_temp', 'min_temp', 'wind_speed', 'snowfall_sum', 'precip_sum',
    'avg_temp_lag1', 'min_temp_lag1', 'wind_speed_lag1', 'snowfall_sum_lag1', 'precip_sum_lag1',
    'avg_temp_lag2', 'min_temp_lag2', 'wind_speed_lag2', 'snowfall_sum_lag2', 'precip_sum_lag2',
    'is_winter', 'cold_index', 'snow_cumulative', 'precip_deficit',
    'extreme_cold', 'extreme_wind', 'heavy_snow',
    'total_livestock', 'livestock_change_pct'
]


class DzudRiskPredictor:
    def __init__(self):
        """Initialize predictor with weather data and model"""
        # Load weather data
        self.weather_data = pd.read_csv('weather_omnogovi_monthly_clean.csv')
        
        # Load model (if exists)
        try:
            self.model = joblib.load('dzud_risk_model_advanced.pkl')
            self.scaler = joblib.load('scaler_advanced.pkl')
            self.has_model = True
        except Exception:
            self.model = None
            self.scaler = None
            self.has_model = False
            print("⚠️  Model not found, using rule-based system")
        
        # Livestock by year (for ML features: total_livestock, livestock_change_pct)
        self.livestock_by_year: Optional[Dict[int, Tuple[float, float]]] = self._load_livestock_by_year()
        
        # Livestock vulnerability weights (эмзэг байдал)
        self.livestock_weights = {
            'sheep': 1.2,    # хонь - эмзэг
            'goat': 1.3,     # ямаа - хамгийн эмзэг
            'cattle': 0.9,   # үхэр - тэсвэртэй
            'horse': 0.8,    # адуу - тэсвэртэй
            'camel': 0.6     # тэмээ - хамгийн тэсвэртэй
        }
    
    def _load_livestock_by_year(self) -> Optional[Dict[int, Tuple[float, float]]]:
        """Load yearly livestock for Ömnögovi: year -> (total_livestock, livestock_change_pct)"""
        try:
            livestock = pd.read_csv('livestock_omnogovi.csv')
            omnogovi = livestock[
                (livestock['Бүс'].str.strip() == 'Өмнөговь') |
                (livestock['Бүс'] == '               Өмнөговь')
            ]
            if 'Малын төрөл' in omnogovi.columns:
                omnogovi = omnogovi[omnogovi['Малын төрөл'] == 'Бүгд']
            omnogovi = omnogovi[['Он', 'Утга']].copy()
            omnogovi.columns = ['year', 'total_livestock']
            omnogovi = omnogovi.sort_values('year').drop_duplicates('year')
            omnogovi['livestock_change_pct'] = omnogovi['total_livestock'].pct_change() * 100
            out = {}
            for _, row in omnogovi.iterrows():
                out[int(row['year'])] = (float(row['total_livestock']), float(row['livestock_change_pct']) if pd.notna(row['livestock_change_pct']) else 0.0)
            return out
        except Exception:
            return None
    
    def find_nearest_location(self, lat: float, lon: float) -> Dict:
        """Find nearest weather station"""
        # Calculate distance to all locations
        self.weather_data['distance'] = np.sqrt(
            (self.weather_data['lat'] - lat)**2 + 
            (self.weather_data['lon'] - lon)**2
        )
        
        # Get nearest location
        nearest = self.weather_data.loc[self.weather_data['distance'].idxmin()]
        
        return {
            'aimag': nearest['aimag'],
            'soum': nearest['soum'],
            'lat': nearest['lat'],
            'lon': nearest['lon'],
            'distance_km': nearest['distance'] * 111  # degrees to km
        }
    
    def get_current_weather(self, lat: float, lon: float, month: int = None) -> Dict:
        """Get weather features for location and month"""
        if month is None:
            month = datetime.now().month
        
        # Find nearest location
        location = self.find_nearest_location(lat, lon)
        
        # Get weather for this location and month
        weather = self.weather_data[
            (self.weather_data['soum'] == location['soum']) &
            (self.weather_data['month'] == month)
        ].sort_values('year', ascending=False).iloc[0]
        
        return {
            'location': location,
            'month': month,
            'year': int(weather['year']),
            'avg_temp': float(weather['avg_temp']),
            'min_temp': float(weather['min_temp']),
            'wind_speed': float(weather['wind_speed']),
            'snowfall_sum': float(weather['snowfall_sum']),
            'precip_sum': float(weather['precip_sum'])
        }
    
    def calculate_weather_risk(self, weather: Dict) -> Tuple[float, List[str]]:
        """Calculate weather-based risk score (0-100)
        Зуд зөвхөн өвлийн сарууд (11, 12, 1, 2, 3) дээр тооцоологдоно
        """
        # Зун (4-10 сар) - зуд байхгүй
        if weather['month'] not in [11, 12, 1, 2, 3]:
            return 0, ["Зун - зудын эрсдэл байхгүй"]
        
        score = 0
        reasons = []
        
        # Temperature risk
        if weather['min_temp'] < -25:
            score += 35
            reasons.append(f"Хамгийн бага температур маш бага ({weather['min_temp']:.1f}°C)")
        elif weather['min_temp'] < -20:
            score += 25
            reasons.append(f"Хамгийн бага температур бага ({weather['min_temp']:.1f}°C)")
        elif weather['min_temp'] < -15:
            score += 15
            reasons.append(f"Температур доогуур ({weather['min_temp']:.1f}°C)")
        
        # Wind risk
        if weather['wind_speed'] > 18:
            score += 25
            reasons.append(f"Салхи маш хүчтэй ({weather['wind_speed']:.1f} м/с)")
        elif weather['wind_speed'] > 15:
            score += 15
            reasons.append(f"Салхи хүчтэй ({weather['wind_speed']:.1f} м/с)")
        elif weather['wind_speed'] > 12:
            score += 10
            reasons.append(f"Салхи дунд зэрэг ({weather['wind_speed']:.1f} м/с)")
        
        # Snowfall risk
        if weather['snowfall_sum'] > 10:
            score += 20
            reasons.append(f"Их цас орсон ({weather['snowfall_sum']:.1f} мм)")
        elif weather['snowfall_sum'] > 5:
            score += 10
            reasons.append(f"Цас орсон ({weather['snowfall_sum']:.1f} мм)")
        
        # Precipitation deficit (drought)
        if weather['precip_sum'] < 5:
            score += 15
            reasons.append(f"Хур тунадас маш бага ({weather['precip_sum']:.1f} мм)")
        elif weather['precip_sum'] < 10:
            score += 8
            reasons.append(f"Хур тунадас бага ({weather['precip_sum']:.1f} мм)")
        
        # Cold index (wind chill)
        cold_index = weather['min_temp'] - (weather['wind_speed'] * 0.5)
        if cold_index < -30:
            score += 15
            reasons.append(f"Хүйтний индекс өндөр ({cold_index:.1f})")
        
        return min(score, 100), reasons
    
    def calculate_livestock_exposure(self, livestock: Dict) -> Tuple[float, int]:
        """Calculate livestock exposure score (0-100)"""
        total_count = 0
        weighted_sum = 0
        
        for animal_type, count in livestock.items():
            if count > 0 and animal_type in self.livestock_weights:
                total_count += count
                weighted_sum += count * self.livestock_weights[animal_type]
        
        if total_count == 0:
            return 0, 0
        
        # Normalize to 0-100 scale
        # Assume 1000 animals = 50 points baseline
        exposure_score = min((weighted_sum / 1000) * 50, 100)
        
        return exposure_score, total_count
    
    def _score_to_level(self, score: float) -> Dict:
        """Convert 0-100 score to level, label, color"""
        if score < 25:
            level, label, color = 0, "Бага", "green"
        elif score < 50:
            level, label, color = 1, "Дунд", "yellow"
        elif score < 75:
            level, label, color = 2, "Өндөр", "orange"
        else:
            level, label, color = 3, "Маш өндөр", "red"
        return {'level': level, 'label': label, 'color': color}
    
    def calculate_final_risk(self, weather_risk: float, exposure_score: float) -> Dict:
        """Calculate final risk score and level (rule-based)"""
        final_score = (weather_risk * 0.7) + (exposure_score * 0.3)
        lev = self._score_to_level(final_score)
        return {
            'score': round(final_score, 1),
            'level': lev['level'],
            'label': lev['label'],
            'color': lev['color'],
            'weather_risk': round(weather_risk, 1),
            'exposure_score': round(exposure_score, 1)
        }
    
    def _build_ml_features(self, soum: str, year: int, month: int) -> Optional[pd.DataFrame]:
        """Build 24 ML features for (soum, year, month). Returns one row DataFrame or None."""
        df = self.weather_data[self.weather_data['soum'] == soum].sort_values(['year', 'month']).copy()
        if df.empty:
            return None
        row = df[(df['year'] == year) & (df['month'] == month)]
        if row.empty:
            return None
        row = row.iloc[0]
        # Previous month (lag1), two months back (lag2)
        def prev_month(y: int, m: int, k: int):
            for _ in range(k):
                if m <= 1:
                    y, m = y - 1, 12
                else:
                    m -= 1
            return y, m
        y1, m1 = prev_month(year, month, 1)
        y2, m2 = prev_month(year, month, 2)
        row1 = df[(df['year'] == y1) & (df['month'] == m1)]
        row2 = df[(df['year'] == y2) & (df['month'] == m2)]
        if row1.empty or row2.empty:
            return None
        row1, row2 = row1.iloc[0], row2.iloc[0]
        # Snow cumulative for this year up to this month
        year_snow = df[(df['year'] == year) & (df['month'] <= month)]['snowfall_sum'].sum()
        # Livestock for this year
        if not self.livestock_by_year or year not in self.livestock_by_year:
            return None
        total_livestock, livestock_change_pct = self.livestock_by_year[year]
        cold_index = float(row['min_temp']) - (float(row['wind_speed']) * 0.5)
        is_winter = 1 if month in (11, 12, 1, 2, 3) else 0
        precip_deficit = 20.0 - float(row['precip_sum'])
        data = {
            'avg_temp': float(row['avg_temp']),
            'min_temp': float(row['min_temp']),
            'wind_speed': float(row['wind_speed']),
            'snowfall_sum': float(row['snowfall_sum']),
            'precip_sum': float(row['precip_sum']),
            'avg_temp_lag1': float(row1['avg_temp']),
            'min_temp_lag1': float(row1['min_temp']),
            'wind_speed_lag1': float(row1['wind_speed']),
            'snowfall_sum_lag1': float(row1['snowfall_sum']),
            'precip_sum_lag1': float(row1['precip_sum']),
            'avg_temp_lag2': float(row2['avg_temp']),
            'min_temp_lag2': float(row2['min_temp']),
            'wind_speed_lag2': float(row2['wind_speed']),
            'snowfall_sum_lag2': float(row2['snowfall_sum']),
            'precip_sum_lag2': float(row2['precip_sum']),
            'is_winter': is_winter,
            'cold_index': cold_index,
            'snow_cumulative': year_snow,
            'precip_deficit': precip_deficit,
            'extreme_cold': 1 if float(row['min_temp']) < -25 else 0,
            'extreme_wind': 1 if float(row['wind_speed']) > 18 else 0,
            'heavy_snow': 1 if float(row['snowfall_sum']) > 10 else 0,
            'total_livestock': total_livestock,
            'livestock_change_pct': livestock_change_pct
        }
        return pd.DataFrame([data])[ML_FEATURE_COLS]
    
    def _predict_ml_probability(self, soum: str, year: int, month: int) -> Optional[float]:
        """Return P(dzud=1) from ML model, or None if not available."""
        if not self.has_model or self.model is None or self.scaler is None:
            return None
        X = self._build_ml_features(soum, year, month)
        if X is None or X.empty:
            return None
        try:
            model_name = type(self.model).__name__
            if 'Logistic' in model_name:
                X_scaled = self.scaler.transform(X)
                proba = self.model.predict_proba(X_scaled)[0, 1]
            else:
                proba = self.model.predict_proba(X)[0, 1]
            return float(proba)
        except Exception:
            return None
    
    def get_recommendations(self, risk_level: int, livestock: Dict, weather: Dict) -> Dict:
        """Generate action recommendations by livestock type"""
        recommendations = {}
        
        # Sheep and Goats (хонь, ямаа)
        if livestock.get('sheep', 0) > 0 or livestock.get('goat', 0) > 0:
            if risk_level >= 2:  # High risk
                recommendations['sheep_goat'] = [
                    "🏠 Салхи, хүйтнээс хамгаалах байр бэлтгэх",
                    "🌾 Нэмэлт тэжээл нөөцлөх (өвс, тэжээл)",
                    "💧 Усны хангамж шалгах",
                    "👥 Сүрэг бүлэглэн хамгаалах"
                ]
            else:
                recommendations['sheep_goat'] = [
                    "✓ Өвөлжилтийн бэлтгэл хангалттай эсэх шалгах",
                    "✓ Тэжээлийн нөөц хангалттай байх"
                ]
        
        # Cattle (үхэр)
        if livestock.get('cattle', 0) > 0:
            if risk_level >= 2:
                recommendations['cattle'] = [
                    "🏠 Хашаа, байр бэлтгэх",
                    "💧 Ус, тэжээлийн нөөц нэмэгдүүлэх",
                    "🌡️ Дулаан хадгалах арга хэмжээ"
                ]
            else:
                recommendations['cattle'] = [
                    "✓ Хэвийн өвөлжилтийн бэлтгэл"
                ]
        
        # Horses (адуу)
        if livestock.get('horse', 0) > 0:
            if risk_level >= 2:
                recommendations['horse'] = [
                    "🏃 Нүүх боломжтой газар бэлтгэх",
                    "🌾 Тэжээлийн нөөц",
                    "💧 Усны эх үүсвэр"
                ]
            else:
                recommendations['horse'] = [
                    "✓ Хэвийн өвөлжилт"
                ]
        
        # Camels (тэмээ)
        if livestock.get('camel', 0) > 0:
            recommendations['camel'] = [
                "✓ Тэмээ хамгийн тэсвэртэй",
                "✓ Ердийн арчилгаа хангалттай"
            ]
        
        # General recommendations
        general = []
        if risk_level >= 3:
            general.append("🚨 АНХААРУУЛГА: Маш өндөр эрсдэл!")
            general.append("📍 Эрсдэл багатай газар руу нүүх боломжийг судлах")
        if risk_level >= 2:
            general.append("⚠️  Цаг агаарын мэдээг тогтмол хянах")
            general.append("📞 Орон нутгийн мал эмнэлэгтэй холбоо барих")
        
        recommendations['general'] = general
        
        return recommendations
    
    def predict(self, lat: float, lon: float, livestock: Dict, month: int = None) -> Dict:
        """Main prediction function. Uses hybrid (rule-based + ML) when model is available."""
        weather = self.get_current_weather(lat, lon, month)
        weather_risk, weather_reasons = self.calculate_weather_risk(weather)
        exposure_score, total_livestock = self.calculate_livestock_exposure(livestock)
        risk = self.calculate_final_risk(weather_risk, exposure_score)
        
        # Hybrid: combine rule-based score with ML probability (50/50) when available
        ml_prob = self._predict_ml_probability(
            weather['location']['soum'],
            weather['year'],
            weather['month']
        )
        if ml_prob is not None:
            rule_score = risk['score']
            ml_score = ml_prob * 100
            hybrid_score = 0.5 * rule_score + 0.5 * ml_score
            lev = self._score_to_level(hybrid_score)
            risk = {
                'score': round(hybrid_score, 1),
                'level': lev['level'],
                'label': lev['label'],
                'color': lev['color'],
                'weather_risk': risk['weather_risk'],
                'exposure_score': risk['exposure_score'],
                'ml_probability': round(ml_prob, 4),
                'risk_source': 'hybrid'
            }
        else:
            risk['ml_probability'] = None
            risk['risk_source'] = 'rule_based'
        
        recommendations = self.get_recommendations(risk['level'], livestock, weather)
        result = {
            'location': weather['location'],
            'weather': weather,
            'risk': risk,
            'livestock': {
                'total': total_livestock,
                'breakdown': livestock,
                'exposure_score': exposure_score
            },
            'top_reasons': weather_reasons[:3],
            'recommendations': recommendations,
            'confidence': 'өндөр' if risk.get('risk_source') == 'hybrid' else ('дунд' if self.has_model else 'бага'),
            'note': 'Энэ нь туршилтын тооцоолол юм. Бодит мэдээлэл дээр үндэслэнэ үү.'
        }
        return result

    # ------------------------------------------------------------------
    # БОДИТ ЦАГ АГААРЫН ӨГӨГДӨЛ — Open-Meteo API
    # ------------------------------------------------------------------

    def fetch_real_weather(self, lat: float, lon: float, year: int, month: int) -> Optional[Dict]:
        """
        Open-Meteo archive API-аас тухайн сарын бодит цаг агаарын өгөгдөл татна.
        Ирээдүйн сар бол forecast API ашиглана.
        Буцаах: {'avg_temp', 'min_temp', 'wind_speed', 'snowfall_sum', 'precip_sum'} эсвэл None
        """
        today = date.today()
        is_future = (year > today.year) or (year == today.year and month > today.month)
        is_current = (year == today.year and month == today.month)

        try:
            if is_future or is_current:
                # Forecast API — 16 хоног хүртэл
                url = "https://api.open-meteo.com/v1/forecast"
                # Сарын эхний болон сүүлийн өдрийг тооцно
                _, days_in_month = monthrange(year, month)
                start = date(year, month, 1)
                end = date(year, month, days_in_month)
                # Forecast API зөвхөн 16 хоног хүртэл — хэтэрсэн бол ERA5 ашиглана
                days_ahead = (start - today).days
                if days_ahead > 15:
                    # ERA5 archive-д байхгүй ирээдүйн сар — түүхэн дундажаар fallback
                    return None
                params = {
                    'latitude': lat, 'longitude': lon,
                    'daily': ['temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
                              'precipitation_sum', 'snowfall_sum', 'windspeed_10m_max'],
                    'timezone': 'Asia/Ulaanbaatar',
                    'start_date': str(start), 'end_date': str(end),
                }
            else:
                # Archive API — түүхэн бодит өгөгдөл
                url = "https://archive-api.open-meteo.com/v1/archive"
                _, days_in_month = monthrange(year, month)
                params = {
                    'latitude': lat, 'longitude': lon,
                    'daily': ['temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
                              'precipitation_sum', 'snowfall_sum', 'windspeed_10m_max'],
                    'timezone': 'Asia/Ulaanbaatar',
                    'start_date': f"{year}-{month:02d}-01",
                    'end_date': f"{year}-{month:02d}-{days_in_month:02d}",
                }

            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            daily = resp.json().get('daily', {})

            temps_min = [v for v in daily.get('temperature_2m_min', []) if v is not None]
            temps_mean = [v for v in daily.get('temperature_2m_mean', []) if v is not None]
            winds = [v for v in daily.get('windspeed_10m_max', []) if v is not None]
            snow = [v for v in daily.get('snowfall_sum', []) if v is not None]
            precip = [v for v in daily.get('precipitation_sum', []) if v is not None]

            if not temps_min:
                return None

            return {
                'avg_temp': round(float(np.mean(temps_mean)) if temps_mean else float(np.mean(temps_min)), 2),
                'min_temp': round(float(np.min(temps_min)), 2),
                'wind_speed': round(float(np.mean(winds)) if winds else 0.0, 2),
                'snowfall_sum': round(float(np.sum(snow)) if snow else 0.0, 2),
                'precip_sum': round(float(np.sum(precip)) if precip else 0.0, 2),
                'data_source': 'forecast' if (is_future or is_current) else 'archive',
            }
        except Exception as e:
            print(f"⚠️  Open-Meteo fetch алдаа ({year}-{month:02d}): {e}")
            return None

    def predict_forecast_months(self, lat: float, lon: float, livestock: Dict, months: int = 3) -> List[Dict]:
        """
        Одоогийн болон ирээдүйн N сарын эрсдэлийг бодит/forecast цаг агаарын өгөгдлөөр тооцоолно.
        Бодит өгөгдөл татаж чадахгүй бол түүхэн дундажаар fallback хийнэ.

        Буцаах: list of {month, year, month_name, risk_score, risk_level, risk_label,
                          weather, data_source, reasons}
        """
        location = self.find_nearest_location(lat, lon)
        exposure_score, total_livestock_count = self.calculate_livestock_exposure(livestock)

        today = date.today()
        results = []

        MONTH_NAMES = ['', 'Нэгдүгээр', 'Хоёрдугаар', 'Гуравдугаар', 'Дөрөвдүгээр',
                       'Тавдугаар', 'Зургадугаар', 'Долдугаар', 'Наймдугаар',
                       'Есдүгээр', 'Аравдугаар', 'Арван нэгдүгээр', 'Арван хоёрдугаар']

        for i in range(months):
            # i=0 → одоогийн сар, i=1,2,3 → ирээдүйн сарууд
            target_month = ((today.month - 1 + i) % 12) + 1
            target_year = today.year + ((today.month - 1 + i) // 12)

            # 1. Бодит/forecast өгөгдөл татах
            real = self.fetch_real_weather(lat, lon, target_year, target_month)

            if real:
                weather_dict = {
                    'location': location,
                    'month': target_month,
                    'year': target_year,
                    'avg_temp': real['avg_temp'],
                    'min_temp': real['min_temp'],
                    'wind_speed': real['wind_speed'],
                    'snowfall_sum': real['snowfall_sum'],
                    'precip_sum': real['precip_sum'],
                }
                data_source = real['data_source']
            else:
                # Fallback: түүхэн дундаж
                weather_dict = self.get_current_weather(lat, lon, month=target_month)
                weather_dict['year'] = target_year
                data_source = 'historical_avg'

            # 2. Эрсдэл тооцоолох
            weather_risk, reasons = self.calculate_weather_risk(weather_dict)
            risk = self.calculate_final_risk(weather_risk, exposure_score)

            # 3. ML hybrid (түүхэн жилийн өгөгдлөөр — ирээдүйд ML feature байхгүй тул skip)
            risk['ml_probability'] = None
            risk['risk_source'] = 'rule_based_realtime'

            results.append({
                'month': target_month,
                'year': target_year,
                'month_name': MONTH_NAMES[target_month],
                'is_current': i == 0,
                'risk_score': risk['score'],
                'risk_level': risk['level'],
                'risk_label': risk['label'],
                'risk_color': risk['color'],
                'weather': {
                    'avg_temp': weather_dict['avg_temp'],
                    'min_temp': weather_dict['min_temp'],
                    'wind_speed': weather_dict['wind_speed'],
                    'snowfall_sum': weather_dict['snowfall_sum'],
                    'precip_sum': weather_dict['precip_sum'],
                },
                'data_source': data_source,
                'reasons': reasons[:3],
            })

        return results


# Example usage
if __name__ == "__main__":
    predictor = DzudRiskPredictor()
    
    # Test case
    result = predictor.predict(
        lat=43.5,
        lon=104.4,
        livestock={
            'sheep': 200,
            'goat': 150,
            'cattle': 50,
            'horse': 30,
            'camel': 10
        },
        month=1  # January
    )
    
    print("="*60)
    print("ЗУДЫН ЭРСДЭЛИЙН ҮНЭЛГЭЭ")
    print("="*60)
    print(f"\n📍 Байршил: {result['location']['soum']}, {result['location']['aimag']}")
    print(f"   Координат: {result['location']['lat']:.2f}, {result['location']['lon']:.2f}")
    print(f"\n🌡️  Цаг агаар ({result['weather']['month']}-р сар, {result['weather']['year']}):")
    print(f"   Дундаж температур: {result['weather']['avg_temp']:.1f}°C")
    print(f"   Хамгийн бага температур: {result['weather']['min_temp']:.1f}°C")
    print(f"   Салхины хурд: {result['weather']['wind_speed']:.1f} м/с")
    print(f"   Цас: {result['weather']['snowfall_sum']:.1f} мм")
    print(f"   Хур тунадас: {result['weather']['precip_sum']:.1f} мм")
    print(f"\n🎯 ЭРСДЭЛИЙН ДҮН: {result['risk']['score']}/100")
    print(f"   Түвшин: {result['risk']['label']} ({result['risk']['color']})")
    print(f"   Итгэлцүүр: {result['confidence']}")
    print(f"\n📊 Дэлгэрэнгүй:")
    print(f"   Цаг агаарын эрсдэл: {result['risk']['weather_risk']}/100")
    print(f"   Малын өртөлт: {result['risk']['exposure_score']}/100")
    print(f"\n🐑 Малын тоо: {result['livestock']['total']} толгой")
    for animal, count in result['livestock']['breakdown'].items():
        if count > 0:
            print(f"   {animal}: {count}")
    print(f"\n⚠️  Гол шалтгаанууд:")
    for i, reason in enumerate(result['top_reasons'], 1):
        print(f"   {i}. {reason}")
    print(f"\n💡 Зөвлөмж:")
    for category, recs in result['recommendations'].items():
        if recs:
            print(f"\n   {category.upper()}:")
            for rec in recs:
                print(f"      {rec}")
    print(f"\n📝 {result['note']}")
    print("="*60)
