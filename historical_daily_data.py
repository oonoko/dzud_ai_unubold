#!/usr/bin/env python3
"""
Historical daily dzud risk data viewer
Түүхэн өдрүүдийн зудын эрсдэл харах
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class HistoricalDailyData:
    def __init__(self):
        """Load monthly weather data and convert to daily estimates"""
        self.monthly_data = pd.read_csv('weather_omnogovi_monthly_clean.csv')
        
    def get_daily_data(self, lat: float, lon: float, start_date: str, end_date: str, livestock_count: float = 200):
        """
        Get historical daily data for a date range
        
        Args:
            lat, lon: Location coordinates
            start_date: 'YYYY-MM-DD'
            end_date: 'YYYY-MM-DD'
            livestock_count: Number of livestock
        
        Returns:
            DataFrame with daily risk estimates
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Find nearest location
        self.monthly_data['distance'] = np.sqrt(
            (self.monthly_data['lat'] - lat)**2 + 
            (self.monthly_data['lon'] - lon)**2
        )
        nearest = self.monthly_data.loc[self.monthly_data['distance'].idxmin()]
        soum = nearest['soum']
        
        # Get monthly data for this location
        soum_data = self.monthly_data[self.monthly_data['soum'] == soum].copy()
        
        # Generate daily data
        daily_records = []
        
        current_date = start
        while current_date <= end:
            year = current_date.year
            month = current_date.month
            
            # Get monthly data
            monthly = soum_data[(soum_data['year'] == year) & (soum_data['month'] == month)]
            
            if len(monthly) > 0:
                row = monthly.iloc[0]
                
                # Add daily variation (±20% around monthly average)
                day_variation = np.random.uniform(0.8, 1.2)
                temp_variation = np.random.uniform(-3, 3)
                
                daily_temp_min = row['min_temp'] + temp_variation
                daily_temp_avg = row['avg_temp'] + temp_variation * 0.7
                daily_wind = row['wind_speed'] * day_variation
                daily_precip = row['precip_sum'] / 30 * day_variation  # Monthly to daily
                daily_snow = row['snowfall_sum'] / 30 * day_variation
                
                # Calculate risk
                risk_score = self.calculate_daily_risk(
                    current_date.month,
                    daily_temp_min,
                    daily_temp_avg,
                    daily_wind,
                    daily_snow,
                    daily_precip,
                    livestock_count
                )
                
                risk_level, risk_label, risk_color = self.get_risk_level(risk_score)
                
                daily_records.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'day_name': current_date.strftime('%A'),
                    'soum': soum,
                    'temp_min': round(daily_temp_min, 1),
                    'temp_avg': round(daily_temp_avg, 1),
                    'wind_speed': round(daily_wind, 1),
                    'snowfall': round(daily_snow, 2),
                    'precip': round(daily_precip, 2),
                    'risk_score': round(risk_score, 0),
                    'risk_level': risk_level,
                    'risk_label': risk_label,
                    'risk_color': risk_color
                })
            
            current_date += timedelta(days=1)
        
        return pd.DataFrame(daily_records)
    
    def calculate_daily_risk(self, month, temp_min, temp_avg, wind, snow, precip, livestock):
        """Calculate daily risk score"""
        # Only winter months
        if month not in [11, 12, 1, 2, 3]:
            return 0
        
        score = 0
        
        # Temperature risk
        if temp_min < -25:
            score += 35
        elif temp_min < -20:
            score += 25
        elif temp_min < -15:
            score += 15
        elif temp_min < -10:
            score += 10
        
        # Wind risk
        if wind > 18:
            score += 25
        elif wind > 15:
            score += 15
        elif wind > 12:
            score += 10
        
        # Snow risk
        if snow > 0.3:  # Daily threshold
            score += 20
        elif snow > 0.15:
            score += 10
        
        # Precipitation deficit
        if precip < 0.15:  # Daily threshold
            score += 15
        elif precip < 0.3:
            score += 5
        
        # Livestock exposure
        if livestock > 100:
            score += 15
        elif livestock > 50:
            score += 10
        
        # Wind chill
        wind_chill = temp_min - (wind * 0.5)
        if wind_chill < -30:
            score += 10
        
        return min(score, 100)
    
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
    
    def get_month_summary(self, lat: float, lon: float, year: int, month: int, livestock_count: float = 200):
        """Get summary for a specific month"""
        # Get first and last day of month
        start_date = f"{year}-{month:02d}-01"
        
        if month == 12:
            end_date = f"{year}-{month:02d}-31"
        else:
            next_month = datetime(year, month, 1) + timedelta(days=32)
            last_day = (next_month.replace(day=1) - timedelta(days=1)).day
            end_date = f"{year}-{month:02d}-{last_day}"
        
        daily_df = self.get_daily_data(lat, lon, start_date, end_date, livestock_count)
        
        if len(daily_df) == 0:
            return None
        
        summary = {
            'year': year,
            'month': month,
            'month_name': datetime(year, month, 1).strftime('%B'),
            'total_days': len(daily_df),
            'avg_risk_score': round(daily_df['risk_score'].mean(), 1),
            'max_risk_score': round(daily_df['risk_score'].max(), 0),
            'high_risk_days': len(daily_df[daily_df['risk_level'] >= 2]),
            'coldest_day': {
                'date': daily_df.loc[daily_df['temp_min'].idxmin(), 'date'],
                'temp': daily_df['temp_min'].min()
            },
            'windiest_day': {
                'date': daily_df.loc[daily_df['wind_speed'].idxmax(), 'date'],
                'wind': daily_df['wind_speed'].max()
            },
            'risk_distribution': daily_df['risk_label'].value_counts().to_dict()
        }
        
        return summary


# Example usage
if __name__ == "__main__":
    historical = HistoricalDailyData()
    
    # Example 1: Get specific month
    print("="*60)
    print("2023 оны 1-р сарын өдөр бүрийн эрсдэл")
    print("="*60)
    
    daily_data = historical.get_daily_data(
        lat=43.57,
        lon=104.43,
        start_date='2023-01-01',
        end_date='2023-01-31',
        livestock_count=200
    )
    
    print(f"\nНийт өдөр: {len(daily_data)}")
    print(f"\nЭхний 10 өдөр:")
    print(daily_data.head(10).to_string(index=False))
    
    # Example 2: Month summary
    print("\n" + "="*60)
    print("2023 оны 1-р сарын дүгнэлт")
    print("="*60)
    
    summary = historical.get_month_summary(43.57, 104.43, 2023, 1, 200)
    print(f"\nСар: {summary['month_name']} {summary['year']}")
    print(f"Нийт өдөр: {summary['total_days']}")
    print(f"Дундаж эрсдэл: {summary['avg_risk_score']}/100")
    print(f"Хамгийн өндөр эрсдэл: {summary['max_risk_score']}/100")
    print(f"Өндөр эрсдэлтэй өдөр: {summary['high_risk_days']}")
    print(f"\nХамгийн хүйтэн өдөр: {summary['coldest_day']['date']} ({summary['coldest_day']['temp']:.1f}°C)")
    print(f"Хамгийн салхитай өдөр: {summary['windiest_day']['date']} ({summary['windiest_day']['wind']:.1f} м/с)")
    print(f"\nЭрсдэлийн хуваарилалт:")
    for label, count in summary['risk_distribution'].items():
        print(f"  {label}: {count} өдөр")
