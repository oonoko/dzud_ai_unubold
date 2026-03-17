#!/usr/bin/env python3
"""
Fetch weather data from Open-Meteo for all Omnogovi soums
"""

import pandas as pd
import requests
import time
from datetime import datetime

# Read soum list
soums = pd.read_csv("soum_list_last2.csv")

start_date = "2015-01-01"
end_date = "2024-12-31"

all_data = []

for idx, row in soums.iterrows():
    aimag = row["aimag"]
    soum = row["soum"]
    lat = row["lat"]
    lon = row["lon"]
    
    print(f"Fetching {soum} ({lat}, {lon})...")
    
    try:
        # Open-Meteo API
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_mean,temperature_2m_min,wind_speed_10m_max,snowfall_sum,precipitation_sum",
            "timezone": "Asia/Ulaanbaatar"
        }
        
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        if "daily" not in data:
            print(f"  ⚠️  No data for {soum}")
            continue
        
        daily = data["daily"]
        df = pd.DataFrame({
            "aimag": aimag,
            "soum": soum,
            "lat": lat,
            "lon": lon,
            "date": daily["time"],
            "avg_temp": daily["temperature_2m_mean"],
            "min_temp": daily["temperature_2m_min"],
            "wind_speed": daily["wind_speed_10m_max"],
            "snowfall_sum": daily["snowfall_sum"],
            "precip_sum": daily["precipitation_sum"]
        })
        
        all_data.append(df)
        print(f"  ✅ {len(df)} rows")
        
        time.sleep(5)  # 5 second delay to avoid rate limit
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        continue

if all_data:
    # Combine all data
    result = pd.concat(all_data, ignore_index=True)
    
    # Convert date to year/month
    result["date"] = pd.to_datetime(result["date"])
    result["year"] = result["date"].dt.year
    result["month"] = result["date"].dt.month
    
    # Group by month and aggregate
    monthly = result.groupby(["aimag", "soum", "lat", "lon", "year", "month"]).agg({
        "avg_temp": "mean",
        "min_temp": "min",
        "wind_speed": "mean",
        "snowfall_sum": "sum",
        "precip_sum": "sum"
    }).reset_index()
    
    # Round values
    monthly["avg_temp"] = monthly["avg_temp"].round(3)
    monthly["min_temp"] = monthly["min_temp"].round(1)
    monthly["wind_speed"] = monthly["wind_speed"].round(3)
    monthly["snowfall_sum"] = monthly["snowfall_sum"].round(2)
    monthly["precip_sum"] = monthly["precip_sum"].round(1)
    
    # Save
    monthly.to_csv("weather_omnogovi_monthly_last2.csv", index=False)
    print(f"\n✅ Saved weather_omnogovi_monthly_last2.csv")
    print(f"   Total rows: {len(monthly)}")
    print(f"   Soums: {monthly['soum'].nunique()}")
    print(f"\nSample:")
    print(monthly.head(10))
else:
    print("❌ No data collected")
