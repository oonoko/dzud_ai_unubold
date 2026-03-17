#!/usr/bin/env python3
"""
Fetch weather data from Meteostat for Omnogovi soums
"""

from datetime import datetime
import pandas as pd
import time
from meteostat import Point, Daily

soums = pd.read_csv("soum_list_full.csv")

start = datetime(2015, 1, 1)
end   = datetime(2024, 12, 31)

all_rows = []
log_lines = []

for _, r in soums.iterrows():
    aimag = r["aimag"]
    soum  = r["soum"]
    lat   = float(r["lat"])
    lon   = float(r["lon"])

    try:
        point = Point(lat, lon)
        df = Daily(point, start, end).fetch()

        if df is None or df.empty:
            log_lines.append(f"[WARN] {soum}: no data")
            continue

        df = df.reset_index()  # time багана
        df["aimag"] = aimag
        df["soum"] = soum
        df["lat"] = lat
        df["lon"] = lon

        all_rows.append(df)
        log_lines.append(f"[OK] {soum}: {len(df)} rows")

        time.sleep(0.3)

    except Exception as e:
        log_lines.append(f"[ERR] {soum}: {e}")

if all_rows:
    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv("weather_omnogovi_daily.csv", index=False, encoding="utf-8")
    print("✅ Saved weather_omnogovi_daily.csv", out.shape)
    print(out.head())
else:
    print("❌ No weather data collected")

with open("weather_fetch.log", "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))

print("📝 Saved weather_fetch.log")
