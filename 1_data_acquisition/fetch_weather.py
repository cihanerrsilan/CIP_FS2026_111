

import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from pathlib import Path

# Setup Open-Meteo client with cache
cache_session = requests_cache.CachedSession(".weather_cache", expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo     = openmeteo_requests.Client(session=retry_session)


# ZRH airport coordinates
LAT = 47.4647
LON = 8.5492

url    = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude":   LAT,
    "longitude":  LON,
    "start_date": "2025-10-01",
    "end_date":   "2025-12-31",
    "hourly": [
        "temperature_2m",          # air temperature °C
        "precipitation",           # rainfall/snowfall mm
        "windspeed_10m",           # wind speed km/h
        "windgusts_10m",           # wind gusts km/h
        "visibility",              # visibility metres
        "cloudcover",              # cloud cover %
        "snowfall",                # snowfall cm
        "weathercode",             # WMO weather code
        "relative_humidity_2m",    # humidity %
    ],
    "timezone": "UTC",
    "wind_speed_unit": "kmh",
}

print("Fetching weather data from Open-Meteo...")
responses = openmeteo.weather_api(url, params=params)
response  = responses[0]

print(f"  Coordinates : {response.Latitude():.2f}°N  {response.Longitude():.2f}°E")
print(f"  Timezone    : {response.Timezone()}")
print(f"  UTC offset  : {response.UtcOffsetSeconds()} s")

hourly = response.Hourly()

weather_df = pd.DataFrame({
    "time":                pd.date_range(
                               start=pd.to_datetime(hourly.Time(),   unit="s", utc=True),
                               end=  pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                               freq=pd.Timedelta(seconds=hourly.Interval()),
                               inclusive="left"
                           ),
    "temperature_2m":       hourly.Variables(0).ValuesAsNumpy(),
    "precipitation":        hourly.Variables(1).ValuesAsNumpy(),
    "windspeed_10m":        hourly.Variables(2).ValuesAsNumpy(),
    "windgusts_10m":        hourly.Variables(3).ValuesAsNumpy(),
    "visibility":           hourly.Variables(4).ValuesAsNumpy(),
    "cloudcover":           hourly.Variables(5).ValuesAsNumpy(),
    "snowfall":             hourly.Variables(6).ValuesAsNumpy(),
    "weathercode":          hourly.Variables(7).ValuesAsNumpy(),
    "relative_humidity_2m": hourly.Variables(8).ValuesAsNumpy(),
})

#Save
weather_df.to_csv("ZRH_Weather_2025.csv", index=False)
print(f"\nSaved {len(weather_df):,} rows to ZRH_Weather_2025.csv")
print(weather_df.head())
print(weather_df.describe().round(2))