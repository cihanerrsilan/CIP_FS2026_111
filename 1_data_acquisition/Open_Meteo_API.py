import requests
import pandas as pd
import os

print("1. Connecting to Open-Meteo API...")

# Zurich Airport (ZRH) coordinates
lat = 47.4647
lon = 8.5492

# Project data range (October 1 - December 31, 2025)
start_date = "2025-10-01"
end_date = "2025-12-31"

url = (f"https://archive-api.open-meteo.com/v1/archive?"
       f"latitude={lat}&longitude={lon}&"
       f"start_date={start_date}&end_date={end_date}&"
       f"hourly=temperature_2m,precipitation,wind_speed_10m&"
       f"timezone=UTC")

print("2. Sending request to the server... (This might take a few seconds)")
response = requests.get(url)

print(f"3. Server Response Code: {response.status_code} (If it's 200, everything is perfect!)")

if response.status_code == 200:
    data = response.json()
    df_weather = pd.DataFrame(data['hourly'])

    # Fix time format
    df_weather['time'] = pd.to_datetime(df_weather['time'], utc=True)
    df_weather['month'] = df_weather['time'].dt.month
    df_weather['day'] = df_weather['time'].dt.day
    df_weather['hour'] = df_weather['time'].dt.hour

    # Rename columns for clarity
    df_weather.rename(columns={
        'temperature_2m': 'temperature_C',
        'precipitation': 'precipitation_mm',
        'wind_speed_10m': 'wind_speed_kmh'
    }, inplace=True)

    # SAVE DIRECTLY TO THE SAME FOLDER
    file_name = "ZRH_Weather_2025.csv"
    df_weather.to_csv(file_name, index=False)

    print(f"\n4.  SUCCESS! Data has been saved to your computer as '{file_name}'.")
    print(f" The exact file location on your Mac: {os.path.abspath(file_name)}")
else:
    print("\n ERROR! Could not fetch the data.")