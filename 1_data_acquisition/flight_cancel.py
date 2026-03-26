import requests
import pandas as pd
import warnings
import time
from datetime import datetime
from pathlib import Path


# Suppress SSL warnings on Mac
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

headers = {
    "x-rapidapi-key": "2818be0ac3msh07bed8a4ea154edp19c29ajsn8fdea3b5d9e8",
    "x-rapidapi-host": "aerodatabox.p.rapidapi.com"
}

# UPDATED: withCancelled is now set to "true" to include canceled flights
querystring = {
    "withLeg": "true",
    "direction": "Departure",
    "withCancelled": "true",
    "withCodeshared": "true",
    "withLocation": "false"
}

# October 1 - December 31, 2025
start_date = pd.to_datetime("2025-10-01")
end_date = pd.to_datetime("2025-12-31")
date_list = pd.date_range(start=start_date, end=end_date).to_pydatetime()

all_flights = []

print(f"Starting data extraction (INCLUDING CANCELED FLIGHTS) for {len(date_list)} days...")

for current_date in date_list:
    date_str = current_date.strftime('%Y-%m-%d')

    # API limits: 12-hour slots
    url_am = f"https://aerodatabox.p.rapidapi.com/flights/airports/iata/ZRH/{date_str}T00:00/{date_str}T11:59"
    url_pm = f"https://aerodatabox.p.rapidapi.com/flights/airports/iata/ZRH/{date_str}T12:00/{date_str}T23:59"

    for url in [url_am, url_pm]:
        response = requests.get(url, headers=headers, params=querystring, verify=False)

        if response.status_code == 200:
            data = response.json()
            if "departures" in data:
                all_flights.extend(data["departures"])
        elif response.status_code == 429:
            print(f"\n❌ Quota Exceeded on {date_str}. Stopping.")
            break

        time.sleep(1)  # Rate limit protection
    print(f"✅ {date_str} completed.")

if all_flights:
    df_final = pd.DataFrame(all_flights)
    # Saving with a new name to distinguish from the previous version
    Path("../data").mkdir(parents=True, exist_ok=True)
    file_name = "../data/ZRH_All_Flights_Inc_Canceled_2025.csv"
    df_final.to_csv(file_name, index=False)
    print(f"\n🎉 SUCCESS! Data saved as '{file_name}'")
    print("Total flights (including potential cancellations):", len(df_final))