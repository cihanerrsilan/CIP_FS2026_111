import requests
import pandas as pd
import warnings
import time
from datetime import datetime, timedelta
from pathlib import Path
import os
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()

# Get the API key from the environment variable
api_key = os.getenv("RAPIDAPI_KEY")

# Suppress SSL warnings on Mac
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

headers = {
    "x-rapidapi-key": api_key,
    "x-rapidapi-host": "aerodatabox.p.rapidapi.com"
}

querystring = {
    "withLeg": "true", "direction": "Departure",
    "withCancelled": "false", "withCodeshared": "true", "withLocation": "false"
}

# Target: October 1 - December 31, 2025
start_date = pd.to_datetime("2025-10-01")
end_date = pd.to_datetime("2025-12-31")

# Create a list of all dates between the start and end dates
date_list = pd.date_range(start=start_date, end=end_date).to_pydatetime()

all_flights = []  # We will collect all fetched flights in this list

print(f"Starting data extraction for a total of {len(date_list)} days...")
print("Waiting 1 second between requests to avoid API rate limits. This process may take 3-4 minutes...\n")

# ATTENTION: TO AVOID WASTING QUOTA, LET'S DO A 3-DAY TEST FIRST.
# Uncomment the line below when running the code for the first time for a 3-day test.
# If everything is fine, comment it out again (put the # back) to fetch the full 3-month data.
# date_list = date_list[:3]

for current_date in date_list:
    date_str = current_date.strftime('%Y-%m-%d')

    # Sending 2 requests per day (AM and PM) due to the API's 12-hour limit
    url_am = f"https://aerodatabox.p.rapidapi.com/flights/airports/iata/ZRH/{date_str}T00:00/{date_str}T11:59"
    url_pm = f"https://aerodatabox.p.rapidapi.com/flights/airports/iata/ZRH/{date_str}T12:00/{date_str}T23:59"

    for url in [url_am, url_pm]:
        response = requests.get(url, headers=headers, params=querystring, verify=False)

        if response.status_code == 200:
            data = response.json()
            if "departures" in data:
                all_flights.extend(data["departures"])
        elif response.status_code == 429:
            print(f"\n❌ API Quota Exceeded! (Error 429). Process stopped on {date_str}.")
            break
        else:
            print(f"An error occurred ({date_str}): {response.status_code}")

        # Sleep for 1 second to prevent being blocked by the system
        time.sleep(1)

    print(f"✅ Data for {date_str} successfully fetched and added.")

# Combine all collected data into a single Pandas DataFrame
if all_flights:
    df_final = pd.DataFrame(all_flights)
    print("\n🎉 ALL DATA SUCCESSFULLY FETCHED AND COMBINED!")
    print("Total Number of Flights:", df_final.shape[0])

    # Save the data as a CSV file to the current directory
    Path("../data").mkdir(parents=True, exist_ok=True)
    file_name = "../data/ZRH_Departure_Delays_2025.csv"
    df_final.to_csv(file_name, index=False)
    print(f"Data successfully saved to your computer as '{file_name}'. We are done with the API phase!")
else:
    print("No data could be fetched.")
