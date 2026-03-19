import pandas as pd
import numpy as np
from datetime import timedelta
import random

print("🚀 Zurich Airport (ZRH) Flight Simulation is Starting...")

# 1. Date Range (According to Feasibility Report: Oct 1 - Dec 31, 2025)
dates = pd.date_range(start="2025-10-01", end="2025-12-31", freq='h')

# 2. Aviation Variables (Realistic data)
airlines = ['Swiss International Air Lines', 'Edelweiss Air', 'Lufthansa', 'EasyJet', 'British Airways']
destinations = ['LHR (London)', 'JFK (New York)', 'FRA (Frankfurt)', 'CDG (Paris)', 'VIE (Vienna)', 'DXB (Dubai)']

simulated_data = []

# 3. Data Generation Loop
for date in dates:
    # Zurich Airport night flight ban (Usually closed between 23:30 - 06:00)
    if 6 <= date.hour <= 23:
        # Have a random number of 2 to 5 flights depart every hour
        hourly_flight_count = random.randint(2, 5)

        for _ in range(hourly_flight_count):
            airline = random.choice(airlines)
            destination = random.choice(destinations)

            # Delay Simulation
            # Most flights depart on time (0-5 min delay), some are heavily delayed
            # In statistics, this is called an "Exponential Distribution"
            delay_minutes = int(np.random.exponential(scale=15))

            # If the delay is less than 15 minutes, it's "On Time", otherwise "Delayed"
            if delay_minutes < 15:
                status = 'On Time'
                delay_minutes = 0  # On-time flights have 0 delay
            else:
                status = 'Delayed'

            # 2% chance for the flight to be cancelled
            if random.random() < 0.02:
                status = 'Cancelled'
                delay_minutes = None  # We don't calculate delay for cancelled flights

            actual_departure = date + timedelta(minutes=delay_minutes) if delay_minutes is not None else None

            # Append the data to the list
            simulated_data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Scheduled_Time': date.strftime('%H:%M'),
                'Actual_Departure': actual_departure.strftime('%H:%M') if actual_departure else 'Cancelled',
                'Airline': airline,
                'Destination': destination,
                'Flight_Status': status,
                'Delay_Minutes': delay_minutes
            })

print("✅ Data successfully simulated!")

# 4. Convert to DataFrame and Save
df = pd.DataFrame(simulated_data)

filename = "zrh_simulated_flight_delays_2025.csv"
df.to_csv(filename, index=False)

print(f"\n🎉 GREAT! A total of {len(df)} flight records have been saved to '{filename}'.")
print("\nFirst 5 Rows:")
print(df.head())