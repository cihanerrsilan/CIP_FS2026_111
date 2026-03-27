import pandas as pd

# 1. Load the datasets
flights_df = pd.read_csv("ZRH_Cleaned_Flights_2025.csv")
weather_df = pd.read_csv("ZRH_Weather_2025.csv")

# 2. Extract the "day of the month" for merging (THE LIFESAVER LINE)
flights_df['scheduled_utc'] = pd.to_datetime(flights_df['scheduled_utc'])
flights_df['dep_day'] = flights_df['scheduled_utc'].dt.day

# 3. Merge the two datasets
merged_df = flights_df.merge(
    weather_df[['month', 'day', 'hour', 'temperature_C', 'precipitation_mm', 'wind_speed_kmh']],
    left_on=['dep_month', 'dep_day', 'dep_hour'],
    right_on=['month', 'day', 'hour'],
    how='left'
)

# 4. Drop redundant columns and save the final dataset
merged_df = merged_df.drop(columns=['month', 'day', 'hour', 'dep_day'])
merged_df.to_csv("ZRH_Flights_with_Weather_2025.csv", index=False)

print("Weather data successfully merged with flight data with zero missing values!")