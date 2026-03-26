import pandas as pd
import numpy as np
import ast
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

def safe_parse(x):
    if isinstance(x, dict):
        return x
    try:
        return ast.literal_eval(str(x))
    except:
        return {}


print("Loading files...")
Path("../data").mkdir(parents=True, exist_ok=True)
df1 = pd.read_csv("../data/ZRH_Departure_Delays_2025.csv")
df2 = pd.read_csv("../data/ZRH_All_Flights_Inc_Canceled_2025.csv")

df1["source"] = "delays"
df2["source"] = "all_flights"

df = pd.concat([df1, df2], ignore_index=True)

# remove exact duplicates based on flight number + departure info
df = df.drop_duplicates(subset=["number", "departure"], keep="last")
print("Rows after merge:", len(df))

# keep only operating flights, remove codeshare duplicates
df = df[df["codeshareStatus"].isin(["IsOperator", "Unknown"])]
print("Rows after removing codeshares:", len(df))


print("Parsing nested columns...")

# departure info
df["dep_dict"] = df["departure"].apply(safe_parse)
df["scheduled_utc"] = df["dep_dict"].apply(lambda x: x.get("scheduledTime", {}).get("utc"))
df["revised_utc"] = df["dep_dict"].apply(lambda x: x.get("revisedTime", {}).get("utc"))
df["gate"] = df["dep_dict"].apply(lambda x: x.get("gate"))

# arrival info
df["arr_dict"] = df["arrival"].apply(safe_parse)
df["dest_iata"] = df["arr_dict"].apply(lambda x: x.get("airport", {}).get("iata"))
df["dest_name"] = df["arr_dict"].apply(lambda x: x.get("airport", {}).get("name"))
df["dest_country"] = df["arr_dict"].apply(lambda x: x.get("airport", {}).get("countryCode"))

# airline info
df["airline_dict"] = df["airline"].apply(safe_parse)
df["airline_name"] = df["airline_dict"].apply(lambda x: x.get("name"))
df["airline_iata"] = df["airline_dict"].apply(lambda x: x.get("iata"))

# aircraft info
df["aircraft_dict"] = df["aircraft"].apply(safe_parse)
df["aircraft_model"] = df["aircraft_dict"].apply(lambda x: x.get("model"))

# convert timestamps
df["scheduled_utc"] = pd.to_datetime(df["scheduled_utc"], errors="coerce", utc=True)
df["revised_utc"] = pd.to_datetime(df["revised_utc"], errors="coerce", utc=True)

# calculate delay in minutes
df["delay_min"] = (df["revised_utc"] - df["scheduled_utc"]).dt.total_seconds() / 60

# keep canceled separately if needed
df_canceled = df[df["status"].isin(["Canceled", "CanceledUncertain"])].copy()
df = df[~df["status"].isin(["Canceled", "CanceledUncertain"])].copy()

print("Canceled flights:", len(df_canceled))
print("Flights left for analysis:", len(df))

# rows without delay can't be used for delay analysis
before = len(df)
df = df.dropna(subset=["delay_min"])
print("Dropped rows with missing delay:", before - len(df))

# make sure isCargo is boolean
df["isCargo"] = df["isCargo"].astype(bool)

# fill missing values in text columns
fill_cols = [
    "airline_name", "airline_iata", "dest_iata",
    "dest_name", "dest_country", "aircraft_model", "gate"
]

for col in fill_cols:
    df[col] = df[col].fillna("Unknown")

# outlier detection
q1 = df["delay_min"].quantile(0.25)
q3 = df["delay_min"].quantile(0.75)
iqr = q3 - q1
lower = q1 - 3 * iqr
upper = q3 + 3 * iqr

df["is_outlier"] = ~df["delay_min"].between(lower, upper)

# feature engineering
df["dep_hour"] = df["scheduled_utc"].dt.hour
df["dep_dow"] = df["scheduled_utc"].dt.dayofweek
df["dep_month"] = df["scheduled_utc"].dt.month
df["is_weekend"] = df["dep_dow"].isin([5, 6]).astype(int)
df["route"] = df["dest_iata"]

# route average delay
route_avg = df.groupby("route")["delay_min"].mean().rename("route_avg_delay")
df = df.merge(route_avg, on="route", how="left")


def aircraft_family(model):
    if pd.isna(model) or model == "Unknown":
        return "Unknown"

    model = str(model).lower()

    if any(x in model for x in ["a320", "a321", "a319", "737", "a220"]):
        return "Narrow-body"
    elif any(x in model for x in ["a330", "a340", "a350", "777", "787", "747", "767"]):
        return "Wide-body"
    elif any(x in model for x in ["atr", "crj", "dash", "embraer", "e1", "e2"]):
        return "Regional"
    else:
        return "Other"


df["aircraft_family"] = df["aircraft_model"].apply(aircraft_family)

# remove helper columns
df = df.drop(columns=[
    "departure", "arrival", "airline", "aircraft", "callSign",
    "dep_dict", "arr_dict", "airline_dict", "aircraft_dict", "source"
], errors="ignore")

# save cleaned dataset
df.to_csv("../data/ZRH_Cleaned_Flights_2025.csv", index=False)

print("Cleaned dataset saved as ZRH_Cleaned_Flights_2025.csv")
print("Final shape:", df.shape)
print(df.head())
