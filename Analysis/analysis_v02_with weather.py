import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from pathlib import Path

Path("../outputs").mkdir(parents=True, exist_ok=True)

df_weather = pd.read_csv("../data/ZRH_Flights_with_Weather_2025.csv")
df_weather = df_weather[df_weather["is_outlier"] == False].copy()
df_weather = df_weather[df_weather["delay_min"].between(-60, 300)].copy()

def evaluate(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"\n── {name} ──")
    print(f"  MAE:  {mean_absolute_error(y_test, preds):.2f}")
    print(f"  RMSE: {root_mean_squared_error(y_test, preds):.2f}")
    print(f"  R²:   {r2_score(y_test, preds):.4f}")
    return model

# ── 1. Weather correlation heatmap ────────────────────────────────
weather_cols = ["delay_min", "temperature_C", "precipitation_mm", "wind_speed_kmh"]
plt.figure(figsize=(6, 5))
sns.heatmap(df_weather[weather_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Weather Variables Correlation with Delay")
plt.savefig("../outputs/weather_correlation.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 2. Bar chart: Average Delay by Wind Speed group ───────────────
df_weather["wind_bin"] = pd.cut(
    df_weather["wind_speed_kmh"],
    bins=[0, 10, 20, 30, 100],
    labels=["0-10", "10-20", "20-30", "30+"]
)

fig, ax = plt.subplots(figsize=(8, 4))
df_weather.groupby("wind_bin", observed=True)["delay_min"].mean().plot(kind="bar", ax=ax, color="steelblue")
ax.set_title("Average Delay by Wind Speed Group")
ax.set_xlabel("Wind Speed (km/h)")
ax.set_ylabel("Avg Delay (min)")
ax.grid(False)
plt.xticks(rotation=0)
plt.savefig("../outputs/bar_wind.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 3. Bar chart: Average Delay by Temperature group ─────────────
df_weather["temp_bin"] = pd.cut(
    df_weather["temperature_C"],
    bins=[-20, -5, 0, 5, 10, 20],
    labels=["Below -5", "-5 to 0", "0 to 5", "5 to 10", "Above 10"]
)

fig, ax = plt.subplots(figsize=(8, 4))
df_weather.groupby("temp_bin", observed=True)["delay_min"].mean().plot(kind="bar", ax=ax, color="steelblue")
ax.set_title("Average Delay by Temperature Group")
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Avg Delay (min)")
ax.grid(False)
plt.xticks(rotation=0)
plt.savefig("../outputs/bar_temp.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 4. Bar chart: Average Delay by Precipitation group ───────────
df_weather["precip_bucket"] = pd.cut(
    df_weather["precipitation_mm"],
    bins=[-0.1, 0, 0.5, 1, 5],
    labels=["None", "Trace(<0.5mm)", "Light(0.5-1mm)", "Moderate(1-5mm)"]
)

fig, ax = plt.subplots(figsize=(8, 4))
df_weather.groupby("precip_bucket", observed=True)["delay_min"].mean().plot(kind="bar", ax=ax, color="steelblue")
ax.set_title("Average Delay by Precipitation Level")
ax.set_xlabel("Precipitation")
ax.set_ylabel("Avg Delay (min)")
ax.grid(False)
plt.xticks(rotation=0)
plt.savefig("../outputs/bar_precip.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 5. Regression: with vs without weather ────────────────────────
features_no_weather   = ["dep_hour", "dep_dow", "dep_month", "is_weekend", "route_avg_delay"]
features_with_weather = features_no_weather + ["temperature_C", "precipitation_mm", "wind_speed_kmh"]

df_w  = df_weather[features_with_weather + ["delay_min"]].dropna()
X_no  = df_w[features_no_weather]
X_yes = df_w[features_with_weather]
y_w   = df_w["delay_min"]

X_tr_n, X_te_n, y_tr_w, y_te_w = train_test_split(X_no,  y_w, test_size=0.2, random_state=42)
X_tr_y, X_te_y, _,      _      = train_test_split(X_yes, y_w, test_size=0.2, random_state=42)

evaluate("LR – Without Weather", LinearRegression(), X_tr_n, X_te_n, y_tr_w, y_te_w)
evaluate("LR – With Weather",    LinearRegression(), X_tr_y, X_te_y, y_tr_w, y_te_w)
evaluate("RF – Without Weather", RandomForestRegressor(n_estimators=100, random_state=42), X_tr_n, X_te_n, y_tr_w, y_te_w)
evaluate("RF – With Weather",    RandomForestRegressor(n_estimators=100, random_state=42), X_tr_y, X_te_y, y_tr_w, y_te_w)

print("\nAll outputs saved to /outputs/")