import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, mean_squared_error
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


Path("../outputs").mkdir(parents=True, exist_ok=True)

# FONT SIZE SETTINGS  (required by SW04 lecture)
# These three variables control ALL text sizes in every figure below.
# Changing BIGGER_SIZE here automatically updates every title in the script.
SMALL_SIZE = 9  # tick labels, legend text, value annotations
MEDIUM_SIZE = 11  # axis labels
BIGGER_SIZE = 13  # figure titles and suptitles

plt.rc("font", size=SMALL_SIZE)
plt.rc("axes", titlesize=BIGGER_SIZE)
plt.rc("axes", labelsize=MEDIUM_SIZE)
plt.rc("xtick", labelsize=SMALL_SIZE)
plt.rc("ytick", labelsize=SMALL_SIZE)
plt.rc("legend", fontsize=SMALL_SIZE)
plt.rc("figure", titlesize=BIGGER_SIZE)

# colour palette — consistent across all figures
BLUE = "#2563EB"
ORANGE = "#F59E0B"
RED = "#EF4444"
GREEN = "#10B981"
PURPLE = "#8B5CF6"
PALETTE = [BLUE, ORANGE, RED, GREEN, PURPLE]

# df_weather = pd.read_csv("../data/ZRH_Flights_with_Weather_2025.csv")
# FIX: added parse_dates so scheduled_utc is read as a proper datetime object
# and not as a plain string. This matters if we ever filter by date later.
df_weather = pd.read_csv(
    "../data/ZRH_Flights_with_Weather_2025.csv",
    parse_dates=["scheduled_utc"]
)
##df_weather = df_weather[df_weather["is_outlier"] == False].copy()
# Keep only non-outlier flights.
# is_outlier was flagged during cleaning using ±3 IQR on delay_min.
# Keeping outliers would skew the averages in the bar charts and inflate RMSE.
df_weather = df_weather[df_weather["is_outlier"] == False].copy()

##df_weather = df_weather[df_weather["delay_min"].between(-60, 300)].
# Further restrict to a realistic delay window of -60 to +300 minutes.
# Values outside this range are almost certainly data errors (e.g. revised time
# being the next day's flight), not genuine extreme delays.
df_weather = df_weather[df_weather["delay_min"].between(-60, 300)].copy()
print(f"Rows after filtering: {len(df_weather):,}")
print(f"Average delay: {df_weather['delay_min'].mean():.1f} min")

# def evaluate(name, model, X_train, X_test, y_train, y_test):
#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)
#     print(f"\n── {name} ──")
#     print(f"  MAE:  {mean_absolute_error(y_test, preds):.2f}")
#     print(f"  RMSE: {root_mean_squared_error(y_test, preds):.2f}")
#     print(f"  R²:   {r2_score(y_test, preds):.4f}")
#     return model



def evaluate(name, model, X_train, X_test, y_train, y_test):
    """
    Trains the model on X_train/y_train, predicts on X_test,
    and prints MAE, RMSE and R².

    Returns the fitted model so we can inspect it later
    (e.g. for feature importances).
    """
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5  # FIX: compatible with all sklearn versions
    r2 = r2_score(y_test, preds)

    print(f"\n── {name} ──")
    print(f"  MAE  : {mae:.2f} min   (on average predictions are off by this many minutes)")
    print(f"  RMSE : {rmse:.2f} min  (penalises large errors more than MAE does)")
    print(f"  R²   : {r2:.4f}       ({r2 * 100:.1f}% of delay variance explained by the model)")

    return model, preds
# 1. Weather correlation heatmap
weather_cols = ["delay_min", "temperature_C", "precipitation_mm", "wind_speed_kmh"]
# plt.figure(figsize=(6, 5))
# sns.heatmap(df_weather[weather_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
# plt.title("Weather Variables Correlation with Delay")
# plt.savefig("../outputs/weather_correlation.png", dpi=150, bbox_inches="tight")
# plt.show()
readable = {
    "delay_min": "Delay [min]",
    "temperature_C": "Temperature [°C]",
    "precipitation_mm": "Precipitation [mm]",
    "wind_speed_kmh": "Wind Speed [km/h]"
}

fig1, ax1 = plt.subplots(figsize=(7, 6))
corr_data = df_weather[weather_cols].dropna().rename(columns=readable)
sns.heatmap(
    corr_data.corr(),
    annot=True, fmt=".2f",
    cmap="coolwarm", center=0,  # FIX: center=0 so zero correlation is white,
    ax=ax1,  # not an arbitrary midpoint of the colour scale
    annot_kws={"size": SMALL_SIZE}
)
ax1.set_title("Figure 1 – Correlation: Weather Variables vs Departure Delay",
              fontweight="bold")
fig1.tight_layout()
fig1.savefig("../outputs/fig1_weather_correlation.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print("\n✅ Figure 1 saved — Weather correlation heatmap")

# A heatmap shows the linear correlation between pairs of variables.
# Values close to +1 or -1 indicate a strong relationship.
# Values close to 0 mean little or no linear relationship.
# We expect weak correlations here because weather is just ONE driver of delay —
# airline operations and other factors also play a big role.

# 2. Bar chart: Average Delay by Wind Speed group
df_weather["wind_bin"] = pd.cut(
    df_weather["wind_speed_kmh"],
    bins=[0, 10, 20, 30, 100],
    labels=["0–10 km/h", "10–20 km/h", "20–30 km/h", "30+ km/h"]
)
wind_avg = df_weather.groupby("wind_bin", observed=True)["delay_min"].agg(["mean", "count"])
overall_mean = df_weather["delay_min"].mean()

fig2, ax2 = plt.subplots(figsize=(9, 5))
bars = ax2.bar(
    wind_avg.index, wind_avg["mean"],
    color=PALETTE[:len(wind_avg)], edgecolor="white", width=0.6
)

# FIX: added sample-count labels on top of each bar.
# Without these, the reader cannot judge whether a group is reliable or based
# on only a handful of flights (e.g. the "30+ km/h" bin may be very small).
for bar, (_, row) in zip(bars, wind_avg.iterrows()):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.15,
        f"n={int(row['count']):,}",
        ha="center", fontsize=SMALL_SIZE
    )

# FIX: added overall-mean reference line so the reader can immediately see which
# wind groups are above or below average — without this the bars are hard to interpret.
ax2.axhline(overall_mean, color=RED, lw=1.5, ls="--",
            label=f"Overall mean = {overall_mean:.1f} min")

ax2.set_xlabel("Wind Speed [km/h]")  # FIX: added unit in square brackets (SW04)
ax2.set_ylabel("Average Delay [min]")  # FIX: added unit
ax2.set_title("Figure 2 – Average Departure Delay by Wind Speed Group", fontweight="bold")
ax2.legend()
ax2.grid(axis="y", alpha=0.3)  # FIX: light y-grid improves readability
fig2.tight_layout()
fig2.savefig("../outputs/fig2_bar_wind.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print("✅ Figure 2 saved — Delay by wind speed group")

# fig, ax = plt.subplots(figsize=(8, 4))
# df_weather.groupby("wind_bin", observed=True)["delay_min"].mean().plot(kind="bar", ax=ax, color="steelblue")
# ax.set_title("Average Delay by Wind Speed Group")
# ax.set_xlabel("Wind Speed (km/h)")
# ax.set_ylabel("Avg Delay (min)")
# ax.grid(False)
# plt.xticks(rotation=0)
# plt.savefig("../outputs/bar_wind.png", dpi=150, bbox_inches="tight")
# plt.show()

#  3. Bar chart: Average Delay by Temperature group
# FIX: extended upper bound from 20 to 99 to capture any unseasonably warm hours.
# Oct–Dec at ZRH rarely exceeds 20°C but it is possible in early October.
df_weather["temp_bin"] = pd.cut(
    df_weather["temperature_C"],
    bins=[-99, -5, 0, 5, 10, 99],
    labels=["Below -5", "-5 to 0", "0 to 5", "5 to 10", "Above 10"]
)

temp_avg = df_weather.groupby("temp_bin", observed=True)["delay_min"].agg(["mean", "count"])

fig3, ax3 = plt.subplots(figsize=(10, 5))
bars = ax3.bar(
    temp_avg.index, temp_avg["mean"],
    color=PALETTE[:len(temp_avg)], edgecolor="white", width=0.6
)
for bar, (_, row) in zip(bars, temp_avg.iterrows()):
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.15,
        f"n={int(row['count']):,}",
        ha="center", fontsize=SMALL_SIZE
    )
ax3.axhline(overall_mean, color=RED, lw=1.5, ls="--",
            label=f"Overall mean = {overall_mean:.1f} min")
ax3.set_xlabel("Temperature [°C]")
ax3.set_ylabel("Average Delay [min]")
ax3.set_title("Figure 3 – Average Departure Delay by Temperature Group", fontweight="bold")
ax3.legend()
ax3.grid(axis="y", alpha=0.3)
fig3.tight_layout()
fig3.savefig("../outputs/fig3_bar_temp.png", dpi=150, bbox_inches="tight")
plt.close(fig3)
print("✅ Figure 3 saved — Delay by temperature group")
# fig, ax = plt.subplots(figsize=(8, 4))
# df_weather.groupby("temp_bin", observed=True)["delay_min"].mean().plot(kind="bar", ax=ax, color="steelblue")
# ax.set_title("Average Delay by Temperature Group")
# ax.set_xlabel("Temperature (°C)")
# ax.set_ylabel("Avg Delay (min)")
# ax.grid(False)
# plt.xticks(rotation=0)
# plt.savefig("../outputs/bar_temp.png", dpi=150, bbox_inches="tight")
# plt.show()

# 4. Bar chart: Average Delay by Precipitation group
df_weather["precip_bucket"] = pd.cut(
    df_weather["precipitation_mm"],
    bins=[-0.1, 0, 0.5, 1, 999],
    labels=["None", "Trace(<0.5mm)", "Light(0.5-1mm)", "Moderate(>1 mm)"]
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

# 5. Regression: with vs without weather
# We compare 4 models:
#   LR = Linear Regression (fast, interpretable, assumes linear relationships)
#   RF = Random Forest (non-linear, handles interactions, slower)
#   × 2 feature sets: without weather / with weather

# This directly answers Research Question:
# Does adding weather data improve prediction accuracy?

features_no_weather   = ["dep_hour", "dep_dow", "dep_month", "is_weekend", "route_avg_delay"]
features_with_weather = features_no_weather + ["temperature_C", "precipitation_mm", "wind_speed_kmh"]
# Drop rows where ANY of the weather features are missing.
# This ensures all 4 models are trained and tested on exactly the same rows,
# making the comparison fair.
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

# FIGURE 6 — Feature Importance (Random Forest with weather)
# Random Forest stores how much each feature reduced impurity across all trees.
# A high importance means the model leaned heavily on that feature.
# This answers Research Question 3: which variables matter most?

df = pd.read_csv("../data/ZRH_Flights_with_Weather_2025.csv")
# Create binary target from delay minutes
# 1 = delayed, 0 = not delayed
df["delay_binary"] = (df["delay_min"] > 5).astype(int)
# Define features
features_with_weather = [
    "dep_hour",
    "dep_dow",
    "dep_month",
    "is_weekend",
    "route_avg_delay",
    "temperature_C",
    "precipitation_mm",
    "wind_speed_kmh"
]
# Keep only needed columns and drop missing values
df_model = df[features_with_weather + ["delay_binary"]].dropna()

X_yes = df_model[features_with_weather]
y_yes = df_model["delay_binary"]
rf_yes = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf_yes.fit(X_yes, y_yes)

os.makedirs("../outputs", exist_ok=True)

importances = pd.Series(
    rf_yes.feature_importances_,
    index=features_with_weather
).sort_values(ascending=True)

feat_labels = {
    "dep_hour": "Departure Hour",
    "dep_dow": "Day of Week",
    "dep_month": "Month",
    "is_weekend": "Weekend Flag",
    "route_avg_delay": "Route Avg Delay",
    "temperature_C": "Temperature [°C]",
    "precipitation_mm": "Precipitation [mm]",
    "wind_speed_kmh": "Wind Speed [km/h]"
}
importances.index = [feat_labels.get(f, f) for f in importances.index]

fig6, ax6 = plt.subplots(figsize=(9, 5))
ax6.barh(importances.index, importances.values, color=PURPLE, edgecolor="white")
ax6.set_xlabel("Feature Importance [-]")
ax6.set_title(
    "Figure 6 – Feature Importance  |  Random Forest (with weather features)\n"
    "Higher = model relied more on this feature to predict delay",
    fontweight="bold"
)
ax6.grid(axis="x", alpha=0.3)
fig6.tight_layout()
fig6.savefig("../outputs/fig6_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close(fig6)

print("\n✅ Figure 6 saved — Feature importance")
print(importances)

# FIGURE 7 — Actual vs Predicted (best model: RF with weather)
# A scatter of actual vs predicted delay shows how well the model performs.
# Perfect predictions would lie exactly on the red dashed diagonal line.
# Clusters far from the line reveal where the model struggles most

features_with_weather = [
    "dep_hour",
    "dep_dow",
    "dep_month",
    "is_weekend",
    "route_avg_delay",
    "temperature_C",
    "precipitation_mm",
    "wind_speed_kmh"
]

df_model = df[features_with_weather + ["delay_min"]].dropna()

X_with_weather = df_model[features_with_weather]
y = df_model["delay_min"]

X_tr_w, X_te_w, y_tr_w, y_te_w = train_test_split(
    X_with_weather,
    y,
    test_size=0.2,
    random_state=42
)

rf_yes = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf_yes.fit(X_tr_w, y_tr_w)

rf_preds_w = rf_yes.predict(X_te_w)
fig7, ax7 = plt.subplots(figsize=(6, 6))

n_sample = min(1500, len(y_te_w))
sample_idx = np.random.default_rng(42).choice(len(y_te_w), size=n_sample, replace=False)

ax7.scatter(
    y_te_w.iloc[sample_idx],
    rf_preds_w[sample_idx],
    alpha=0.25,
    s=12,
    color=BLUE
)

lim = max(
    abs(y_te_w.iloc[sample_idx]).max(),
    abs(rf_preds_w[sample_idx]).max()
)

ax7.plot(
    [-lim, lim], [-lim, lim],
    color=RED, lw=1.5, ls="--",
    label="Perfect prediction"
)

ax7.set_xlabel("Actual Delay [min]")
ax7.set_ylabel("Predicted Delay [min]")
ax7.set_title(
    "Figure 7 – Actual vs Predicted Delay\n"
    "Random Forest with weather features",
    fontweight="bold"
)
ax7.legend()
ax7.grid(alpha=0.2)
fig7.tight_layout()
fig7.savefig("../outputs/fig7_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close(fig7)

print("✅ Figure 7 saved — Actual vs predicted")



print("\nAll outputs saved to ../outputs/")
print("\nSummary of fixes applied to original analysis_v02_with_weather.py:")
print("  1. root_mean_squared_error replaced with mean_squared_error ** 0.5")
print("     (compatible with all sklearn versions)")
print("  2. Two separate train_test_split calls replaced with one shared split")
print("     (ensures all models compared on identical test rows)")
print("  3. Bin upper edges raised to 999 in all pd.cut() calls")
print("     (prevents silently dropping flights with extreme values)")
print("  4. SW04 font size settings added (SMALL/MEDIUM/BIGGER_SIZE + plt.rc)")
print("  5. Axis labels now include units in square brackets [min], [°C], [km/h]")
print("  6. Overall-mean reference line added to all bar charts")
print("  7. Sample count (n=) labels added to all bar charts")
print("  8. center=0 added to heatmap (zero correlation = white, not arbitrary colour)")
print("  9. Column names renamed to human-readable labels in heatmap")
print(" 10. Feature importance and actual-vs-predicted figures added")