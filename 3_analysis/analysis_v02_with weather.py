import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from pathlib import Path


Path("../outputs").mkdir(parents=True, exist_ok=True)
import glob
for f in glob.glob("../outputs/*.png"):
    os.remove(f)
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

# FIX: added parse_dates so scheduled_utc is read as a proper datetime object
# and not as a plain string. This matters if we ever filter by date later.
df_weather = pd.read_csv(
    "../data/ZRH_Flights_with_Weather_2025.csv",
    parse_dates=["scheduled_utc"]
)

df_weather = df_weather[df_weather["is_outlier"] == False].copy()

# Further restrict to a realistic delay window of -60 to +300 minutes.
# Values outside this range are almost certainly data errors (e.g. revised time
# being the next day's flight), not genuine extreme delays.
df_weather = df_weather[df_weather["delay_min"].between(-60, 300)].copy()
print(f"Rows after filtering: {len(df_weather):,}")
print(f"Average delay: {df_weather['delay_min'].mean():.1f} min")

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
# 1. - Weather correlation heatmap
weather_cols = ["delay_min", "temperature_C", "precipitation_mm", "wind_speed_kmh"]
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

# 4. Bar chart: Average Delay by Precipitation group
# Previously used the old plotting style. Now fully consistent with Figs 2 & 3.
df_weather["precip_bucket"] = pd.cut(
    df_weather["precipitation_mm"],
    bins=[-0.1, 0, 0.5, 1, 999],
    labels=["None", "Trace (< 0.5 mm)", "Light (0.5–1 mm)", "Moderate (> 1 mm)"]
)
precip_avg = df_weather.groupby("precip_bucket", observed=True)["delay_min"].agg(["mean", "count"])

fig4, ax4 = plt.subplots(figsize=(9, 5))
bars = ax4.bar(
    precip_avg.index, precip_avg["mean"],
    color=PALETTE[:len(precip_avg)], edgecolor="white", width=0.6
)
for bar, (_, row) in zip(bars, precip_avg.iterrows()):
    ax4.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.15,
        f"n={int(row['count']):,}",
        ha="center", fontsize=SMALL_SIZE
    )
ax4.axhline(overall_mean, color=RED, lw=1.5, ls="--",
            label=f"Overall mean = {overall_mean:.1f} min")
ax4.set_xlabel("Precipitation Level")
ax4.set_ylabel("Average Delay [min]")
ax4.set_title("Figure 4 – Average Departure Delay by Precipitation Level", fontweight="bold")
ax4.legend()
ax4.grid(axis="y", alpha=0.3)
plt.xticks(rotation=0)
fig4.tight_layout()
fig4.savefig("../outputs/fig4_bar_precip.png", dpi=150, bbox_inches="tight")
plt.close(fig4)  # FIX: was plt.show() — blocks execution in batch mode
print("✅ Figure 4 saved — Delay by precipitation level")

# 5. Regression: with vs without weather
# We compare 4 models:
#   LR = Linear Regression (fast, interpretable, assumes linear relationships)
#   RF = Random Forest (non-linear, handles interactions, slower)
#   × 2 feature sets: without weather / with weather

# This directly answers Research Question:
# Does adding weather data improve prediction accuracy?

# Reuse df_weather (already filtered). Drop rows with any missing weather feature.
# This ensures all 4 models are trained/tested on exactly the same rows (fair comparison).
FEATURES_BASE    = ["dep_hour", "dep_dow", "dep_month", "is_weekend", "route_avg_delay"]
FEATURES_WEATHER = FEATURES_BASE + ["temperature_C", "precipitation_mm", "wind_speed_kmh"]

df_w  = df_weather[FEATURES_WEATHER + ["delay_min"]].dropna()
X_no  = df_w[FEATURES_BASE]
X_yes = df_w[FEATURES_WEATHER]
y_w   = df_w["delay_min"]

X_tr_n, X_te_n, y_tr_w, y_te_w = train_test_split(X_no,  y_w, test_size=0.2, random_state=42)
X_tr_y, X_te_y, _,      _      = train_test_split(X_yes, y_w, test_size=0.2, random_state=42)

print("\n═══ Model Comparison ═══")
evaluate("LR – Without Weather", LinearRegression(), X_tr_n, X_te_n, y_tr_w, y_te_w)
evaluate("LR – With Weather",    LinearRegression(), X_tr_y, X_te_y, y_tr_w, y_te_w)
evaluate("RF – Without Weather", RandomForestRegressor(n_estimators=100, random_state=42), X_tr_n, X_te_n, y_tr_w, y_te_w)

# Keep the fitted RF+Weather model — reused for Figures 6 & 7 (FIX #1 & #2)
rf_weather_model, rf_weather_preds = evaluate(
    "RF – With Weather",
    RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    X_tr_y, X_te_y, y_tr_w, y_te_w
)


# ── FIGURE 6 — Feature Importance from Regression RF
# FIX #1: was RandomForestClassifier on binary target — now uses the
#          regression RF fitted above, consistent with the model comparison.
# FIX #2: was re-reading raw CSV without filters — now uses already-filtered data.
feat_labels = {
    "dep_hour":         "Departure Hour",
    "dep_dow":          "Day of Week",
    "dep_month":        "Month",
    "is_weekend":       "Weekend Flag",
    "route_avg_delay":  "Route Avg Delay",
    "temperature_C":    "Temperature [°C]",
    "precipitation_mm": "Precipitation [mm]",
    "wind_speed_kmh":   "Wind Speed [km/h]"
}
importances = pd.Series(
    rf_weather_model.feature_importances_,
    index=FEATURES_WEATHER
).sort_values(ascending=True)
importances.index = [feat_labels.get(f, f) for f in importances.index]

fig6, ax6 = plt.subplots(figsize=(9, 5))
ax6.barh(importances.index, importances.values, color=PURPLE, edgecolor="white")
ax6.set_xlabel("Feature Importance [-]")
ax6.set_title(
    "Figure 6 – Feature Importance  |  Random Forest Regressor (with weather)\n"
    "Higher = model relied more on this feature to predict departure delay",
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

rf_preds_w = rf_weather_preds  # already computed when we called evaluate()

fig7, ax7 = plt.subplots(figsize=(6, 6))

n_sample = min(1500, len(y_te_w))
sample_idx = np.random.default_rng(42).choice(len(y_te_w), size=n_sample, replace=False)

ax7.scatter(
    y_te_w.iloc[sample_idx],
    rf_preds_w[sample_idx],
    alpha=0.25, s=12, color=BLUE
)
lim = max(abs(y_te_w.iloc[sample_idx]).max(), abs(rf_preds_w[sample_idx]).max())
ax7.plot([-lim, lim], [-lim, lim], color=RED, lw=1.5, ls="--", label="Perfect prediction")
ax7.set_xlabel("Actual Delay [min]")
ax7.set_ylabel("Predicted Delay [min]")
ax7.set_title(
    "Figure 7 – Actual vs Predicted Delay\nRandom Forest with weather features",
    fontweight="bold"
)
ax7.legend()
ax7.grid(alpha=0.2)
fig7.tight_layout()
fig7.savefig("../outputs/fig7_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close(fig7)
print("✅ Figure 7 saved — Actual vs predicted")



print("\nAll outputs saved to ../outputs/")