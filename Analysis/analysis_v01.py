import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from pathlib import Path

df = pd.read_csv("../data/ZRH_Cleaned_Flights_2025.csv")
df_model = df[df["is_outlier"] == False].copy()

Path("../outputs").mkdir(parents=True, exist_ok=True)

# 1. EDA: Basic overview
print("Shape:", df.shape)
print(df.describe())
print(df["delay_min"].describe())

# 2. Delay distribution
plt.figure(figsize=(10, 4))
sns.histplot(df[df["delay_min"].between(-30, 180)]["delay_min"], bins=60, kde=True)
plt.title("Departure Delay Distribution (ZRH)")
plt.xlabel("Delay (minutes)")
plt.savefig("../outputs/delay_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

#  3. Average delay by airline
top_airlines = df.groupby("airline_name")["delay_min"].agg(["mean", "count"])
top_airlines = top_airlines[top_airlines["count"] >= 50].sort_values("mean", ascending=False).head(15)

plt.figure(figsize=(10, 5))
top_airlines["mean"].plot(kind="bar")
plt.title("Average Delay by Airline (min 50 flights)")
plt.ylabel("Avg Delay (min)")
plt.xticks(rotation=45, ha="right")
plt.savefig("../outputs/delay_by_airline.png", dpi=150, bbox_inches="tight")
plt.show()

#  4. Delay by hour of day
plt.figure(figsize=(10, 4))
df.groupby("dep_hour")["delay_min"].mean().plot(marker="o")
plt.title("Average Delay by Departure Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Avg Delay (min)")
plt.savefig("../outputs/delay_by_hour.png", dpi=150, bbox_inches="tight")
plt.show()

#  5. Delay by day of week
dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
plt.figure(figsize=(8, 4))
df.groupby("dep_dow")["delay_min"].mean().rename(index=dict(enumerate(dow_labels))).plot(kind="bar")
plt.title("Average Delay by Day of Week")
plt.ylabel("Avg Delay (min)")
plt.xticks(rotation=0)
plt.savefig("../outputs/delay_by_dow.png", dpi=150, bbox_inches="tight")
plt.show()

# 6. Delay by month
plt.figure(figsize=(8, 4))
df.groupby("dep_month")["delay_min"].mean().plot(kind="bar")
plt.title("Average Delay by Month")
plt.ylabel("Avg Delay (min)")
plt.xticks(rotation=0)
plt.savefig("../outputs/delay_by_month.png", dpi=150, bbox_inches="tight")
plt.show()

#  7. Correlation heatmap
num_cols = ["delay_min", "dep_hour", "dep_dow", "dep_month", "is_weekend", "route_avg_delay"]
plt.figure(figsize=(8, 6))
sns.heatmap(df_model[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("../outputs/correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

# 8. Regression modelling
features_baseline = ["dep_hour", "dep_dow", "dep_month", "is_weekend"]
features_extended = features_baseline + ["route_avg_delay"]

X_base = df_model[features_baseline].dropna()
X_ext  = df_model[features_extended].dropna()
y      = df_model.loc[X_ext.index, "delay_min"]

X_base = X_base.loc[X_ext.index]  # align indices

X_tr_b, X_te_b, y_tr, y_te = train_test_split(X_base, y, test_size=0.2, random_state=42)
X_tr_e, X_te_e, _, _       = train_test_split(X_ext,  y, test_size=0.2, random_state=42)

def evaluate(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"\n── {name} ──")
    print(f"  MAE:  {mean_absolute_error(y_test, preds):.2f}")
    print(f"  RMSE: {root_mean_squared_error(y_test, preds):.2f}")
    print(f"  R²:   {r2_score(y_test, preds):.4f}")
    return model

lr_base = evaluate("Linear Regression – Baseline",  LinearRegression(), X_tr_b, X_te_b, y_tr, y_te)
lr_ext  = evaluate("Linear Regression – Extended",  LinearRegression(), X_tr_e, X_te_e, y_tr, y_te)
rf_ext  = evaluate("Random Forest – Extended",      RandomForestRegressor(n_estimators=100, random_state=42), X_tr_e, X_te_e, y_tr, y_te)

# 9. Feature importance (Random Forest)
importances = pd.Series(rf_ext.feature_importances_, index=features_extended).sort_values(ascending=False)
plt.figure(figsize=(7, 4))
importances.plot(kind="bar")
plt.title("Feature Importance – Random Forest")
plt.ylabel("Importance")
plt.xticks(rotation=30, ha="right")
plt.savefig("../outputs/feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()

#scatter plot just to verify
plt.figure(figsize=(10, 5))
plt.scatter(df_model["route_avg_delay"], df_model["delay_min"], alpha=0.1, s=5)
plt.xlabel("Route Average Delay (min)")
plt.ylabel("Actual Delay (min)")
plt.title("Route Avg Delay vs Actual Delay")
plt.ylim(-60, 200)
plt.xlim(-20, 100)
plt.savefig("../outputs/scatter_route_vs_delay.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nAll outputs saved to /outputs/")

#the random forest may create a bias of importance because of the many unique numerical values
# apparently that may be the reason why heap map has low correlation too
#indeed the scatter plot shows NO trend whatsoever. so that was the bias...
#The relationship is weak and largely structureless.
# The Random Forest was exploiting the many split points available in this continuous feature rather
# than a genuine predictive signal.
