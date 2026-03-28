[//]: # ()
[//]: # (# CIP_Group_Project_111)

[//]: # (Data Collection Integration and Preprocessing: Analysis and Prediction of Flight Departure Delays at Zurich Airport &#40;ZRH&#41;)

[//]: # ()
[//]: # (## CSV Data Mapping & Solutions)

[//]: # ()
[//]: # (Here is how your CSV columns match your research goals:)

[//]: # ()
[//]: # (Target Variable &#40;Delay in Minutes&#41;:)

[//]: # ()
[//]: # (Data Source: The departure column.)

[//]: # ()
[//]: # (How to solve: Inside the departure column, you have scheduledTime and revisedTime. By calculating the difference between these two timestamps &#40;Revised - Scheduled&#41;, you can create the "Delay" column which is your target for regression.)

[//]: # ()
[//]: # (Route-Level Analysis &#40;Research Question 2&#41;:)

[//]: # ()
[//]: # (Data Source: The arrival column.)

[//]: # ()
[//]: # (How to solve: This column contains the destination airport's IATA/ICAO codes &#40;e.g., 'HER', 'GRU'&#41;. You can group your data by these codes to calculate the "historical average delay" for each specific route.)

[//]: # ()
[//]: # (Operational Predictors &#40;Research Question 3&#41;:)

[//]: # ()
[//]: # (Data Source: airline, aircraft, and isCargo columns.)

[//]: # ()
[//]: # (How to solve: You have specific aircraft models &#40;e.g., Airbus A320&#41;, airline names &#40;e.g., Edelweiss Air&#41;, and the cargo status. These are the "features" &#40;independent variables&#41; you planned to use in your model.)

[//]: # ()


# ✈️ Flight Departure Delay Analysis – Zurich Airport (ZRH)
### CIP Group Project 111 | FS2026
**Hanieh Jebeli · Elif Gürçinar · Silan Cihaner**

## Overview
 
This project was developed as part of the **CIP course** and focuses on the analysis and prediction of flight departure delays at **Zurich Airport (ZRH)**.
The project combines data acquisition via API, data cleaning and enrichment, exploratory analysis, predictive modelling, and an interactive Streamlit dashboard — all implemented in Python. The goal is to better understand delay patterns and to evaluate whether operational flight information available before departure can be used to predict departure delays.
---
## Dataset at a Glance
 
| Metric | Value |
|---|---|
| Observation period | 01 October – 30 December 2025 |
| Total flights (after cleaning) | 29,896 |
| Unique airlines | 209 |
| Unique routes | 445 |
| Average departure delay | 5.44 min |
| On-time rate (delay ≤ 0 min) | 55.6 % |
| Significantly delayed (> 15 min) | 12.2 % |
 
---
## CSV Data Mapping & Solutions
 
Here is how the raw CSV columns relate to the main research goals of the project.
 
### Target Variable: Delay in Minutes
- **Data source:** `departure`
- **How it is used:**
  The `departure` column contains both `scheduledTime` and `revisedTime`.
  By calculating the difference between these two timestamps, we create the **delay in minutes**, which is the target variable for the prediction task.
 
### Route-Level Analysis
- **Data source:** `arrival`
- **How it is used:**
  The `arrival` column includes destination airport information such as IATA and ICAO codes.
  These values are used to group flights by route and calculate historical average delays for each destination.
 
### Operational Predictors
- **Data source:** `airline`, `aircraft`, `isCargo`
- **How it is used:**
  These columns provide important operational variables for the prediction model, including:
  - airline name and code
  - aircraft model and family (narrow-body / wide-body / regional)
  - cargo status
 
---
## Data and Files
 
The project is based on the following datasets:
 
- `ZRH_Departure_Delays_2025.csv` — commercial departures with actual departure times
- `ZRH_All_Flights_Inc_Canceled_2025.csv` — same structure, including cancelled flights
- `ZRH_Flights_with_Weather_2025.csv` — cleaned and enriched dataset including hourly weather from Open-Meteo
 
Both raw files share the same structure and contain the following main columns:
 
| Column | Description |
|---|---|
| `departure` | Nested object with `scheduledTime`, `revisedTime`, `gate` |
| `arrival` | Nested object with destination IATA, ICAO, name, country |
| `number` | Flight number (e.g. `LX 1234`) |
| `status` | Flight status (Departed, Cancelled, Expected, …) |
| `codeshareStatus` | Whether the flight is an operator or codeshare |
| `isCargo` | Boolean — cargo flight flag |
| `aircraft` | Nested object with model name (e.g. `Airbus A320`) |
| `airline` | Nested object with airline name and IATA code |
| `callSign` | ATC call sign |
 
Several fields contain nested information stored as strings. These values are parsed during preprocessing to extract the variables needed for analysis.
 
---
## Research Questions
 
1. How accurately can departure delay be predicted using operational flight information?
2. Does adding weather data improve prediction accuracy?
3. Which variables most strongly influence departure delay?
 
---
 
## What the Code Does
 
**1. Data Acquisition** — Two scripts fetch flight data from the AeroDataBox API via Apify: one for departed flights, one including cancellations. Results are exported as raw CSVs.
 
**2. Cleaning and Transformation** — `cip_pipeline_final.py` merges the raw files, parses nested JSON columns, computes `delay_min`, flags outliers, derives temporal and route features, joins Open-Meteo weather data, and exports the final dataset.
 
**3. Analysis and Visualisation** — `analysis_v02_with_weather.py` produces weather correlation and delay-by-group charts using matplotlib and seaborn, following SW04 formatting conventions.
 
**4. Predictive Modelling** — Four models (Linear Regression and Random Forest × Baseline and +Weather) are compared on an 80/20 split. Best result: **LR +Weather** (MAE = 7.07 min, R² = 0.066). Route average delay is the strongest predictor.
 
**5. Interactive Dashboard** — `streamlit_app.py` is a Streamlit + Plotly dashboard with four tabs (Delay Explorer, Weather Analysis, Predictive Modelling, Live Predictor).
 
```bash
streamlit run streamlit_app.py
```
 
---
## Requirements
 
```
pandas numpy scipy matplotlib seaborn scikit-learn plotly streamlit python-dotenv requests
```
 
```bash
pip install -r requirements.txt
```
 
---

````
CIP_FS2026_111/
├── 1_data_acquisition/
├── 2_cleaning_transformation  
├── 3_analysis
├── 4_dashboard
├── data
├── outputs
├── .gitignore
├── requirements.txt
└── README.md

 

