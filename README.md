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
 
>**Note on Findings**: Our findings and answers to the above research questions are addressed in details in the documentation file.
---
 
## What the Code Does
 
**1. Data Acquisition** — Two scripts fetch flight data from the AeroDataBox API via Apify: one for departed flights, one including cancellations. Results are exported as raw CSVs.
 
**2. Cleaning and Transformation** — `cip_pipeline.py` merges the raw files, parses nested JSON columns, computes `delay_min`, flags outliers, derives temporal and route features, joins Open-Meteo weather data, and exports the final dataset.
 
**3. Analysis and Visualisation** — `analysis_v02_with_weather.py` produces weather correlation and delay-by-group charts using matplotlib and seaborn, following SW04 formatting conventions.
 
**4. Predictive Modelling** — Four models (Linear Regression and Random Forest × Baseline and +Weather) are compared on an 80/20 split. Best result: **LR +Weather** (MAE = 7.07 min, R² = 0.066). Route average delay is the strongest predictor.
 
>**Note on R²**: Best model explains ~6.6% of delay variance.
>Expected — most delays at hub airports like ZRH come from factors not in pre-departure data (e.g., inbound aircraft delay, crew rotation, reactionary network delays).


**5. Interactive Dashboard** — `dashboard.py` is a Streamlit + Plotly dashboard with four tabs (Delay Explorer, Weather Analysis, Predictive Modelling, Live Predictor).
 
```bash
streamlit run 4_dashboard/dashboard.py
```
The app opens automatically at `http://localhost:8501`.
To disable auto-open: `streamlit run 4_dashboard/dashboard.py --server.headless true`
---

## Live Interactive Dashboard
Instead of a static report, you can access the full predictive analytics suite and interactive visualizations directly via Streamlit Cloud:

**https://zrh-flight-delay.streamlit.app**

### Key Interactive Features:
* **Real-Time Inference:** Input live weather and flight data to receive instant delay predictions based on our trained machine learning models.
* **Dynamic Data Exploration:** Explore historical trends, feature importances ($R^2$, RMSE), and flight distributions through interactive Plotly charts.
* **Integrated Pipeline:** Experience the end-to-end data science lifecycle, from automated API-driven acquisition (AeroDataBox & Open-Meteo) to final model deployment.
---

## Team Contributions

This project was developed using a highly collaborative approach. Every major step, from defining the pipeline architecture to debugging the final code, was handled collectively. **Furthermore, the final project documentation, the structuring of this README file, and the overall management of the GitHub repository were built, edited, and finalized completely together as a joint effort.**

While we all reviewed and contributed to the entire codebase and deliverables, our individual focus areas for technical leadership were distributed as follows:

* **Silan Cihaner:**
  * **Primary Focus:** Data Acquisition & Engineering
  * **Key Contributions:** Led the API integration process, successfully extracting flight and weather data from AeroDataBox and Open-Meteo. Handled RapidAPI rate limits, implemented robust `requests` loops with try-except blocks, and managed the initial data extraction and SQLite caching mechanisms.

* **Hanieh Jebeli:**
  * **Primary Focus:** Data Preprocessing & Dashboard Integration
  * **Key Contributions:** Managed the core data cleaning pipeline using `pandas`, including the IQR methodology for extreme delay outliers and complex feature engineering (e.g., calculating `route_avg_delay`). Took the lead on structuring the final interactive Streamlit dashboard application.

* **Elif Gürçinar:**
  * **Primary Focus:** Exploratory Analysis & Predictive Modeling
  * **Key Contributions:** Directed the statistical analysis and exploratory data visualizations. Evaluated the machine learning models (Linear Regression and Random Forest), extracted feature importances, and ensured the models effectively addressed the core research questions.
 
    
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
├── 1_data_acquisition
├── 2_cleaning_transformation  
├── 3_analysis
├── 4_dashboard
├── 5_final_documentation
├── data
├── outputs
├── .gitignore
├── requirements.txt
└── README.md
````

---

## Academic Context
This project was developed as part of the **Master of Science in Data Science** program at the **Lucerne University of Applied Sciences and Arts (HSLU)**, specifically for the *Data Collection, Integration and Preprocessing (CIP)* course. 

* **Author / Contributor:** Hanieh Jebeli · Elif Gürçinar · Silan Cihaner (FS2026 - Group 111)
* **Methodology:** End-to-end data pipeline engineering, predictive modeling (Linear Regressio & Random Forest), and interactive dashboard deployment.
* **Primary Data Sources:** Aviation metrics via AeroDataBox API & localized meteorological data via Open-Meteo API.
 

