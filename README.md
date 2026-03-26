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


# CIP_Group_Project_111

## Data Collection Integration and Preprocessing: Analysis and Prediction of Flight Departure Delays at Zurich Airport (ZRH)

This project was developed as part of the **CIP course** and focuses on the analysis and prediction of flight departure delays at **Zurich Airport (ZRH)**.

The project combines data preparation, exploratory analysis, and predictive modeling in Python. The goal is to better understand delay patterns and to evaluate whether operational flight information can be used to predict departure delays.

## CSV Data Mapping & Solutions

Here is how the CSV columns relate to the main research goals of the project.

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
  - aircraft model
  - cargo status

## Data and files

The project is based on the following datasets:

- `ZRH_Departure_Delays_2025.csv`
- `ZRH_All_Flights_Inc_Canceled_2025.csv`

Both files share the same structure and contain the following main columns:

- `departure`
- `arrival`
- `number`
- `status`
- `codeshareStatus`
- `isCargo`
- `aircraft`
- `airline`
- `callSign`

Several fields contain nested information stored as strings. These values are parsed during preprocessing to extract the variables needed for analysis.

## Research questions

The project is built around the following research questions:

1. **What temporal patterns can be observed in departure delays at Zurich Airport?**  
   This includes differences by hour of day, weekday, and month.

2. **How do routes and destinations influence flight delays?**  
   The project compares delay behavior across different destination airports.

3. **Can departure delays be predicted using operational flight information?**  
   A simple machine learning model is used to test whether variables such as airline, aircraft type, route, and cargo status can explain delay patterns.

## What the code does

The workflow of the project is divided into three main stages.

### 1. Data cleaning and preprocessing

The preprocessing script:

- loads and merges the two CSV files
- removes duplicate entries and codeshare duplicates
- parses nested columns stored as strings
- extracts useful information such as:
  - scheduled departure time
  - revised departure time
  - destination airport
  - airline name
  - aircraft model
  - gate
- calculates delay in minutes
- checks missing values and datatypes
- identifies outliers
- creates additional features for analysis and modeling

The cleaned dataset is then exported for the next project steps.

### 2. Analysis and visualization



### 3. Prediction model





## Project structure

```bash
CIP_Group_Project_111/
│
├── data_cleaning.py
├── analysis_visualization.py
├── modeling.py
├── requirements.txt
├── README.md
├── ZRH_Departure_Delays_2025.csv
├── ZRH_All_Flights_Inc_Canceled_2025.csv
└── ZRH_Cleaned_Flights_2025.csv
