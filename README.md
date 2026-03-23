# CIP_Group_Project_111
Data Collection Integration and Preprocessing: Analysis and Prediction of Flight Departure Delays at Zurich Airport (ZRH)

## CSV Data Mapping & Solutions

Here is how your CSV columns match your research goals:

Target Variable (Delay in Minutes):

Data Source: The departure column.

How to solve: Inside the departure column, you have scheduledTime and revisedTime. By calculating the difference between these two timestamps (Revised - Scheduled), you can create the "Delay" column which is your target for regression.

Route-Level Analysis (Research Question 2):

Data Source: The arrival column.

How to solve: This column contains the destination airport's IATA/ICAO codes (e.g., 'HER', 'GRU'). You can group your data by these codes to calculate the "historical average delay" for each specific route.

Operational Predictors (Research Question 3):

Data Source: airline, aircraft, and isCargo columns.

How to solve: You have specific aircraft models (e.g., Airbus A320), airline names (e.g., Edelweiss Air), and the cargo status. These are the "features" (independent variables) you planned to use in your model.
