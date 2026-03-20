# CIP_Group_Project_FIRSTTASK
Data Collection Integration and Preprocessing: Analysis and Prediction of Flight Departure Delays at Zurich Airport (ZRH)

For new_selenium.pyd 
## Data Acquisition Challenges & Simulation Strategy

### 1. The Roadblock: Anti-Scraping Mechanisms and Paywalls
During the data acquisition phase for Zurich Airport (ZRH) historical flight departures (Oct-Dec 2025), we encountered significant barriers typical in real-world data science projects:
* **Premium API Paywalls:** Official aviation data providers (such as Flightradar24, AviationStack, and RapidAPI) restrict access to historical scheduled and actual departure times behind steep enterprise paywalls ($50 - $100+/month). 
* **Missing Features in Open-Source APIs:** Free alternatives like the OpenSky Network API provide actual radar detection times (`firstSeen`) but completely lack the `Scheduled Departure` time feature, which is mathematically essential for calculating our target variable: **Delay**.
* **Dynamic Web Scraping Limitations:** We successfully engineered a Python Selenium & BeautifulSoup crawler to scrape historical data from flight tracking websites. However, the platform employed a dynamic data-masking mechanism for non-premium users. While it successfully returned the accurate flight network skeleton for our requested dates (real airlines, valid flight numbers, correct destinations, and scheduled times), it forcibly overwrote the date values to the "current day" and masked the `Actual Departure` times with null values.

### 2. Why We Chose Data Simulation (Mitigation Strategy)
Instead of abandoning the research questions or violating the project constraints by manually downloading CSVs, we pivoted to an advanced Data Imputation and Simulation strategy, directly aligning with the CIP project guidelines which permit data simulation. 

This approach is highly effective and scientifically sound because:
1. **The Structural Skeleton is 100% Real:** Our scraped dataset contains 2,760 rows of authentic ZRH flight schedules. The relationships between airlines, routes, and scheduled flight frequencies are perfectly preserved.
2. **Predictive Modeling Focus:** The primary goal of this project is to apply Python programming, data transformation, and machine learning pipelines. Simulating the target variable (`Delay`) allows us to showcase advanced data preprocessing and modeling techniques without being bottlenecked by commercial paywalls.

### 3. Simulation Methodology
To generate the missing `Actual Departure` times and create a realistic target variable, we apply the following methodology using `pandas` and `numpy`:
* **Data Cleansing:** We programmatically strip the masked "current day" strings from the scheduled times and merge them with our crawler's verified historical date index to reconstruct the true `Datetime` objects.
* **Statistical Delay Generation:** Flight delays are not normally distributed; they typically follow a heavily right-skewed distribution (most flights depart on time, a few suffer minor delays, and very few experience severe delays). We utilize statistical distributions (such as the Gamma or Exponential distribution via `numpy.random`) to synthetically generate realistic delay minutes.
* **Target Variable Creation:** The simulated delay minutes are added to the authentic `Scheduled Departure` times to engineer the final `Actual Departure` timestamps, providing a complete and robust dataset ready for exploratory data analysis (EDA) and predictive modeling.
