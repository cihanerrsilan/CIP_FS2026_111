from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import io  # ADDED: This will fix the Pandas file path error

print("Starting Selenium, please wait...")

# Browser settings
chrome_options = Options()
# chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-blink-features=AutomationControlled")

# Initialize ChromeDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# Generate all days (92 days) between Oct 1 and Dec 31, 2025
start_date = "2025-10-01"
end_date = "2025-12-31"
date_list = pd.date_range(start=start_date, end=end_date).strftime('%Y-%m-%d').tolist()

print(f"A {len(date_list)}-day data scraping marathon is beginning!")
print("This process might take about 10-15 minutes, sit back and relax...\n")

all_flights_data = []

try:
    for target_date in date_list:
        url = f"https://www.flightera.net/en/airport/Zurich/LSZH/departure/{target_date}"

        driver.get(url)

        # Wait randomly between 3 and 6 seconds
        sleep_time = random.uniform(3, 6)
        time.sleep(sleep_time)

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        flight_table = soup.find('table')

        if flight_table:
            # FIXED: Wrap the HTML string in io.StringIO()
            html_string = str(flight_table)
            df_day = pd.read_html(io.StringIO(html_string))[0]

            # Add the date to the data
            df_day['Date'] = target_date

            all_flights_data.append(df_day)
            print(f"[SUCCESS] {target_date}: {len(df_day)} flights found and added.")
        else:
            print(f"[WARNING] {target_date}: Table not found (Could be an empty day or bot protection).")

except Exception as e:
    print(f"\n[ERROR] An unexpected error occurred during the loop: {e}")

finally:
    driver.quit()
    print("\nSelenium session closed.")

    if len(all_flights_data) > 0:
        print("Combining all daily data into a single file...")
        final_df = pd.concat(all_flights_data, ignore_index=True)

        csv_filename = "ZRH_All_Departures_Oct_Dec_2025.csv"
        final_df.to_csv(csv_filename, index=False)

        print(f"\n🎉 PERFECT! Successfully scraped a total of {len(final_df)} flight records.")
        print(f"The data has been saved as '{csv_filename}'.")
    else:
        print("Unfortunately, no data could be scraped.")