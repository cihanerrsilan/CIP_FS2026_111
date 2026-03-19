
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from selenium.common.exceptions import TimeoutException

print("✈️ RadarBox Flight Data Bot is Starting...")

# --- 1. CREATE DATE LIST (Date Range) ---
start_date = "2025-10-01"
end_date = "2025-10-03"  # Set 3 days for testing
dates = pd.date_range(start=start_date, end=end_date)

all_flights = []

# --- 2. GHOST BROWSER SETTINGS ---
options = uc.ChromeOptions()
options.add_argument("--incognito")
options.add_argument("--disable-blink-features=AutomationControlled")
# Mimic a real human
options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

driver = uc.Chrome(options=options)

try:
    # --- 3. LOOP BEGINS ---
    for date in dates:
        formatted_date = date.strftime("%Y-%m-%d")  # e.g.: 2025-10-01

        # DYNAMIC URL: RadarBox Zurich (ZRH/LSZH) Departures
        dynamic_url = f"https://www.radarbox.com/data/airports/LSZH/departures?date={formatted_date}"

        print(f"\n🌐 Opening page: {dynamic_url}")
        driver.get(dynamic_url)

        # Human-like wait (Critical to bypass bot protection)
        wait_time = random.uniform(8, 14)
        print(f"⏳ Waiting {wait_time:.1f} seconds...")
        time.sleep(wait_time)

        try:
            # Wait max 30 seconds for the table or flight list to load
            WebDriverWait(driver, 80).until(EC.presence_of_element_located((By.TAG_NAME, "table")))
            print("✅ Table found, extracting data...")

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            tables = soup.find_all('table')

            if tables:
                flight_table = tables[0]
                rows = flight_table.find_all('tr')

                for row in rows[1:]:  # Skip the header
                    cells = row.find_all(['td', 'th'])
                    cell_texts = [cell.text.strip() for cell in cells]
                    if cell_texts and any(cell_texts):
                        # Add the date to the end of the row to know which day it belongs to
                        cell_texts.append(formatted_date)
                        all_flights.append(cell_texts)
                print(f"👍 Data for {formatted_date} added to the main list.")
            else:
                print("⚠️ No table found on the page.")

        except TimeoutException:
            # IF THE SITE IS SLOW OR BLOCKED US, DON'T CRASH THE CODE, SKIP TO NEXT DAY!
            print(f"⚠️ TIMEOUT: Table for {formatted_date} did not load. Skipping to the next day...")
            continue

            # --- 4. COMBINE AND SAVE DATA ---
    if all_flights:
        df = pd.DataFrame(all_flights)
        filename = "zrh_radarbox_flights_oct_2025.csv"
        df.to_csv(filename, index=False, header=False)
        print(f"\n🎉 GREAT! A total of {len(df)} flights were saved to '{filename}'.")
    else:
        print("\n❌ Unfortunately, no data could be extracted for any date. The site might have blocked us.")

except Exception as e:
    print(f"🔥 A critical error occurred: {e}")

finally:
    print("Closing the browser.")
    driver.quit()