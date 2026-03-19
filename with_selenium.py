from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time

print("Web Scraping Bot is waking up, please wait...")

# --- Configuration ---
# FlightRadar24 Zurich Airport page
TARGET_URL = "https://www.flightradar24.com/data/airports/zrh"

# --- Setup Chrome options to avoid detection ---
options = Options()
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
# options.add_argument("--headless")  # Headless modda çalıştırmak isterseniz yorumu kaldırın

# --- Initialize the driver ---
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

print(f"Navigating to target URL: {TARGET_URL}")
driver.get(TARGET_URL)

# --- Wait for the page to load and find the departures table or link ---
try:
    # Önce sayfanın yüklenmesini bekle (body etiketi)
    WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    print("Page loaded successfully.")

    # Sayfada "Departures" linkini bul ve tıkla (genellikle böyle bir sekme vardır)
    # Farklı olasılıkları deneyelim:
    try:
        departures_link = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.LINK_TEXT, "Departures"))
        )
        departures_link.click()
        print("Clicked on 'Departures' link.")
        time.sleep(3)  # Tablonun yüklenmesi için bekle
    except:
        print("No 'Departures' link found, trying to find table directly.")

    # Şimdi tabloyu bulmaya çalış
    WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "table")))
    print("Table found.")
except Exception as e:
    print(f"Error: Page did not load a table within the time limit. Exception: {e}")
    driver.quit()
    exit()

# --- Get page source and parse with BeautifulSoup ---
soup = BeautifulSoup(driver.page_source, 'html.parser')

# --- Find all tables on the page ---
tables = soup.find_all('table')

if len(tables) > 0:
    # Genellikle uçuş verileri ilk tablodadır, ancak birden fazla tablo olabilir
    # Hangi tablonun doğru olduğunu anlamak için başlıkları inceleyebiliriz
    flight_table = None
    for table in tables:
        # Eğer tablo içinde "Flight", "Time", "Destination" gibi kelimeler varsa
        if any(keyword in str(table) for keyword in ["Flight", "Time", "Destination", "Airlines"]):
            flight_table = table
            break
    if flight_table is None:
        flight_table = tables[0]  # İlk tabloyu dene

    rows = flight_table.find_all('tr')

    data = []
    for row in rows:
        cells = row.find_all(['td', 'th'])
        cell_texts = [cell.text.strip() for cell in cells]
        if cell_texts:
            data.append(cell_texts)

    if len(data) > 1:
        # Başlık satırını bulmaya çalış (genellikle ilk satır th içerir)
        # Eğer ilk satırda th varsa, onu başlık olarak kullan
        if rows and rows[0].find_all('th'):
            df = pd.DataFrame(data[1:], columns=data[0])
        else:
            # Yoksa sütunları varsayılan isimlerle adlandır
            df = pd.DataFrame(data[1:], columns=[f"Column_{i}" for i in range(len(data[0]))])

        print("\nSUCCESS! Here are the first 5 rows of scraped data:")
        print(df.head())

        # Save data to CSV
        filename = "zrh_flights_flightradar24.csv"
        df.to_csv(filename, index=False)
        print(f"\nGreat! Data successfully saved to '{filename}'")
    else:
        print("Table found but it appears empty.")
else:
    print("No HTML tables found on the page. The site structure might be different.")

# --- Close the browser ---
driver.quit()