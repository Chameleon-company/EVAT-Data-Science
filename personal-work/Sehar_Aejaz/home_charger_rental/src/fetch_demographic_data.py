
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time

# Set up Selenium driver 
driver = webdriver.Chrome()
wait = WebDriverWait(driver, 10)

# Base URL
base_url = "https://www.abs.gov.au/census/find-census-data/quickstats/2021/LGA11300"


# Storage for all town data
all_data = []

for town in correct_towns:
    try:
        driver.get(base_url)
        time.sleep(3)  

        # Attempt dropdown search unless auto-loaded (e.g. Brunswick - North)
        try:
            search_input = wait.until(EC.presence_of_element_located(
                (By.CSS_SELECTOR, 'input[placeholder="Enter a location, postcode or geography"]')))
            search_input.clear()
            search_input.send_keys(town)
            time.sleep(2)
            search_input.send_keys(Keys.ARROW_DOWN)
            search_input.send_keys(Keys.ENTER)
            time.sleep(4) 
        except Exception:
            # Skip dropdown if not applicable
            print(f"'{town}' likely auto-loaded, continuing without dropdown...")

        # Scrape data from the loaded page
        soup = BeautifulSoup(driver.page_source, "html.parser")
        rows = soup.select("tbody tr")

        data = {}
        for row in rows:
            th = row.find("th")
            td = row.find("td")
            if th and td:
                key = th.get_text(strip=True)
                value = td.get_text(strip=True)
                if key == "All private dwellings":
                    data["All Private Dwellings"] = value
                elif key == "Median weekly household income":
                    data["Median Weekly Household Income"] = value
                elif key == "Average number of motor vehicles per dwelling":
                    data["Average Motor Vehicles per Dwelling"] = value

        result = {
            "Town": town.replace(" - ", " (") + ")" if " - " in town else town,
            "All Private Dwellings": data.get("All Private Dwellings", "N/A"),
            "Median Weekly Household Income": data.get("Median Weekly Household Income", "N/A"),
            "Average Motor Vehicles per Dwelling": data.get("Average Motor Vehicles per Dwelling", "N/A")
        }

        all_data.append(result)
        print(f"✓ Scraped: {result}")

    except Exception as e:
        print(f"Error with {town}: {e}")

# Finish
driver.quit()


more_info = pd.DataFrame(all_data)
more_info.to_csv("Info_for_PCZ.csv", index = False)
