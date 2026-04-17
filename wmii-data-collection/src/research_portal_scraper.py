from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from pathlib import Path
import time
import csv

DATA_DIR = Path(__file__).parent.parent / "data"


class ResearchPortalScraper:
    def __init__(self, headless=False):
        self.headless = headless
        self.driver = None
        self.wait = None
        self.all_scientists = []

    def setup_driver(self):
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--window-size=1920,1080")
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.wait = WebDriverWait(self.driver, 20)
        print("Chrome driver initialized")

    def navigate_to_portal(self):
        url = (
            "https://researchportal.amu.edu.pl/globalResultList.seam"
            "?q=&oa=false&r=author&tab=PEOPLE&conversationPropagation=begin&lang=pl"
            "&qp=openAccess%3Dfalse"
        )
        print(f"Navigating to: {url}")
        self.driver.get(url)
        time.sleep(3)
        print("Page loaded")

    def wait_for_dynamic_content(self):
        try:
            self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, "ui-treenode-content")))
            print("Filter panel detected")
            time.sleep(2)
            return True
        except TimeoutException:
            print("Timeout waiting for filter panel")
            return False

    def expand_parent_node(self, parent_text="Szkoła Nauk Ścisłych"):
        try:
            print(f"Looking for parent node: '{parent_text}'...")
            for node in self.driver.find_elements(By.CLASS_NAME, "ui-treenode-content"):
                try:
                    label = node.find_element(By.CLASS_NAME, "ui-treenode-label")
                    if parent_text not in label.text:
                        continue
                    print(f"Found: {label.text}")
                    toggler = node.find_element(By.CLASS_NAME, "ui-tree-toggler")
                    if toggler.get_attribute("aria-expanded") == "false":
                        self.driver.execute_script("arguments[0].scrollIntoView(true);", toggler)
                        time.sleep(0.5)
                        toggler.click()
                        print("Parent node expanded")
                        time.sleep(2)
                    else:
                        print("Parent node already expanded")
                    return True
                except NoSuchElementException:
                    continue
            print(f"Could not find parent node: {parent_text}")
            return False
        except Exception as e:
            print(f"Error expanding parent node: {e}")
            return False

    def find_and_click_faculty_checkbox(self):
        try:
            print("Looking for 'Wydział Matematyki i Informatyki' checkbox...")
            nodes = self.driver.find_elements(By.CLASS_NAME, "ui-treenode-content")
            print(f"Found {len(nodes)} tree nodes")
            for node in nodes:
                try:
                    label = node.find_element(By.CLASS_NAME, "ui-treenode-label")
                    if "Wydział Matematyki i Informatyki" not in label.text:
                        continue
                    print(f"Found target: {label.text}")
                    checkbox = node.find_element(By.CSS_SELECTOR, "input[type='checkbox']")
                    self.driver.execute_script("arguments[0].scrollIntoView(true);", checkbox)
                    time.sleep(0.5)
                    checkbox.click()
                    print("Checkbox clicked")
                    time.sleep(1)
                    return True
                except NoSuchElementException:
                    continue
            print("Could not find faculty checkbox")
            return False
        except Exception as e:
            print(f"Error finding checkbox: {e}")
            return False

    def click_filter_button(self):
        try:
            print("Looking for filter button...")
            try:
                btn = self.driver.find_element(By.ID, "applySearchFiltersButton")
            except NoSuchElementException:
                btn = next(
                    (b for b in self.driver.find_elements(By.TAG_NAME, "button") if "Filtruj" in b.text),
                    None,
                )
                if not btn:
                    print("Could not find filter button")
                    return False
            self.driver.execute_script("arguments[0].scrollIntoView(true);", btn)
            time.sleep(0.5)
            btn.click()
            print("Filter applied")
            time.sleep(5)
            return True
        except Exception as e:
            print(f"Error clicking filter button: {e}")
            return False

    def extract_scientists_data(self):
        scientists = []
        try:
            print("\nExtracting scientists from current page...")
            self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, "ui-dataview-row")))
            rows = self.driver.find_elements(By.CLASS_NAME, "ui-dataview-row")
            print(f"Found {len(rows)} scientists on page")

            for idx, row in enumerate(rows):
                try:
                    d = {}
                    try:
                        link = row.find_element(By.CSS_SELECTOR, "a.authorNameLink")
                        d["profile_url"] = link.get_attribute("href")
                        d["profile_id"] = d["profile_url"].split("/")[-1].split("?")[0]
                    except NoSuchElementException:
                        d["profile_url"] = None
                        d["profile_id"] = None

                    try:
                        d["academic_title"] = row.find_element(
                            By.CSS_SELECTOR, "span.authorName[lang='pl']"
                        ).text.strip()
                    except NoSuchElementException:
                        d["academic_title"] = None

                    name_spans = row.find_elements(By.CSS_SELECTOR, "span.authorName")
                    d["first_name"] = name_spans[1].text.strip() if len(name_spans) >= 2 else None
                    d["last_name"] = name_spans[2].text.strip() if len(name_spans) >= 3 else None

                    try:
                        d["full_name"] = row.find_element(By.CSS_SELECTOR, "a.authorNameLink").text.strip()
                    except NoSuchElementException:
                        d["full_name"] = None

                    try:
                        d["position"] = row.find_element(
                            By.CSS_SELECTOR, "p.possitionInfo span.authorAffil"
                        ).text.strip()
                    except NoSuchElementException:
                        d["position"] = None

                    affiliations = []
                    try:
                        for li in row.find_element(
                            By.CSS_SELECTOR, "ul.unstyled-list"
                        ).find_elements(By.TAG_NAME, "li"):
                            try:
                                a = li.find_element(By.CSS_SELECTOR, "a")
                                href = a.get_attribute("href")
                                affiliations.append({
                                    "name": a.text.strip(),
                                    "url": href,
                                    "id": href.split("/")[-1].split("?")[0] if href else None,
                                })
                            except NoSuchElementException:
                                affiliations.append({"name": li.text.strip(), "url": None, "id": None})
                    except NoSuchElementException:
                        pass
                    d["affiliations"] = affiliations

                    try:
                        d["image_url"] = row.find_element(
                            By.CSS_SELECTOR, "div.authorGlobalSearchImage img"
                        ).get_attribute("src")
                    except NoSuchElementException:
                        d["image_url"] = None

                    scientists.append(d)
                    print(f"  [{idx+1}] {d.get('full_name', 'Unknown')}")

                except Exception as e:
                    print(f"  Error on scientist {idx+1}: {e}")
                    continue

            return scientists
        except Exception as e:
            print(f"Error extracting scientists: {e}")
            return []

    def get_pagination_info(self):
        try:
            current = int(
                self.driver.find_element(By.CSS_SELECTOR, "span.page-inplace-input button").text.split()[0]
            )
            total = int(
                self.driver.find_element(By.CSS_SELECTOR, "span.entitiesDataListTotalPages").text
            )
            return current, total
        except Exception:
            return None, None

    def click_next_page(self):
        try:
            btn = self.driver.find_element(By.CSS_SELECTOR, "a.ui-paginator-next")
            if "ui-state-disabled" in btn.get_attribute("class"):
                print("Last page reached")
                return False
            self.driver.execute_script("arguments[0].scrollIntoView(true);", btn)
            time.sleep(0.5)
            btn.click()
            print("Clicked next page")
            time.sleep(3)
            self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, "ui-dataview-row")))
            return True
        except Exception as e:
            print(f"Error clicking next page: {e}")
            return False

    def save_to_csv(self):
        if not self.all_scientists:
            print("No data to save")
            return False
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        out = DATA_DIR / "scientists_data.csv"
        try:
            with open(out, "w", newline="", encoding="utf-8") as f:
                fieldnames = [
                    "profile_id", "full_name", "academic_title", "first_name",
                    "last_name", "position", "profile_url", "image_url", "affiliations",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for s in self.all_scientists:
                    row = s.copy()
                    if isinstance(row.get("affiliations"), list):
                        row["affiliations"] = "; ".join(a.get("name", "") for a in row["affiliations"])
                    writer.writerow(row)
            print(f"Saved to {out}")
            return True
        except Exception as e:
            print(f"Error saving CSV: {e}")
            return False

    def take_screenshot(self, filename="screenshot.png"):
        try:
            self.driver.save_screenshot(filename)
            print(f"Screenshot: {filename}")
        except Exception as e:
            print(f"Screenshot error: {e}")

    def run(self):
        try:
            self.setup_driver()
            self.navigate_to_portal()

            if not self.wait_for_dynamic_content():
                print("Failed to load dynamic content")
                return False
            if not self.expand_parent_node("Szkoła Nauk Ścisłych"):
                self.take_screenshot("error_parent_expansion.png")
                return False
            if not self.find_and_click_faculty_checkbox():
                self.take_screenshot("error_checkbox.png")
                return False
            if not self.click_filter_button():
                self.take_screenshot("error_filter.png")
                return False

            page_num = 1
            while True:
                print(f"\n── Page {page_num} ──")
                self.all_scientists.extend(self.extract_scientists_data())
                print(f"Total so far: {len(self.all_scientists)}")

                current, total = self.get_pagination_info()
                if current and total:
                    print(f"Page {current}/{total}")
                    if current >= total:
                        print("Last page reached")
                        break

                if not self.click_next_page():
                    break
                page_num += 1
                time.sleep(2)

            print(f"\nScraping complete — {len(self.all_scientists)} scientists")
            self.save_to_csv()
            return True

        except Exception as e:
            print(f"Error: {e}")
            self.take_screenshot("error_screenshot.png")
            return False
        finally:
            if self.driver:
                self.driver.quit()
                print("Browser closed")


if __name__ == "__main__":
    scraper = ResearchPortalScraper(headless=False)
    success = scraper.run()
    print("\nDone!" if success else "\nFailed — check errors above.")
