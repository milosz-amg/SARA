from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import time
import json
import csv


class ResearchPortalScraper:
    def __init__(self, headless=False):
        self.headless = headless
        self.driver = None
        self.wait = None
        self.all_scientists = []
        
    def setup_driver(self):
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument('--headless')
        
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument('--window-size=1920,1080')
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.wait = WebDriverWait(self.driver, 20)
        
        print("Chrome driver initialized successfully")
        
    def navigate_to_portal(self):
        url = "https://researchportal.amu.edu.pl/globalResultList.seam?q=&oa=false&r=author&tab=PEOPLE&conversationPropagation=begin&lang=pl&qp=openAccess%3Dfalse"
        
        print(f"Navigating to: {url}")
        self.driver.get(url)
        
        time.sleep(3)
        print("Page loaded successfully")
        
    def wait_for_dynamic_content(self):
        try:
            self.wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "ui-treenode-content"))
            )
            print("Filter panel detected")
            time.sleep(2)
            return True
        except TimeoutException:
            print("Timeout waiting for filter panel")
            return False
    
    def expand_parent_node(self, parent_text="Szkoła Nauk Ścisłych"):
        try:
            print(f"Looking for parent node: '{parent_text}'...")
            tree_nodes = self.driver.find_elements(By.CLASS_NAME, "ui-treenode-content")
            
            for node in tree_nodes:
                try:
                    label = node.find_element(By.CLASS_NAME, "ui-treenode-label")
                    label_text = label.text
                    
                    if parent_text in label_text:
                        print(f"Found parent node: {label_text}")
                        
                        toggler = node.find_element(By.CLASS_NAME, "ui-tree-toggler")
                        
                        aria_expanded = toggler.get_attribute("aria-expanded")
                        print(f"Parent node aria-expanded: {aria_expanded}")
                        
                        if aria_expanded == "false":
                            print("Expanding parent node...")
                            self.driver.execute_script("arguments[0].scrollIntoView(true);", toggler)
                            time.sleep(0.5)
                            
                            toggler.click()
                            print("Parent node expanded!")
                            time.sleep(2)
                            return True
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
            tree_nodes = self.driver.find_elements(By.CLASS_NAME, "ui-treenode-content")
            print(f"Found {len(tree_nodes)} tree nodes")
            
            for node in tree_nodes:
                try:
                    label = node.find_element(By.CLASS_NAME, "ui-treenode-label")
                    label_text = label.text
                    
                    if "Wydział Matematyki i Informatyki" in label_text:
                        print(f"Found target faculty: {label_text}")
                        
                        checkbox = node.find_element(By.CSS_SELECTOR, "input[type='checkbox']")
                        self.driver.execute_script("arguments[0].scrollIntoView(true);", checkbox)
                        time.sleep(0.5)
                        
                        checkbox.click()
                        print("Checkbox clicked successfully!")
                        
                        time.sleep(1)
                        return True
                        
                except NoSuchElementException:
                    continue
            
            print("Could not find the target faculty checkbox")
            return False
            
        except Exception as e:
            print(f"Error finding checkbox: {e}")
            return False
            
    def click_filter_button(self):
        try:
            print("Looking for filter button...")
            try:
                filter_button = self.driver.find_element(By.ID, "applySearchFiltersButton")
                print("Found filter button by ID")
            except NoSuchElementException:
                buttons = self.driver.find_elements(By.TAG_NAME, "button")
                filter_button = None
                for button in buttons:
                    if "Filtruj" in button.text:
                        filter_button = button
                        print("Found filter button by text")
                        break
                
                if not filter_button:
                    print("Could not find filter button")
                    return False
            
            self.driver.execute_script("arguments[0].scrollIntoView(true);", filter_button)
            time.sleep(0.5)
            filter_button.click()
            print("Filter button clicked!")
            time.sleep(3)
            print("Waiting for filtered results to load...")
            time.sleep(2)
            
            print("Filter applied successfully!")
            print(f"Current URL: {self.driver.current_url}")
            
            return True
            
        except Exception as e:
            print(f"Error clicking filter button: {e}")
            return False
    
    def extract_scientists_data(self):
        scientists = []
        
        try:
            print("\nExtracting scientists data from current page...")
            
            self.wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "ui-dataview-row"))
            )
            
            scientist_rows = self.driver.find_elements(By.CLASS_NAME, "ui-dataview-row")
            print(f"Found {len(scientist_rows)} scientists on this page")
            
            for idx, row in enumerate(scientist_rows):
                try:
                    scientist_data = {}
                    
                    try:
                        profile_link = row.find_element(By.CSS_SELECTOR, "a.authorNameLink")
                        scientist_data['profile_url'] = profile_link.get_attribute('href')
                        scientist_data['profile_id'] = profile_link.get_attribute('href').split('/')[-1].split('?')[0]
                    except NoSuchElementException:
                        scientist_data['profile_url'] = None
                        scientist_data['profile_id'] = None
                    
                    try:
                        title_element = row.find_element(By.CSS_SELECTOR, "span.authorName[lang='pl']")
                        scientist_data['academic_title'] = title_element.text.strip()
                    except NoSuchElementException:
                        scientist_data['academic_title'] = None
                    
                    try:
                        name_elements = row.find_elements(By.CSS_SELECTOR, "span.authorName")
                        if len(name_elements) >= 2:
                            scientist_data['first_name'] = name_elements[1].text.strip()
                        else:
                            scientist_data['first_name'] = None
                    except NoSuchElementException:
                        scientist_data['first_name'] = None
                    
                    try:
                        name_elements = row.find_elements(By.CSS_SELECTOR, "span.authorName")
                        if len(name_elements) >= 3:
                            scientist_data['last_name'] = name_elements[2].text.strip()
                        else:
                            scientist_data['last_name'] = None
                    except NoSuchElementException:
                        scientist_data['last_name'] = None
                    
                    try:
                        full_name_link = row.find_element(By.CSS_SELECTOR, "a.authorNameLink")
                        scientist_data['full_name'] = full_name_link.text.strip()
                    except NoSuchElementException:
                        scientist_data['full_name'] = None
                    
                    try:
                        position_element = row.find_element(By.CSS_SELECTOR, "p.possitionInfo span.authorAffil")
                        scientist_data['position'] = position_element.text.strip()
                    except NoSuchElementException:
                        scientist_data['position'] = None
                    
                    affiliations = []
                    try:
                        affil_list = row.find_element(By.CSS_SELECTOR, "ul.unstyled-list")
                        affil_items = affil_list.find_elements(By.TAG_NAME, "li")
                        
                        for affil_item in affil_items:
                            affil_data = {}
                            try:
                                affil_link = affil_item.find_element(By.CSS_SELECTOR, "a")
                                affil_data['name'] = affil_link.text.strip()
                                affil_data['url'] = affil_link.get_attribute('href')
                                affil_data['id'] = affil_link.get_attribute('href').split('/')[-1].split('?')[0] if affil_link.get_attribute('href') else None
                            except NoSuchElementException:
                                affil_data['name'] = affil_item.text.strip()
                                affil_data['url'] = None
                                affil_data['id'] = None
                            
                            affiliations.append(affil_data)
                    except NoSuchElementException:
                        pass
                    
                    scientist_data['affiliations'] = affiliations
                    
                    try:
                        img_element = row.find_element(By.CSS_SELECTOR, "div.authorGlobalSearchImage img")
                        scientist_data['image_url'] = img_element.get_attribute('src')
                    except NoSuchElementException:
                        scientist_data['image_url'] = None
                    
                    scientists.append(scientist_data)
                    print(f"  [{idx+1}] Extracted: {scientist_data.get('full_name', 'Unknown')}")
                    
                except Exception as e:
                    print(f"  Error extracting scientist {idx+1}: {e}")
                    continue
            
            return scientists
            
        except Exception as e:
            print(f"Error extracting scientists data: {e}")
            return []
    
    def get_pagination_info(self):
        try:
            current_page_element = self.driver.find_element(By.CSS_SELECTOR, "span.page-inplace-input button")
            current_page = int(current_page_element.text.split()[0])
            total_pages_element = self.driver.find_element(By.CSS_SELECTOR, "span.entitiesDataListTotalPages")
            total_pages = int(total_pages_element.text)
            
            return current_page, total_pages
        except Exception as e:
            print(f"Error getting pagination info: {e}")
            return None, None
    
    def click_next_page(self):
        try:
            next_button = self.driver.find_element(By.CSS_SELECTOR, "a.ui-paginator-next")
            if "ui-state-disabled" in next_button.get_attribute("class"):
                print("Next button is disabled (last page reached)")
                return False
            
            self.driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
            time.sleep(0.5)
            
            next_button.click()
            print("Clicked next page button")
            time.sleep(3)
            
            self.wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "ui-dataview-row"))
            )
            
            return True
            
        except Exception as e:
            print(f"Error clicking next page: {e}")
            return False
    
    def save_data_to_json(self, filename="./data/scientists_data.json"):
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.all_scientists, f, ensure_ascii=False, indent=2)
            print(f"\nData saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving to JSON: {e}")
            return False
    
    def save_data_to_csv(self, filename="./data/scientists_data.csv"):
        try:
            if not self.all_scientists:
                print("No data to save")
                return False
            
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['profile_id', 'full_name', 'academic_title', 'first_name', 
                             'last_name', 'position', 'profile_url', 'image_url', 
                             'affiliations']
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for scientist in self.all_scientists:
                    row = scientist.copy()
                    if 'affiliations' in row and isinstance(row['affiliations'], list):
                        row['affiliations'] = '; '.join([a.get('name', '') for a in row['affiliations']])
                    writer.writerow(row)
            
            print(f"Data saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving to CSV: {e}")
            return False
            
    def take_screenshot(self, filename="screenshot.png"):
        try:
            self.driver.save_screenshot(filename)
            print(f"Screenshot saved to {filename}")
        except Exception as e:
            print(f"Error taking screenshot: {e}")
            
    def run(self):
        try:
            self.setup_driver()
            self.navigate_to_portal()
            
            if not self.wait_for_dynamic_content():
                print("Failed to load dynamic content")
                return False
            
            if not self.expand_parent_node("Szkoła Nauk Ścisłych"):
                print("Failed to expand parent node")
                self.take_screenshot("error_parent_expansion.png")
                return False
            
            if not self.find_and_click_faculty_checkbox():
                print("Failed to click checkbox")
                self.take_screenshot("error_before_checkbox.png")
                return False
            
            if not self.click_filter_button():
                print("Failed to click filter button")
                self.take_screenshot("error_before_filter.png")
                return False
            
            current_page, total_pages = self.get_pagination_info()
            if current_page and total_pages:
                print(f"Pagination: Page {current_page} of {total_pages}")
            
            page_num = 1
            while True:
                print(f"Processing Page {page_num}")
                scientists = self.extract_scientists_data()
                self.all_scientists.extend(scientists)
                print(f"\nTotal scientists collected so far: {len(self.all_scientists)}")
                
                current_page, total_pages = self.get_pagination_info()
                if current_page and total_pages:
                    print(f"Current position: Page {current_page} of {total_pages}")
                    
                    if current_page >= total_pages:
                        print("\nReached last page!")
                        break
                
                if not self.click_next_page():
                    print("\nNo more pages available")
                    break
                
                page_num += 1
                time.sleep(2)
            
            print(f"SCRAPING COMPLETED!")
            print(f"Total scientists extracted: {len(self.all_scientists)}")
            
            # self.save_data_to_json()
            self.save_data_to_csv()
            
            return True
            
        except Exception as e:
            print(f"Error during scraping: {e}")
            self.take_screenshot("error_screenshot.png")
            return False
            
        finally:
            if self.driver:
                self.driver.quit()
                print("Browser closed")

if __name__ == "__main__":
    scraper = ResearchPortalScraper(headless=False)
    success = scraper.run()
    
    if success:
        print("\nScraping completed successfully!")
    else:
        print("\nScraping failed. Check error messages above.")