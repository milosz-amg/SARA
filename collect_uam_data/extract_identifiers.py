from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import csv
import json
import time
import re


class ProfileIdentifierExtractor:
    def __init__(self, csv_filename="scientists_data.csv", headless=False):
        self.csv_filename = csv_filename
        self.headless = headless
        self.driver = None
        self.wait = None
        self.scientists_with_identifiers = []
        
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
        self.wait = WebDriverWait(self.driver, 15)
        
        print("Chrome driver initialized successfully")
    
    def read_scientists_from_csv(self):
        scientists = []
        try:
            with open(self.csv_filename, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    scientists.append(row)
            print(f"Loaded {len(scientists)} scientists from {self.csv_filename}")
            return scientists
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return []
    
    def extract_identifiers_from_profile(self, profile_url):
        identifiers = {
            'orcid': None,
            'google_scholar': None,
            'scopus': None,
            'europepmc': None,
            'crossref': None,
            'researchgate': None,
            'other_links': []
        }
        
        try:
            print(f"  Navigating to profile: {profile_url}")
            self.driver.get(profile_url)
            time.sleep(2)
            
            try:
                self.wait.until(
                    EC.presence_of_element_located((By.ID, "socialMediaPanel"))
                )
            except TimeoutException:
                print("  No social media panel found")
                return identifiers
            
            try:
                social_panel = self.driver.find_element(By.ID, "socialMediaPanel")
            except NoSuchElementException:
                print("  Social media panel not found")
                return identifiers
            
            links = social_panel.find_elements(By.TAG_NAME, "a")
            
            for link in links:
                try:
                    href = link.get_attribute('href')
                    text = link.text.strip()
                    
                    if not href:
                        continue
                    
                    if 'orcid.org' in href:
                        orcid_match = re.search(r'(\d{4}-\d{4}-\d{4}-\d{3}[0-9X])', href)
                        if orcid_match:
                            identifiers['orcid'] = orcid_match.group(1)
                            print(f"  Found ORCID: {identifiers['orcid']}")
                    
                    elif 'scholar.google' in href:
                        scholar_match = re.search(r'user=([^&]+)', href)
                        if scholar_match:
                            identifiers['google_scholar'] = {
                                'id': scholar_match.group(1),
                                'url': href
                            }
                            print(f"  Found Google Scholar: {scholar_match.group(1)}")
                    
                    elif 'scopus.com' in href:
                        scopus_match = re.search(r'authorId=(\d+)', href)
                        if scopus_match:
                            identifiers['scopus'] = {
                                'id': scopus_match.group(1),
                                'url': href
                            }
                            print(f"  Found Scopus: {scopus_match.group(1)}")
                        else:
                            if 'orcidId' in href:
                                identifiers['scopus'] = {
                                    'id': 'via_orcid',
                                    'url': href
                                }
                                print(f"  Found Scopus (via ORCID)")
                    elif 'europepmc.org' in href:
                        identifiers['europepmc'] = href
                        print(f"  Found EuropePMC link")
                    elif 'crossref.org' in href:
                        identifiers['crossref'] = href
                        print(f"  Found Crossref link")
                    elif 'researchgate.net' in href:
                        identifiers['researchgate'] = href
                        print(f"  Found ResearchGate link")
                    elif any(domain in href for domain in [
                        'academia.edu', 'linkedin.com', 'twitter.com', 
                        'github.com', 'publons.com', 'mendeley.com'
                    ]):
                        identifiers['other_links'].append({
                            'text': text,
                            'url': href
                        })
                        print(f"  Found other link: {text}")
                
                except Exception as e:
                    print(f"  Error extracting link: {e}")
                    continue
            
            return identifiers
            
        except Exception as e:
            print(f"  Error extracting identifiers: {e}")
            return identifiers
    
    def process_all_scientists(self):
        scientists = self.read_scientists_from_csv()
        
        if not scientists:
            print("No scientists to process")
            return
        
        total = len(scientists)
        
        print(f"Starting to extract identifiers for {total} scientists")
        
        for idx, scientist in enumerate(scientists):
            print(f"\n[{idx+1}/{total}] Processing: {scientist.get('full_name', 'Unknown')}")
            print("-" * 70)
            
            profile_url = scientist.get('profile_url')
            
            if not profile_url:
                print("  No profile URL available")
                self.scientists_with_identifiers.append({
                    **scientist,
                    'identifiers': None
                })
                continue
            
            identifiers = self.extract_identifiers_from_profile(profile_url)
            
            scientist_with_ids = {
                **scientist,
                'identifiers': identifiers
            }
            
            self.scientists_with_identifiers.append(scientist_with_ids)
            
            time.sleep(1)
        
        print(f"Extraction completed for {total} scientists!")
    
    def save_to_json(self, filename="./data/scientists_with_identifiers.json"):
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.scientists_with_identifiers, f, ensure_ascii=False, indent=2)
            print(f"\nData saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving to JSON: {e}")
            return False
    
    def save_to_csv(self, filename="scientists_with_identifiers.csv"):
        try:
            if not self.scientists_with_identifiers:
                print("No data to save")
                return False
            
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'profile_id', 'full_name', 'academic_title', 'first_name', 
                    'last_name', 'position', 'profile_url', 'image_url', 
                    'affiliations', 'orcid', 'google_scholar_id', 'google_scholar_url',
                    'scopus_id', 'scopus_url', 'europepmc', 'crossref', 
                    'researchgate', 'other_links'
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for scientist in self.scientists_with_identifiers:
                    row = {
                        'profile_id': scientist.get('profile_id'),
                        'full_name': scientist.get('full_name'),
                        'academic_title': scientist.get('academic_title'),
                        'first_name': scientist.get('first_name'),
                        'last_name': scientist.get('last_name'),
                        'position': scientist.get('position'),
                        'profile_url': scientist.get('profile_url'),
                        'image_url': scientist.get('image_url'),
                        'affiliations': scientist.get('affiliations'),
                        'orcid': None,
                        'google_scholar_id': None,
                        'google_scholar_url': None,
                        'scopus_id': None,
                        'scopus_url': None,
                        'europepmc': None,
                        'crossref': None,
                        'researchgate': None,
                        'other_links': None
                    }
                    
                    if scientist.get('identifiers'):
                        ids = scientist['identifiers']
                        row['orcid'] = ids.get('orcid')
                        
                        if isinstance(ids.get('google_scholar'), dict):
                            row['google_scholar_id'] = ids['google_scholar'].get('id')
                            row['google_scholar_url'] = ids['google_scholar'].get('url')
                        
                        if isinstance(ids.get('scopus'), dict):
                            row['scopus_id'] = ids['scopus'].get('id')
                            row['scopus_url'] = ids['scopus'].get('url')
                        
                        row['europepmc'] = ids.get('europepmc')
                        row['crossref'] = ids.get('crossref')
                        row['researchgate'] = ids.get('researchgate')
                        
                        if ids.get('other_links'):
                            row['other_links'] = '; '.join([
                                f"{link['text']}: {link['url']}" 
                                for link in ids['other_links']
                            ])
                    
                    writer.writerow(row)
            
            print(f"Data saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving to CSV: {e}")
            return False
    
    def generate_statistics(self):
        total = len(self.scientists_with_identifiers)
        
        stats = {
            'total_scientists': total,
            'with_orcid': 0,
            'with_google_scholar': 0,
            'with_scopus': 0,
            'with_europepmc': 0,
            'with_crossref': 0,
            'with_researchgate': 0,
            'with_other': 0,
            'with_no_identifiers': 0
        }
        
        for scientist in self.scientists_with_identifiers:
            ids = scientist.get('identifiers')
            if not ids:
                stats['with_no_identifiers'] += 1
                continue
            
            has_any = False
            
            if ids.get('orcid'):
                stats['with_orcid'] += 1
                has_any = True
            
            if ids.get('google_scholar'):
                stats['with_google_scholar'] += 1
                has_any = True
            
            if ids.get('scopus'):
                stats['with_scopus'] += 1
                has_any = True
            
            if ids.get('europepmc'):
                stats['with_europepmc'] += 1
                has_any = True
            
            if ids.get('crossref'):
                stats['with_crossref'] += 1
                has_any = True
            
            if ids.get('researchgate'):
                stats['with_researchgate'] += 1
                has_any = True
            
            if ids.get('other_links'):
                stats['with_other'] += 1
                has_any = True
            
            if not has_any:
                stats['with_no_identifiers'] += 1
        
        print("IDENTIFIER STATISTICS")
        print(f"Total scientists: {stats['total_scientists']}")
        print(f"With ORCID: {stats['with_orcid']} ({stats['with_orcid']/total*100:.1f}%)")
        print(f"With Google Scholar: {stats['with_google_scholar']} ({stats['with_google_scholar']/total*100:.1f}%)")
        print(f"With Scopus: {stats['with_scopus']} ({stats['with_scopus']/total*100:.1f}%)")
        print(f"With EuropePMC: {stats['with_europepmc']} ({stats['with_europepmc']/total*100:.1f}%)")
        print(f"With Crossref: {stats['with_crossref']} ({stats['with_crossref']/total*100:.1f}%)")
        print(f"With ResearchGate: {stats['with_researchgate']} ({stats['with_researchgate']/total*100:.1f}%)")
        print(f"With other links: {stats['with_other']} ({stats['with_other']/total*100:.1f}%)")
        print(f"With no identifiers: {stats['with_no_identifiers']} ({stats['with_no_identifiers']/total*100:.1f}%)")
        
        return stats
    
    def run(self):
        try:
            self.setup_driver()
            self.process_all_scientists()
            self.generate_statistics()
            # self.save_to_json()
            self.save_to_csv()
            
            print("\nAll operations completed successfully!")
            
            return True
            
        except Exception as e:
            print(f"Error during extraction: {e}")
            return False
            
        finally:
            if self.driver:
                self.driver.quit()
                print("Browser closed")


if __name__ == "__main__":
    extractor = ProfileIdentifierExtractor(
        csv_filename="./data/scientists_data.csv",
        headless=False
    )
    
    success = extractor.run()
    
    if success:
        print("EXTRACTION COMPLETED SUCCESSFULLY!")
        print("Output files:")
        print("  - scientists_with_identifiers.csv")
    else:
        print("\nExtraction failed. Check error messages above.")
