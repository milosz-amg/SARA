from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from pathlib import Path
import csv
import time
import re

DATA_DIR = Path(__file__).parent.parent / "data"


class ProfileIdentifierExtractor:
    def __init__(self, headless=False):
        self.csv_input = DATA_DIR / "scientists_data.csv"
        self.csv_output = DATA_DIR / "scientists_with_identifiers.csv"
        self.headless = headless
        self.driver = None
        self.wait = None
        self.results = []

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
        self.wait = WebDriverWait(self.driver, 15)
        print("Chrome driver initialized")

    def read_scientists(self):
        try:
            with open(self.csv_input, encoding="utf-8") as f:
                scientists = list(csv.DictReader(f))
            print(f"Loaded {len(scientists)} scientists from {self.csv_input}")
            return scientists
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return []

    def extract_identifiers_from_profile(self, profile_url):
        ids = {
            "orcid": None,
            "google_scholar": None,
            "scopus": None,
            "europepmc": None,
            "crossref": None,
            "researchgate": None,
            "other_links": [],
        }

        try:
            print(f"  Visiting: {profile_url}")
            self.driver.get(profile_url)
            time.sleep(2)

            try:
                self.wait.until(EC.presence_of_element_located((By.ID, "socialMediaPanel")))
            except TimeoutException:
                print("  No social media panel")
                return ids

            try:
                panel = self.driver.find_element(By.ID, "socialMediaPanel")
            except NoSuchElementException:
                return ids

            for link in panel.find_elements(By.TAG_NAME, "a"):
                try:
                    href = link.get_attribute("href")
                    text = link.text.strip()
                    if not href:
                        continue

                    if "orcid.org" in href:
                        m = re.search(r"(\d{4}-\d{4}-\d{4}-\d{3}[0-9X])", href)
                        if m:
                            ids["orcid"] = m.group(1)
                            print(f"  ORCID: {ids['orcid']}")

                    elif "scholar.google" in href:
                        m = re.search(r"user=([^&]+)", href)
                        if m:
                            ids["google_scholar"] = {"id": m.group(1), "url": href}
                            print(f"  Google Scholar: {m.group(1)}")

                    elif "scopus.com" in href:
                        m = re.search(r"authorId=(\d+)", href)
                        ids["scopus"] = {
                            "id": m.group(1) if m else "via_orcid",
                            "url": href,
                        }
                        print(f"  Scopus: {ids['scopus']['id']}")

                    elif "europepmc.org" in href:
                        ids["europepmc"] = href
                        print("  EuropePMC found")

                    elif "crossref.org" in href:
                        ids["crossref"] = href
                        print("  Crossref found")

                    elif "researchgate.net" in href:
                        ids["researchgate"] = href
                        print("  ResearchGate found")

                    elif any(d in href for d in [
                        "academia.edu", "linkedin.com", "twitter.com",
                        "github.com", "publons.com", "mendeley.com",
                    ]):
                        ids["other_links"].append({"text": text, "url": href})
                        print(f"  Other: {text}")

                except Exception as e:
                    print(f"  Link error: {e}")
                    continue

            return ids

        except Exception as e:
            print(f"  Error: {e}")
            return ids

    def extract_orcid_from_scopus(self, scopus_url):
        if not scopus_url:
            return None
        try:
            print(f"  → Checking Scopus for ORCID: {scopus_url}")
            self.driver.get(scopus_url)
            time.sleep(3)

            for link in self.driver.find_elements(By.CSS_SELECTOR, "a[href*='orcid.org']"):
                m = re.search(r"orcid\.org/(\d{4}-\d{4}-\d{4}-\d{3}[0-9X])", link.get_attribute("href"))
                if m:
                    print(f"  ✓ ORCID from Scopus: {m.group(1)}")
                    return m.group(1)

            m = re.search(r"\b(\d{4}-\d{4}-\d{4}-\d{3}[0-9X])\b", self.driver.page_source)
            if m:
                print(f"  ✓ ORCID from Scopus page source: {m.group(1)}")
                return m.group(1)

            print("  ✗ No ORCID on Scopus profile")
            return None
        except Exception as e:
            print(f"  Scopus error: {e}")
            return None

    def process_all_scientists(self):
        scientists = self.read_scientists()
        if not scientists:
            print("No scientists to process")
            return

        total = len(scientists)
        print(f"Processing {total} scientists...")

        for idx, scientist in enumerate(scientists):
            print(f"\n[{idx+1}/{total}] {scientist.get('full_name', 'Unknown')}")
            print("-" * 60)

            profile_url = scientist.get("profile_url")
            if not profile_url:
                print("  No profile URL")
                self.results.append({**scientist, "identifiers": None})
                continue

            ids = self.extract_identifiers_from_profile(profile_url)

            if not ids.get("orcid") and ids.get("scopus"):
                orcid = self.extract_orcid_from_scopus(ids["scopus"].get("url"))
                if orcid:
                    ids["orcid"] = orcid

            self.results.append({**scientist, "identifiers": ids})
            time.sleep(1)

        print(f"\nExtraction complete for {total} scientists")

    def save_to_csv(self):
        scientists_with_orcid = [
            s for s in self.results
            if s.get("identifiers") and s["identifiers"].get("orcid")
        ]

        print(f"\n{len(scientists_with_orcid)}/{len(self.results)} scientists have ORCID")

        if not scientists_with_orcid:
            print("Nothing to save")
            return False

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            fieldnames = [
                "profile_id", "full_name", "academic_title", "first_name",
                "last_name", "position", "profile_url", "image_url", "affiliations",
                "orcid", "google_scholar_id", "google_scholar_url",
                "scopus_id", "scopus_url", "europepmc", "crossref",
                "researchgate", "other_links",
            ]

            with open(self.csv_output, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for s in scientists_with_orcid:
                    ids = s.get("identifiers") or {}
                    row = {
                        "profile_id":         s.get("profile_id"),
                        "full_name":          s.get("full_name"),
                        "academic_title":     s.get("academic_title"),
                        "first_name":         s.get("first_name"),
                        "last_name":          s.get("last_name"),
                        "position":           s.get("position"),
                        "profile_url":        s.get("profile_url"),
                        "image_url":          s.get("image_url"),
                        "affiliations":       s.get("affiliations"),
                        "orcid":              ids.get("orcid"),
                        "google_scholar_id":  (ids.get("google_scholar") or {}).get("id"),
                        "google_scholar_url": (ids.get("google_scholar") or {}).get("url"),
                        "scopus_id":          (ids.get("scopus") or {}).get("id"),
                        "scopus_url":         (ids.get("scopus") or {}).get("url"),
                        "europepmc":          ids.get("europepmc"),
                        "crossref":           ids.get("crossref"),
                        "researchgate":       ids.get("researchgate"),
                        "other_links":        "; ".join(
                            f"{l['text']}: {l['url']}" for l in ids.get("other_links") or []
                        ) or None,
                    }
                    writer.writerow(row)

            print(f"Saved to {self.csv_output}")
            return True
        except Exception as e:
            print(f"Error saving CSV: {e}")
            return False

    def print_statistics(self):
        total = len(self.results)
        if total == 0:
            return

        def count(key):
            return sum(1 for s in self.results if (s.get("identifiers") or {}).get(key))

        print("\nSTATISTICS")
        print(f"  Total scientists:      {total}")
        print(f"  With ORCID:            {count('orcid')}  ({count('orcid')/total*100:.1f}%)")
        print(f"  With Google Scholar:   {count('google_scholar')}  ({count('google_scholar')/total*100:.1f}%)")
        print(f"  With Scopus:           {count('scopus')}  ({count('scopus')/total*100:.1f}%)")
        print(f"  With EuropePMC:        {count('europepmc')}  ({count('europepmc')/total*100:.1f}%)")
        print(f"  With Crossref:         {count('crossref')}  ({count('crossref')/total*100:.1f}%)")
        print(f"  With ResearchGate:     {count('researchgate')}  ({count('researchgate')/total*100:.1f}%)")

    def run(self):
        try:
            self.setup_driver()
            self.process_all_scientists()
            self.print_statistics()
            self.save_to_csv()
            print("\nAll done!")
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False
        finally:
            if self.driver:
                self.driver.quit()
                print("Browser closed")


if __name__ == "__main__":
    extractor = ProfileIdentifierExtractor(headless=False)
    success = extractor.run()
    print("\nSuccess!" if success else "\nFailed - check errors above.")
