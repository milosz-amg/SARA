import requests
import time
import re
import csv
import pandas as pd
from pathlib import Path
from typing import List, Dict

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait

DATA_DIR = Path(__file__).parent.parent / "data"

ORCID_SOURCES = [
    DATA_DIR / "scientists_with_identifiers.csv",
    DATA_DIR / "wmii_orcid.csv",
]

OUTPUT_ALL   = DATA_DIR / "wmii_publications.csv"
OUTPUT_CLEAN = DATA_DIR / "wmii_publications_with_abstracts.csv"


def has_abstract(val) -> bool:
    return bool(val and str(val).strip())

def has_doi(val) -> bool:
    return bool(val and str(val).strip())

def save_both(df: pd.DataFrame):
    df.to_csv(OUTPUT_ALL, index=False, encoding="utf-8")
    df[df["abstract"].apply(has_abstract)].to_csv(OUTPUT_CLEAN, index=False, encoding="utf-8")


# ── OpenAlex API scraper ─────────────────────────────────────────────────────

class OpenAlexScraper:
    def __init__(self):
        self.base_url = "https://api.openalex.org"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "OpenAlexScraper/1.0 (mailto:your@email.com)",
            "Accept": "application/json",
        })

    def get_works_for_orcid(self, orcid: str) -> List[Dict]:
        print(f"\n{'='*60}")
        print(f"Processing ORCID: {orcid}")
        print(f"{'='*60}")

        all_works = []
        page = 1
        per_page = 200

        while True:
            try:
                response = self.session.get(
                    f"{self.base_url}/works",
                    params={
                        "filter": f"authorships.author.orcid:{orcid}",
                        "per-page": per_page,
                        "page": page,
                    },
                )
                response.raise_for_status()
                data = response.json()

                results = data.get("results", [])
                total_count = data.get("meta", {}).get("count", 0)

                if not results:
                    break

                print(f"  Page {page}: {len(results)} works (total available: {total_count})")

                for work in results:
                    try:
                        all_works.append(self._process_work(work, orcid))
                    except Exception as e:
                        print(f"    Error processing work: {e}")
                        continue

                if len(all_works) >= total_count:
                    break

                page += 1
                time.sleep(0.1)

            except Exception as e:
                print(f"  Error fetching page {page}: {e}")
                break

        with_abstract = sum(1 for w in all_works if w.get("abstract"))
        print(f"  ✓ {len(all_works)} works, {with_abstract} with abstracts")
        return all_works

    def _process_work(self, work: Dict, main_orcid: str) -> Dict:
        work_id = work.get("id", "").split("/")[-1]
        primary_location = work.get("primary_location") or {}
        source = primary_location.get("source") or {}
        topics = work.get("topics", [])
        keywords = work.get("keywords", [])
        authorships = work.get("authorships", [])

        co_authors = []
        co_author_orcids = []
        for authorship in authorships:
            author = authorship.get("author") or {}
            author_orcid = author.get("orcid", "")
            if author_orcid:
                m = re.search(r"(\d{4}-\d{4}-\d{4}-\d{3}[0-9X])", author_orcid)
                if m:
                    author_orcid = m.group(1)
            if author_orcid and author_orcid != main_orcid:
                name = author.get("display_name", "")
                if name:
                    co_authors.append(name)
                    co_author_orcids.append(author_orcid)

        return {
            "main_author_orcid":  main_orcid,
            "openalex_id":        work_id,
            "title":              work.get("title") or "",
            "publication_year":   work.get("publication_year") or "",
            "publication_date":   work.get("publication_date") or "",
            "doi":                work.get("doi") or "",
            "type":               work.get("type") or "",
            "cited_by_count":     work.get("cited_by_count") or 0,
            "journal":            source.get("display_name") or "",
            "topics":             "; ".join(t.get("display_name", "") for t in topics[:5] if t.get("display_name")),
            "co_authors":         "; ".join(co_authors),
            "co_author_orcids":   "; ".join(co_author_orcids),
            "num_co_authors":     len(co_authors),
            "abstract":           self._extract_abstract(work),
            "keywords":           "; ".join(k.get("display_name", "") for k in keywords if k.get("display_name")),
        }

    def _extract_abstract(self, work: Dict) -> str:
        inverted = work.get("abstract_inverted_index") or {}
        if not inverted:
            return ""
        try:
            word_positions = []
            for word, positions in inverted.items():
                if isinstance(positions, list):
                    for pos in positions:
                        word_positions.append((pos, word))
                else:
                    word_positions.append((positions, word))
            word_positions.sort(key=lambda x: x[0])
            return " ".join(word for _, word in word_positions).strip()
        except Exception:
            return ""

    def scrape_orcids(self, orcids: List[str]) -> List[Dict]:
        print(f"\nStarting API scrape for {len(orcids)} ORCIDs")
        all_results = []

        for idx, orcid in enumerate(orcids):
            print(f"\nProgress: {idx+1}/{len(orcids)}")
            try:
                works = self.get_works_for_orcid(orcid)
                all_results.extend(works)
                save_both(pd.DataFrame(all_results))
                print(f"  Progress saved — {len(all_results)} total works so far")
            except Exception as e:
                print(f"  ERROR for {orcid}: {e}")
                continue
            time.sleep(0.5)

        return all_results


# ── Selenium DOI abstract fetcher ────────────────────────────────────────────

class DOIAbstractFetcher:
    SELECTORS = [
        "div.abstract.author",
        "div#abs0010",
        "section.abstract",
        "div.Abstract",
        "section[data-title='Abstract']",
        "div#Abs1-content",
        "section#Abs1",
        "section.article-section__abstract",
        "div.article-section__content",
        "div.art-abstract",
        "section.html-abstract",
        "div.abstract-text",
        "div.abstract",
        "div[id*='abstract']",
        "section[id*='abstract']",
        "div[class*='abstract']",
        "section[class*='abstract']",
    ]

    def __init__(self, headless=True):
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        self.driver = webdriver.Chrome(options=chrome_options)
        self.wait = WebDriverWait(self.driver, 10)

    def fetch(self, doi_url: str) -> str:
        if not doi_url or not str(doi_url).strip():
            return ""
        try:
            print(f"    Fetching: {str(doi_url)[:70]}...")
            self.driver.get(str(doi_url))
            time.sleep(3)

            for selector in self.SELECTORS:
                try:
                    elem = self.driver.find_element(By.CSS_SELECTOR, selector)
                    paragraphs = elem.find_elements(By.TAG_NAME, "p")
                    text = (
                        " ".join(p.text for p in paragraphs if p.text.strip())
                        if paragraphs else elem.text
                    )
                    text = re.sub(r"^Abstract\s*", "", text, flags=re.IGNORECASE).strip()
                    if len(text) > 50:
                        print(f"      ✓ Found ({len(text)} chars)")
                        return text
                except Exception:
                    continue

            print("      ✗ Not found")
            return ""
        except Exception as e:
            print(f"      ✗ Error: {str(e)[:60]}")
            return ""

    def close(self):
        if self.driver:
            self.driver.quit()

    def fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        no_abstract = ~df["abstract"].apply(has_abstract)
        has_doi_mask = df["doi"].fillna("").str.strip().ne("")
        missing_idx = df[no_abstract & has_doi_mask].index

        print(f"\nDOI fallback: {len(missing_idx)} records missing abstract but have DOI")

        if len(missing_idx) == 0:
            return df

        success = 0

        for i, idx in enumerate(missing_idx):
            title = str(df.at[idx, "title"] or "")[:60]
            doi   = str(df.at[idx, "doi"] or "")
            print(f"  [{i+1}/{len(missing_idx)}] {title}...")

            abstract = self.fetch(doi)
            if abstract:
                df.at[idx, "abstract"] = abstract
                success += 1

            if (i + 1) % 10 == 0:
                save_both(df)
                print(f"  Progress saved ({success} fetched so far)")

            time.sleep(2)

        save_both(df)
        print(f"  DOI fallback done — {success}/{len(missing_idx)} abstracts fetched")
        return df


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_orcids() -> List[str]:
    for path in ORCID_SOURCES:
        if path.exists():
            print(f"Loading ORCIDs from {path}")
            with open(path, encoding="utf-8") as f:
                orcids = [
                    row["orcid"].strip()
                    for row in csv.DictReader(f)
                    if row.get("orcid", "").strip()
                ]
            print(f"  Loaded {len(orcids)} ORCIDs")
            return orcids

    print("ERROR: No ORCID source file found")
    print(f"  Checked: {[str(p) for p in ORCID_SOURCES]}")
    return []


def fill_abstracts_from_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_title_norm"] = df["title"].fillna("").str.lower().str.strip()
    df["_doi_norm"]   = df["doi"].fillna("").str.strip()
    filled = 0

    for col in ["_title_norm", "_doi_norm"]:
        df_sub = df[df[col] != ""]
        for _, group in df_sub.groupby(col):
            if len(group) < 2:
                continue
            have = group[group["abstract"].apply(has_abstract)]
            missing = group[~group["abstract"].apply(has_abstract)]
            if len(have) > 0 and len(missing) > 0:
                abstract_to_copy = have.iloc[0]["abstract"]
                for idx in missing.index:
                    if not has_abstract(df.at[idx, "abstract"]):
                        df.at[idx, "abstract"] = abstract_to_copy
                        filled += 1

    df = df.drop(columns=["_title_norm", "_doi_norm"])
    print(f"  Filled {filled} abstracts from duplicates")
    return df


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    orcids = load_orcids()
    if not orcids:
        return

    # Step A: OpenAlex API
    scraper = OpenAlexScraper()
    results = scraper.scrape_orcids(orcids)

    if not results:
        print("No results — aborting")
        return

    df = pd.DataFrame(results)
    total = len(df)

    with_abstract = df["abstract"].apply(has_abstract).sum()
    print(f"\nAfter API:          {with_abstract}/{total} have abstracts")

    # Step B: fill from duplicates
    df = fill_abstracts_from_duplicates(df)
    with_abstract = df["abstract"].apply(has_abstract).sum()
    print(f"After dedup fill:   {with_abstract}/{total} have abstracts")
    save_both(df)
    print("Checkpoint saved after dedup fill")

    # Step C: Selenium DOI fallback
    doi_fetcher = DOIAbstractFetcher(headless=True)
    try:
        df = doi_fetcher.fill_missing(df)
    finally:
        doi_fetcher.close()

    with_abstract_final = df["abstract"].apply(has_abstract).sum()
    print(f"After DOI fallback: {with_abstract_final}/{total} have abstracts")

    save_both(df)

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"{'='*60}")
    print(f"  Total works:            {total}")
    print(f"  With abstracts:         {with_abstract_final} ({with_abstract_final/total*100:.1f}%)")
    print(f"  All records:            {OUTPUT_ALL}")
    print(f"  With abstracts only:    {OUTPUT_CLEAN}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
