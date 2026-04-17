import requests
import time
import re
import csv
import pandas as pd
from pathlib import Path
from typing import List, Dict

DATA_DIR = Path(__file__).parent.parent / "data"

# Input: ORCIDs to scrape. Pipeline generates this from scientists_with_identifiers.csv,
# but wmii_orcid.csv (pre-existing seed file) is used as fallback if the full
# pipeline hasn't been run yet.
ORCID_SOURCES = [
    DATA_DIR / "scientists_with_identifiers.csv",  # preferred: output of step 2
    DATA_DIR / "wmii_orcid.csv",                   # fallback: pre-existing seed
]

OUTPUT_FILE = DATA_DIR / "wmii_publications.csv"


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
            "title":              work.get("title", ""),
            "publication_year":   work.get("publication_year", ""),
            "publication_date":   work.get("publication_date", ""),
            "doi":                work.get("doi", ""),
            "type":               work.get("type", ""),
            "cited_by_count":     work.get("cited_by_count", 0),
            "journal":            source.get("display_name", ""),
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
                # Save progress after each ORCID
                pd.DataFrame(all_results).to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
                print(f"  Progress saved — {len(all_results)} total works so far")
            except Exception as e:
                print(f"  ERROR for {orcid}: {e}")
                continue
            time.sleep(0.5)

        return all_results


def load_orcids() -> List[str]:
    for path in ORCID_SOURCES:
        if path.exists():
            print(f"Loading ORCIDs from {path}")
            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                orcids = [
                    row["orcid"].strip()
                    for row in reader
                    if row.get("orcid", "").strip()
                ]
            print(f"  Loaded {len(orcids)} ORCIDs")
            return orcids

    print("ERROR: No ORCID source file found")
    print(f"  Checked: {[str(p) for p in ORCID_SOURCES]}")
    return []


def fill_abstracts_from_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Copy abstracts from duplicate records (same title or DOI) that do have one."""
    df = df.copy()
    df["_title_norm"] = df["title"].str.lower().str.strip()
    filled = 0

    def has_abstract(val):
        return bool(val and str(val).strip())

    for col in ["_title_norm", "doi"]:
        if col == "doi":
            df_sub = df[df["doi"].notna() & (df["doi"].str.strip() != "")]
        else:
            df_sub = df

        for key, group in df_sub.groupby(col):
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

    df = df.drop(columns=["_title_norm"])
    print(f"  Filled {filled} abstracts from duplicates")
    return df


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    orcids = load_orcids()
    if not orcids:
        return

    scraper = OpenAlexScraper()
    results = scraper.scrape_orcids(orcids)

    if not results:
        print("No results — aborting")
        return

    df = pd.DataFrame(results)

    total = len(df)
    with_abstract_before = df["abstract"].apply(lambda x: bool(x and str(x).strip())).sum()
    print(f"\nBefore dedup fill: {with_abstract_before}/{total} have abstracts")

    df = fill_abstracts_from_duplicates(df)

    with_abstract_after = df["abstract"].apply(lambda x: bool(x and str(x).strip())).sum()
    print(f"After dedup fill:  {with_abstract_after}/{total} have abstracts")

    # Keep only records WITH abstracts as the final clean dataset
    df_final = df[df["abstract"].apply(lambda x: bool(x and str(x).strip()))].copy()

    df_final.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"{'='*60}")
    print(f"  Total works found:      {total}")
    print(f"  With abstracts:         {with_abstract_after} ({with_abstract_after/total*100:.1f}%)")
    print(f"  Final dataset:          {len(df_final)} records")
    print(f"  Output:                 {OUTPUT_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
