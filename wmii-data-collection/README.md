# SARA — UAM WMiI Research Data Pipeline

Collects publication data for WMiI UAM faculty members from the UAM Research Portal and OpenAlex API.

## Pipeline overview

```
Step 1: research_portal_scraper.py
  → Scrapes scientist profiles from UAM Research Portal
  → Output: data/scientists_data.csv

Step 2: extract_identifiers.py
  → Visits each profile, extracts ORCID + Scopus + Scholar IDs
  → Output: data/scientists_with_identifiers.csv  ← key intermediate file

Step 3: filter_data.py  [optional — requires large OpenAlex dump files]
  → Filters pre-downloaded OpenAlex dumps by faculty ORCIDs
  → Output: data/wmii_authors.json, data/wmii_works.json

Step 4: fetch_abstracts.py
  → Calls OpenAlex API for each ORCID, fetches all publications + abstracts
  → Fills missing abstracts from duplicate records (same title or DOI)
  → For still-missing abstracts: opens DOI links via Selenium, scrapes publisher pages
     (ScienceDirect, Springer, Wiley, MDPI, IEEE and others)
  → Falls back to data/wmii_orcid.csv if Step 2 hasn't run yet
  → Output: data/wmii_publications.csv                ← all records
            data/wmii_publications_with_abstracts.csv ← only records with abstracts
```

## Project structure

```
.
├── run_pipeline.sh              # Run the full pipeline
├── requirements.txt
├── src/
│   ├── research_portal_scraper.py   # Step 1
│   ├── extract_identifiers.py       # Step 2
│   ├── filter_data.py               # Step 3 (optional)
│   └── fetch_abstracts.py           # Step 4
└── data/
    ├── wmii_orcid.csv               # Seed ORCID list (pre-existing fallback)
    ├── uam_authors.json             # [optional input] OpenAlex authors dump
    ├── uam_works.json               # [optional input] OpenAlex works dump
    ├── scientists_data.csv          # [output] Step 1
    ├── scientists_with_identifiers.csv      # [output] Step 2
    ├── wmii_authors.json            # [output] Step 3
    ├── wmii_works.json              # [output] Step 3
    ├── wmii_publications.csv        # [output] Step 4 — all records
    └── wmii_publications_with_abstracts.csv # [output] Step 4 — abstracts only
```

## Output files

| File | Description |
|------|-------------|
| `scientists_with_identifiers.csv` | Faculty profiles with ORCID, Scopus, Scholar IDs |
| `wmii_publications.csv` | All publications including those with missing abstracts |
| `wmii_publications_with_abstracts.csv` | Publications with abstracts only — ready for analysis |
| `wmii_authors.json` | OpenAlex author records (Step 3 only) |
| `wmii_works.json` | OpenAlex work records (Step 3 only) |

## Setup

```bash
pip install -r requirements.txt
chmod +x run_pipeline.sh
```

Optional — download OpenAlex dump files for Step 3:
> https://uam-my.sharepoint.com/:f:/r/personal/jakpas3_st_amu_edu_pl/Documents/SARA?csf=1&web=1&e=RlhsKV

Place `uam_authors.json` and `uam_works.json` in `data/`.

## Usage

```bash
# Full pipeline (Steps 1–4)
./run_pipeline.sh

# Or run steps individually:
python src/research_portal_scraper.py   # Step 1
python src/extract_identifiers.py       # Step 2
python src/filter_data.py               # Step 3 (optional)
python src/fetch_abstracts.py           # Step 4

# Step 4 can be run standalone using the seed file:
# data/wmii_orcid.csv is used automatically if scientists_with_identifiers.csv
# doesn't exist yet
```

## Column reference — wmii_publications.csv

| Column | Description |
|--------|-------------|
| `main_author_orcid` | Faculty member's ORCID |
| `openalex_id` | OpenAlex work ID |
| `title` | Publication title |
| `publication_year` | Year |
| `publication_date` | Full date |
| `doi` | DOI link |
| `type` | article / book-chapter / etc. |
| `cited_by_count` | Citation count |
| `journal` | Journal or venue name |
| `topics` | Research topics (semicolon-separated) |
| `co_authors` | Co-author names (semicolon-separated) |
| `co_author_orcids` | Co-author ORCIDs (semicolon-separated) |
| `num_co_authors` | Number of co-authors |
| `abstract` | Full abstract text |
| `keywords` | Keywords (semicolon-separated) |
