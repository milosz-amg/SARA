# UAM Research Data Collection Pipeline

## Pipeline Steps

The pipeline consists of 4 automated steps:

1. **Web Scraping** - Scrape scientist profiles from UAM Research Portal
2. **Identifier Extraction** - Extract academic identifiers from each profile
3. **Author Filtering** - Filter OpenAlex authors by ORCID
4. **Works Filtering** - Filter OpenAlex works by author ORCID

## Prerequisites

### Required Data Files

Download the following files from the shared folder and place them in the `data/` directory:

- `uam_authors.json` - OpenAlex authors dataset
- `uam_works.json` - OpenAlex works dataset

**Download link:** https://uam-my.sharepoint.com/:f:/r/personal/jakpas3_st_amu_edu_pl/Documents/SARA?csf=1&web=1&e=RlhsKV

### Python Dependencies

```bash
pip install selenium webdriver-manager
```
```bash
chmod +x collect_and_filter_data.sh
```

## Usage

### Run Complete Pipeline

Execute all 4 steps automatically:

```bash
./collect_and_filter_data.sh
```

### Run Individual Scripts

If you need to run steps individually:

```bash
# Step 1: Scrape Research Portal
python research_portal_scraper.py

# Step 2: Extract Identifiers
python extract_identifiers.py

# Step 3: Filter Authors
python filter_authors.py --csv ./data/scientists_with_identifiers.csv \
                        --json ./data/uam_authors.json \
                        --output ./data/wmii_authors.json

# Step 4: Filter Works
python filter_works.py --csv ./data/scientists_with_identifiers.csv \
                      --json ./data/uam_works.json \
                      --output ./data/wmii_works.json
```

## Scripts Description

### `research_portal_scraper.py`

Scrapes scientist profiles from UAM Research Portal:
- Navigates to the portal
- Filters by "Wydział Matematyki i Informatyki"
- Extracts profile data from all pages
- Saves results to CSV


### `extract_identifiers.py`

Visits each scientist's profile page and extracts academic identifiers:
- ORCID
- Google Scholar ID
- Scopus Author ID
- ResearchGate, EuropePMC, Crossref links


### `filter_authors.py`

Filters the large OpenAlex authors dataset:
- Reads ORCID numbers from scraped data
- Matches them against `uam_authors.json`
- Outputs only matching author records

### `filter_works.py`

Filters the large OpenAlex works dataset:
- Reads ORCID numbers from scraped data
- Finds all works authored by these researchers
- Outputs only relevant publications

## Output Files

### Primary Outputs

| File | Description | Format |
|------|-------------|--------|
| `scientists_data.csv` | Basic scientist profiles from portal | CSV |
| `scientists_with_identifiers.csv` | Profiles with extracted identifiers | CSV |
| `wmii_authors.json` | Filtered OpenAlex author records | JSON | Variable |
| `wmii_works.json` | Filtered OpenAlex publication records | JSON | Variable |

### Detailed File Descriptions

#### `data/scientists_data.csv`
Contains basic information scraped from the research portal:

- `profile_id` - Unique portal identifier
- `full_name` - Full name with academic title
- `academic_title` - Title (prof., dr hab., dr, mgr)
- `first_name` - First name
- `last_name` - Last name  
- `position` - Academic position
- `profile_url` - Link to portal profile
- `image_url` - Profile photo URL
- `affiliations` - Departmental affiliations

#### `data/scientists_with_identifiers.csv`

Extends the basic data with academic identifiers:

- All fields from `scientists_data.csv`
- `orcid` - ORCID identifier (e.g., 0000-0002-1234-5678)
- `google_scholar_id` - Google Scholar user ID
- `google_scholar_url` - Full Google Scholar profile URL
- `scopus_id` - Scopus Author ID
- `scopus_url` - Full Scopus profile URL
- `europepmc` - EuropePMC profile link
- `crossref` - Crossref profile link
- `researchgate` - ResearchGate profile link
- `other_links` - Additional academic profiles

#### `data/wmii_authors.json`

Filtered subset of OpenAlex authors matching faculty ORCID numbers.

#### `data/wmii_works.json`

Filtered subset of OpenAlex publications authored by faculty members.

## Project Structure

```
.
├── collect_and_filter_data.sh          # Main pipeline orchestrator
├── research_portal_scraper.py          # Step 1: Web scraping
├── extract_identifiers.py              # Step 2: Identifier extraction
├── filter_authors.py                   # Step 3: Author filtering
├── filter_works.py                     # Step 4: Works filtering
├── README.md
└── data/
    ├── uam_authors.json                # [Required] Input: OpenAlex authors
    ├── uam_works.json                  # [Required] Input: OpenAlex works
    ├── scientists_data.csv             # [Output] Step 1 results
    ├── scientists_with_identifiers.csv # [Output] Step 2 results
    ├── wmii_authors.json               # [Output] Step 3 results
    └── wmii_works.json                 # [Output] Step 4 results
```
