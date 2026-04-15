# OpenAlex Abstract Scraper Project

This project scrapes publication data and abstracts from OpenAlex API for researchers based on their ORCID IDs.

## 📁 Project Structure

```
abstracts/
├── abstract_scraper.py              # Main API scraper (uses OpenAlex API)
├── fetch_missing_abstracts.py       # Fetches missing abstracts from DOI links (Selenium)
├── seperate.py                      # Splits data by abstract availability
├── requirements.txt                 # Python dependencies
├── docker-compose.yml               # Docker setup fo DB
├── load_dump.sh                     # Database loading script
├── data/                            # Data directory
│   ├── wmii_orcid.csv               # Input: List of ORCID IDs to scrape
│   ├── openalex_all_results.csv     # Output: Raw scraped data
│   ├── openalex_all_results_complete.csv  # Output: Data with fetched abstracts
│   ├── openalex_all_results_statistics.csv # Output: Per-ORCID statistics
│   ├── titles_with_abstracts.csv    # Output: Records WITH abstracts
│   ├── titles_without_abstracts.csv # Output: Records WITHOUT abstracts
│   └── devdb_backup_new.dump        # Database backup
└── scripts/
    └── init-db.sh                   # Database initialization script
```

## CSV Files Explained

### Input Files

| File | Description |
|------|-------------|
| `wmii_orcid.csv` | **Input file** containing ORCID IDs to scrape. Format: one column named `orcid` with ORCID numbers (e.g., 0000-0003-3542-8982) |

### Output Files

| File | Description | Created By |
|------|-------------|------------|
| `openalex_all_results.csv` | **Raw scraped data** from OpenAlex API. Contains all publication metadata but may have missing abstracts. | `abstract_scraper.py` |
| `openalex_all_results_complete.csv` | **Enhanced data** with abstracts fetched from DOI links. More complete than raw results. | `fetch_missing_abstracts.py` |
| `openalex_all_results_statistics.csv` | **Per-ORCID statistics** showing total works, abstracts found/missing, citation counts for each researcher. | `abstract_scraper.py` |
| `titles_with_abstracts.csv` | **Filtered data** containing ONLY records that have abstracts. Ready for analysis. | `seperate.py` |
| `titles_without_abstracts.csv` | **Filtered data** containing ONLY records that are still missing abstracts. Can be used for further processing. | `seperate.py` |

### Column Structure

All output CSV files contain the following columns:

- `main_author_orcid` - The researcher's ORCID ID
- `openalex_id` - OpenAlex work ID (e.g., W4200584002)
- `title` - Publication title
- `publication_year` - Year published
- `publication_date` - Full publication date
- `doi` - Digital Object Identifier (DOI link)
- `type` - Publication type (article, book, etc.)
- `cited_by_count` - Number of citations
- `journal` - Journal/venue name
- `topics` - Research topics (semicolon-separated)
- `co_authors` - Co-author names (semicolon-separated)
- `co_author_orcids` - Co-author ORCID IDs (semicolon-separated)
- `num_co_authors` - Number of co-authors
- `abstract` - Publication abstract (if available)
- `keywords` - Keywords (semicolon-separated)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Scraper

```bash
# Step 1: Scrape data from OpenAlex API
python abstract_scraper.py
# Creates: openalex_all_results.csv, openalex_all_results_statistics.csv

# Step 2: Fetch missing abstracts from DOI links (optional, uses Selenium)
python fetch_missing_abstracts.py
# Creates: openalex_all_results_complete.csv

# Step 3: Split data by abstract availability (optional)
python seperate.py
# Creates: titles_with_abstracts.csv, titles_without_abstracts.csv
```

## Workflow

```
ORCID IDs (wmii_orcid.csv)
        ↓
    API Scraper (abstract_scraper.py)
        ↓
Raw Results (openalex_all_results.csv) + Statistics
        ↓
Abstract Fetcher (fetch_missing_abstracts.py) [optional]
        ↓
Complete Results (openalex_all_results_complete.csv)
        ↓
    Separator (seperate.py) [optional]
        ↓
    ┌───────────────────┴────────────────────┐
    ↓                                        ↓
With Abstracts                      Without Abstracts
(titles_with_abstracts.csv)        (titles_without_abstracts.csv)
```

## Scripts

### `abstract_scraper.py`
- **Purpose**: Main scraper using OpenAlex API
- **Input**: `data/wmii_orcid.csv`
- **Output**: `openalex_all_results.csv`, `openalex_all_results_statistics.csv`
- **Features**:
  - Fetches all publications for each ORCID
  - Extracts co-author ORCIDs
  - Converts abstract inverted index to readable text
  - Generates per-ORCID statistics
  - Fast (API-based, no browser needed)

### `fetch_missing_abstracts.py`
- **Purpose**: Fetches abstracts from DOI links using Selenium
- **Input**: `openalex_all_results.csv`
- **Output**: `openalex_all_results_complete.csv`
- **Features**:
  - Only processes records with missing abstracts
  - Supports multiple publishers (ScienceDirect, Springer, Wiley, etc.)
  - Saves progress every 10 records
  - Shows success/failure statistics

### `seperate.py`
- **Purpose**: Splits data by abstract availability
- **Input**: `openalex_all_results_complete.csv`
- **Output**: `titles_with_abstracts.csv`, `titles_without_abstracts.csv`
- **Features**:
  - Fills abstracts from duplicates (if one version has abstract, copies to others)
  - Splits into two clean datasets
  - Shows before/after statistics

## Database

The project includes PostgreSQL setup:
- `docker-compose.yml` - Docker configuration
- `scripts/init-db.sh` - Database initialization
- `data/devdb_backup_new.dump` - Database backup
