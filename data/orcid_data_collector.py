#!/usr/bin/env python3
"""
ORCID Data Collector - Updated for New Database Schema

Fetches researcher data from ORCID API and stores it in SQLite database
matching the new schema design with separate Institutions and Emails tables.
"""

import sqlite3
import pandas as pd
import requests
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import time
import logging

# Configure logging to file
import datetime
log_filename = f"orcid_collection_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()  # Also keep console output
    ]
)
logger = logging.getLogger(__name__)

# Set debug logging for development
def enable_debug_logging():
    """Enable debug logging to see detailed API responses."""
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers:
        handler.setLevel(logging.DEBUG)

# ORCID API Configuration
HEADERS = {"Accept": "application/json"}
BASE_URL = "https://pub.orcid.org/v3.0"

# Simple rate limiting for Anonymous API: 12 req/sec, 25k reads/day
SIMPLE_DELAY = 0.1  # 100ms between requests (10 req/sec, safely under 12)

class ORCIDDataCollector:
    def __init__(self, db_path: str = "scientists.db"):
        """Initialize the ORCID data collector with database connection."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.request_count = 0
        self.setup_database()
        
    def setup_database(self):
        """Create database tables according to new schema."""
        cursor = self.conn.cursor()
        
        # Researchers table (updated schema)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Researchers (
                id TEXT PRIMARY KEY,
                orcid_id TEXT UNIQUE,
                first_name TEXT,
                last_name TEXT,
                full_name TEXT,
                country TEXT,
                current_affiliation TEXT,
                degree TEXT,
                field TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                bio TEXT
            )
        ''')
        
        # Sources table (unchanged)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Sources (
                id TEXT PRIMARY KEY,
                researcher_id TEXT,
                platform TEXT,
                source_id TEXT,
                endpoint TEXT,
                last_fetched TIMESTAMP,
                last_status TEXT,
                active BOOLEAN,
                FOREIGN KEY (researcher_id) REFERENCES Researchers(id)
            )
        ''')
        
        # Institutions table (NEW)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Institutions (
                id TEXT PRIMARY KEY,
                ror_id TEXT UNIQUE,
                grid_id TEXT,
                name TEXT,
                aliases TEXT,
                country TEXT,
                city TEXT,
                organization_type TEXT,
                url TEXT,
                parent_ror TEXT
            )
        ''')
        
        # Affiliations table (updated with institution FK)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Affiliations (
                id TEXT PRIMARY KEY,
                researcher_id TEXT,
                institution_id TEXT,
                department TEXT,
                role TEXT,
                country TEXT,
                start_date DATE,
                end_date DATE,
                FOREIGN KEY (researcher_id) REFERENCES Researchers(id),
                FOREIGN KEY (institution_id) REFERENCES Institutions(id)
            )
        ''')
        
        # Education table (unchanged)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Education (
                id TEXT PRIMARY KEY,
                researcher_id TEXT,
                degree TEXT,
                field TEXT,
                institution TEXT,
                country TEXT,
                start_date DATE,
                end_date DATE,
                FOREIGN KEY (researcher_id) REFERENCES Researchers(id)
            )
        ''')
        
        # Keywords table (unchanged)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Keywords (
                id TEXT PRIMARY KEY,
                researcher_id TEXT,
                keyword TEXT,
                FOREIGN KEY (researcher_id) REFERENCES Researchers(id)
            )
        ''')
        
        # Emails table (NEW)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Emails (
                id TEXT PRIMARY KEY,
                researcher_id TEXT,
                email TEXT,
                FOREIGN KEY (researcher_id) REFERENCES Researchers(id)
            )
        ''')
        
        # Publications table (updated with abstract)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Publications (
                id TEXT PRIMARY KEY,
                title TEXT,
                journal TEXT,
                doi TEXT,
                year TEXT,
                source TEXT,
                project_id TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                abstract TEXT
            )
        ''')
        
        # Publication_author_project table (NEW - enhanced junction)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Publication_author_project (
                id TEXT PRIMARY KEY,
                publication_id TEXT,
                researcher_id TEXT,
                author_order INTEGER,
                is_corresponding TEXT,
                evidence_score INTEGER,
                FOREIGN KEY (publication_id) REFERENCES Publications(id),
                FOREIGN KEY (researcher_id) REFERENCES Researchers(id)
            )
        ''')
        
        self.conn.commit()
        logger.info("Database tables created successfully")

    def safe_get(self, data: dict, *keys) -> Optional[Any]:
        """Safely extract nested dictionary values."""
        for key in keys:
            data = data.get(key) if isinstance(data, dict) else {}
        return data or None

    def safe_date_component(self, value, default="01") -> str:
        """Extract date component safely."""
        return value.get("value") if isinstance(value, dict) and value.get("value") else default

    def fetch_orcid_person(self, orcid_id: str) -> Dict:
        """Fetch person data from ORCID API with rate limit handling."""
        max_retries = 3
        backoff_delay = 1
        
        for attempt in range(max_retries):
            time.sleep(SIMPLE_DELAY)
            self.request_count += 1
            url = f"{BASE_URL}/{orcid_id}/person"
            
            try:
                response = requests.get(url, headers=HEADERS)
                if response.status_code == 429:  # Rate limit exceeded
                    wait_time = backoff_delay * (2 ** attempt)
                    logger.warning(f"Rate limit hit for {orcid_id}, waiting {wait_time}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                    
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    logger.error(f"Error fetching person data for {orcid_id}: {e}")
                    return {}
                else:
                    time.sleep(backoff_delay * (2 ** attempt))
        
        return {}

    def fetch_orcid_employment(self, orcid_id: str) -> Dict:
        """Fetch employment data from ORCID API."""
        time.sleep(SIMPLE_DELAY)
        self.request_count += 1
        url = f"{BASE_URL}/{orcid_id}/employments"
        try:
            response = requests.get(url, headers=HEADERS)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching employment for {orcid_id}: {e}")
            return {}

    def fetch_orcid_education(self, orcid_id: str) -> Dict:
        """Fetch education data from ORCID API."""
        time.sleep(SIMPLE_DELAY)
        self.request_count += 1
        url = f"{BASE_URL}/{orcid_id}/educations"
        try:
            response = requests.get(url, headers=HEADERS)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching education for {orcid_id}: {e}")
            return {}

    def fetch_orcid_works(self, orcid_id: str) -> Dict:
        """Fetch works/publications from ORCID API."""
        time.sleep(SIMPLE_DELAY)
        self.request_count += 1
        url = f"{BASE_URL}/{orcid_id}/works"
        try:
            response = requests.get(url, headers=HEADERS)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching works for {orcid_id}: {e}")
            return {}

    def get_or_create_institution(self, institution_name: str, country: str = None) -> str:
        """Get existing institution or create new one."""
        cursor = self.conn.cursor()
        
        # Try to find existing institution
        cursor.execute("SELECT id FROM Institutions WHERE name = ?", (institution_name,))
        result = cursor.fetchone()
        
        if result:
            return result[0]
        
        # Create new institution
        institution_id = str(uuid.uuid4())
        cursor.execute('''
            INSERT INTO Institutions (id, name, country)
            VALUES (?, ?, ?)
        ''', (institution_id, institution_name, country))
        self.conn.commit()
        
        return institution_id

    def parse_orcid_person(self, orcid_id: str, person_json: Dict, employment_json: Dict, education_json: Dict) -> Dict:
        """Parse ORCID person data according to new schema."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        researcher_id = str(uuid.uuid4())

        # Basic person info
        first = self.safe_get(person_json, "name", "given-names", "value")
        last = self.safe_get(person_json, "name", "family-name", "value")
        full_name = f"{first} {last}".strip() if first or last else ""
        
        # Country from address
        address_list = self.safe_get(person_json, "addresses", "address") or []
        country = self.safe_get(address_list[0], "country", "value") if address_list else None
        
        # Bio from person data
        bio = self.safe_get(person_json, "biography", "content")

        # Researcher record
        researcher = {
            "id": researcher_id,
            "orcid_id": orcid_id,
            "first_name": first,
            "last_name": last,
            "full_name": full_name,
            "country": country,
            "current_affiliation": None,  # Will be set from most recent employment
            "degree": None,  # Will be set from highest education
            "field": None,   # Will be set from most recent education field
            "created_at": now,
            "updated_at": now,
            "bio": bio
        }

        # Sources
        sources = [{
            "id": str(uuid.uuid4()),
            "researcher_id": researcher_id,
            "platform": "ORCID",
            "source_id": orcid_id,
            "endpoint": f"{BASE_URL}/{orcid_id}/person",
            "last_fetched": now,
            "last_status": "200",
            "active": True
        }]

        # Keywords
        keywords = []
        for kw in self.safe_get(person_json, "keywords", "keyword") or []:
            content = self.safe_get(kw, "content")
            if content:
                keywords.append({
                    "id": str(uuid.uuid4()),
                    "researcher_id": researcher_id,
                    "keyword": content
                })

        # Emails
        emails = []
        for email in self.safe_get(person_json, "emails", "email") or []:
            email_addr = self.safe_get(email, "email")
            if email_addr:
                emails.append({
                    "id": str(uuid.uuid4()),
                    "researcher_id": researcher_id,
                    "email": email_addr
                })

        # Affiliations with institution references
        affiliations = []
        most_recent_affiliation = None
        most_recent_start = None

        for group in self.safe_get(employment_json, "affiliation-group") or []:
            for summary in self.safe_get(group, "summaries") or []:
                emp = self.safe_get(summary, "employment-summary")
                org = self.safe_get(emp, "organization")
                address = self.safe_get(org, "address")
                
                institution_name = self.safe_get(org, "name")
                if not institution_name:
                    continue
                    
                # Get or create institution
                institution_id = self.get_or_create_institution(
                    institution_name, 
                    self.safe_get(address, "country")
                )

                start = self.safe_get(emp, "start-date")
                end = self.safe_get(emp, "end-date")

                start_date = None
                if start:
                    start_date = f"{self.safe_date_component(self.safe_get(start, 'year'), '1900')}-" \
                                 f"{self.safe_date_component(self.safe_get(start, 'month'))}-" \
                                 f"{self.safe_date_component(self.safe_get(start, 'day'))}"

                end_date = None
                if self.safe_get(end, "year"):
                    end_date = f"{self.safe_date_component(self.safe_get(end, 'year'))}-" \
                               f"{self.safe_date_component(self.safe_get(end, 'month'))}-" \
                               f"{self.safe_date_component(self.safe_get(end, 'day'))}"

                affiliations.append({
                    "id": str(uuid.uuid4()),
                    "researcher_id": researcher_id,
                    "institution_id": institution_id,
                    "department": self.safe_get(emp, "department-name"),
                    "role": self.safe_get(emp, "role-title"),
                    "country": self.safe_get(address, "country"),
                    "start_date": start_date,
                    "end_date": end_date
                })

                # Track most recent affiliation
                if not end_date and start_date:
                    parsed_start = pd.to_datetime(start_date, errors="coerce")
                    if parsed_start and (most_recent_start is None or parsed_start > most_recent_start):
                        most_recent_affiliation = institution_name
                        most_recent_start = parsed_start

        researcher["current_affiliation"] = most_recent_affiliation

        # Education
        education = []
        highest_degree = None
        most_recent_field = None

        for group in self.safe_get(education_json, "affiliation-group") or []:
            for summary in self.safe_get(group, "summaries") or []:
                edu = self.safe_get(summary, "education-summary")
                org = self.safe_get(edu, "organization")
                address = self.safe_get(org, "address")

                degree = self.safe_get(edu, "role-title")
                field = self.safe_get(edu, "department-name")

                start = self.safe_get(edu, "start-date")
                end = self.safe_get(edu, "end-date")

                start_date = None
                if start:
                    start_date = f"{self.safe_date_component(self.safe_get(start, 'year'), '1900')}-" \
                                 f"{self.safe_date_component(self.safe_get(start, 'month'))}-" \
                                 f"{self.safe_date_component(self.safe_get(start, 'day'))}"

                end_date = None
                if self.safe_get(end, "year"):
                    end_date = f"{self.safe_date_component(self.safe_get(end, 'year'))}-" \
                               f"{self.safe_date_component(self.safe_get(end, 'month'))}-" \
                               f"{self.safe_date_component(self.safe_get(end, 'day'))}"

                education.append({
                    "id": str(uuid.uuid4()),
                    "researcher_id": researcher_id,
                    "degree": degree,
                    "field": field,
                    "institution": self.safe_get(org, "name"),
                    "country": self.safe_get(address, "country"),
                    "start_date": start_date,
                    "end_date": end_date
                })

                # Track highest degree and most recent field
                if degree and ("PhD" in degree or "Ph.D" in degree or "Doctor" in degree):
                    highest_degree = degree
                    if field:
                        most_recent_field = field

        researcher["degree"] = highest_degree
        researcher["field"] = most_recent_field

        return {
            "researcher": researcher,
            "sources": sources,
            "keywords": keywords,
            "emails": emails,
            "affiliations": affiliations,
            "education": education
        }

    def parse_orcid_works(self, works_json: Dict, researcher_id: str) -> Dict:
        """Parse ORCID works/publications data."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        publications = []
        publication_authors = []

        for group in self.safe_get(works_json, "group") or []:
            summaries = self.safe_get(group, "work-summary") or []
            if not summaries:
                continue

            summary = summaries[0]
            pub_id = str(uuid.uuid4())

            title = self.safe_get(summary, "title", "title", "value")
            journal = self.safe_get(summary, "journal-title", "value")
            year = self.safe_get(summary, "publication-date", "year", "value")

            # Extract DOI
            doi = None
            for ext in self.safe_get(summary, "external-ids", "external-id") or []:
                ext_type = self.safe_get(ext, "external-id-type")
                ext_val = self.safe_get(ext, "external-id-value")
                if ext_type and ext_type.lower() == "doi" and ext_val:
                    doi = f"https://doi.org/{ext_val}"
                    break

            publications.append({
                "id": pub_id,
                "title": title,
                "journal": journal,
                "doi": doi,
                "year": year,
                "source": "ORCID",
                "project_id": None,
                "created_at": now,
                "updated_at": now,
                "abstract": None  # ORCID API doesn't provide abstracts in summary
            })

            publication_authors.append({
                "id": str(uuid.uuid4()),
                "publication_id": pub_id,
                "researcher_id": researcher_id,
                "author_order": None,
                "is_corresponding": None,
                "evidence_score": None
            })

        return {
            "publications": publications,
            "publication_authors": publication_authors
        }

    def insert_researcher_data(self, data: Dict, publications_data: Dict):
        """Insert parsed researcher data into database."""
        cursor = self.conn.cursor()
        
        try:
            # Insert researcher
            cursor.execute('''
                INSERT INTO Researchers 
                (id, orcid_id, first_name, last_name, full_name, country, current_affiliation, 
                 degree, field, created_at, updated_at, bio)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data["researcher"]["id"], data["researcher"]["orcid_id"],
                data["researcher"]["first_name"], data["researcher"]["last_name"],
                data["researcher"]["full_name"], data["researcher"]["country"],
                data["researcher"]["current_affiliation"], data["researcher"]["degree"],
                data["researcher"]["field"], data["researcher"]["created_at"],
                data["researcher"]["updated_at"], data["researcher"]["bio"]
            ))

            # Insert sources
            for source in data["sources"]:
                cursor.execute('''
                    INSERT INTO Sources 
                    (id, researcher_id, platform, source_id, endpoint, last_fetched, last_status, active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', tuple(source.values()))

            # Insert keywords
            for keyword in data["keywords"]:
                cursor.execute('''
                    INSERT INTO Keywords (id, researcher_id, keyword)
                    VALUES (?, ?, ?)
                ''', tuple(keyword.values()))

            # Insert emails
            for email in data["emails"]:
                cursor.execute('''
                    INSERT INTO Emails (id, researcher_id, email)
                    VALUES (?, ?, ?)
                ''', tuple(email.values()))

            # Insert affiliations
            for affiliation in data["affiliations"]:
                cursor.execute('''
                    INSERT INTO Affiliations 
                    (id, researcher_id, institution_id, department, role, country, start_date, end_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', tuple(affiliation.values()))

            # Insert education
            for edu in data["education"]:
                cursor.execute('''
                    INSERT INTO Education 
                    (id, researcher_id, degree, field, institution, country, start_date, end_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', tuple(edu.values()))

            # Insert publications
            for pub in publications_data["publications"]:
                cursor.execute('''
                    INSERT INTO Publications 
                    (id, title, journal, doi, year, source, project_id, created_at, updated_at, abstract)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', tuple(pub.values()))

            # Insert publication authors
            for pub_author in publications_data["publication_authors"]:
                cursor.execute('''
                    INSERT INTO Publication_author_project 
                    (id, publication_id, researcher_id, author_order, is_corresponding, evidence_score)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', tuple(pub_author.values()))

            self.conn.commit()
            logger.info(f"Successfully inserted data for researcher {data['researcher']['orcid_id']}")

        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error inserting data for {data['researcher']['orcid_id']}: {e}")
            raise

    def collect_researcher_data(self, orcid_id: str):
        """Collect and store data for a single researcher."""
        logger.info(f"Collecting data for {orcid_id}")
        
        # Fetch data from ORCID API
        person_data = self.fetch_orcid_person(orcid_id)
        employment_data = self.fetch_orcid_employment(orcid_id)
        education_data = self.fetch_orcid_education(orcid_id)
        works_data = self.fetch_orcid_works(orcid_id)

        if not person_data:
            logger.warning(f"No person data found for {orcid_id}")
            return

        # Parse the data
        researcher_data = self.parse_orcid_person(orcid_id, person_data, employment_data, education_data)
        publications_data = self.parse_orcid_works(works_data, researcher_data["researcher"]["id"])

        # Insert into database
        self.insert_researcher_data(researcher_data, publications_data)

    def get_orcid_ids_by_query(self, query: str, count: int = 100, country: str = None) -> List[str]:
        """Search for ORCID IDs by query with optional country filter."""
        headers = {"Accept": "application/json"}
        rows = []
        start = 0
        
        # Add country filter to query
        search_query = query
        if country:
            search_query = f"{query} AND current-institution-affiliation-name:*{country}*"
        
        logger.debug(f"Starting search for query: {search_query}")
        
        while len(rows) < count:
            time.sleep(SIMPLE_DELAY)
            self.request_count += 1
            url = f"https://pub.orcid.org/v3.0/expanded-search/?q={search_query}&start={start}&rows=100"
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                results = data.get("expanded-result", [])
                total_found = data.get("num-found", 0)
                
                logger.debug(f"API response: found {len(results)} results (total available: {total_found}, start: {start})")
                
                if not results:
                    logger.info(f"No more results for query: {search_query}")
                    break
                    
                new_ids = [res["orcid-id"] for res in results]
                rows.extend(new_ids)
                start += 100
                
                logger.debug(f"Collected {len(new_ids)} new IDs, total: {len(rows)}")
                
                # If we got less than 100 results, we've reached the end
                if len(results) < 100:
                    logger.info(f"Reached end of results for query: {search_query}")
                    break
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Error searching ORCID for query '{search_query}': {e}")
                break
            except Exception as e:
                logger.error(f"Unexpected error searching ORCID for query '{search_query}': {e}")
                break
                
        final_results = rows[:count] if rows else []
        logger.info(f"Query '{search_query}' returned {len(final_results)} researchers (requested: {count})")
        return final_results

    def get_polish_researchers(self, max_per_pattern: int = 10000) -> List[str]:
        """Get Polish researchers using various Polish institution patterns."""
        # Search patterns for Polish institutions
        polish_patterns = [
            # Country/language anchors
            "current-institution-affiliation-name:*Poland*",
            "current-institution-affiliation-name:*Polska*",
            "current-institution-affiliation-name:*Rzeczpospolita*",
            "current-institution-affiliation-name:*PL*",

            # Generic org types (PL + EN)
            "current-institution-affiliation-name:*Uniwersytet*",
            "current-institution-affiliation-name:*University*",
            "current-institution-affiliation-name:*Politechnika*",
            "current-institution-affiliation-name:*University of Technology*",
            "current-institution-affiliation-name:*Akademia*",
            "current-institution-affiliation-name:*Academy*",
            "current-institution-affiliation-name:*Instytut*",
            "current-institution-affiliation-name:*Institute*",
            "current-institution-affiliation-name:*PAN*",
            "current-institution-affiliation-name:*Polish Academy of Sciences*",
            "current-institution-affiliation-name:*Sieć Badawcza Łukasiewicz*",
            "current-institution-affiliation-name:*Łukasiewicz*",
            "current-institution-affiliation-name:*Państwowy Instytut Badawczy*",
            "current-institution-affiliation-name:*National Research Institute*",
            "current-institution-affiliation-name:*Narodowe Centrum*",
            "current-institution-affiliation-name:*Centrum Badan*",
            "current-institution-affiliation-name:*Centrum Badań*",
            "current-institution-affiliation-name:*Collegium Medicum*",
            "current-institution-affiliation-name:*Medical University*",
            "current-institution-affiliation-name:*Uniwersytet Medyczny*",
            "current-institution-affiliation-name:*Uniwersytet Przyrodniczy*",
            "current-institution-affiliation-name:*University of Life Sciences*",
            "current-institution-affiliation-name:*Uniwersytet Ekonomiczny*",
            "current-institution-affiliation-name:*University of Economics*",
            "current-institution-affiliation-name:*Szkoła Główna Handlowa*",
            "current-institution-affiliation-name:*SGH*",
            "current-institution-affiliation-name:*AWF*",
            "current-institution-affiliation-name:*Academy of Physical Education*",
            "current-institution-affiliation-name:*Polish Geological Institute*",
            "current-institution-affiliation-name:*National Centre for Nuclear Research*",
            "current-institution-affiliation-name:*Narodowe Centrum Badań Jądrowych*",

            # Major city anchors (EN + PL + diacritics + common misspellings)
            "current-institution-affiliation-name:*Warsaw*",
            "current-institution-affiliation-name:*Warszawa*",
            "current-institution-affiliation-name:*Krakow*",
            "current-institution-affiliation-name:*Kraków*",
            "current-institution-affiliation-name:*Wroclaw*",
            "current-institution-affiliation-name:*Wrocław*",
            "current-institution-affiliation-name:*Poznan*",
            "current-institution-affiliation-name:*Poznań*",
            "current-institution-affiliation-name:*Gdansk*",
            "current-institution-affiliation-name:*Gdańsk*",
            "current-institution-affiliation-name:*Lodz*",
            "current-institution-affiliation-name:*Łódź*",
            "current-institution-affiliation-name:*Szczecin*",
            "current-institution-affiliation-name:*Gdynia*",
            "current-institution-affiliation-name:*Bydgoszcz*",
            "current-institution-affiliation-name:*Toruń*",
            "current-institution-affiliation-name:*Torun*",
            "current-institution-affiliation-name:*Lublin*",
            "current-institution-affiliation-name:*Białystok*",
            "current-institution-affiliation-name:*Bialystok*",
            "current-institution-affiliation-name:*Rzeszów*",
            "current-institution-affiliation-name:*Rzeszow*",
            "current-institution-affiliation-name:*Olsztyn*",
            "current-institution-affiliation-name:*Katowice*",
            "current-institution-affiliation-name:*Opole*",
            "current-institution-affiliation-name:*Kielce*",
            "current-institution-affiliation-name:*Zielona Góra*",
            "current-institution-affiliation-name:*Zielona Gora*",
            "current-institution-affiliation-name:*Gliwice*",
            "current-institution-affiliation-name:*Częstochowa*",
            "current-institution-affiliation-name:*Czestochowa*",
            "current-institution-affiliation-name:*Sopot*",
            "current-institution-affiliation-name:*Radom*",
            "current-institution-affiliation-name:*Płock*",
            "current-institution-affiliation-name:*Plock*",
            "current-institution-affiliation-name:*Tarnów*",
            "current-institution-affiliation-name:*Tarnow*",
        ]

        
        all_ids = set()
        
        for pattern in polish_patterns:
            logger.info(f"🔍 Searching: {pattern}")
            ids = self.get_orcid_ids_by_query(pattern, count=max_per_pattern)
            if ids is not None:
                all_ids.update(ids)
                logger.info(f"✅ Found {len(ids)} researchers with pattern: {pattern} (total unique: {len(all_ids)})")
            else:
                logger.warning(f"⚠️ No results for pattern: {pattern}")
                
        logger.info(f"🎯 Total unique researchers from all patterns: {len(all_ids)}")
        return list(all_ids)

    def get_researchers_by_institution(self, institution_names: List[str], max_per_institution: int = 999999) -> List[str]:
        """Get researchers by specific institution names."""
        all_ids = set()
        
        for institution in institution_names:
            query = f'current-institution-affiliation-name:"{institution}"'
            logger.info(f"🏛️ Searching institution: {institution}")
            ids = self.get_orcid_ids_by_query(query, count=max_per_institution)
            if ids is not None:
                all_ids.update(ids)
                logger.info(f"✅ Found {len(ids)} researchers at {institution} (total unique: {len(all_ids)})")
            else:
                logger.warning(f"⚠️ No results for institution: {institution}")
            
        logger.info(f"🎯 Total unique researchers from all institutions: {len(all_ids)}")
        return list(all_ids)

    def collect_diverse_researchers(self, topics: List[str], per_topic: int = 70):
        """Collect diverse researchers across multiple topics."""
        all_ids = set()
        
        for topic in topics:
            logger.info(f"Searching for researchers in: {topic}")
            ids = self.get_orcid_ids_by_query(topic.replace(" ", "+"), count=per_topic)
            all_ids.update(ids)
            time.sleep(1)  # Rate limiting
            
        logger.info(f"Found {len(all_ids)} unique ORCID IDs")
        return list(all_ids)

    def collect_batch_data(self, orcid_ids: List[str]):
        """Collect data for multiple researchers with rate limiting."""
        total = len(orcid_ids)
        
        for i, orcid_id in enumerate(orcid_ids, 1):
            try:
                logger.info(f"👨‍🔬 Processing {i}/{total}: {orcid_id}")
                self.collect_researcher_data(orcid_id)
                
                # Log progress every 10 researchers
                if i % 10 == 0:
                    progress_pct = (i / total) * 100
                    logger.info(f"📊 Progress: {i}/{total} researchers ({progress_pct:.1f}%) - {self.request_count} API calls made")
                    
            except Exception as e:
                logger.error(f"❌ Failed to process {orcid_id}: {e}")
                continue

    def save_orcid_ids_to_file(self, orcid_ids: List[str], filename: str):
        """Save ORCID IDs to a file for future processing."""
        try:
            with open(filename, 'w') as f:
                for orcid_id in orcid_ids:
                    f.write(f"{orcid_id}\n")
            logger.info(f"Saved {len(orcid_ids)} ORCID IDs to {filename}")
        except Exception as e:
            logger.error(f"Failed to save ORCID IDs to {filename}: {e}")

    def load_orcid_ids_from_file(self, filename: str) -> List[str]:
        """Load ORCID IDs from a file."""
        try:
            with open(filename, 'r') as f:
                ids = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(ids)} ORCID IDs from {filename}")
            return ids
        except Exception as e:
            logger.error(f"Failed to load ORCID IDs from {filename}: {e}")
            return []

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

def main():
    """Main function to collect Polish researchers."""
    # Major Polish universities and institutions
    polish_institutions = [
        # Flagship universities
        "University of Warsaw",
        "Uniwersytet Warszawski",
        "Jagiellonian University",
        "Uniwersytet Jagielloński",
        "Adam Mickiewicz University in Poznań",
        "Uniwersytet im. Adama Mickiewicza w Poznaniu",
        "Nicolaus Copernicus University in Toruń",
        "Uniwersytet Mikołaja Kopernika w Toruniu",
        "University of Wrocław",
        "Uniwersytet Wrocławski",
        "University of Gdańsk",
        "Uniwersytet Gdański",
        "University of Łódź",
        "Uniwersytet Łódzki",
        "University of Silesia in Katowice",
        "Uniwersytet Śląski w Katowicach",
        "Maria Curie-Skłodowska University",
        "Uniwersytet Marii Curie‑Skłodowskiej",
        "University of Szczecin",
        "Uniwersytet Szczeciński",
        "University of Warmia and Mazury in Olsztyn",
        "Uniwersytet Warmińsko‑Mazurski w Olsztynie",
        "University of Rzeszów",
        "Uniwersytet Rzeszowski",
        "University of Białystok",
        "Uniwersytet w Białymstoku",
        "Kazimierz Wielki University in Bydgoszcz",
        "Uniwersytet Kazimierza Wielkiego w Bydgoszczy",
        "Jan Kochanowski University in Kielce",
        "Uniwersytet Jana Kochanowskiego w Kielcach",
        "University of Zielona Góra",
        "Uniwersytet Zielonogórski",
        "Opole University",
        "Uniwersytet Opolski",
        "University of Silesia in Katowice",
        "Collegium Medicum in Katowice",  # some profiles include CM units
        "Cardinal Stefan Wyszyński University in Warsaw",
        "Uniwersytet Kardynała Stefana Wyszyńskiego w Warszawie",
        "The John Paul II Catholic University of Lublin",
        "Katolicki Uniwersytet Lubelski Jana Pawła II (KUL)",

        # Universities of Technology / Polytechnics
        "Warsaw University of Technology",
        "Politechnika Warszawska",
        "AGH University of Krakow",
        "Akademia Górniczo‑Hutnicza (AGH)",
        "Wrocław University of Science and Technology",
        "Politechnika Wrocławska",
        "Poznan University of Technology",
        "Politechnika Poznańska",
        "Gdańsk University of Technology",
        "Politechnika Gdańska",
        "Lodz University of Technology",
        "Politechnika Łódzka",
        "Silesian University of Technology",
        "Politechnika Śląska",
        "Cracow University of Technology",
        "Politechnika Krakowska",
        "Bialystok University of Technology",
        "Politechnika Białostocka",
        "Rzeszow University of Technology",
        "Politechnika Rzeszowska",
        "Opole University of Technology",
        "Politechnika Opolska",
        "Częstochowa University of Technology",
        "Politechnika Częstochowska",
        "Kielce University of Technology",
        "Politechnika Świętokrzyska",
        "Koszalin University of Technology",
        "Politechnika Koszalińska",
        "Bydgoszcz University of Science and Technology",
        "Politechnika Bydgoska Jana i Jędrzeja Śniadeckich",
        "Siedlce University of Natural Sciences and Humanities",
        "Uniwersytet w Siedlcach",

        # Economics / Management
        "SGH Warsaw School of Economics",
        "Szkoła Główna Handlowa w Warszawie",
        "Cracow University of Economics",
        "Uniwersytet Ekonomiczny w Krakowie",
        "Poznań University of Economics and Business",
        "Uniwersytet Ekonomiczny w Poznaniu",
        "Wrocław University of Economics and Business",
        "Uniwersytet Ekonomiczny we Wrocławiu",
        "University of Economics in Katowice",
        "Uniwersytet Ekonomiczny w Katowicach",

        # Life sciences / Agriculture / Veterinary
        "Warsaw University of Life Sciences (SGGW)",
        "Szkoła Główna Gospodarstwa Wiejskiego w Warszawie",
        "Poznań University of Life Sciences",
        "Uniwersytet Przyrodniczy w Poznaniu",
        "University of Agriculture in Krakow",
        "Uniwersytet Rolniczy w Krakowie",
        "Wrocław University of Environmental and Life Sciences",
        "Uniwersytet Przyrodniczy we Wrocławiu",
        "University of Life Sciences in Lublin",
        "Uniwersytet Przyrodniczy w Lublinie",
        "West Pomeranian University of Technology in Szczecin",
        "Zachodniopomorski Uniwersytet Technologiczny w Szczecinie",
        "University of Bielsko‑Biala",
        "Akademia Techniczno‑Humanistyczna w Bielsku‑Białej",
        "National Veterinary Research Institute",
        "Państwowy Instytut Weterynaryjny – PIB",
        "Institute of Plant Protection – National Research Institute",
        "Instytut Ochrony Roślin – PIB",

        # Medical universities & colleges
        "Medical University of Warsaw",
        "Warszawski Uniwersytet Medyczny",
        "Medical University of Gdańsk",
        "Gdański Uniwersytet Medyczny",
        "Medical University of Łódź",
        "Uniwersytet Medyczny w Łodzi",
        "Poznan University of Medical Sciences",
        "Uniwersytet Medyczny im. Karola Marcinkowskiego w Poznaniu",
        "Wroclaw Medical University",
        "Uniwersytet Medyczny im. Piastów Śląskich we Wrocławiu",
        "Medical University of Lublin",
        "Uniwersytet Medyczny w Lublinie",
        "Medical University of Bialystok",
        "Uniwersytet Medyczny w Białymstoku",
        "Pomeranian Medical University in Szczecin",
        "Pomorski Uniwersytet Medyczny w Szczecinie",
        "Medical University of Silesia in Katowice",
        "Śląski Uniwersytet Medyczny w Katowicach",
        "Jagiellonian University Medical College",
        "Collegium Medicum Uniwersytetu Jagiellońskiego",
        "Nicolaus Copernicus University Collegium Medicum in Bydgoszcz",
        "Collegium Medicum UMK w Bydgoszczy",

        # Maritime & transport
        "Gdynia Maritime University",
        "Uniwersytet Morski w Gdyni",
        "Maritime University of Szczecin",
        "Akademia Morska w Szczecinie",
        "Military University of Technology",
        "Wojskowa Akademia Techniczna",
        "Polish Naval Academy",
        "Akademia Marynarki Wojennej w Gdyni",

        # Physical education (AWF)
        "University of Physical Education in Warsaw",
        "Akademia Wychowania Fizycznego Józefa Piłsudskiego w Warszawie",
        "University School of Physical Education in Krakow",
        "Akademia Wychowania Fizycznego w Krakowie",
        "University School of Physical Education in Wrocław",
        "Akademia Wychowania Fizycznego we Wrocławiu",
        "University School of Physical Education in Poznań",
        "Akademia Wychowania Fizycznego w Poznaniu",
        "Academy of Physical Education and Sport in Gdańsk",
        "Akademia Wychowania Fizycznego i Sportu w Gdańsku",
        "Academy of Physical Education in Katowice",
        "Akademia Wychowania Fizycznego w Katowicach",

        # Arts / Music / Film
        "Academy of Fine Arts in Warsaw",
        "Akademia Sztuk Pięknych w Warszawie",
        "Jan Matejko Academy of Fine Arts in Krakow",
        "Akademia Sztuk Pięknych im. Jana Matejki w Krakowie",
        "Eugeniusz Geppert Academy of Fine Arts in Wrocław",
        "Akademia Sztuk Pięknych we Wrocławiu",
        "Academy of Fine Arts in Gdańsk",
        "Akademia Sztuk Pięknych w Gdańsku",
        "University of the Arts Poznań",
        "Uniwersytet Artystyczny w Poznaniu",
        "Strzemiński Academy of Fine Arts Łódź",
        "Akademia Sztuk Pięknych w Łodzi",
        "Krzysztof Kieślowski Film School (University of Silesia)",
        "Państwowa Wyższa Szkoła Filmowa, Telewizyjna i Teatralna w Łodzi",
        "Fryderyk Chopin University of Music",
        "Uniwersytet Muzyczny Fryderyka Chopina",
        "Krzysztof Penderecki Academy of Music in Krakow",
        "Akademia Muzyczna w Krakowie",
        "Karol Szymanowski Academy of Music in Katowice",
        "Akademia Muzyczna w Katowicach",
        "Ignacy Jan Paderewski Academy of Music in Poznań",
        "Akademia Muzyczna w Poznaniu",
        "Grażyna and Kiejstut Bacewicz Academy of Music in Łódź",
        "Akademia Muzyczna w Łodzi",
        "Stanislaw Moniuszko Academy of Music in Gdańsk",
        "Akademia Muzyczna w Gdańsku",
        "Karol Lipiński Academy of Music in Wrocław",
        "Akademia Muzyczna we Wrocławiu",
        "Feliks Nowowiejski Academy of Music in Bydgoszcz",
        "Akademia Muzyczna w Bydgoszczy",
        "Art Academy of Szczecin",
        "Akademia Sztuki w Szczecinie",

        # PAN (Polish Academy of Sciences) – examples (there are many more)
        "Institute of Physics, Polish Academy of Sciences",
        "Instytut Fizyki PAN",
        "Institute of Biochemistry and Biophysics, PAS",
        "Instytut Biochemii i Biofizyki PAN",
        "Nencki Institute of Experimental Biology, PAS",
        "Instytut Biologii Doświadczalnej im. M. Nenckiego PAN",
        "Institute of Bioorganic Chemistry, PAS",
        "Instytut Chemii Bioorganicznej PAN",
        "Institute of Organic Chemistry, PAS",
        "Instytut Chemii Organicznej PAN",
        "Institute of Physical Chemistry, PAS",
        "Instytut Chemii Fizycznej PAN",
        "Institute of High Pressure Physics, PAS",
        "Instytut Wysokich Ciśnień PAN",
        "Institute of Mathematics, PAS",
        "Instytut Matematyczny PAN",
        "Systems Research Institute, PAS",
        "Instytut Badań Systemowych PAN",
        "Institute of Computer Science, PAS",
        "Instytut Podstaw Informatyki PAN",
        "Institute of Fundamental Technological Research, PAS",
        "Instytut Podstawowych Problemów Techniki PAN",
        "Institute of Philosophy and Sociology, PAS",
        "Instytut Filozofii i Socjologii PAN",
        "Institute of Literary Research, PAS",
        "Instytut Badań Literackich PAN",
        "Institute of Polish Language, PAS",
        "Instytut Języka Polskiego PAN",
        "Institute of Geological Sciences, PAS",
        "Instytut Nauk Geologicznych PAN",
        "Institute of Geophysics, PAS",
        "Instytut Geofizyki PAN",
        "Space Research Centre, PAS",
        "Centrum Badań Kosmicznych PAN",
        "Institute of Oceanology, PAS",
        "Instytut Oceanologii PAN",
        "Mossakowski Medical Research Institute, PAS",
        "Instytut Medycyny Doświadczalnej i Klinicznej im. Mossakowskiego PAN",

        # Major national labs / networks / govt research
        "National Centre for Nuclear Research (NCBJ)",
        "Narodowe Centrum Badań Jądrowych",
        "Łukasiewicz Research Network – Institute of Aviation",
        "Sieć Badawcza Łukasiewicz – Instytut Lotnictwa",
        "Łukasiewicz – Industrial Research Institute for Automation and Measurements PIAP",
        "Sieć Badawcza Łukasiewicz – PIAP",
        "Łukasiewicz – Institute of Microelectronics and Photonics",
        "Łukasiewicz – Instytut Mikroelektroniki i Fotoniki",
        "Łukasiewicz – PORT Polish Center for Technology Development",
        "PORT Polski Ośrodek Rozwoju Technologii",
        "Łukasiewicz – Metal Forming Institute",
        "Łukasiewicz – Instytut Obróbki Plastycznej",
        "Institute of Meteorology and Water Management – NRI (IMGW‑PIB)",
        "Instytut Meteorologii i Gospodarki Wodnej – PIB",
        "Polish Geological Institute – NRI (PIG‑PIB)",
        "Państwowy Instytut Geologiczny – PIB",
        "Institute of Environmental Protection – NRI (IOŚ‑PIB)",
        "Instytut Ochrony Środowiska – PIB",
        "National Medicines Institute",
        "Narodowy Instytut Leków",
        "Nofer Institute of Occupational Medicine",
        "Instytut Medycyny Pracy im. prof. J. Nofera",
        "Jerzy Haber Institute of Catalysis and Surface Chemistry, PAS",
        "Instytut Katalizy i Fizykochemii Powierzchni im. J. Habera PAN",
        "Jerzy Kukuczka Academy of Physical Education in Katowice",
        "Akademia Wychowania Fizycznego im. Jerzego Kukuczki w Katowicach",
        "Institute of Aviation Medicine",
        "Wojskowy Instytut Medycyny Lotniczej",
        "Institute of Cardiology (NICardio)",
        "Narodowy Instytut Kardiologii",
        "Institute of Hematology and Transfusion Medicine",
        "Instytut Hematologii i Transfuzjologii",
        "National Institute of Public Health PZH‑PIB",
        "Narodowy Instytut Zdrowia Publicznego PZH‑PIB",
        "International Institute of Molecular and Cell Biology in Warsaw (IIMCB)",
        "Międzynarodowy Instytut Biologii Molekularnej i Komórkowej w Warszawie",
        "OPI National Research Institute",
        "Ośrodek Przetwarzania Informacji – PIB",
        "Poznań Supercomputing and Networking Center (PSNC)",
        "Poznańskie Centrum Superkomputerowo‑Sieciowe (PCSS)",
        "Interdisciplinary Centre for Mathematical and Computational Modelling, UW (ICM)",
        "Interdyscyplinarne Centrum Modelowania Matematycznego i Komputerowego UW",

        # Museums / cultural science units with research profiles
        "National Museum in Krakow",
        "Muzeum Narodowe w Krakowie",
        "National Museum in Warsaw",
        "Muzeum Narodowe w Warszawie",

        # Teacher / pedagogy (incl. new names)
        "University of the National Education Commission in Krakow",
        "Uniwersytet Komisji Edukacji Narodowej w Krakowie",
        "Pedagogical University of Krakow (legacy name)",
        "Akademia Pedagogiczna w Krakowie (legacy)",

        # Extra regional public universities & academies
        "Pomeranian University in Słupsk",
        "Akademia Pomorska w Słupsku",
        "University of Applied Sciences in Nysa",
        "Państwowa Akademia Nauk Stosowanych w Nysie",
        "State University of Applied Sciences in Krosno",
        "Państwowa Akademia Nauk Stosowanych w Krośnie",
        "Jacob of Paradies University in Gorzów Wielkopolski",
        "Akademia im. Jakuba z Paradyża w Gorzowie Wielkopolskim",
        "Academy of Silesia in Katowice (AEiS)",
        "Akademia Śląska w Katowicach",
    ]

    collector = ORCIDDataCollector("polish_scientists.db")
    
    try:
        logger.info(f"🚀 Starting unlimited Polish researcher collection...")
        logger.info(f"📄 Logging to: {log_filename}")
        
        # Load existing IDs from file if available
        ids_file = "polish_researchers_all.txt"
        existing_ids = collector.load_orcid_ids_from_file(ids_file)
        
        if existing_ids:
            logger.info(f"📋 Using existing IDs file with {len(existing_ids)} researchers")
            all_polish_ids = set(existing_ids)
            pattern_ids = []
            institution_ids = []
        else:
            logger.info("🔍 Starting fresh collection...")
            
            # Method 1: Search by general Polish patterns
            logger.info("🔍 Searching by Polish institution patterns...")
            pattern_ids = collector.get_polish_researchers()  # No limit!
            logger.info(f"📊 Found {len(pattern_ids)} researchers via patterns")
            
            # Method 2: Search by specific institutions
            logger.info("🏛️ Searching by specific Polish institutions...")
            institution_ids = collector.get_researchers_by_institution(polish_institutions)  # No limit!
            logger.info(f"📊 Found {len(institution_ids)} researchers via institutions")
            
            # Combine and deduplicate
            all_polish_ids = set(pattern_ids + institution_ids)
            logger.info(f"🎯 Total unique Polish researchers found: {len(all_polish_ids)}")
            
            # Save all found IDs for future use
            collector.save_orcid_ids_to_file(list(all_polish_ids), ids_file)
        
        # Collect detailed data for ALL researchers (no limits!)
        researchers_to_collect = list(all_polish_ids)
        
        logger.info(f"=== UNLIMITED COLLECTION SUMMARY ===")
        logger.info(f"• Found via patterns: {len(pattern_ids)}")
        logger.info(f"• Found via institutions: {len(institution_ids)}")
        logger.info(f"• Total unique researchers: {len(all_polish_ids)}")
        logger.info(f"• Will collect ALL {len(researchers_to_collect)} researchers (no limits!)")
        logger.info(f"• Estimated time: {len(researchers_to_collect) * 2} seconds (~{len(researchers_to_collect) * 2 / 60:.1f} minutes)")
        logger.info(f"• Estimated API calls: {len(researchers_to_collect) * 5} calls")
        logger.info("=====================================")
        
        # Note: IDs are saved above when doing fresh collection
        
        if researchers_to_collect:
            collector.collect_batch_data(researchers_to_collect)
            logger.info("✅ Polish researcher data collection completed successfully!")
        else:
            logger.warning("⚠️ No researchers to collect!")
        
    except KeyboardInterrupt:
        logger.info("Data collection interrupted by user")
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
    finally:
        collector.close()

def collect_polish_sample():
    """Quick function to collect a small sample of Polish researchers."""
    collector = ORCIDDataCollector("polish_sample.db")
    
    try:
        logger.info("Collecting Polish researchers sample...")
        polish_researchers = collector.get_polish_researchers(count=30)
        logger.info(f"Found {len(polish_researchers)} Polish researchers")
        
        if polish_researchers:
            collector.collect_batch_data(polish_researchers[:15])
            logger.info("Sample collection completed!")
        else:
            logger.warning("No Polish researchers found")
            
    except Exception as e:
        logger.error(f"Sample collection failed: {e}")
    finally:
        collector.close()

if __name__ == "__main__":
    main()