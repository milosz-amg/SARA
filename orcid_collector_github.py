#!/usr/bin/env python3
"""
ORCID Data Collector - GitHub Actions Optimized Version

Fetches researcher data from ORCID API and stores it in SQLite database
Optimized for cloud environments with batch processing and artifact management.
"""

import sqlite3
import pandas as pd
import requests
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import time
import logging
import os
import sys

# GitHub Actions optimized logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # GitHub Actions console only
    ]
)
logger = logging.getLogger(__name__)

# Configuration for GitHub Actions
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '6250'))  # Max researchers per run
MAX_API_CALLS = int(os.getenv('MAX_API_CALLS', '25000'))  # Daily API limit
DB_PATH = os.getenv('DB_PATH', 'data/polish_scientists.db')
IDS_FILE = os.getenv('IDS_FILE', 'data/polish_researchers_all.txt')

# ORCID API configuration
BASE_URL = "https://pub.orcid.org/v3.0"
HEADERS = {"Accept": "application/json"}
SIMPLE_DELAY = 0.08  # Slightly faster for cloud (12.5 req/sec vs 10)

class ORCIDDataCollectorGitHub:
    """ORCID Data Collector optimized for GitHub Actions."""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn = None
        self.request_count = 0
        self.setup_database()
        logger.info(f"🚀 GitHub Actions ORCID Collector initialized")
        logger.info(f"📊 Batch size: {BATCH_SIZE} researchers")
        logger.info(f"🔄 Max API calls: {MAX_API_CALLS}")
    
    def setup_database(self):
        """Set up SQLite database with the new schema."""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # Create Researchers table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS Researchers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            orcid_id TEXT UNIQUE NOT NULL,
            given_names TEXT,
            family_name TEXT,
            credit_name TEXT,
            other_names TEXT,
            country TEXT,
            keywords TEXT,
            external_ids TEXT,
            researcher_urls TEXT,
            bio TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create Institutions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS Institutions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            city TEXT,
            region TEXT,
            country TEXT,
            ror TEXT,
            parent_ror TEXT,
            disambiguated_id TEXT,
            disambiguation_source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create Emails table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS Emails (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            researcher_id INTEGER,
            email TEXT NOT NULL,
            is_primary BOOLEAN DEFAULT FALSE,
            is_verified BOOLEAN DEFAULT FALSE,
            visibility TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (researcher_id) REFERENCES Researchers (id)
        )
        ''')
        
        # Create Publications table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS Publications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            researcher_id INTEGER,
            title TEXT,
            journal TEXT,
            publication_date TEXT,
            doi TEXT,
            external_ids TEXT,
            url TEXT,
            type TEXT,
            visibility TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (researcher_id) REFERENCES Researchers (id)
        )
        ''')
        
        # Create Affiliations table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS Affiliations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            researcher_id INTEGER,
            institution_id INTEGER,
            department TEXT,
            role_title TEXT,
            start_date TEXT,
            end_date TEXT,
            type TEXT,
            visibility TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (researcher_id) REFERENCES Researchers (id),
            FOREIGN KEY (institution_id) REFERENCES Institutions (id)
        )
        ''')
        
        self.conn.commit()
        logger.info("✅ Database schema initialized")

    def load_orcid_ids_from_file(self, filename: str) -> List[str]:
        """Load ORCID IDs from a file."""
        try:
            if not os.path.exists(filename):
                logger.warning(f"📄 File {filename} not found")
                return []
                
            with open(filename, 'r') as f:
                ids = [line.strip() for line in f if line.strip()]
            logger.info(f"📋 Loaded {len(ids)} ORCID IDs from {filename}")
            return ids
        except Exception as e:
            logger.error(f"❌ Failed to load ORCID IDs from {filename}: {e}")
            return []

    def get_processed_researchers(self) -> set:
        """Get ORCID IDs of researchers already in database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT orcid_id FROM Researchers")
        processed = {row[0] for row in cursor.fetchall()}
        logger.info(f"📊 Found {len(processed)} researchers already in database")
        return processed

    def collect_batch_data_github(self, orcid_ids: List[str]):
        """Collect data with GitHub Actions constraints."""
        # Check which researchers are already processed
        processed = self.get_processed_researchers()
        remaining = [orcid_id for orcid_id in orcid_ids if orcid_id not in processed]
        
        # Apply batch size limit
        if len(remaining) > BATCH_SIZE:
            remaining = remaining[:BATCH_SIZE]
            logger.info(f"🔄 Limited to {BATCH_SIZE} researchers for this run")
        
        total_to_process = len(remaining)
        already_done = len(processed.intersection(set(orcid_ids)))
        
        logger.info(f"📊 GitHub Actions batch info:")
        logger.info(f"   • Already processed: {already_done}")
        logger.info(f"   • Will process now: {total_to_process}")
        logger.info(f"   • Estimated API calls: {total_to_process * 4}")
        logger.info(f"   • Estimated runtime: {total_to_process * 0.5:.1f} minutes")
        
        if not remaining:
            logger.info("✅ All researchers in batch already processed!")
            return
        
        success_count = 0
        error_count = 0
        
        for i, orcid_id in enumerate(remaining, 1):
            # Check API call limit
            if self.request_count >= MAX_API_CALLS:
                logger.warning(f"🛑 API call limit ({MAX_API_CALLS}) reached, stopping")
                break
                
            try:
                logger.info(f"👨‍🔬 Processing {i}/{total_to_process}: {orcid_id}")
                self.collect_researcher_data(orcid_id)
                success_count += 1
                
                # Progress logging for GitHub Actions
                if i % 100 == 0 or i == total_to_process:
                    progress_pct = (i / total_to_process) * 100
                    logger.info(f"📊 Progress: {i}/{total_to_process} ({progress_pct:.1f}%) - {self.request_count} API calls")
                    
            except Exception as e:
                logger.error(f"❌ Failed to process {orcid_id}: {e}")
                error_count += 1
                continue
        
        logger.info(f"🎯 Batch completed: {success_count} success, {error_count} errors")
        logger.info(f"📊 Total API calls made: {self.request_count}")

    def fetch_orcid_person(self, orcid_id: str) -> Dict:
        """Fetch person data from ORCID API with rate limit handling."""
        time.sleep(SIMPLE_DELAY)
        self.request_count += 1
        url = f"{BASE_URL}/{orcid_id}/person"
        try:
            response = requests.get(url, headers=HEADERS)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching person data for {orcid_id}: {e}")
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
        """Fetch works data from ORCID API."""
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

    def collect_researcher_data(self, orcid_id: str):
        """Collect and store all data for a single researcher."""
        logger.info(f"Collecting data for {orcid_id}")
        
        # Fetch all data
        person_data = self.fetch_orcid_person(orcid_id)
        employment_data = self.fetch_orcid_employment(orcid_id)
        education_data = self.fetch_orcid_education(orcid_id)
        works_data = self.fetch_orcid_works(orcid_id)
        
        # Store in database
        if person_data:
            self.store_researcher_data(orcid_id, person_data, employment_data, education_data, works_data)
            logger.info(f"Successfully inserted data for researcher {orcid_id}")
        else:
            logger.warning(f"No person data found for {orcid_id}")

    def store_researcher_data(self, orcid_id: str, person_data: Dict, employment_data: Dict, education_data: Dict, works_data: Dict):
        """Store researcher data in database."""
        cursor = self.conn.cursor()
        
        # Extract person information
        name_data = person_data.get("name", {}) if person_data else {}
        given_names = name_data.get("given-names", {}).get("value") if name_data.get("given-names") else None
        family_name = name_data.get("family-name", {}).get("value") if name_data.get("family-name") else None
        credit_name = name_data.get("credit-name", {}).get("value") if name_data.get("credit-name") else None
        
        # Extract other names
        other_names_list = name_data.get("other-names", {}).get("other-name", []) if name_data.get("other-names") else []
        other_names = ", ".join([name.get("content", "") for name in other_names_list if name.get("content")])
        
        # Extract country
        addresses = person_data.get("addresses", {}).get("address", []) if person_data else []
        country = addresses[0].get("country", {}).get("value") if addresses else None
        
        # Extract keywords
        keywords_data = person_data.get("keywords", {}).get("keyword", []) if person_data else []
        keywords = ", ".join([kw.get("content", "") for kw in keywords_data if kw.get("content")])
        
        # Extract bio
        bio_data = person_data.get("biography", {}) if person_data else {}
        bio = bio_data.get("content") if bio_data else None
        
        # Insert researcher
        cursor.execute('''
        INSERT OR REPLACE INTO Researchers 
        (orcid_id, given_names, family_name, credit_name, other_names, country, keywords, bio, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (orcid_id, given_names, family_name, credit_name, other_names, country, keywords, bio, datetime.now().isoformat()))
        
        researcher_id = cursor.lastrowid
        self.conn.commit()

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

def main_github():
    """Main function optimized for GitHub Actions."""
    logger.info("🚀 Starting GitHub Actions ORCID collection...")
    
    collector = ORCIDDataCollectorGitHub()
    
    try:
        # Load existing researcher IDs
        all_ids = collector.load_orcid_ids_from_file(IDS_FILE)
        
        if not all_ids:
            logger.error("❌ No researcher IDs found. Please ensure polish_researchers_all.txt exists.")
            sys.exit(1)
        
        logger.info(f"📋 Found {len(all_ids)} total researchers to process")
        
        # Process batch
        collector.collect_batch_data_github(all_ids)
        
        # Print final stats for GitHub Actions
        processed = collector.get_processed_researchers()
        remaining = len(all_ids) - len(processed)
        completion_pct = (len(processed) / len(all_ids)) * 100
        
        logger.info("=" * 50)
        logger.info("🎯 GITHUB ACTIONS RUN SUMMARY")
        logger.info(f"📊 Total researchers: {len(all_ids)}")
        logger.info(f"✅ Processed: {len(processed)}")
        logger.info(f"⏳ Remaining: {remaining}")
        logger.info(f"📈 Completion: {completion_pct:.1f}%")
        logger.info(f"🔄 API calls made: {collector.request_count}")
        
        if remaining > 0:
            days_remaining = (remaining // BATCH_SIZE) + 1
            logger.info(f"⏰ Estimated days to completion: {days_remaining}")
        else:
            logger.info("🎉 Collection completed!")
        
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"❌ GitHub Actions run failed: {e}")
        sys.exit(1)
    finally:
        collector.close()

if __name__ == "__main__":
    main_github()