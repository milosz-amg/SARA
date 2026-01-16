import pandas as pd
import time
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class AbstractFetcher:
    def __init__(self, headless=True):
        chrome_options = Options()
        if headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.wait = WebDriverWait(self.driver, 10)
    
    def fetch_abstract_from_doi(self, doi_url: str) -> str:
        if not doi_url or not doi_url.strip():
            return ''
        
        abstract = ""
        
        try:
            print(f"    Fetching from: {doi_url[:50]}...")
            self.driver.get(doi_url)
            time.sleep(3)
            
            abstract_selectors = [
                # ScienceDirect
                "div.abstract.author",
                "div#abs0010",
                "section.abstract",
                "div.Abstract",
                
                # Springer
                "section[data-title='Abstract']",
                "div#Abs1-content",
                "section#Abs1",
                
                # Wiley
                "section.article-section__abstract",
                "div.article-section__content",
                
                # MDPI
                "div.art-abstract",
                "section.html-abstract",
                
                # IEEE
                "div.abstract-text",
                
                # Generic
                "div.abstract",
                "section.abstract",
                "div[id*='abstract']",
                "section[id*='abstract']",
                "div[class*='abstract']",
                "section[class*='abstract']",
                "article section:first-of-type",
            ]
            
            for selector in abstract_selectors:
                try:
                    abstract_elem = self.driver.find_element(By.CSS_SELECTOR, selector)
                    
                    paragraphs = abstract_elem.find_elements(By.TAG_NAME, "p")
                    if paragraphs:
                        abstract = ' '.join([p.text for p in paragraphs if p.text.strip()])
                    else:
                        abstract = abstract_elem.text
                    
                    abstract = re.sub(r'^Abstract\s*', '', abstract, flags=re.IGNORECASE).strip()
                    
                    if abstract and len(abstract) > 50:
                        print(f"      ✓ Abstract found ({len(abstract)} chars)")
                        break
                    else:
                        abstract = ""
                        
                except:
                    continue
            
            if not abstract:
                print(f"      ✗ No abstract found on page")
                
        except Exception as e:
            print(f"      ✗ Error: {str(e)[:50]}")
        
        return abstract
    
    def process_csv_file(self, input_csv: str, output_csv: str = None):
        if output_csv is None:
            output_csv = input_csv.replace('.csv', '_with_abstracts.csv')
        
        print(f"\n{'='*60}")
        print(f"Loading CSV: {input_csv}")
        print(f"{'='*60}")
        
        df = pd.read_csv(input_csv)
        
        print(f"Total records: {len(df)}")
        
        missing_abstract_mask = df['abstract'].isna() | (df['abstract'] == '') | (df['abstract'].str.strip() == '')
        missing_abstract_df = df[missing_abstract_mask]
        
        print(f"Records with missing abstracts: {len(missing_abstract_df)}")
        
        has_doi_mask = missing_abstract_df['doi'].notna() & (missing_abstract_df['doi'] != '') & (missing_abstract_df['doi'].str.strip() != '')
        fetchable_df = missing_abstract_df[has_doi_mask]
        
        print(f"Records with DOI (can attempt fetch): {len(fetchable_df)}")
        print(f"\n{'='*60}")
        print(f"Starting abstract fetching...")
        print(f"{'='*60}\n")
        
        stats = {
            'attempted': 0,
            'successful': 0,
            'failed': 0
        }
        
        for idx, row in fetchable_df.iterrows():
            stats['attempted'] += 1
            
            print(f"[{stats['attempted']}/{len(fetchable_df)}] {row['openalex_id']}")
            print(f"  Title: {row['title'][:60]}...")
            
            abstract = self.fetch_abstract_from_doi(row['doi'])
            
            if abstract:
                df.at[idx, 'abstract'] = abstract
                stats['successful'] += 1
            else:
                stats['failed'] += 1
            
            if stats['attempted'] % 10 == 0:
                df.to_csv(output_csv, index=False, encoding='utf-8')
                print(f"\n  💾 Progress saved ({stats['successful']} successful so far)\n")
            
            time.sleep(2)
        
        df.to_csv(output_csv, index=False, encoding='utf-8')
        
        self.print_summary(stats, output_csv)
        
        return df
    
    def print_summary(self, stats: dict, output_file: str):
        print(f"\n{'='*60}")
        print(f"ABSTRACT FETCHING COMPLETE")
        print(f"{'='*60}")
        print(f"Total attempted:        {stats['attempted']}")
        print(f"✓ Successfully fetched: {stats['successful']} ({stats['successful']/max(stats['attempted'],1)*100:.1f}%)")
        print(f"✗ Failed to fetch:      {stats['failed']} ({stats['failed']/max(stats['attempted'],1)*100:.1f}%)")
        print(f"\n✓ Results saved to: {output_file}")
        print(f"{'='*60}")
    
    def close(self):
        """Close the browser"""
        if self.driver:
            self.driver.quit()


if __name__ == "__main__":
    fetcher = AbstractFetcher(headless=False)
    
    try:
        input_csv = './data/openalex_all_results.csv'
        output_csv = './data/openalex_all_results_complete.csv'
        
        print(f"Abstract Fetcher")
        print(f"Input:  {input_csv}")
        print(f"Output: {output_csv}")
        
        df = fetcher.process_csv_file(input_csv, output_csv)
        
        total_records = len(df)
        with_abstract = df['abstract'].apply(lambda x: bool(x and str(x).strip())).sum()
        
        print(f"\nFinal Statistics:")
        print(f"  Total records:        {total_records}")
        print(f"  With abstracts:       {with_abstract} ({with_abstract/total_records*100:.1f}%)")
        print(f"  Still missing:        {total_records - with_abstract} ({(total_records - with_abstract)/total_records*100:.1f}%)")
        
    finally:
        fetcher.close()
        print("\n✓ Browser closed")
