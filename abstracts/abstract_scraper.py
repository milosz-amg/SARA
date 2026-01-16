import requests
import time
import pandas as pd
from typing import List, Dict
import re


class OpenAlexAPIScraper:
    def __init__(self):
        self.base_url = "https://api.openalex.org"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'OpenAlexScraper/1.0 (mailto:your@email.com)',
            'Accept': 'application/json'
        })
    
    def get_works_for_orcid(self, orcid: str) -> List[Dict]:
        """Get all works for an ORCID using the API"""
        print(f"\n{'='*60}")
        print(f"Processing ORCID: {orcid}")
        print(f"{'='*60}")
        
        all_works = []
        page = 1
        per_page = 200
        
        stats = {
            'total_works': 0,
            'with_abstract': 0,
            'without_abstract': 0,
            'with_doi': 0,
            'without_doi': 0,
            'with_co_authors': 0,
            'without_co_authors': 0,
            'total_co_authors': 0,
            'errors': 0
        }
        
        while True:
            url = f"{self.base_url}/works"
            params = {
                'filter': f'authorships.author.orcid:{orcid}',
                'per-page': per_page,
                'page': page
            }
            
            print(f"Fetching page {page}...")
            
            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                results = data.get('results', [])
                meta = data.get('meta', {})
                total_count = meta.get('count', 0)
                
                if not results:
                    break
                
                print(f"  Found {len(results)} works on page {page} (Total available: {total_count})")
                
                for idx, work in enumerate(results):
                    try:
                        work_id = work.get('id', '').split('/')[-1]
                        print(f"    Processing {work_id}...")
                        
                        work_data = self.process_work(work, orcid)
                        all_works.append(work_data)
                        
                        stats['total_works'] += 1
                        if work_data['abstract'] and len(work_data['abstract'].strip()) > 0:
                            stats['with_abstract'] += 1
                        else:
                            stats['without_abstract'] += 1
                        
                        if work_data['doi'] and len(work_data['doi'].strip()) > 0:
                            stats['with_doi'] += 1
                        else:
                            stats['without_doi'] += 1
                        
                        if work_data['num_co_authors'] > 0:
                            stats['with_co_authors'] += 1
                            stats['total_co_authors'] += work_data['num_co_authors']
                        else:
                            stats['without_co_authors'] += 1
                            
                    except Exception as e:
                        print(f"    Error processing work {idx}: {str(e)}")
                        stats['errors'] += 1
                        continue
                
                if len(all_works) >= total_count:
                    print(f"  ✓ Got all {total_count} works")
                    break
                
                page += 1
                time.sleep(0.1)
                
            except Exception as e:
                print(f"  Error fetching page {page}: {str(e)}")
                stats['errors'] += 1
                break
        
        self.print_orcid_summary(orcid, stats)
        
        return all_works
    
    def print_orcid_summary(self, orcid: str, stats: Dict):
        print(f"\n{'='*60}")
        print(f"SUMMARY FOR ORCID: {orcid}")
        print(f"{'='*60}")
        print(f"Total works found:        {stats['total_works']}")
        print(f"")
        print(f"Abstracts:")
        print(f"  ✓ With abstract:        {stats['with_abstract']} ({stats['with_abstract']/max(stats['total_works'],1)*100:.1f}%)")
        print(f"  ✗ Missing abstract:     {stats['without_abstract']} ({stats['without_abstract']/max(stats['total_works'],1)*100:.1f}%)")
        print(f"")
        print(f"DOI Links:")
        print(f"  ✓ With DOI:             {stats['with_doi']} ({stats['with_doi']/max(stats['total_works'],1)*100:.1f}%)")
        print(f"  ✗ Missing DOI:          {stats['without_doi']} ({stats['without_doi']/max(stats['total_works'],1)*100:.1f}%)")
        print(f"")
        print(f"Co-authors:")
        print(f"  Works with co-authors:  {stats['with_co_authors']}")
        print(f"  Works without:          {stats['without_co_authors']}")
        print(f"  Total co-authors:       {stats['total_co_authors']}")
        if stats['with_co_authors'] > 0:
            print(f"  Avg per work:           {stats['total_co_authors']/stats['with_co_authors']:.1f}")
        print(f"")
        if stats['errors'] > 0:
            print(f"⚠ Errors encountered:    {stats['errors']}")
        print(f"{'='*60}")
    
    def process_work(self, work: Dict, main_orcid: str) -> Dict:
        """Process a single work and extract relevant data"""
        try:
            work_id = work['id'].split('/')[-1]
            
            work_data = {
                'main_author_orcid': main_orcid,
                'openalex_id': work_id,
                'title': work.get('title', ''),
                'publication_year': work.get('publication_year', ''),
                'publication_date': work.get('publication_date', ''),
                'doi': work.get('doi', ''),
                'type': work.get('type', ''),
                'cited_by_count': work.get('cited_by_count', 0),
            }
            
            primary_location = work.get('primary_location')
            if primary_location:
                source = primary_location.get('source', {})
                if source:
                    work_data['journal'] = source.get('display_name', '')
                else:
                    work_data['journal'] = ''
            else:
                work_data['journal'] = ''
            
            topics = work.get('topics', [])
            topic_names = [t.get('display_name', '') for t in topics if t.get('display_name')]
            work_data['topics'] = '; '.join(topic_names[:5])
            
            authorships = work.get('authorships', [])
            co_authors = []
            co_author_orcids = []
            
            for authorship in authorships:
                author = authorship.get('author', {})
                author_orcid = author.get('orcid', '')
                
                if author_orcid:
                    orcid_match = re.search(r'(\d{4}-\d{4}-\d{4}-\d{3}[0-9X])', author_orcid)
                    if orcid_match:
                        author_orcid = orcid_match.group(1)
                
                if author_orcid and author_orcid != main_orcid:
                    author_name = author.get('display_name', '')
                    if author_name:
                        co_authors.append(author_name)
                        co_author_orcids.append(author_orcid)
            
            work_data['co_authors'] = '; '.join(co_authors)
            work_data['co_author_orcids'] = '; '.join(co_author_orcids)
            work_data['num_co_authors'] = len(co_authors)
            
            abstract = self.extract_abstract(work)
            work_data['abstract'] = abstract
            
            if abstract:
                print(f"      ✓ Abstract extracted ({len(abstract)} chars)")
            else:
                if work.get('abstract_inverted_index'):
                    print(f"      ⚠ Abstract exists but extraction failed")
                else:
                    print(f"      ✗ No abstract available in API")
            
            keywords = work.get('keywords', [])
            keyword_names = [k.get('display_name', '') for k in keywords if k.get('display_name')]
            work_data['keywords'] = '; '.join(keyword_names)
            
            return work_data
            
        except Exception as e:
            print(f"      Error processing work: {str(e)}")
            return {
                'main_author_orcid': main_orcid,
                'openalex_id': work.get('id', '').split('/')[-1],
                'title': work.get('title', 'Error'),
                'publication_year': '',
                'publication_date': '',
                'doi': '',
                'type': '',
                'cited_by_count': 0,
                'journal': '',
                'topics': '',
                'co_authors': '',
                'co_author_orcids': '',
                'num_co_authors': 0,
                'abstract': '',
                'keywords': ''
            }
    
    def extract_abstract(self, work: Dict) -> str:
        """Convert inverted index abstract to readable text"""
        abstract_inverted = work.get('abstract_inverted_index', {})
        
        if not abstract_inverted:
            return ''
        
        try:
            word_positions = []
            
            for word, positions in abstract_inverted.items():
                if isinstance(positions, list):
                    for pos in positions:
                        word_positions.append((pos, word))
                else:
                    word_positions.append((positions, word))
            
            word_positions.sort(key=lambda x: x[0])
            
            abstract = ' '.join([word for pos, word in word_positions])
            
            abstract = ' '.join(abstract.split())
            
            return abstract
            
        except Exception as e:
            print(f"      Warning: Could not reconstruct abstract: {str(e)}")
            return ''
    
    def scrape_all_orcids(self, csv_file: str, output_file: str = 'openalex_api_results.csv'):
        df = pd.read_csv(csv_file)
        orcids = df['orcid'].tolist()
        
        print(f"\n{'='*60}")
        print(f"Starting API scraping for {len(orcids)} ORCIDs")
        print(f"{'='*60}")
        
        all_results = []
        overall_stats = {
            'total_orcids': len(orcids),
            'total_works': 0,
            'with_abstract': 0,
            'without_abstract': 0,
            'with_doi': 0,
            'without_doi': 0,
            'orcid_errors': 0
        }
        
        for idx, orcid in enumerate(orcids):
            print(f"\nProgress: {idx+1}/{len(orcids)}")
            
            try:
                works = self.get_works_for_orcid(orcid)
                all_results.extend(works)
                
                overall_stats['total_works'] += len(works)
                for work in works:
                    if work['abstract'] and len(str(work['abstract']).strip()) > 0:
                        overall_stats['with_abstract'] += 1
                    else:
                        overall_stats['without_abstract'] += 1
                    if work['doi'] and len(str(work['doi']).strip()) > 0:
                        overall_stats['with_doi'] += 1
                    else:
                        overall_stats['without_doi'] += 1
                
                self.save_results(all_results, output_file)
                print(f"✓ Saved {len(works)} works. Total so far: {len(all_results)}")
                
            except Exception as e:
                print(f"✗ ERROR processing {orcid}: {str(e)}")
                overall_stats['orcid_errors'] += 1
                continue
            
            time.sleep(0.5)
        
        self.print_overall_summary(overall_stats)
        
        stats_file = output_file.replace('.csv', '_statistics.csv')
        self.save_statistics_summary(all_results, stats_file)
        
        print(f"\n✓ Results saved to: {output_file}")
        
        return all_results
    
    def print_overall_summary(self, stats: Dict):
        """Print overall summary statistics for all ORCIDs"""
        print(f"\n\n{'='*60}")
        print(f"OVERALL SUMMARY - ALL ORCIDs")
        print(f"{'='*60}")
        print(f"Total ORCIDs processed:     {stats['total_orcids']}")
        print(f"Total works found:          {stats['total_works']}")
        print(f"")
        print(f"Abstracts:")
        print(f"  ✓ With abstract:          {stats['with_abstract']} ({stats['with_abstract']/max(stats['total_works'],1)*100:.1f}%)")
        print(f"  ✗ Missing abstract:       {stats['without_abstract']} ({stats['without_abstract']/max(stats['total_works'],1)*100:.1f}%)")
        print(f"")
        print(f"DOI Links:")
        print(f"  ✓ With DOI:               {stats['with_doi']} ({stats['with_doi']/max(stats['total_works'],1)*100:.1f}%)")
        print(f"  ✗ Missing DOI:            {stats['without_doi']} ({stats['without_doi']/max(stats['total_works'],1)*100:.1f}%)")
        print(f"")
        print(f"Average works per ORCID:    {stats['total_works']/max(stats['total_orcids'],1):.1f}")
        if stats['orcid_errors'] > 0:
            print(f"⚠ ORCIDs with errors:       {stats['orcid_errors']}")
        print(f"{'='*60}")
        print(f"\n✓ Scraping complete!")
    
    def save_results(self, results: List[Dict], output_file: str):
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False, encoding='utf-8')
    
    def save_statistics_summary(self, all_results: List[Dict], summary_file: str = 'statistics_summary.csv'):
        df = pd.DataFrame(all_results)
        
        if df.empty:
            return
        
        stats_per_orcid = []
        
        for orcid in df['main_author_orcid'].unique():
            orcid_data = df[df['main_author_orcid'] == orcid]
            
            total = len(orcid_data)
            with_abstract = orcid_data['abstract'].apply(lambda x: bool(x and str(x).strip())).sum()
            with_doi = orcid_data['doi'].apply(lambda x: bool(x and str(x).strip())).sum()
            total_citations = orcid_data['cited_by_count'].sum()
            
            stats_per_orcid.append({
                'orcid': orcid,
                'total_works': total,
                'with_abstract': with_abstract,
                'without_abstract': total - with_abstract,
                'abstract_percentage': f"{with_abstract/total*100:.1f}%",
                'with_doi': with_doi,
                'without_doi': total - with_doi,
                'doi_percentage': f"{with_doi/total*100:.1f}%",
                'total_citations': total_citations,
                'avg_citations_per_work': f"{total_citations/total:.1f}"
            })
        
        stats_df = pd.DataFrame(stats_per_orcid)
        stats_df.to_csv(summary_file, index=False, encoding='utf-8')
        print(f"\n✓ Statistics summary saved to: {summary_file}")
    
    def get_single_work_details(self, work_id: str) -> Dict:
        url = f"{self.base_url}/works/{work_id}"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching work {work_id}: {str(e)}")
            return {}


if __name__ == "__main__":
    scraper = OpenAlexAPIScraper()
    
    csv_file = './data/wmii_orcid.csv'
    output_file = 'openalex_all_results.csv'
    
    print(f"Starting scraper...")
    print(f"Input file: {csv_file}")
    print(f"Output file: {output_file}")
    
    results = scraper.scrape_all_orcids(csv_file, output_file)
    
    print(f"\n{'='*60}")
    print(f"ALL DONE!")
    print(f"{'='*60}")
    print(f"✓ Main results: {output_file}")
    print(f"✓ Statistics: {output_file.replace('.csv', '_statistics.csv')}")
    print(f"✓ Total publications scraped: {len(results)}")
    print(f"{'='*60}")
