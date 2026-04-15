import csv
import json
import sys

class SimpleWorksFilter:
    def __init__(self, csv_filename="./data/scientists_with_identifiers.csv", 
                 json_filename="./data/uam_works.json",
                 output_filename="./data/wmii_works.json"):
        self.csv_filename = csv_filename
        self.json_filename = json_filename
        self.output_filename = output_filename
        self.orcid_set = set()
        self.matched_works = []
        
    def extract_orcids_from_csv(self):
        print(f"Reading ORCID numbers from {self.csv_filename}...")
        
        try:
            with open(self.csv_filename, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    orcid = row.get('orcid')
                    if orcid and orcid.strip():
                        orcid_clean = orcid.strip()
                        
                        if orcid_clean.startswith('http'):
                            self.orcid_set.add(orcid_clean)
                            orcid_no_prefix = orcid_clean.replace('https://orcid.org/', '')
                            self.orcid_set.add(orcid_no_prefix)
                        else:
                            self.orcid_set.add(orcid_clean)
                            self.orcid_set.add(f"https://orcid.org/{orcid_clean}")
            
            print(f"Found {len(self.orcid_set) // 2} unique ORCID numbers in CSV")
            return True
            
        except FileNotFoundError:
            print(f"Error: File '{self.csv_filename}' not found")
            return False
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return False
    
    def normalize_orcid(self, orcid):
        if not orcid:
            return None
        orcid = orcid.strip()
        return orcid.replace('https://orcid.org/', '').replace('http://orcid.org/', '')
    
    def work_has_matching_author(self, work):
        if 'authorships' not in work:
            return False
        
        for authorship in work['authorships']:
            if 'author' not in authorship:
                continue
            
            author = authorship['author']
            author_orcid = author.get('orcid')
            
            if author_orcid:
                normalized_orcid = self.normalize_orcid(author_orcid)
                
                if (author_orcid in self.orcid_set or 
                    normalized_orcid in self.orcid_set or
                    f"https://orcid.org/{normalized_orcid}" in self.orcid_set):
                    return True
        
        return False
    
    def filter_json_data(self):
        print(f"\nReading and filtering {self.json_filename}...")
        
        try:
            with open(self.json_filename, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    
                    if isinstance(data, dict):
                        if 'results' in data:
                            data = data['results']
                        elif 'data' in data:
                            data = data['data']
                        elif 'works' in data:
                            data = data['works']
                    
                    if not isinstance(data, list):
                        print("Error: JSON file doesn't contain a list of works")
                        return False
                    
                    print(f"Loaded {len(data)} works from JSON file")
                    
                except json.JSONDecodeError:
                    print("Trying to read as JSONL (line-delimited JSON)...")
                    f.seek(0)
                    data = []
                    for line in f:
                        if line.strip():
                            try:
                                data.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                    
                    print(f"Loaded {len(data)} works from JSONL file")
            
            print("\nFiltering works by author ORCID...")
            
            for idx, work in enumerate(data):
                if (idx + 1) % 1000 == 0:
                    print(f"  Processed {idx + 1}/{len(data)} works...")
                
                if self.work_has_matching_author(work):
                    self.matched_works.append(work)
            
            print(f"\nFound {len(self.matched_works)} works with matching authors")
            return True
            
        except FileNotFoundError:
            print(f"Error: File '{self.json_filename}' not found")
            return False
        except Exception as e:
            print(f"Error reading JSON: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_filtered_data(self):
        print(f"\nSaving filtered data to {self.output_filename}...")
        
        try:
            with open(self.output_filename, 'w', encoding='utf-8') as f:
                json.dump(self.matched_works, f, ensure_ascii=False, indent=2)
            
            print(f"Saved {len(self.matched_works)} works to {self.output_filename}")
            return True
            
        except Exception as e:
            print(f"Error saving JSON: {e}")
            return False
    
    def run(self):
        if not self.extract_orcids_from_csv():
            return False
        
        if len(self.orcid_set) == 0:
            print("No ORCID numbers found in CSV file")
            return False
        
        if not self.filter_json_data():
            return False
        
        if not self.save_filtered_data():
            return False
        
        print(f"Filtering completed successfully!")
        print(f"{len(self.matched_works)} works saved to {self.output_filename}")
        
        return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Filter OpenAlex works JSON data by ORCID numbers from CSV'
    )
    parser.add_argument(
        '--csv',
        default='./data/scientists_with_identifiers.csv',
        help='Input CSV file with ORCID numbers (default: scientists_with_identifiers.csv)'
    )
    parser.add_argument(
        '--json',
        default='./data/uam_works.json',
        help='Input JSON file with all works (default: uam_works.json)'
    )
    parser.add_argument(
        '--output',
        default='./data/wmii_works.json',
        help='Output JSON file (default: wmii_works.json)'
    )
    
    args = parser.parse_args()
    
    filter_obj = SimpleWorksFilter(
        csv_filename=args.csv,
        json_filename=args.json,
        output_filename=args.output
    )
    
    success = filter_obj.run()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()