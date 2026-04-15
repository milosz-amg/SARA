import csv
import json
import sys

class OpenAlexFilter:
    def __init__(self, csv_filename="./data/scientists_with_identifiers.csv", 
                 json_filename="./data/uam_authors.json",
                 output_filename="./data/wmii_authors.json"):
        self.csv_filename = csv_filename
        self.json_filename = json_filename
        self.output_filename = output_filename
        self.orcid_set = set()
        self.matched_scientists = []
        
    def extract_orcids_from_csv(self):
        print(f"Reading ORCID numbers from {self.csv_filename}...")
        
        try:
            with open(self.csv_filename, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    orcid = row.get('orcid')
                    if orcid and orcid.strip():
                        orcid_clean = orcid.strip()
                        self.orcid_set.add(orcid_clean)
                        
                        if not orcid_clean.startswith('http'):
                            self.orcid_set.add(f"https://orcid.org/{orcid_clean}")
            
            print(f"Found {len(self.orcid_set)} unique ORCID numbers in CSV")
            return True
            
        except FileNotFoundError:
            print(f"Error: File '{self.csv_filename}' not found")
            return False
        except Exception as e:
            print(f"Error reading CSV: {e}")
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
                        elif 'scientists' in data:
                            data = data['scientists']
                    
                    if not isinstance(data, list):
                        print("Error: JSON file doesn't contain a list of scientists")
                        return False
                    
                    print(f"Loaded {len(data)} scientists from JSON file")
                    
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
                    
                    print(f"Loaded {len(data)} scientists from JSONL file")
            
            print("\nFiltering scientists by ORCID...")
            matched_count = 0
            
            for scientist in data:
                scientist_orcid = None
                
                if 'orcid' in scientist:
                    scientist_orcid = scientist['orcid']
                elif 'ids' in scientist and 'orcid' in scientist['ids']:
                    scientist_orcid = scientist['ids']['orcid']
                
                if scientist_orcid:
                    scientist_orcid_clean = scientist_orcid.strip()
                    
                    if (scientist_orcid_clean in self.orcid_set or
                        scientist_orcid_clean.replace('https://orcid.org/', '') in self.orcid_set):
                        
                        self.matched_scientists.append(scientist)
                        matched_count += 1
                        
                        name = scientist.get('display_name', 'Unknown')
                        print(f"  Matched: {name} ({scientist_orcid_clean})")
            
            print(f"\nFound {matched_count} matching scientists")
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
                json.dump(self.matched_scientists, f, ensure_ascii=False, indent=2)
            
            print(f"Saved {len(self.matched_scientists)} scientists to {self.output_filename}")
            return True
            
        except Exception as e:
            print(f"Error saving JSON: {e}")
            return False
    
    def generate_statistics(self):
        print("FILTERING STATISTICS")
        print(f"ORCID numbers in CSV: {len(self.orcid_set)}")
        print(f"Scientists matched: {len(self.matched_scientists)}")
        
        if len(self.orcid_set) > 0:
            match_rate = (len(self.matched_scientists) / len(self.orcid_set)) * 100
            print(f"Match rate: {match_rate:.1f}%")
        
        if len(self.matched_scientists) < len(self.orcid_set):
            missing = len(self.orcid_set) - len(self.matched_scientists)
            print(f"⚠ Missing: {missing} ORCID numbers not found in JSON file")
        
        print(f"{'='*70}\n")
    
    def save_unmatched_orcids(self, filename="unmatched_orcids.txt"):
        matched_orcids = set()
        
        for scientist in self.matched_scientists:
            if 'orcid' in scientist:
                orcid = scientist['orcid'].strip()
                matched_orcids.add(orcid)
                matched_orcids.add(orcid.replace('https://orcid.org/', ''))
            elif 'ids' in scientist and 'orcid' in scientist['ids']:
                orcid = scientist['ids']['orcid'].strip()
                matched_orcids.add(orcid)
                matched_orcids.add(orcid.replace('https://orcid.org/', ''))
        
        unmatched = []
        for orcid in self.orcid_set:
            orcid_clean = orcid.replace('https://orcid.org/', '')
            if orcid_clean not in matched_orcids and orcid not in matched_orcids:
                unmatched.append(orcid)
        
        if unmatched:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("# ORCID numbers from CSV that were not found in JSON file\n")
                    f.write(f"# Total: {len(unmatched)}\n\n")
                    for orcid in sorted(unmatched):
                        f.write(f"{orcid}\n")
                
                print(f"Saved {len(unmatched)} unmatched ORCID numbers to {filename}")
            except Exception as e:
                print(f"Error saving unmatched ORCIDs: {e}")
    
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
        
        self.generate_statistics()
        self.save_unmatched_orcids()
        
        print("Filtering completed successfully!")
        print(f"\nOutput files:")
        print(f"  - {self.output_filename}")
        print(f"  - unmatched_orcids.txt")
        
        return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Filter OpenAlex JSON data by ORCID numbers from CSV'
    )
    parser.add_argument(
        '--csv',
        default='./data/scientists_with_identifiers.csv',
        help='Input CSV file with ORCID numbers (default: scientists_with_identifiers.csv)'
    )
    parser.add_argument(
        '--json',
        default='./data/uam_authors.json',
        help='Input JSON file with all scientists (default: openalex_data.json)'
    )
    parser.add_argument(
        '--output',
        default='./data/wmii_authors.json',
        help='Output JSON file (default: wmii_authors.json)'
    )
    
    args = parser.parse_args()
    
    filter_obj = OpenAlexFilter(
        csv_filename=args.csv,
        json_filename=args.json,
        output_filename=args.output
    )
    
    success = filter_obj.run()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
