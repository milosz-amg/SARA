import pandas as pd


def fill_abstracts_from_duplicates_by_title_and_doi(input_csv: str):
    print(f"\n{'='*60}")
    print(f"Processing Duplicates and Splitting by Abstract Status")
    print(f"{'='*60}")
    print(f"Input file: {input_csv}")
    
    df = pd.read_csv(input_csv)
    
    print(f"Total records: {len(df)}")
    
    df['title_normalized'] = df['title'].str.lower().str.strip()
    
    has_abstract_before = (df['abstract'].notna() & 
                          (df['abstract'] != '') & 
                          (df['abstract'].str.strip() != '')).sum()
    
    print(f"Records with abstract BEFORE: {has_abstract_before}")
    print(f"Records without abstract BEFORE: {len(df) - has_abstract_before}")
    
    print(f"\n{'='*60}")
    print(f"STEP 1: Finding and filling abstracts from duplicates")
    print(f"{'='*60}")
    
    filled_count = 0
    
    print(f"\nProcessing duplicates by TITLE...")
    title_groups = df.groupby('title_normalized')
    
    for title, group in title_groups:
        if len(group) > 1:  # Has duplicates
            has_abstract_mask = group['abstract'].notna() & \
                               (group['abstract'] != '') & \
                               (group['abstract'].str.strip() != '')
            
            versions_with_abstract = group[has_abstract_mask]
            versions_without_abstract = group[~has_abstract_mask]
            
            if len(versions_with_abstract) > 0 and len(versions_without_abstract) > 0:
                abstract_to_copy = versions_with_abstract.iloc[0]['abstract']
                
                for idx in versions_without_abstract.index:
                    if pd.isna(df.at[idx, 'abstract']) or str(df.at[idx, 'abstract']).strip() == '':
                        df.at[idx, 'abstract'] = abstract_to_copy
                        filled_count += 1
                
                print(f"  ✓ Filled {len(versions_without_abstract)} record(s) for title: {title[:60]}...")
    
    print(f"\nProcessing duplicates by DOI...")
    
    df_with_doi = df[df['doi'].notna() & (df['doi'] != '') & (df['doi'].str.strip() != '')]
    
    if len(df_with_doi) > 0:
        doi_groups = df_with_doi.groupby('doi')
        
        for doi, group in doi_groups:
            if len(group) > 1:
                has_abstract_mask = group['abstract'].notna() & \
                                   (group['abstract'] != '') & \
                                   (group['abstract'].str.strip() != '')
                
                versions_with_abstract = group[has_abstract_mask]
                versions_without_abstract = group[~has_abstract_mask]
                
                if len(versions_with_abstract) > 0 and len(versions_without_abstract) > 0:
                    abstract_to_copy = versions_with_abstract.iloc[0]['abstract']
                    
                    for idx in versions_without_abstract.index:
                        if pd.isna(df.at[idx, 'abstract']) or str(df.at[idx, 'abstract']).strip() == '':
                            df.at[idx, 'abstract'] = abstract_to_copy
                            filled_count += 1
                    
                    print(f"  ✓ Filled {len(versions_without_abstract)} record(s) for DOI: {doi[:60]}...")
    
    df = df.drop('title_normalized', axis=1)
    
    print(f"\n{'='*60}")
    print(f"Summary of filling:")
    print(f"{'='*60}")
    print(f"Total abstracts filled: {filled_count}")
    
    has_abstract_after = (df['abstract'].notna() & 
                         (df['abstract'] != '') & 
                         (df['abstract'].str.strip() != '')).sum()
    
    print(f"\nRecords with abstract AFTER: {has_abstract_after} (was {has_abstract_before})")
    print(f"Records without abstract AFTER: {len(df) - has_abstract_after} (was {len(df) - has_abstract_before})")
    print(f"Improvement: +{has_abstract_after - has_abstract_before} abstracts")
    
    print(f"\n{'='*60}")
    print(f"STEP 2: Splitting into two files")
    print(f"{'='*60}")
    
    has_abstract_mask = df['abstract'].notna() & \
                        (df['abstract'] != '') & \
                        (df['abstract'].str.strip() != '')
    
    df_with_abstract = df[has_abstract_mask].copy()
    df_without_abstract = df[~has_abstract_mask].copy()
    
    print(f"\nRecords WITH abstracts: {len(df_with_abstract)}")
    print(f"Records WITHOUT abstracts: {len(df_without_abstract)}")
    
    with_abstracts_file = 'titles_with_abstracts.csv'
    without_abstracts_file = 'titles_without_abstracts.csv'
    
    df_with_abstract.to_csv(with_abstracts_file, index=False, encoding='utf-8')
    print(f"\n✓ Saved to: {with_abstracts_file}")
    
    if len(df_without_abstract) > 0:
        df_without_abstract.to_csv(without_abstracts_file, index=False, encoding='utf-8')
        print(f"✓ Saved to: {without_abstracts_file}")
    else:
        print(f"✓ No records without abstracts - skipping {without_abstracts_file}")
    
    print(f"\n{'='*60}")
    print(f"Examples from titles_with_abstracts.csv:")
    print(f"{'-'*60}")
    
    for idx, row in df_with_abstract.head(3).iterrows():
        print(f"\nTitle: {row['title'][:60]}...")
        print(f"ORCID: {row.get('main_author_orcid', 'N/A')}")
        print(f"Year: {row.get('publication_year', 'N/A')}")
        print(f"Abstract length: {len(str(row['abstract']))} chars")
    
    if len(df_without_abstract) > 0:
        print(f"\n{'='*60}")
        print(f"Examples from titles_without_abstracts.csv:")
        print(f"{'-'*60}")
        
        for idx, row in df_without_abstract.head(3).iterrows():
            print(f"\nTitle: {row['title'][:60]}...")
            print(f"ORCID: {row.get('main_author_orcid', 'N/A')}")
            print(f"Year: {row.get('publication_year', 'N/A')}")
            print(f"DOI: {row.get('doi', 'No DOI')[:60] if row.get('doi') else 'No DOI'}")
    
    print(f"\n{'='*60}")
    print(f"✓ Complete!")
    print(f"{'='*60}\n")
    
    return df_with_abstract, df_without_abstract


def generate_summary_report(input_file: str, with_file: str, without_file: str):
    """Generate a summary report of the processing"""
    
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY REPORT")
    print(f"{'='*60}")
    
    df_original = pd.read_csv(input_file)
    df_with = pd.read_csv(with_file)
    
    try:
        df_without = pd.read_csv(without_file)
    except:
        df_without = pd.DataFrame()
    
    print(f"\nOriginal file: {input_file}")
    print(f"  Total records: {len(df_original)}")
    
    print(f"\nOutput file 1: {with_file}")
    print(f"  Records with abstracts: {len(df_with)}")
    print(f"  Percentage: {len(df_with)/len(df_original)*100:.1f}%")
    
    if len(df_without) > 0:
        print(f"\nOutput file 2: {without_file}")
        print(f"  Records without abstracts: {len(df_without)}")
        print(f"  Percentage: {len(df_without)/len(df_original)*100:.1f}%")
    else:
        print(f"\nOutput file 2: {without_file}")
        print(f"  Records without abstracts: 0 (100% have abstracts!)")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    input_file = './data/openalex_all_results_complete.csv'
    
    print("\n" + "="*60)
    print("DUPLICATE ABSTRACT FILLER & SPLITTER")
    print("="*60)
    print("\nProcess:")
    print("1. Find duplicates by TITLE or DOI")
    print("2. Copy abstracts from versions that have them")
    print("3. Split into two files:")
    print("   - titles_with_abstracts.csv")
    print("   - titles_without_abstracts.csv")
    print("="*60)
    
    df_with, df_without = fill_abstracts_from_duplicates_by_title_and_doi(input_file)
    
    generate_summary_report(input_file, './data/titles_with_abstracts.csv', './data/titles_without_abstracts.csv')
    
    print("✓ All done!")