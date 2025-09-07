import logging
from dotenv import load_dotenv

from semantic_scholar.utils.authors import AuthorSearcher

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main function demonstrating the improved approach."""
    
    test_cases = [
        "Kim jest Patryk Żywica?",
        "Artykuł napisany przez Jan Kowalski i Maria Nowak z Uniwersytetu Warszawskiego.",
        "Badania prowadzone przez zespół pod kierunkiem prof. Anna Kowalczyk.",
        "Einstein opublikował teorię względności w 1905 roku.",
        "Współpraca między Dr. Smith i Prof. Johnson doprowadziła do przełomowych odkryć."
    ]
    
    methods = ["spacy" , "openai"]#, "hybrid"]
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Testing with {method.upper()} NER")
        print('='*50)
        
        try:
            with AuthorSearcher(ner_method=method, cache_enabled=True) as searcher:
                
                for test_text in test_cases:
                    print(f"\n--- Processing: {test_text} ---")
                    
                    results = searcher.find_authors_in_text(
                        text=test_text,
                        limit_per_name=3
                    )
                    
                    searcher.print_results(results)
                    
        except Exception as e:
            logger.error(f"Failed to test {method} method: {e}")
            continue

if __name__ == "__main__":
    main()