import unicodedata
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

from semantic_scholar.custom_ner import CustomNER, NERError
from semantic_scholar.semanticscholar_api_client import SemanticScholarAPIClient, SemanticScholarAPIError
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AuthorCandidate:
    """Structured representation of an author candidate."""
    name: str
    author_id: str
    h_index: Optional[int] = None
    paper_count: Optional[int] = None
    url: Optional[str] = None
    affiliation: Optional[str] = None
    
    def __str__(self) -> str:
        parts = [f"Name: {self.name}", f"ID: {self.author_id}"]
        if self.h_index is not None:
            parts.append(f"h-Index: {self.h_index}")
        if self.paper_count is not None:
            parts.append(f"Papers: {self.paper_count}")
        if self.affiliation:
            parts.append(f"Affiliation: {self.affiliation}")
        if self.url:
            parts.append(f"URL: {self.url}")
        return " | ".join(parts)

class AuthorSearcher:
    """Class for searching and identifying authors using NER and Semantic Scholar."""
    
    def __init__(
        self, 
        ner_method: str = "hybrid",
        openai_model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        cache_enabled: bool = True
    ):
        """
        Initialize the author searcher.
        
        Args:
            ner_method: NER method to use ('spacy', 'openai', 'hybrid')
            openai_model: OpenAI model for NER
            api_key: Semantic Scholar API key (optional but recommended)
            cache_enabled: Enable caching for better performance
        """
        self.ner_method = ner_method
        self.cache_enabled = cache_enabled
        
        # Initialize NER
        try:
            self.ner = CustomNER(
                method=ner_method, 
                openai_model=openai_model,
                cache_enabled=cache_enabled
            )
        except Exception as e:
            logger.error(f"Failed to initialize NER: {e}")
            raise
        
        # Initialize Semantic Scholar client
        try:
            self.ss_client = SemanticScholarAPIClient(api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to initialize Semantic Scholar client: {e}")
            raise
    
    @staticmethod
    def strip_accents(text: str) -> str:
        """Remove accents from text for better matching."""
        return "".join(
            c for c in unicodedata.normalize("NFD", text) 
            if unicodedata.category(c) != "Mn"
        )
    
    @staticmethod
    def normalize_name(name: str) -> str:
        """Normalize name for deduplication."""
        return AuthorSearcher.strip_accents(name.strip()).lower()
    
    def extract_potential_names(self, text: str) -> List[str]:
        """
        Extract potential author names from text.
        
        Instead of guessing if something is a person, we extract all named entities
        and let Semantic Scholar API determine if they're actual authors.
        """
        try:
            entities = self.ner.extract_entities(text)
        except NERError as e:
            logger.error(f"NER extraction failed: {e}")
            return []
        
        # Extract all text from named entities (not just PERSON labels)
        # This is more inclusive and lets the API do the filtering
        potential_names = []
        
        for entity in entities:
            text_content = entity.get("text", "").strip()
            if not text_content:
                continue
            
            # Basic filters to remove obviously non-name entities
            if self._is_likely_name(text_content):
                potential_names.append(text_content)
        
        # Deduplicate while preserving original casing
        return self._deduplicate_names(potential_names)
    
    def _is_likely_name(self, text: str) -> bool:
        """
        Basic heuristics to filter out obviously non-name entities.
        Much more permissive than the original approach.
        """
        # Remove very short or very long strings
        if len(text) < 2 or len(text) > 100:
            return False
        
        # Remove strings that are clearly not names
        text_lower = text.lower()
        
        # Skip common non-name words
        non_names = {
            'today', 'yesterday', 'tomorrow', 'now', 'here', 'there',
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december',
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'http', 'https', 'www', 'com', 'org', 'net', 'edu',
        }
        
        if text_lower in non_names:
            return False
        
        # Skip URLs and email-like patterns
        if any(pattern in text_lower for pattern in ['http', 'www.', '@', '.com', '.org', '.net']):
            return False
        
        # Skip numbers-only
        if text.replace(' ', '').replace('-', '').replace('.', '').isdigit():
            return False
        
        return True
    
    def _deduplicate_names(self, names: List[str]) -> List[str]:
        """Deduplicate names while preserving original casing."""
        seen = set()
        unique = []
        
        for name in names:
            normalized = self.normalize_name(name)
            if normalized not in seen and normalized:  # Skip empty normalized names
                seen.add(normalized)
                unique.append(name)
        
        return unique
    
    def search_semantic_scholar_candidates(
        self, 
        name: str, 
        limit: int = 5
    ) -> List[AuthorCandidate]:
        """Search for author candidates in Semantic Scholar."""
        try:
            logger.debug(f"Searching Semantic Scholar for: {name}")
            response = self.ss_client.search_authors(name, limit=limit)
            authors_data = response.get("data", [])
            
            candidates = []
            for author in authors_data:
                candidate = AuthorCandidate(
                    name=author.get("name", ""),
                    author_id=author.get("authorId", ""),
                    h_index=author.get("hIndex"),
                    paper_count=author.get("paperCount"),
                    url=author.get("url"),
                    affiliation=self._extract_affiliation(author)
                )
                candidates.append(candidate)
            
            return candidates
            
        except SemanticScholarAPIError as e:
            logger.warning(f"Semantic Scholar API error for '{name}': {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error searching for '{name}': {e}")
            return []
    
    def _extract_affiliation(self, author_data: Dict) -> Optional[str]:
        """Extract affiliation information from author data."""
        affiliations = author_data.get("affiliations", [])
        if affiliations and isinstance(affiliations, list):
            # Return the first affiliation name
            first_affil = affiliations[0]
            if isinstance(first_affil, dict):
                return first_affil.get("name")
            elif isinstance(first_affil, str):
                return first_affil
        return None
    
    def find_authors_in_text(
        self, 
        text: str, 
        limit_per_name: int = 3,
        min_confidence_score: float = 0.0
    ) -> Dict[str, List[AuthorCandidate]]:
        """
        Find potential authors mentioned in text.
        
        Args:
            text: Input text to analyze
            limit_per_name: Maximum candidates per name
            min_confidence_score: Minimum confidence for filtering results
            
        Returns:
            Dictionary mapping extracted names to their candidates
        """
        logger.info(f"Analyzing text: {text[:100]}...")
        
        # Extract potential names
        potential_names = self.extract_potential_names(text)
        logger.info(f"Found {len(potential_names)} potential names: {potential_names}")
        
        results = {}
        
        for name in potential_names:
            candidates = self.search_semantic_scholar_candidates(name, limit_per_name)
            
            # Filter by confidence if we have scoring
            if min_confidence_score > 0:
                # This would require implementing confidence scoring
                # For now, we'll include all results
                pass
            
            if candidates:
                results[name] = candidates
                logger.info(f"Found {len(candidates)} candidates for '{name}'")
            else:
                logger.info(f"No candidates found for '{name}'")
        
        return results
    
    def print_results(self, results: Dict[str, List[AuthorCandidate]]):
        """Print search results in a formatted way."""
        if not results:
            print("No author candidates found.")
            return
        
        for query_name, candidates in results.items():
            print(f"\n=== Results for: {query_name} ===")
            if not candidates:
                print("  (no candidates)")
                continue
            
            for i, candidate in enumerate(candidates, 1):
                print(f"  {i}. {candidate}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Clean up resources if needed
        pass