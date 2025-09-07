import os
import json
import hashlib
import logging
from typing import List, Dict, Optional, Union, Set
from pathlib import Path
import time
from semantic_scholar.utils.ner_utils import Entity
import spacy
from openai import OpenAI
from semantic_scholar.utils.utils import extract_json_array


class NERError(Exception):
    """Base exception for NER operations."""
    pass

class ModelNotAvailableError(NERError):
    """Raised when required model is not available."""
    pass

class CustomNER:    
    def __init__(
        self, 
        method: str = "spacy", 
        openai_model: str = "gpt-4o-mini",
        language: str = "pl",
        cache_enabled: bool = True,
        max_text_length: int = 10000,
        confidence_threshold: float = 0.0,
        normalize_labels: bool = True
    ):
        """
        Initialize CustomNER.
        
        Args:
            method: NER method ('spacy', 'openai', or 'hybrid')
            openai_model: OpenAI model to use
            language: Language code (pl, en, etc.)
            cache_enabled: Enable result caching
            max_text_length: Maximum text length to process
            confidence_threshold: Minimum confidence for entities
            normalize_labels: Standardize entity labels
        """
        self.method = method.lower()
        self.openai_model = openai_model
        self.language = language
        self.cache_enabled = cache_enabled
        self.max_text_length = max_text_length
        self.confidence_threshold = confidence_threshold
        self.normalize_labels = normalize_labels
        
        self.logger = logging.getLogger(__name__)
        
        if cache_enabled:
            self._cache = {}
            self._cache_file = Path(f".ner_cache_{language}_{method}.json")
            self._load_cache()
        
        self._init_models()
    
    def _init_models(self):
        """Initialize the selected NER models."""
        if self.method in ("spacy", "hybrid"):
            self._init_spacy()
        if self.method in ("openai", "hybrid"):
            self._init_openai()
    
    def _init_spacy(self):
        """Initialize SpaCy model with better error handling."""
        if spacy is None:
            raise ModelNotAvailableError("SpaCy is not installed. Install with: pip install spacy")
        
        model_map = {
            "pl": "pl_core_news_sm",
            "en": "en_core_web_sm", 
            "de": "de_core_news_sm",
            "fr": "fr_core_news_sm",
            "es": "es_core_news_sm"
        }
        
        model_name = model_map.get(self.language)
        if not model_name:
            raise ValueError(f"Unsupported language for SpaCy: {self.language}")
        
        try:
            self.nlp = spacy.load(model_name)
            self.logger.info(f"Loaded SpaCy model: {model_name}")
        except OSError:
            error_msg = (
                f"SpaCy model '{model_name}' not found. Install it with:\n"
                f"python -m spacy download {model_name}"
            )
            self.logger.error(error_msg)
            raise ModelNotAvailableError(error_msg)
        
        if "ner" not in self.nlp.pipe_names:
            self.logger.warning("NER component not found in SpaCy pipeline")
    
    def _init_openai(self):
        """Initialize OpenAI client with better configuration."""
        if OpenAI is None:
            raise ModelNotAvailableError("OpenAI library is not installed. Install with: pip install openai")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ModelNotAvailableError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
            )
        
        try:
            self.client = OpenAI(api_key=api_key)
            self.client.models.list()
            self.logger.info(f"OpenAI client initialized with model: {self.openai_model}")
        except Exception as e:
            raise ModelNotAvailableError(f"Failed to initialize OpenAI client: {str(e)}")
    
    def _load_cache(self):
        """Load cached results from file."""
        if self._cache_file.exists():
            try:
                with open(self._cache_file, 'r', encoding='utf-8') as f:
                    self._cache = json.load(f)
                self.logger.debug(f"Loaded {len(self._cache)} cached results")
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")
                self._cache = {}
    
    def _save_cache(self):
        """Save cache to file."""
        if not self.cache_enabled:
            return
        
        try:
            with open(self._cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        key_data = f"{self.method}:{self.language}:{self.openai_model}:{text}"
        return hashlib.md5(key_data.encode('utf-8')).hexdigest()
    
    def _normalize_label(self, label: str) -> str:
        """Normalize entity labels to standard format."""
        if not self.normalize_labels:
            return label
        return self.STANDARD_LABELS.get(label, label.upper())
    
    def _validate_text(self, text: str) -> str:
        """Validate and preprocess input text."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        text = text.strip()
        if len(text) > self.max_text_length:
            self.logger.warning(f"Text truncated from {len(text)} to {self.max_text_length} characters")
            text = text[:self.max_text_length]
        
        return text
    
    # ---------- Public API ----------
    
    def extract_entities(self, text: str, return_objects: bool = False) -> Union[List[Dict], List[Entity]]:
        """
        Extract entities from text.
        
        Args:
            text: Input text
            return_objects: Return Entity objects instead of dictionaries
            
        Returns:
            List of entities as dictionaries or Entity objects
        """
        text = self._validate_text(text)
        
        if self.cache_enabled:
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                cached_result = self._cache[cache_key]
                entities = [Entity(**e) for e in cached_result]
                return entities if return_objects else [e.to_dict() for e in entities]
        
        try:
            if self.method == "spacy":
                entities = self._extract_with_spacy(text)
            elif self.method == "openai":
                entities = self._extract_with_openai(text)
            elif self.method == "hybrid":
                entities = self._extract_hybrid(text)
            else:
                raise ValueError(f"Unsupported method: {self.method}")
            
            # Filter by confidence threshold
            if self.confidence_threshold > 0:
                entities = [e for e in entities if (e.confidence or 1.0) >= self.confidence_threshold]
            
            # Cache results
            if self.cache_enabled and cache_key:
                self._cache[cache_key] = [e.to_dict() for e in entities]
                if len(self._cache) % 10 == 0:  # Save cache every 10 entries
                    self._save_cache()
            
            return entities if return_objects else [e.to_dict() for e in entities]
            
        except Exception as e:
            self.logger.error(f"Entity extraction failed: {str(e)}")
            raise NERError(f"Failed to extract entities: {str(e)}")
    
    def extract_entity_types(self, text: str) -> Set[str]:
        """Get unique entity types found in text."""
        entities = self.extract_entities(text, return_objects=True)
        return {entity.label for entity in entities}
    
    def filter_entities(
        self, 
        text: str, 
        entity_types: Optional[List[str]] = None,
        min_length: int = 1,
        exclude_patterns: Optional[List[str]] = None
    ) -> List[Entity]:
        """
        Extract and filter entities by type and other criteria.
        
        Args:
            text: Input text
            entity_types: List of entity types to include
            min_length: Minimum entity text length
            exclude_patterns: Patterns to exclude (simple substring matching)
            
        Returns:
            Filtered list of Entity objects
        """
        entities = self.extract_entities(text, return_objects=True)
        
        filtered = []
        for entity in entities:
            if entity_types and entity.label not in entity_types:
                continue
            if len(entity.text) < min_length:
                continue
            if exclude_patterns:
                if any(pattern.lower() in entity.text.lower() for pattern in exclude_patterns):
                    continue
            
            filtered.append(entity)
        
        return filtered
    
    def print_entities(self, text: str, show_positions: bool = True):
        """Print entities in a formatted way."""
        entities = self.extract_entities(text, return_objects=True)
        
        print(f"Text: {text}")
        print(f"Found {len(entities)} entities:")
        
        for entity in entities:
            if show_positions and entity.start is not None and entity.end is not None:
                pos_info = f" (pos: {entity.start}-{entity.end})"
            else:
                pos_info = ""
            
            confidence_info = f" [conf: {entity.confidence:.2f}]" if entity.confidence else ""
            print(f"  - {entity.text} ({entity.label}){pos_info}{confidence_info}")
        print()
    
    def get_statistics(self) -> Dict[str, Union[int, float]]:
        """Get processing statistics."""
        stats = {
            "method": self.method,
            "language": self.language,
            "cached_results": len(self._cache) if self.cache_enabled else 0,
        }
        return stats
    
    # ---------- Backend Methods ----------
    
    def _extract_with_spacy(self, text: str) -> List[Entity]:
        """Extract entities using SpaCy."""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entity = Entity(
                text=ent.text.strip(),
                label=self._normalize_label(ent.label_),
                start=ent.start_char,
                end=ent.end_char,
                confidence=getattr(ent, 'score', None)
            )
            entities.append(entity)
        
        return entities
    
    def _extract_with_openai(self, text: str, max_retries: int = 2) -> List[Entity]:
        """Extract entities using OpenAI with improved error handling."""
        if self.language == "pl":
            prompt = (
                "Wykonaj rozpoznawanie nazw własnych (NER) w poniższym tekście polskim. "
                "Zwróć TYLKO tablicę JSON z obiektami zawierającymi pola: text, label. "
                "Używaj standardowych etykiet: PERSON, ORGANIZATION, LOCATION, DATE, TIME, MONEY, PERCENT, MISCELLANEOUS.\n\n"
                f'Tekst: "{text}"'
            )
        else:
            prompt = (
                "Perform named entity recognition (NER) on the following text. "
                "Return ONLY a JSON array of objects with fields: text, label. "
                "Use standard labels: PERSON, ORGANIZATION, LOCATION, DATE, TIME, MONEY, PERCENT, MISCELLANEOUS.\n\n"
                f'Text: "{text}"'
            )
        
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": "You are an expert at named entity recognition. Output only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    response_format={"type": "json_object"} if "gpt-4" in self.openai_model else None
                )
                
                content = response.choices[0].message.content.strip()
                
                try:
                    data = json.loads(content)
                    if isinstance(data, dict):
                        entities_data = data.get('entities', data.get('results', []))
                    else:
                        entities_data = data
                    
                    if not isinstance(entities_data, list):
                        entities_data = []
                    
                except json.JSONDecodeError:
                    entities_data = extract_json_array(content)
                
                entities = []
                for item in entities_data:
                    if isinstance(item, dict) and 'text' in item and 'label' in item:
                        entity = Entity(
                            text=item['text'].strip(),
                            label=self._normalize_label(item['label']),
                            confidence=item.get('confidence')
                        )
                        entities.append(entity)
                
                return entities
                
            except Exception as e:
                self.logger.warning(f"OpenAI NER attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries:
                    raise NERError(f"OpenAI NER failed after {max_retries + 1} attempts: {str(e)}")
                time.sleep(1)
        
        return []
    
    def _extract_hybrid(self, text: str) -> List[Entity]:
        """Extract entities using both SpaCy and OpenAI, then merge results."""
        spacy_entities = self._extract_with_spacy(text)
        openai_entities = self._extract_with_openai(text)
        
        seen_texts = set()
        merged_entities = []
        
        for entity in spacy_entities:
            if entity.text.lower() not in seen_texts:
                seen_texts.add(entity.text.lower())
                merged_entities.append(entity)
        
        for entity in openai_entities:
            if entity.text.lower() not in seen_texts:
                seen_texts.add(entity.text.lower())
                merged_entities.append(entity)
        
        return merged_entities
    
    # ---------- Context Manager ----------
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save cache and cleanup."""
        if self.cache_enabled:
            self._save_cache()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'cache_enabled') and self.cache_enabled:
            self._save_cache
