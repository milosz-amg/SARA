from typing import List, Dict, Optional, Union, Set
from dataclasses import dataclass


STANDARD_LABELS = {
        "persName": "PERSON",
        "placeName": "LOCATION", 
        "orgName": "ORGANIZATION",
        "geogName": "LOCATION",
        "date": "DATE",
        "time": "TIME",
        "PERSON": "PERSON",
        "ORG": "ORGANIZATION", 
        "GPE": "LOCATION",
        "LOC": "LOCATION",
        "MISC": "MISCELLANEOUS",
        "DATE": "DATE",
        "TIME": "TIME",
        "MONEY": "MONEY",
        "PERCENT": "PERCENT"
    }


@dataclass
class Entity:
    """Structured entity representation."""
    text: str
    label: str
    start: Optional[int] = None
    end: Optional[int] = None
    confidence: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Union[str, int, float]]:
        """Convert to dictionary format."""
        result = {"text": self.text, "label": self.label}
        if self.start is not None:
            result["start"] = self.start
        if self.end is not None:
            result["end"] = self.end
        if self.confidence is not None:
            result["confidence"] = self.confidence
        return result