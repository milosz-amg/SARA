import openai
import json
import re
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class Entity:
    type: str
    value: str
    confidence: float = 1.0
    normalized_value: Optional[str] = None

class EntityExtractor:
    def __init__(self, api_key: str, logger, model: str = "gpt-4o"):
        self.client = openai.OpenAI(api_key=api_key)
        self.logger = logger
        self.model = model
    
    def create_extraction_prompt(self, text: str) -> str:
        prompt = f"""
                Jesteś ekspertem do ekstrakcji encji, zaprojektowanym do identyfikacji i normalizacji encji z tekstu w języku polskim i angielskim na potrzeby zapytań do bazy danych.

                Wydobądź encje z poniższego tekstu i sklasyfikuj je do następujących kategorii:
                - PERSON: Pełne imiona i nazwiska osób (imię + nazwisko)
                - ORGANIZATION: Firmy, uczelnie, instytucje, wydziały
                - LOCATION: Miasta, kraje, adresy, miejsca
                - JOB_TITLE: Stanowiska zawodowe, role, tytuły
                - DATE: Daty, okresy czasu, lata
                - PROJECT: KONKRETNE projekty (zwykle z nazwą własną, akronimem/ID/grantem, liderem lub ramami czasowymi)
                - DEPARTMENT: Konkretne wydziały/jednostki
                - TOPIC: Dziedziny, obszary badawcze, technologie, metody („machine learning”, „NLP”, „chemia kwantowa”)
                - OTHER: Inne encje przydatne do zapytań w bazie danych

                WAŻNE ZASADY (doprecyzowanie PROJECT vs TOPIC):
                - Oznacz jako PROJECT tylko wtedy, gdy występuje nazwa własna projektu lub identyfikator (np. „Horizon 2020 Grant 12345”, „Projekt ‘SmartCity’”, „NCN OPUS 2021/AB/…”, „Project Apollo”).
                - Ogólne dziedziny/tematy/technologie NIE są projektami → klasyfikuj jako TOPIC (np. „machine learning”, „blockchain”, „mikroserwisy”, „sieci neuronowe”).
                - Jeśli nie masz pewności, wybierz TOPIC zamiast PROJECT.

                Tekst do analizy: "{text}"

                Zwróć odpowiedź jako poprawny obiekt JSON o dokładnie takiej strukturze:
                {{
                    "entities": [
                        {{
                            "type": "PERSON",
                            "value": "wydobyty fragment tekstu",
                            "normalized_value": "znormalizowana forma, jeśli różni się od value",
                            "confidence": "wartość pewności ekstrakcji (0-100)"
                        }}
                    ],
                }}

                Upewnij się, że JSON jest poprawny i zawiera wszystkie istotne encje przydatne do zapytań w bazie danych.
            """
        return prompt

    
    def extract_entities(self, text: str, max_retries: int = 3) -> Dict:
        if not text or not text.strip():
            return {"entities": []}
        
        prompt = self.create_extraction_prompt(text)
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a precise entity extraction system. Always return valid JSON."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1000,
                    response_format={"type": "json_object"}
                )
                
                result_text = response.choices[0].message.content
                
                result = json.loads(result_text)
                result = self._post_process_entities(result)
                
                return result
                    
            except Exception as e:
                self.logger.error(f"API call failed on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise
                    
        return {"entities": [], "query_intent": "extraction_failed"}
    
    def _post_process_entities(self, result: Dict) -> Dict:
        if "entities" not in result:
            return result
            
        processed_entities = []
        
        for entity in result["entities"]:
            if "type" not in entity or "value" not in entity:
                continue
                
            entity_obj = Entity(
                type=entity["type"],
                value=entity["value"].strip(),
                confidence=entity.get("confidence", 0.9),
                normalized_value=entity.get("normalized_value", "").strip() or None
            )
            
            entity_obj = self._apply_normalization_rules(entity_obj)
            
            processed_entities.append({
                "type": entity_obj.type,
                "value": entity_obj.value,
                "normalized_value": entity_obj.normalized_value,
                "confidence": entity_obj.confidence
            })
        
        result["entities"] = processed_entities
        return result
    
    def _apply_normalization_rules(self, entity: Entity) -> Entity:
        if entity.type == "PERSON":
            parts = entity.value.split()
            normalized_parts = [part.capitalize() for part in parts if part]
            entity.normalized_value = " ".join(normalized_parts)
            
        return entity
