# 🔍 SARA Semantic Search Module

System wyszukiwania semantycznego dla projektu SARA - umożliwia inteligentne wyszukiwanie naukowców i publikacji na podstawie zapytań w języku naturalnym.

## ✨ Funkcjonalności

### 1. **Wyszukiwanie Naukowców** (`search_similar_authors`)
Znajdź naukowców na podstawie opisu zainteresowań badawczych.

```python
from semantic_search import search_similar_authors

results = search_similar_authors(
    query="machine learning expert in computer vision",
    top_k=10,
    min_works=20,
    min_citations=1000,
    institution="Warsaw"
)

for author in results:
    print(f"{author['display_name']} - h-index: {author['h_index']}")
    print(f"Similarity: {author['similarity']:.4f}")
```

**Parametry:**
- `query` (str) - zapytanie w języku naturalnym
- `top_k` (int) - liczba wyników do zwrócenia (domyślnie: 10)
- `min_works` (int) - minimalna liczba publikacji (domyślnie: 0)
- `min_citations` (int) - minimalna liczba cytowań (domyślnie: 0)
- `institution` (str) - filtruj po nazwie instytucji (opcjonalne)

---

### 2. **Wyszukiwanie Publikacji** (`search_similar_works`)
Znajdź publikacje naukowe na podstawie opisu tematu badawczego.

```python
from semantic_search import search_similar_works

results = search_similar_works(
    query="deep learning for medical image segmentation",
    top_k=10,
    year_from=2020,
    year_to=2024,
    work_type="article",
    min_citations=50
)

for work in results:
    print(f"{work['title']}")
    print(f"Year: {work['publication_year']}, Citations: {work['cited_by_count']}")
    print(f"Similarity: {work['similarity']:.4f}")
```

**Parametry:**
- `query` (str) - zapytanie w języku naturalnym
- `top_k` (int) - liczba wyników (domyślnie: 10)
- `year_from` (int) - od roku (opcjonalne)
- `year_to` (int) - do roku (opcjonalne)
- `work_type` (str) - typ publikacji: "article", "book", "proceedings" (opcjonalne)
- `min_citations` (int) - minimalna liczba cytowań (domyślnie: 0)

---

### 3. **Rekomendacja Współpracowników** (`recommend_collaborators`)
Znajdź potencjalnych współpracowników do projektu badawczego.

```python
from semantic_search import recommend_collaborators

project = """
We are developing a new AI system for climate modeling and prediction.
We need experts in climate science, machine learning, and big data analytics.
"""

results = recommend_collaborators(
    project_description=project,
    top_k=5,
    min_h_index=15,
    exclude_institution="My University"
)

for author in results:
    print(f"{author['display_name']} ({author['impact_level']})")
    print(f"Institution: {author['last_known_institution_name']}")
    print(f"Expertise: {', '.join(author['top_research_areas'][:3])}")
    print(f"Match Score: {author['similarity']:.4f}")
```

**Parametry:**
- `project_description` (str) - opis projektu badawczego
- `top_k` (int) - liczba rekomendacji (domyślnie: 10)
- `min_h_index` (int) - minimalny h-index (domyślnie: 0)
- `institution` (str) - preferuj tę instytucję (opcjonalne)
- `exclude_institution` (str) - wyklucz tę instytucję (opcjonalne)

**Impact Levels:**
- `High Impact`: h-index ≥ 50
- `Mid Impact`: h-index ≥ 20
- `Emerging`: h-index ≥ 10
- `Early Career`: h-index < 10

---

## 🚀 Szybki Start

### Wymagania
```bash
pip install psycopg2-binary torch transformers
```

### Podstawowe użycie
```bash
# Proste wyszukiwanie z linii komend
python semantic_search.py "quantum computing researcher"

# Uruchomienie interaktywnego demo
python demo_semantic_search.py
```

### Demo interaktywne
```bash
python demo_semantic_search.py
```

Demo zawiera 7 przykładów:
1. Basic Author Search
2. Filtered Author Search
3. Search by Institution
4. Search Research Papers
5. Medical Research
6. Collaborator Recommendations
7. Cross-Disciplinary Search

---

## 🔧 Architektura

### Baza Danych
- **PostgreSQL 15** z rozszerzeniem **pgvector 0.8.1**
- **438,000 autorów** z embeddingami (1024-dim)
- **1,528,000 publikacji** z embeddingami (1024-dim)
- **HNSW indexes** dla szybkiego wyszukiwania (cosine similarity)

### Model Embeddingów
- **Qwen3-Embedding-0.6B**
- Wymiar wektorów: 1024
- Normalizacja: L2
- Pooling: mean pooling ostatniej warstwy

### Co jest embeddowane?

**Autorzy:**
```
Name: [display_name]
Institution: [institution]
Metrics: [works] works, [citations] citations, h-index [h]
Research areas: [top 5 concepts]
```

**Publikacje:**
```
[title]
[abstract - pierwsze 500 słów]
Type: [type], Year: [year]
Topics: [top 5 concepts]
```

---

## 📊 Przykładowe Wyniki

### Wyszukiwanie: "cardiology researcher"
```
1. Piotr Ponikowski
   🏛️  Wroclaw Medical University
   📊 1,456 works, 265,330 citations, h-index: 174
   🎯 Similarity: 0.8254
   🔬 Cardiology, Internal Medicine, Heart Failure
```

### Wyszukiwanie: "quantum computing"
```
1. "Quantum algorithms for optimization problems"
   📅 Year: 2023 | Type: article
   📚 Citations: 145
   🎯 Similarity: 0.8912
   🔬 Quantum Computing, Optimization, Algorithm Design
```

---

## 🔍 Funkcje Pomocnicze

### Szczegóły autora
```python
from semantic_search import get_author_details

author = get_author_details("A5017272571")
print(author['display_name'])
print(author['h_index'])
```

### Szczegóły publikacji
```python
from semantic_search import get_work_details

work = get_work_details("W2123456789")
print(work['title'])
print(work['abstract_inverted_index'])
```

---

## ⚡ Wydajność

- **Wyszukiwanie**: ~0.5-1s (z ładowaniem modelu: ~3s przy pierwszym uruchomieniu)
- **HNSW Index**: O(log n) dla wyszukiwania
- **Batch Processing**: Możliwe równoległe zapytania
- **GPU**: Wspierane (automatyczna detekcja CUDA)

---

## 🛠️ Konfiguracja

### Parametry bazy danych
Edytuj w `semantic_search.py`:
```python
DB_PARAMS = {
    'host': 'localhost',
    'port': 5432,
    'database': 'devdb',
    'user': 'devuser',
    'password': 'devpass'
}
```

### Model embeddingów
```python
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DIMENSIONS = 1024
```

---

## 📝 Przykłady Użycia

### Use Case 1: Znalezienie eksperta do projektu
```python
results = search_similar_authors(
    "expert in climate modeling and environmental science",
    top_k=5,
    min_h_index=10
)
```

### Use Case 2: Literatura do grant proposal
```python
papers = search_similar_works(
    "renewable energy and sustainability",
    top_k=20,
    year_from=2020,
    min_citations=100
)
```

### Use Case 3: Międzynarodowa współpraca
```python
collaborators = recommend_collaborators(
    "AI for healthcare diagnostics",
    top_k=10,
    min_h_index=20,
    exclude_institution="University of Warsaw"  # Szukamy poza własną uczelnią
)
```

---

## 🐛 Troubleshooting

### Model się nie ładuje
```bash
# Sprawdź połączenie z Hugging Face
pip install --upgrade transformers torch

# Ręczne pobranie
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B')"
```

### Błąd połączenia z bazą
```bash
# Sprawdź czy kontener PostgreSQL działa
docker ps | grep pg-local

# Restart kontenera
docker restart pg-local
```

### Wolne wyszukiwanie
```sql
-- Sprawdź czy indeksy HNSW są aktywne
SELECT indexname, indexdef FROM pg_indexes
WHERE tablename IN ('authors', 'works')
AND indexname LIKE '%embedding%';
```

---

## 🎯 Następne kroki (Milestone 2.4)

- [ ] Integracja z LLM do generowania spersonalizowanych rekomendacji
- [ ] Dodanie kontekstu projektowego (NCN grants, UAM priorities)
- [ ] Cache'owanie wyników częstych zapytań
- [ ] API endpoint (FastAPI)
- [ ] Web UI (Streamlit)

---

## 📚 Dokumentacja

### Similarity Score
- Range: 0.0 - 1.0
- Metryka: Cosine Similarity (1 - cosine distance)
- > 0.85: Bardzo silne dopasowanie
- 0.75-0.85: Dobre dopasowanie
- 0.65-0.75: Umiarkowane dopasowanie
- < 0.65: Słabe dopasowanie

### Filtry
Wszystkie filtry są opcjonalne i łączą się operatorem AND.

---

## 👥 Autorzy
Projekt SARA - Scientific Assistant for Research and Analysis
Data utworzenia modułu: 2025-10-11

---

## 📄 Licencja
Internal research tool - UAM
