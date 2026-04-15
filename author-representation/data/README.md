# Dane

## Zawartość folderu

| Plik | Opis | Rozmiar | Źródło |
|------|------|---------|--------|
| `arxiv_papers.db` | SQLite — 18 880 paperów ArXiv z 118 kategorii (tytuł, abstrakt, kategorie, daty) | ~38 MB | Generowany przez `scripts/01_fetch_arxiv_papers.py` + `02_create_database.py` |
| `categories.txt` | Oficjalna taksonomia ArXiv (118 podkategorii pod 8 grupami) | ~40 KB | [arxiv.org/category_taxonomy](https://arxiv.org/category_taxonomy) |
| `titles_with_abstracts.csv` | 3 440 publikacji 115 naukowców WMI (tytuł + abstrakt + topics + keywords + co-authors) | ~4.6 MB | OpenAlex API po ORCID |
| `scientists_with_identifiers.csv` | 164 pracowników WMI UAM (imię, zakład, ORCID, Scopus ID) | ~120 KB | Scraping [wmi.amu.edu.pl](https://wmi.amu.edu.pl) |
| `paper_embeddings_cosent.npy` | Cache embeddingów paperów WMI z modelu fine-tuned (3440 × 768) | ~11 MB | Generowany przez `scripts/09_pca_authors_multiclusters.py` (pierwsze uruchomienie) |

## Odtworzenie danych

### 1. Dane ArXiv (faza 1)

```bash
# Pobiera 200 paperów × 118 kategorii (~3.5 min, rate limit 3s)
python scripts/01_fetch_arxiv_papers.py --all-categories

# Tworzy SQLite z pobranych JSON
python scripts/02_create_database.py
```

Po tym kroku powinien pojawić się `data/arxiv_papers.db` z 18 880 paperów.

### 2. Dane WMI (już w repo)

Pliki `titles_with_abstracts.csv` i `scientists_with_identifiers.csv` są commitowane — nie wymagają pobierania.

Do ich pierwotnego wygenerowania użyto:
- scraper profili WMI UAM (nie dołączony — osobny projekt `collect_uam_data/`)
- OpenAlex API (klient w osobnym projekcie `abstracts/`)

### 3. Cache embeddingów

Pierwsze uruchomienie `scripts/09_pca_authors_multiclusters.py` generuje `paper_embeddings_cosent.npy` (wymaga modelu fine-tuned — patrz `models/README.md`). Cache jest commitowany, więc wizualizacje można odtworzyć od razu.

## Dane pomijane w repo (>100 MB)

- `data/raw/papers_*.json` — 118 surowych odpowiedzi ArXiv API (regenerowalne skryptem `01_fetch_arxiv_papers.py`)
- `data/embeddings/*/embeddings.npy` — embeddingi 9 modeli (faza 2, regenerowalne skryptem `03_generate_embeddings.py`)
- `data/finetune/` — pary/tryplety do fine-tuningu (~230 MB, regenerowalne skryptem `05_prepare_finetune_data.py`)
