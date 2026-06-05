# SARA - pipeline zbierania danych WMiI UAM

Pobiera dane o publikacjach pracowników naukowych WMiI UAM z Portalu Badawczego
UAM i OpenAlex API.

## Przegląd pipeline'u

```
Krok 1: research_portal_scraper.py
  -> scraping profili pracowników z Portalu Badawczego UAM
  -> wyjście: data/scientists_data.csv

Krok 2: extract_identifiers.py
  -> wchodzi na każdy profil, wyciąga identyfikatory ORCID + Scopus + Scholar
  -> wyjście: data/scientists_with_identifiers.csv  <- kluczowy plik pośredni

Krok 3: filter_data.py  [opcjonalny - wymaga dużych dumpów OpenAlex]
  -> filtruje pobrane wcześniej dumpy OpenAlex po ORCID-ach pracowników WMiI
  -> wyjście: data/wmii_authors.json, data/wmii_works.json

Krok 4: fetch_abstracts.py
  -> wywołuje OpenAlex API dla każdego ORCID-a, pobiera wszystkie publikacje + abstrakty
  -> uzupełnia brakujące abstrakty z rekordów duplikatów (ten sam tytuł lub DOI)
  -> dla wciąż brakujących abstraktów: otwiera linki DOI przez Selenium
     i scrapuje strony wydawców (ScienceDirect, Springer, Wiley, MDPI, IEEE i inne)
  -> jeżeli Krok 2 nie został uruchomiony, używa data/wmii_orcid.csv jako fallback
  -> wyjście: data/wmii_publications.csv                <- wszystkie rekordy
              data/wmii_publications_with_abstracts.csv <- tylko rekordy z abstraktami
```

## Struktura katalogu

```
.
├── run_pipeline.sh                       # uruchomienie pełnego pipeline'u
├── requirements.txt
├── src/
│   ├── research_portal_scraper.py        # Krok 1
│   ├── extract_identifiers.py            # Krok 2
│   ├── filter_data.py                    # Krok 3 (opcjonalny)
│   └── fetch_abstracts.py                # Krok 4
└── data/
    ├── wmii_orcid.csv                    # lista ORCID-ów (fallback startowy)
    ├── uam_authors.json                  # [wejście opcjonalne] dump autorów OpenAlex
    ├── uam_works.json                    # [wejście opcjonalne] dump prac OpenAlex
    ├── scientists_data.csv               # [wyjście] Krok 1
    ├── scientists_with_identifiers.csv   # [wyjście] Krok 2
    ├── wmii_authors.json                 # [wyjście] Krok 3
    ├── wmii_works.json                   # [wyjście] Krok 3
    ├── wmii_publications.csv             # [wyjście] Krok 4 - wszystkie rekordy
    └── wmii_publications_with_abstracts.csv  # [wyjście] Krok 4 - tylko z abstraktami
```

## Pliki wynikowe

| Plik | Opis |
|------|------|
| `scientists_with_identifiers.csv` | Profile pracowników wraz z ORCID, Scopus i Scholar ID |
| `wmii_publications.csv` | Wszystkie publikacje (również te bez abstraktu) |
| `wmii_publications_with_abstracts.csv` | Publikacje z abstraktem, gotowe do analizy |
| `wmii_authors.json` | Rekordy autorów z OpenAlex (tylko Krok 3) |
| `wmii_works.json` | Rekordy prac z OpenAlex (tylko Krok 3) |

## Instalacja

```bash
pip install -r requirements.txt
chmod +x run_pipeline.sh
```

Opcjonalnie - pobranie dumpów OpenAlex dla Kroku 3:
> https://uam-my.sharepoint.com/:f:/r/personal/jakpas3_st_amu_edu_pl/Documents/SARA?csf=1&web=1&e=RlhsKV

Plik `uam_authors.json` i `uam_works.json` należy umieścić w `data/`.

## Uruchomienie

```bash
# pełny pipeline (Kroki 1-4)
./run_pipeline.sh

# albo poszczególne kroki:
python src/research_portal_scraper.py   # Krok 1
python src/extract_identifiers.py       # Krok 2
python src/filter_data.py               # Krok 3 (opcjonalny)
python src/fetch_abstracts.py           # Krok 4

# Krok 4 może działać samodzielnie korzystając z pliku startowego:
# data/wmii_orcid.csv jest używany automatycznie, jeżeli
# scientists_with_identifiers.csv jeszcze nie istnieje
```

## Opis kolumn - wmii_publications.csv

| Kolumna | Opis |
|---------|------|
| `main_author_orcid` | ORCID pracownika |
| `openalex_id` | identyfikator pracy w OpenAlex |
| `title` | tytuł publikacji |
| `publication_year` | rok publikacji |
| `publication_date` | pełna data publikacji |
| `doi` | link DOI |
| `type` | typ pracy (article / book-chapter / itd.) |
| `cited_by_count` | liczba cytowań |
| `journal` | nazwa czasopisma / venue |
| `topics` | tematy badawcze (oddzielone średnikami) |
| `co_authors` | nazwiska współautorów (oddzielone średnikami) |
| `co_author_orcids` | ORCID-y współautorów (oddzielone średnikami) |
| `num_co_authors` | liczba współautorów |
| `abstract` | pełny tekst abstraktu |
| `keywords` | słowa kluczowe (oddzielone średnikami) |
