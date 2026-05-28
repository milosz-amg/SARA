# SARA — Search and Research Assistant

Repozytorium spinające część badawczo-rozwojową projektu **SARA**: budowę i ewaluację **mapy dorobku naukowego** pracowników Wydziału Matematyki i Informatyki UAM na podstawie embeddingów tekstowych ich publikacji.

**Autor:** Jakub Paszke · **Wydział:** WMiI UAM

## Cel projektu

Projekt podejmuje dwa powiązane problemy badawcze (pełny opis w artykule naukowym przekazywanym komisji osobno):

1. **Reprezentacja autora** — czy naukowca można wiarygodnie reprezentować pojedynczym punktem (centroidem) w przestrzeni embeddingów, czy potrzebna jest reprezentacja wielopunktowa uwzględniająca wielomodalność profilu badawczego. Proponujemy **reprezentację adaptacyjną**: liczbę punktów dobiera się per autor na podstawie diagnostyki stabilności centroidu i automatycznej dekompozycji multi-cluster.
2. **Wizualizacja** — systematyczne porównanie ośmiu metod redukcji wymiarowości pod kątem jakości projekcji 2D zbiorów publikacji, z autorskim wskaźnikiem **Composite Score** agregującym siedem metryk lokalnych i globalnych.

Najważniejsze rezultaty (na zbiorze N = 3440 publikacji, 115 autorów, 14 jednostek):

- fine-tuning BGE-base (CoSENTLoss + hierarchiczna odległość kategorii ArXiv) poprawił wariancję wyjaśnioną w PCA 2D z **33,11 % do 54,71 %**,
- **85,2 %** autorów z ≥5 publikacjami ma wieloklastrową strukturę profilu (średnio k = 2,59 klastrów),
- NMI względem podziału na zakłady: **0,702**.

## Struktura repozytorium

Projekt jest podzielony na trzy niezależne moduły uruchamiane po kolei (każdy z własnym README i `requirements.txt`) oraz artykuł:

```text
README.md                     ← jesteś tutaj (punkt wejścia, spina całość)
wmii-data-collection/         ← MODUŁ 1: zbieranie danych (profile + publikacje + abstrakty)
author-representation/        ← MODUŁ 2: embeddingi, fine-tuning, reprezentacja adaptacyjna, mapa autorów
publications-visualisation/   ← MODUŁ 3: porównanie metod redukcji wymiarowości + Composite Score
site/                         ← gotowa statyczna strona-explorer (wizualizacje do otwarcia w przeglądarce)
old_data/                     ← archiwum: wcześniejszy prototyp asystenta RAG (poza zakresem oceny)
```

Artykuł naukowy opisujący część badawczą (LaTeX + PDF) jest przekazywany komisji osobno (zgodnie z rekomendacjami dot. materiałów poza repozytorium kodu).

| Moduł | Rola | Wejście → Wyjście | README |
|-------|------|-------------------|--------|
| **wmii-data-collection** | Pobiera profile pracowników z Portalu Badawczego UAM i publikacje z OpenAlex (z abstraktami) | Portal UAM / OpenAlex → `embeddings.npy`, CSV publikacji | [link](wmii-data-collection/README.md) |
| **author-representation** | Generuje i dostraja embeddingi, buduje reprezentację adaptacyjną i interaktywną mapę autorów WMiI | publikacje → mapy HTML, metryki separacji zakładów | [link](author-representation/README.md) |
| **publications-visualisation** | Porównuje 8 metod redukcji wymiarowości i wyłania najlepszą wskaźnikiem Composite Score | `embeddings.npy` → ranking metod DR, `vis_methods.html` | [link](publications-visualisation/README.md) |

**Zależność:** moduł 3 czyta dane wyprodukowane przez moduł 1 (`wmii-data-collection/data/embeddings.npy`). Moduł 2 jest samodzielny (pracuje na własnym zbiorze ArXiv + danych WMiI).

## Wymagania

- Python 3.10+, system Linux/macOS/WSL
- Zależności instalowane per moduł z jego `requirements.txt`
- **GPU (CUDA)** zalecane do generowania embeddingów w module 2; fine-tuning BGE wymaga Google Colab (A100/T4) — szczegóły w [author-representation/README.md](author-representation/README.md)
- Moduły 1 i 3 działają na CPU
- Dostęp do internetu (ArXiv API, OpenAlex API, Portal Badawczy UAM) — **bez kluczy API** (oba API są publiczne)

## Instalacja i uruchomienie demonstracji

Każdy moduł uruchamia się we własnym środowisku wirtualnym. Najprostszy scenariusz demonstracyjny — **porównanie metod wizualizacji na gotowych embeddingach** (moduł 3, działa na CPU, kilka–kilkanaście minut):

```bash
cd publications-visualisation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run_visualisation.sh --no-tsne        # szybki wariant demonstracyjny
```

Pełny pipeline (od zbierania danych) opisują README poszczególnych modułów — kolejność: **moduł 1 → moduł 3** oraz niezależnie **moduł 2**.

## Oczekiwany wynik

- **Najszybsza weryfikacja (bez uruchamiania):** otwórz [site/index.html](site/index.html) w przeglądarce — gotowa statyczna strona-explorer z mapą autorów, eksploracją publikacji i porównaniem metod redukcji wymiarowości.
- **Moduł 3:** w `publications-visualisation/output/` powstają `vis_methods.json` + `vis_methods.html` (interaktywna mapa), `k_metrics.json`, `robustness_results.json`; scorery wypisują ranking metod DR.
- **Moduł 2:** interaktywne mapy autorów WMiI (`results/wmi_authors/*.html`) — gotowe artefakty są dołączone, można je otworzyć w przeglądarce bez ponownego liczenia.

## Dane

- **Dane publiczne, bez kluczy:** ArXiv API i OpenAlex API są otwarte; Portal Badawczy UAM jest scrapowany bez logowania.
- **Gotowe artefakty w repo:** embeddingi WMiI (`wmii-data-collection/data/`), baza ArXiv (`author-representation/data/arxiv_papers.db`), wyniki wizualizacji (`publications-visualisation/output/`) — pozwalają zweryfikować wyniki bez pełnej regeneracji.
- **Duże pliki opcjonalne** (dumpy OpenAlex, model fine-tuned >100 MB) są poza repo; instrukcje pobrania w README modułów. Link do współdzielonego folderu OneDrive podany w [wmii-data-collection/README.md](wmii-data-collection/README.md).

## Reprodukcja lub weryfikacja wyników

Trzy poziomy sprawdzenia (zgodnie z rekomendacjami):

1. **Uruchomienie** — instalacja i przebieg modułu 3 na CPU (scenariusz wyżej).
2. **Demonstracja** — interaktywne mapy autorów (moduł 2) i porównanie metod DR (moduł 3) z dołączonych artefaktów.
3. **Weryfikacja wyników** — metryki z artykułu odtwarzalne ze skryptów ewaluacyjnych: [author-representation/scripts/10_evaluate_dept_separation.py](author-representation/scripts/10_evaluate_dept_separation.py) (separacja zakładów), scorery Composite w module 3.

Pełna regeneracja embeddingów i fine-tuning wymagają GPU/Colab i kilkudziesięciu minut–godzin; do oceny wystarczają dołączone artefakty.

## Testy i jakość rozwiązania

Projekt ma charakter badawczy — jakość weryfikowana jest empirycznie, nie testami jednostkowymi:

- **Moduł 3:** eksperyment odporności na losowość (wiele ziaren + testy istotności Wilcoxona) oraz trzy niezależne scorery (Choquet / Sugeno / trimmed-mean), które powinny dawać spójny ranking.
- **Moduł 2:** metryki separacji zakładów (intra/inter similarity, NN@k, NMI, purity) liczone skryptami ewaluacyjnymi.

## Dokumentacja

- **Artykuł naukowy** (część badawcza, LaTeX + PDF) — przekazywany komisji osobno
- **Dokumentacja modułów:** README + szczegółowe opisy w `author-representation/docs/`
- **Referencja skryptów wizualizacji:** [publications-visualisation/src/README.MD](publications-visualisation/src/README.MD)

## Ograniczenia

- **Wieloskładnikowość:** projekt to trzy moduły uruchamiane osobno; pełny przepływ wymaga uruchomienia modułu 1 przed modułem 3.
- **Zasoby:** generowanie embeddingów i fine-tuning wymagają GPU; fine-tuning realizowany był na Google Colab (A100). Do weryfikacji dołączono gotowe artefakty.
- **Dane zewnętrzne:** zależność od dostępności ArXiv API, OpenAlex API i struktury Portalu Badawczego UAM (scraping może wymagać aktualizacji selektorów).
- **`old_data/`** to wcześniejszy prototyp asystenta (RAG na Azure) — pozostawiony jako archiwum, **nie jest częścią ocenianego rozwiązania**.
