# Wyniki fine-tuningu BGE-base-en-v1.5 dla podobieństwa artykułów ArXiv

## Konfiguracja

| Parametr | Wartość |
|----------|---------|
| Model bazowy | BAAI/bge-base-en-v1.5 (768 wymiarów) |
| Funkcja straty | CoSENTLoss (ciągłe wyniki podobieństwa 0.0–1.0) |
| Liczba artykułów (trening) | 18 880 |
| Epoki | 1 |
| Learning rate | 1e-5 |
| Batch size | 64 |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| GPU | Google Colab A100 |

## Co zmieniliśmy vs poprzednia wersja

1. **Hierarchiczna odległość kategorii** — zamiast binarnego "ta sama / inna kategoria", stosujemy 3-poziomową hierarchię:
   - Ta sama podkategoria (np. cs.AI vs cs.AI) → odległość 0.0
   - Ten sam archiwum, inna podkategoria (np. hep-ph vs hep-th) → 0.33
   - Ta sama grupa główna, inne archiwum (np. astro-ph.CO vs quant-ph) → 0.67
   - Inna grupa (np. cs.AI vs math.AG) → 1.0

2. **Rozmyte labele (multi-label)** — 63.8% artykułów ma >1 kategorię (średnio 2.06). Podobieństwo pary obliczamy jako: `0.5 * Jaccard overlap + 0.5 * średnie najlepsze dopasowanie hierarchiczne`

3. **CoSENTLoss zamiast MultipleNegativesRankingLoss** — przyjmuje ciągłe wyniki (float), uczy rankingu: jeśli `score(A,B) > score(C,D)`, to `sim(A,B)` powinno być > `sim(C,D)`. Poprzedni MNR powodował overfitting.

4. **Stratified sampling par treningowych** — na artykuł: 3 pary z tej samej podkategorii, 2 z tego samego archiwum, 2 z tej samej grupy, 1 z innej grupy

## Wyniki ewaluacji

### Metryki klastrowania (PCA + K-means)

| k | Metryka | Oryginalny | Fine-tuned | Poprawa |
|---|---------|-----------|------------|---------|
| 8 | Main category purity | 65.76% | 70.49% | **+4.74pp** |
| 8 | Subcategory purity | 9.52% | 10.09% | +0.57pp |
| 8 | Fuzzy purity | 74.18% | 80.86% | **+6.68pp** |
| 8 | Silhouette score | 0.381 | 0.438 | **+0.056** |
| 20 | Main category purity | 67.13% | 72.60% | **+5.47pp** |
| 20 | Subcategory purity | 11.43% | 12.93% | +1.50pp |
| 20 | Fuzzy purity | 75.13% | 80.00% | **+4.87pp** |
| 20 | Silhouette score | 0.329 | 0.349 | +0.020 |
| 118 | Main category purity | 68.07% | 73.76% | **+5.69pp** |
| 118 | Subcategory purity | 13.38% | 16.17% | **+2.79pp** |
| 118 | Fuzzy purity | 74.14% | 80.59% | **+6.46pp** |
| 118 | Silhouette score | 0.324 | 0.327 | +0.004 |

### Metryki wyszukiwania (Precision/Recall@k, multi-label)

Sąsiad jest "relevant" jeśli dzieli jakąkolwiek kategorię z zapytaniem. Obliczone na 2000 losowych zapytaniach.

| Metryka | Oryginalny | Fine-tuned | Poprawa |
|---------|-----------|------------|---------|
| P@1 | 0.740 | 0.758 | **+1.80pp** |
| P@3 | 0.707 | 0.730 | **+2.32pp** |
| P@5 | 0.684 | 0.713 | **+2.87pp** |
| P@10 | 0.654 | 0.690 | **+3.61pp** |
| P@20 | 0.625 | 0.667 | **+4.29pp** |
| R@1 | 0.0015 | 0.0015 | +0.00pp |
| R@3 | 0.0043 | 0.0044 | +0.01pp |
| R@5 | 0.0069 | 0.0071 | +0.02pp |
| R@10 | 0.0129 | 0.0136 | +0.06pp |
| R@20 | 0.0244 | 0.0260 | +0.16pp |

### Variance explained (PCA, 2 komponenty)

| Model | Variance explained |
|-------|-------------------|
| Oryginalny | 12.19% |
| Fine-tuned | 29.37% |

Fine-tuning sprawił, że embeddingi lepiej separują się w 2D — więcej informacji o strukturze kategorii jest uchwycone w pierwszych 2 komponentach PCA.

## Kluczowe wnioski

1. **Precision@k rośnie z k** — poprawa jest większa przy wyższych k (+1.8pp dla P@1, +4.3pp dla P@20), co oznacza, że model lepiej grupuje artykuły tematycznie w szerszym sąsiedztwie

2. **Recall jest niski w wartościach bezwzględnych** — to oczekiwane, bo przy 18 880 artykułach z szerokim nakładaniem kategorii, liczba "relevant" artykułów na zapytanie jest ogromna (tysiące), więc R@20 z natury będzie mały

3. **Fuzzy purity ~80%** — 4 na 5 artykułów w klastrze dzieli przynajmniej jedną kategorię z dominującą grupą klastra

4. **Brak overfittingu** — w przeciwieństwie do poprzedniego treningu z MNR (3 epoki, lr=2e-5), CoSENTLoss z 1 epoką i lr=1e-5 dawał stabilnie malejący validation loss

5. **Subcategory purity jest niska** — to oczekiwane przy 118 podkategoriach i 8/20 klastrach. Przy k=118 jest 16.2% (vs 13.4% dla oryginału, poprawa +2.8pp)

## Wizualizacja PCA — autorzy WMI UAM

### Dane

| Parametr | Wartość |
|----------|---------|
| Autorzy (łącznie) | 115 naukowców WMI UAM |
| Autorzy (>=5 prac, "reliable") | 106 |
| Autorzy (<5 prac, "low confidence") | 9 |
| Prace | 3 440 (tytuł + abstrakt) |
| Średnia prac/autor | 29.9 |
| Mediana prac/autor | 19.0 |
| Zakłady/pracownie | 19 jednostek organizacyjnych |

### Metoda

- Embedding per-paper: tytuł + abstrakt zakodowany modelem (768 wymiarów)
- **Mean pooling per autor** — embedding autora = średnia znormalizowanych embeddingów jego prac
- PCA do 2 komponentów, K-means (k=5) na reliable autorach

### Variance explained (PCA, 2 komponenty — autorzy WMI)

| Model | Variance explained |
|-------|-------------------|
| BGE-base (oryginalny) | 33.11% |
| BGE-base (CoSENT fine-tuned) | **54.71%** |

Fine-tuned model wyjaśnia ponad połowę wariancji w zaledwie 2 komponentach (+21.6pp vs oryginalny). Oznacza to, że po fine-tuningu autorzy o podobnej tematyce badań (np. z tego samego zakładu) są wyraźniej zgrupowani na wykresie 2D.

### Obserwacje

1. **Variance explained jest znacznie wyższy niż na artykułach ArXiv** (54.7% vs 29.4%) — to naturalne, bo mean pooling per autor wygładza szum pojedynczych prac i uwydatnia główne tematy badawcze
2. **Autorzy z <5 pracami** mają niestabilne embeddingi (za mało danych do sensownego uśrednienia) — na wykresie są domyślnie ukryci, z opcją włączenia checkboxem
3. **Zakłady/pracownie widoczne w hoverze** — pozwala szybko ocenić czy embedding dobrze separuje jednostki organizacyjne WMI
4. Największy zakład to **Zakład Sztucznej Inteligencji** (21 osób), potem **Szkoła Doktorska** (10), **Zakład Matematyki Dyskretnej** (9)

### Interaktywna wizualizacja

`PCA_GRAPH/pca_authors_finetuned.html` — Plotly HTML z:
- Side-by-side: oryginalny BGE vs CoSENT fine-tuned
- K-means klastry (5 kolorów)
- Wyszukiwarka naukowców z autocomplete
- Checkbox "Show authors with <5 papers"
- Hover: imię, zakład/pracownia, liczba prac, koordynaty PCA

## Pliki źródłowe

- Hierarchia kategorii: `ArXiv/scripts/utils/category_hierarchy.py`
- Przygotowanie danych: `ArXiv/scripts/05_prepare_finetune_data.py --mode scored`
- Trening: `ArXiv/scripts/06_finetune_bge.py --loss cosent`
- Ewaluacja: `ArXiv/scripts/07_evaluate_finetuned.py`
- Notebook Colab: `ArXiv/notebooks/finetune_bge_colab.ipynb`
- Model: `ArXiv/models/bge-base-cosent-finetuned/final/`
- Surowe wyniki JSON: `ArXiv/results/finetuned_comparison/comparison.json`
