# Publications Visualisation — porównanie metod redukcji wymiarowości

Moduł systemu **SARA**. Rzutuje embeddingi publikacji naukowców WMiI UAM do 2D wieloma metodami redukcji wymiarowości (DR), ocenia każdą metodę zestawem metryk jakości i wyłania zwycięzcę przez złożony scoring (Choquet / Sugeno / trimmed-mean).

## Cel

Odpowiedź na pytanie: *która metoda redukcji wymiarowości daje najwierniejszą 2D-mapę dorobku naukowego?* Pipeline:

1. dobiera optymalną liczbę klastrów `k` (6 metod złożonych w jeden wskaźnik),
2. rzutuje embeddingi 8 metodami DR (UMAP, t-SNE, PaCMAP, Isomap, Spectral, PCA, PCA-8D, LDA) i liczy 7 metryk jakości,
3. sprawdza odporność t-SNE/UMAP/PaCMAP na losowość (wiele ziaren + testy Wilcoxona),
4. agreguje metryki całką Choqueta i porównuje z wariantami Sugeno / trimmed-mean.

## Wymagania

- Python 3.10+
- Pakiety z `requirements.txt` (numpy, pandas, scipy, scikit-learn, umap-learn, pacmap)
- Dane wejściowe z modułu [wmii-data-collection](../wmii-data-collection/) — `embeddings.npy` + `embeddings_metadata.csv`
- CPU wystarcza; pełny przebieg z t-SNE i eksperymentem odporności to rząd kilkunastu minut

## Instalacja

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uruchomienie demonstracji

```bash
./run_visualisation.sh                 # pełny pipeline (auto k, wszystkie metody)
./run_visualisation.sh --no-tsne       # szybciej (pomija t-SNE i eksperyment odporności)
./run_visualisation.sh --k 9           # pomija dobór k, używa k=9
```

Skrypt wymaga `embeddings.npy` i `embeddings_metadata.csv` w `../wmii-data-collection/data/`. Uruchamianie skryptów pojedynczo opisuje [src/README.MD](src/README.MD).

## Oczekiwany wynik

Po przebiegu w `output/` powstają:

| Plik | Zawartość |
|------|-----------|
| `k_metrics.json` | metryki doboru `k` + konsensus |
| `vis_methods.json` | współrzędne 2D i metryki jakości dla 8 metod DR |
| `robustness_results.json` | wyniki eksperymentu odporności (mean ± std, win counts, Wilcoxon) |
| `vis_methods.html` | interaktywna przeglądarka — otwórz w przeglądarce |

## Reprodukcja / weryfikacja wyników

Gotowe artefakty leżą w [output/](output/) — można je obejrzeć bez ponownego liczenia. Scoringi działają na zapisanym `vis_methods.json`:

```bash
python src/choquet_composite.py    --input output/vis_methods.json   # rekomendowany scorer
python src/sugeno_composite.py     --input output/vis_methods.json
python src/trimmed_composite.py    --input output/vis_methods.json
python src/metric_correlations.py  --input output/vis_methods.json
```

## Testy

Moduł nie ma testów jednostkowych — weryfikacja jest empiryczna: eksperyment odporności (`robustness_experiment.py`) z testami istotności Wilcoxona oraz porównanie trzech niezależnych scorerów (Choquet / Sugeno / trimmed-mean), które powinny dawać spójny ranking.

## Ograniczenia

- Wejście pochodzi z modułu `wmii-data-collection` — bez `embeddings.npy` pipeline nie ruszy.
- t-SNE i eksperyment odporności są kosztowne; do szybkiej iteracji użyj `--no-tsne`.
- `vis_methods.html` to statyczna przeglądarka wczytująca `vis_methods.json` — oba pliki muszą być w tym samym katalogu.
