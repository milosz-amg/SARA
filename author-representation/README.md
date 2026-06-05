# author-representation

Moduł 2 systemu **SARA**. Generuje embeddingi publikacji naukowców WMiI UAM,
dostraja model embeddingowy BGE-base na hierarchicznej taksonomii ArXiv
i buduje adaptacyjną reprezentację autora (single-point lub multi-cluster
w zależności od stabilności centroidu i struktury klastrowej publikacji).
Wynikiem są interaktywne mapy autorów (HTML) oraz metryki separacji
jednostek organizacyjnych WMiI.

## Szybki start

```bash
pip install -r requirements.txt
```

**Gotowe wizualizacje do otwarcia w przeglądarce:**

- [results/wmi_authors/pca_authors_finetuned.html](results/wmi_authors/pca_authors_finetuned.html) - single-point (model oryginalny vs dostrojony)
- [results/wmi_authors/pca_authors_multiclusters.html](results/wmi_authors/pca_authors_multiclusters.html) - multi-cluster + porównanie A vs B

**Regeneracja wizualizacji** (wymaga modelu dostrojonego, patrz
[models/bge-base-cosent-finetuned/README.md](models/bge-base-cosent-finetuned/README.md)):

```bash
python scripts/08_pca_authors_finetuned.py
python scripts/09_pca_authors_multiclusters.py
python scripts/10_evaluate_dept_separation.py
```

## Pipeline

Skrypty w `scripts/` uruchamiane są sekwencyjnie (numeracja 01-13):

| Etap | Opis | Skrypty |
|------|------|---------|
| **Dane ArXiv** | Pobranie 118 podkategorii z ArXiv API i utworzenie bazy SQLite | `01_fetch_arxiv_papers.py`, `02_create_database.py` |
| **Porównanie modeli embeddingowych** | Generowanie embeddingów dla 9 modeli i ocena klastrowania ArXiv | `03_generate_embeddings.py`, `04_pca_clustering.py` |
| **Fine-tuning BGE** | CoSENTLoss z hierarchiczną odległością kategorii ArXiv | `05_prepare_finetune_data.py`, `06_finetune_bge.py`, `07_evaluate_finetuned.py` |
| **Mapa autorów WMiI** | Mean pooling + automatyczna dekompozycja multi-cluster | `08_pca_authors_finetuned.py`, `09_pca_authors_multiclusters.py` |
| **Ewaluacja separacji jednostek** | Metryki intra/inter, NN@k, NMI, purity | `10_evaluate_dept_separation.py` |
| **Schemat adaptacyjny** | Diagnostyka stabilności i klasyfikacja SINGLE / MULTI / LOW_CONF / AMBIGUOUS | `11_compute_stability.py`, `12_adaptive_decision.py`, `13_evaluate_variants.py` |

## Struktura katalogu

```
author-representation/
├── README.md
├── requirements.txt
├── src/                   # kod importowalny (klient ArXiv, embedding utils, wizualizacje)
├── scripts/               # pipeline (uruchomienia numerowane 01-13)
├── configs/               # parametry (ścieżki, lista modeli, batch sizes)
├── notebooks/             # Colab (większe modele wymagają A100/T4)
├── data/                  # dane wejściowe i cache (patrz data/README.md)
├── models/                # artefakty modelu dostrojonego (poza repo, patrz README modułu)
├── results/               # wyniki eksperymentów
│   ├── model_comparison/
│   ├── finetuning/
│   ├── wmi_authors/
│   └── figures/
└── docs/                  # dokumentacja techniczna
```

## Wymagania

- Python 3.10+
- GPU (CUDA) dla inferencji embeddingów (16 GB VRAM na wszystkie modele)
- ok. 20 GB dysku (dane, embeddingi, model dostrojony)
- Fine-tuning BGE realizowany na Google Colab (A100/T4)

## Wyniki kluczowe

| Metryka | Wartość |
|---------|---------|
| Liczba autorów (z publikacjami) | 115 |
| Liczba publikacji | 3 440 |
| Jednostki organizacyjne (z jednoznaczną afiliacją) | 14 (92 autorów) |
| Autorzy z dekompozycją wieloklastrową (k > 1) | 98 (85,2 %) |
| Łączna liczba cluster-points | 298 (średnio 2,6 na autora) |
| Variance explained (PCA 2D, autorzy WMiI) | 54,71 % po dostrojeniu vs 33,11 % bez |
| NMI vs podział na jednostki | 0,702 |
| Main-category purity (k = 8, ArXiv) | 70,49 % (+4,74 pp po dostrojeniu) |
| Fuzzy purity (k = 8, ArXiv) | 80,86 % (+6,68 pp) |
| Precision@20 (ArXiv) | 0,667 (+4,29 pp) |
