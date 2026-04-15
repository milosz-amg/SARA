# Author Representation — wizualizacja profili badawczych naukowców WMI UAM

Projekt w ramach systemu **SARA** (*Search and Research Assistant*). Dostarcza interaktywną mapę naukowców Wydziału Matematyki i Informatyki UAM, opartą na embeddingach tekstowych ich dorobku naukowego.

**Autor:** Jakub Paszke
**Wydział:** WMiI UAM

## Cel

Przekształcenie publikacji naukowych 115 pracowników WMI UAM (≈3440 artykułów) w dwuwymiarową mapę pozwalającą na:
- identyfikację pokrewieństwa badawczego między naukowcami,
- wykrycie interdyscyplinarności (multi-cluster view — jeden naukowiec jako *kilka* punktów),
- porównanie dwóch naukowców (A vs B) pod kątem profilu tematycznego,
- ewaluację, na ile podział na zakłady/pracownie odpowiada rzeczywistej bliskości tematycznej.

## Szybki start

```bash
pip install -r requirements.txt
```

**Wizualizacja gotowa** — otwórz w przeglądarce:
- [results/wmi_authors/pca_authors_finetuned.html](results/wmi_authors/pca_authors_finetuned.html) — single-point (orig vs fine-tuned)
- [results/wmi_authors/pca_authors_multiclusters.html](results/wmi_authors/pca_authors_multiclusters.html) — multi-cluster + porównanie A vs B

**Regeneracja wizualizacji** (wymaga modelu fine-tuned — patrz [models/README.md](models/bge-base-cosent-finetuned/README.md)):

```bash
python scripts/08_pca_authors_finetuned.py
python scripts/09_pca_authors_multiclusters.py
python scripts/10_evaluate_dept_separation.py
```

## Pipeline (fazy 1–6)

| Faza | Opis | Skrypty |
|------|------|---------|
| **1. Dane ArXiv** | Pobranie 118 kategorii × 200 prac (18 880 total) z ArXiv API | `01_fetch_arxiv_papers.py`, `02_create_database.py` |
| **2. Porównanie 9 modeli** | Test BGE, Qwen, Mistral, Specter, GTE, Nomic, MPNet na 2 397 paperach | `03_generate_embeddings.py`, `04_pca_clustering.py` |
| **3. Fine-tuning BGE** | CoSENTLoss + hierarchiczna odległość kategorii ArXiv (3 poziomy) | `05_prepare_finetune_data.py`, `06_finetune_bge.py`, `07_evaluate_finetuned.py` |
| **4. Wizualizacja WMI** | Mean pooling + multi-cluster (auto-k via silhouette) | `08_pca_authors_finetuned.py`, `09_pca_authors_multiclusters.py` |
| **5. Ewaluacja zakładów** | Separacja 14 zakładów WMI: intra/inter sim, NN@k, NMI | `10_evaluate_dept_separation.py` |
| **6. Dokument** | Opis postępów dla promotora (LaTeX + PDF) | [docs/opis_postepow_j_paszke.tex](docs/opis_postepow_j_paszke.tex) |

Zwycięzca fazy 2: **BAAI/bge-base-en-v1.5** (768-dim, 70.55% main-category purity).
Po fine-tuningu (faza 3): main purity 73.76% (+5.69 pp), fuzzy purity 80.59%, variance explained (2 PC) 54.71% na autorach WMI.

## Struktura

Zgodnie z zasadą **kod ≠ dane ≠ wyniki** (otwarta nauka, Żywica):

```
author-representation/
├── README.md              ← jesteś tutaj
├── requirements.txt       ← zależności Python
├── src/                   ← kod importowalny (utils, API client)
├── scripts/               ← pipeline (uruchomienia numerowane 01–10)
├── configs/               ← parametry (ścieżki, modele, batch sizes)
├── notebooks/             ← Colab (duże modele wymagają A100/T4)
├── data/                  ← dane wejściowe (lub instrukcje pobrania)
├── models/                ← artefakty modelu (poza repo, patrz README)
├── results/               ← wyniki eksperymentów
│   ├── model_comparison/     faza 2
│   ├── finetuning/           faza 3
│   ├── wmi_authors/          fazy 4–5
│   └── figures/              rysunki do dokumentu
└── docs/                  ← dokumentacja, decyzje, referencje
```

## Wymagania

- Python 3.10+
- GPU (CUDA) dla inferencji embeddingów — 16 GB VRAM na wszystkie modele
- Ok. 20 GB dysku (dane + embeddingi + model fine-tuned)
- Dla fazy 3 (fine-tuning na 18 880 paperach): **Google Colab A100** (pay-as-you-go, ~$12 wystarcza)

## Wyniki kluczowe

| Metryka | Wartość |
|---------|---------|
| Autorów WMI | 115 (z publikacjami) |
| Prac WMI | 3 440 |
| Zakładów/pracowni WMI | 14 (z jednoznaczną afiliacją: 92 autorów) |
| Multi-cluster: autorów z k>1 | 98 (85 %) |
| Multi-cluster: cluster-points | 298 (avg 2.6 na autora) |
| Variance explained (PCA 2D) | 54.71 % (fine-tuned) vs 33.11 % (oryginalny) |
| NMI fine-tuned vs zakłady | 0.702 (+0.030 po fine-tuningu) |
| Fine-tuning: purity (k=8) | 70.49 % (+4.74 pp); fuzzy 80.86 % (+6.68 pp) |
| Fine-tuning: P@20 | 0.667 (+4.29 pp) |

