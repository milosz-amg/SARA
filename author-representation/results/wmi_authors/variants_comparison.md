# Porównanie wariantów reprezentacji autorów na separacji zakładów

**Data:** 2026-05-04 12:12

**Zbiór ewaluacyjny:** 92 autorów z jednoznacznym przypisaniem do 14 zakładów/pracowni WMI (afiliacja zaczynająca się od 'Zakład' lub 'Pracownia').

**Embeddingi:** BGE-base-en-v1.5 fine-tuned (CoSENTLoss + hierarchiczna odległość ArXiv).

## Definicje wariantów

- **baseline** — 1 centroid (mean pooling) per autor (115 punktów dla pełnego zbioru).
- **aggressive** — multi-cluster (k>1) dla *każdego* autora z ≥5 pracami i silhouette_max ≥ 0,15; pozostali autorzy: 1 centroid.
- **adaptive** — decyzja per autor wg `policy.csv` (SINGLE/MULTI/LOW_CONF/AMBIGUOUS) opartej na sygnale silhouette (struktura klastrowa) i stab(5) (stabilność profilu, Rolewski 2024 sec. 1.4.3).

## Wyniki

| Wariant | Punkty | NMI | Purity | NN@1 | NN@3 | NN@5 | Silhouette |
|---------|-------:|----:|-------:|-----:|-----:|-----:|-----------:|
| baseline | 92 | 0.7015 | 0.6196 | 0.6630 | 0.5870 | 0.4783 | 0.0604 |
| aggressive | 259 | 0.5411 | 0.5830 | 0.5907 | 0.5444 | 0.5483 | -0.0012 |
| adaptive | 242 | 0.5038 | 0.5661 | 0.5992 | 0.5413 | 0.5413 | -0.0268 |

## Rozkład decyzji w wariancie adaptive

| Decyzja | Autorów | % | Punktów |
|---------|--------:|--:|--------:|
| SINGLE | 7 | 6.1 | 7 |
| MULTI | 98 | 85.2 | 281 |
| LOW_CONF | 9 | 7.8 | 9 |
| AMBIGUOUS | 1 | 0.9 | 1 |
| **Razem** | **115** | **100,0** | **298** |

### Objaśnienia metryk

- **NMI** — Normalized Mutual Information między klastrami K-means (k=14) a prawdziwymi zakładami (0=losowe, 1=idealne).
- **Purity** — odsetek autorów w klastrach należących do dominującego zakładu.
- **NN@k** — odsetek punktów, których większość z k najbliższych sąsiadów (z wykluczeniem punktów tego samego autora) należy do tego samego zakładu.
- **Silhouette** — jakość separacji zakładów w pełnej przestrzeni 768D (prawdziwe etykiety zakładów).
