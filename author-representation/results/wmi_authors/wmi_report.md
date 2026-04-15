# Ewaluacja separacji zakładów WMI w embeddingach

**Data:** 2026-02-19 17:55
**Model bazowy:** BAAI/bge-base-en-v1.5
**Model fine-tuned:** ArXiv/models/bge-base-cosent-finetuned/final
**Autorzy z konkretnym zakładem:** 92
**Zakłady/pracownie:** 14
**Prace (łącznie):** 3440

## Metryki globalne

| Metryka | Oryginalny | Fine-tuned | Diff |
|---------|-----------|------------|------|
| Mean intra-dept cos-sim | 0.8579 | 0.9665 | +0.1086 |
| Mean inter-dept cos-sim | 0.7559 | 0.9113 | +0.1555 |
| Intra/Inter ratio | 1.1350 | 1.0605 | -0.0745 |
| NN@1 dept accuracy | 0.7283 | 0.6630 | -0.0652 |
| NN@3 dept accuracy | 0.6522 | 0.5870 | -0.0652 |
| NN@5 dept accuracy | 0.6087 | 0.4783 | -0.1304 |
| Dept purity (K-means) | 0.6413 | 0.6196 | -0.0217 |
| NMI | 0.6719 | 0.7015 | +0.0296 |
| Silhouette (true depts) | 0.0643 | 0.0604 | -0.0039 |

### Objaśnienia metryk

- **Mean intra-dept cos-sim** — średnie cosine similarity między autorami z tego samego zakładu (wyższe = zakład bardziej spójny)
- **Mean inter-dept cos-sim** — średnie cosine similarity między autorami z różnych zakładów (niższe = lepsze rozdzielenie)
- **Intra/Inter ratio** — stosunek intra do inter (wyższy = lepsza separacja)
- **NN@k dept accuracy** — % autorów których większość z top-k najbliższych sąsiadów jest z tego samego zakładu
- **Dept purity** — K-means (k=liczba zakładów): % autorów w klastrze należących do dominującego zakładu
- **NMI** — Normalized Mutual Information między klastrami K-means a prawdziwymi zakładami (0=losowe, 1=idealne)
- **Silhouette** — jakość klastrów wg prawdziwych zakładów w pełnej 768D przestrzeni (-1 do 1)

## Breakdown per zakład

| Zakład | N | Intra (orig) | Intra (FT) | Diff | Najbliższy zakład (FT) | Near (orig) | Near (FT) |
|--------|---|-------------|-----------|------|----------------------|------------|----------|
| P. Algorytmiki | 4 | 0.9052 | 0.9814 | +0.0761 | Z. Teorii Algorytmów i Bezpieczeństwa Danych | 0.8736 | 0.9762 |
| P. Logiki i Filozofii Informatyki | 2 | 0.8205 | 0.9452 | +0.1248 | Z. Sztucznej Inteligencji | 0.9108 | 0.9764 |
| Z. Algebry i Teorii Liczb | 6 | 0.8706 | 0.9666 | +0.0960 | Z. Analizy Nieliniowej i Topologii Stosowanej | 0.9449 | 0.9861 |
| Z. Analizy Funkcjonalnej | 8 | 0.9155 | 0.9854 | +0.0699 | Z. Analizy Matematycznej | 0.9750 | 0.9964 |
| Z. Analizy Matematycznej | 5 | 0.8719 | 0.9712 | +0.0993 | Z. Analizy Funkcjonalnej | 0.9750 | 0.9964 |
| Z. Analizy Nieliniowej i Topologii Stosowanej | 6 | 0.9061 | 0.9816 | +0.0755 | Z. Analizy Matematycznej | 0.9731 | 0.9956 |
| Z. Arytmetycznej Geometrii Algebraicznej | 4 | 0.8603 | 0.9414 | +0.0811 | Z. Geometrii Algebraicznej i Diofantycznej | 0.9651 | 0.9930 |
| Z. Geometrii Algebraicznej i Diofantycznej | 5 | 0.8423 | 0.9530 | +0.1107 | Z. Arytmetycznej Geometrii Algebraicznej | 0.9651 | 0.9930 |
| Z. Matematyki Dyskretnej | 9 | 0.8794 | 0.9714 | +0.0920 | Z. Teorii Algorytmów i Bezpieczeństwa Danych | 0.9591 | 0.9845 |
| Z. Przestrzeni Funkcyjnych i Równań Różniczkowych | 5 | 0.9187 | 0.9894 | +0.0706 | Z. Teorii Operatorów | 0.9631 | 0.9958 |
| Z. Statystyki Matematycznej i Analizy Danych | 5 | 0.8680 | 0.9269 | +0.0589 | P. Logiki i Filozofii Informatyki | 0.9264 | 0.9679 |
| Z. Sztucznej Inteligencji | 21 | 0.8344 | 0.9625 | +0.1281 | P. Logiki i Filozofii Informatyki | 0.9264 | 0.9764 |
| Z. Teorii Algorytmów i Bezpieczeństwa Danych | 5 | 0.7713 | 0.9393 | +0.1680 | Z. Matematyki Dyskretnej | 0.9591 | 0.9845 |
| Z. Teorii Operatorów | 7 | 0.9316 | 0.9909 | +0.0593 | Z. Przestrzeni Funkcyjnych i Równań Różniczkowych | 0.9715 | 0.9958 |

### Objaśnienia kolumn

- **Intra** — średnie cosine similarity wewnątrz zakładu (spójność)
- **Najbliższy zakład** — zakład o najbliższym centroidzie (centroid-to-centroid cosine similarity)
- **Near** — cosine similarity do tego najbliższego zakładu (niższe = lepiej oddzielony)
