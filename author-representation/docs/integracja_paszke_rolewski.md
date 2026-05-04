# Analiza integracji: Paszke ↔ Rolewski

Dokument roboczy — jak pogodzić moje wyniki (wizualizacja autorów, fine-tuning BGE, multi-cluster) z wynikami Rolewskiego (stabilność centroidu, porównanie modeli, diagnostyka profili) w jednej pracy magisterskiej wg planu `plan_pracy_rolewski_paszke.md`.

---

## 1. Co każdy z nas zrobił — mapowanie wkładów

### Rolewski — warstwa **diagnostyczna** (Rozdział 2 planu)

| Obszar | Co zbadał | Artefakty |
|--------|-----------|-----------|
| **6 modeli embeddingowych** | SPECTER, MiniLM-L6/L12, MPNet-Base, BGE-Small, Multilingual-MiniLM — porównane na tytułach, abstraktach, N-gramach | `embedding_comparison_experiment.ipynb` |
| **Projekcja JL** | Redukcja 768D→384D przez macierz Johnsona–Lindenstraussa (zachowuje odległości) | `concatenate_embedings_experiment.ipynb` |
| **Konkatenacja modeli** | Profile autorów jako konkatenacja embeddingów z kilku modeli naraz | `concatenate_embedings_experiment.ipynb` |
| **Test współautorów** | Hipoteza: współautorzy mają bliższe centroidy niż pary losowe → **silna separacja dla 2 modeli bazowych** | praca mgr, s. ? |
| **Zgodność Jaccarda** | Jak stabilne są top-k sąsiadów autora między różnymi konfiguracjami modeli? | `scientist_profile_comparison.ipynb` |
| **Stabilność centroidu** | sim(centroid_k, centroid_all) dla k=3, 5, 10, 20, 50, 100 na top-N naukowcach → **przy ilu pracach centroid jest wiarygodny** | `pubs_num_stability.ipynb` |
| **Heatmapy naukowców** | Wizualizacja wewnętrznej spójności dorobku pojedynczych osób | `scientist_heatmaps.ipynb` |
| **Ulepszona wizualizacja publikacji** | Choquet integral + fuzzy measures do agregowania 7 metryk jakości redukcji wymiarowości (ulepszenie Wujca) | `publications-visualisation/` |
| **Pipeline danych WMI** | Pełny scraper: Portal Badawczy UAM → OpenAlex API → abstrakt scraper (Selenium, ScienceDirect/Springer/Wiley/MDPI/IEEE) | `wmii-data-collection/` |

**Kluczowa teza Rolewskiego:** pojedynczy centroid per autor jest wiarygodny tylko dla części populacji. Zmienia się to dramatycznie w zależności od liczby prac, interdyscyplinarności, wybranego modelu. Potrzebna jest **warstwa diagnostyczna**, która dla każdego autora mówi: "ta reprezentacja jest stabilna" vs "nie ufaj temu centroidowi".

### Paszke — warstwa **konstrukcyjna** (Rozdział 3 planu)

| Obszar | Co zrobiłem | Artefakty |
|--------|-------------|-----------|
| **9 modeli embeddingowych** | BGE-base/large, Qwen 0.6B/4B, Mistral-7B, GTE-large, Nomic, Specter, MPNet — na 2 397 paperach ArXiv | `results/model_comparison/` |
| **Fine-tuning BGE** | CoSENTLoss + hierarchiczna odległość kategorii ArXiv (3 poziomy) + rozmyte multi-label (63.8% paperów ma >1 kategorię) | `scripts/06_finetune_bge.py` |
| **Ewaluacja fine-tuned** | +5.69 pp purity, +6.68 pp fuzzy purity, +4.29 pp P@20, +17.18 pp variance (12.2%→29.4% na 2D PCA) | `results/finetuning/wyniki.md` |
| **Single-point na autorach WMI** | Mean pooling 3440 paperów → 115 centroidów → PCA 2D (variance 54.7%) | `scripts/08_pca_authors_finetuned.py` |
| **Multi-cluster** | Auto-k via silhouette (threshold 0.15), K-means per autor → 298 cluster-points, 98 autorów z k>1 (85%) | `scripts/09_pca_authors_multiclusters.py` |
| **Ewaluacja zakładów WMI** | NMI 0.702, NN@1 0.663, per-dept breakdown 14 zakładów (bliźniacze: Analiza Funkcjonalna ↔ Matematyczna 0.996) | `scripts/10_evaluate_dept_separation.py`, `results/wmi_authors/wmi_report.md` |
| **Interaktywne wizualizacje** | Plotly HTML z wyszukiwarką autocomplete, porównaniem A vs B (żółty/cyan), toggle cluster links | `results/wmi_authors/*.html` |

**Kluczowa teza Paszkego:** gdy autor ma odrębne kierunki badawcze, pojedynczy punkt na mapie trafia w pustą przestrzeń "pomiędzy" tymi kierunkami. Rozwiązanie: reprezentuj autora jako wiele punktów (po jednym na klaster tematyczny), połączonych liniami. 85% autorów WMI ma ≥2 sensowne klastry w fine-tuned przestrzeni.

---

## 2. Jak te prace się ze sobą łączą

Plan Rolewskiego (5 rozdziałów) definiuje jasny **pipeline systemowy**:

```
        ┌─────────────────────────────────────────────────────────────────┐
        │                    WSPÓLNY INPUT (Rozdział 1)                    │
        │      3440 paperów × 115 autorów WMI + metadane zakładów          │
        └─────────────────────────────────────────────────────────────────┘
                                       │
                          ┌────────────┼────────────┐
                          │            │            │
                          ▼            ▼            ▼
               ┌──────────────┐ ┌────────────┐ ┌──────────────┐
               │ Embeddingi   │ │ Embeddingi │ │  Embeddingi  │
               │ per praca    │ │ per praca  │ │   per praca  │
               │ (6 modeli    │ │ (9 modeli  │ │  (fine-tuned │
               │  bazowych)   │ │  ArXiv)    │ │   CoSENT)    │
               └──────┬───────┘ └──────┬─────┘ └──────┬───────┘
                      │                │              │
                      │     ROLEWSKI   │              │    PASZKE
                      │    (Rozdz. 2)  │              │  (Rozdz. 3)
                      ▼                ▼              ▼
        ┌──────────────────────────────────────┐  ┌─────────────────────┐
        │  WARSTWA DIAGNOSTYCZNA               │  │ WARSTWA             │
        │                                      │  │ KONSTRUKCYJNA       │
        │  Dla każdego autora oblicza cechy:   │  │                     │
        │  • liczba publikacji                 │  │ Wariant SINGLE:     │
        │  • sim(centroid_k, centroid_all)     │  │   mean pooling      │
        │  • średni Jaccard top-k sąsiadów     │  │   → 1 punkt         │
        │  • rozproszenie (var, std)           │  │                     │
        │  • sygnały interdyscyplinarności     │  │ Wariant MULTI:      │
        │                                      │  │   auto-k silhouette │
        │  Decyzja: SINGLE / MULTI /           │  │   → N punktów       │
        │           LOW_CONF / AMBIGUOUS       │  │                     │
        └──────────────────┬───────────────────┘  └──────────┬──────────┘
                           │                                 │
                           │                                 │
                           ▼                                 ▼
                        ┌──────────────────────────────────────┐
                        │ INTEGRACJA (Rozdz. 4)                │
                        │                                      │
                        │ author_representation_policy.csv:    │
                        │   orcid, n_papers, stability_score,  │
                        │   jaccard, dispersion, decision      │
                        │                                      │
                        │ → zasila wariant Paszkego:           │
                        │   SINGLE/LOW_CONF → 1 punkt          │
                        │   MULTI → uruchom multi-cluster      │
                        │   AMBIGUOUS → wariant zachowawczy    │
                        └──────────────────┬───────────────────┘
                                           ▼
                        ┌──────────────────────────────────────┐
                        │  FINALNA WIZUALIZACJA (Rozdz. 4)     │
                        │  • adaptive representation map       │
                        │  • explainability layer              │
                        │  • integracja z mapą artykułów       │
                        └──────────────────────────────────────┘
```

**To jest sens planu Rolewskiego** i jest bardzo elegancki. Moja dotychczasowa praca robi multi-cluster dla **wszystkich** autorów (85% dostaje k>1 przy threshold 0.15). Rolewski sugeruje: **nie rób tego dla wszystkich — rób tylko dla tych, którzy tego potrzebują na podstawie diagnostyki**.

---

## 3. Punkty zgodności — gdzie nasze prace potwierdzają się nawzajem

### 3.1 Próg MIN_PAPERS

- **Rolewski (eksperyment 3, `pubs_num_stability.ipynb`)**: przy k=5 prac sim(centroid_5, centroid_all) osiąga ~0.95 dla większości modeli; przy k=10 już ~0.99.
- **Paszke (`MIN_PAPERS_FOR_CLUSTERING=5`)**: autorzy z <5 pracami traktowani jako "low confidence", nie są dzieleni na klastry.
- **Konkluzja wspólna:** próg **5 prac** jest empirycznie uzasadnionym minimum do stabilnej reprezentacji centroidowej.

### 3.2 BGE jako model wybrany do produkcji

- **Rolewski** testował BGE-**Small** (384-dim) na WMI — solidny wynik.
- **Paszke** testował BGE-**base** (768-dim) na ArXiv — **zwycięzca spośród 9 modeli** (70.55% main purity k=118).
- **Paszke** dofinetiunował BGE-base na hierarchii ArXiv — poprawa +5.69 pp purity.
- **Konkluzja wspólna:** rodzina BGE jest optymalnym wyborem dla zadania embedowania paperów naukowych (768-dim, szybki, konsekwentnie najlepszy).

### 3.3 Interdyscyplinarność ≠ liczba prac

- **Rolewski** (cyt. z planu, Rozdz. 2): *"sama liczba prac nie jest wystarczająca do oceny jakości profilu — równie ważna okazuje się interdyscyplinarność autora"*.
- **Paszke** (multi-cluster stats): autorzy z wieloma klastrami mają bardzo różną liczbę prac — np. Żywica 26 prac → 4 sensowne klastry (decision making, robotics, fuzzy optimization, ovarian cancer).
- **Konkluzja wspólna:** potrzebna jest **metryka interdyscyplinarności niezależna od liczby prac** — u Paszkego to wartość silhouette przy k>1; u Rolewskiego to rozproszenie embeddingów lub Jaccard między modelami.

### 3.4 Pojedynczy centroid jest niewystarczający dla większości

- **Rolewski** (wniosek z Rozdz. 5): *"pojedynczy centroid dobrze działa dla autorów o stabilnym i jednorodnym profilu"* → implikacja: dla niestabilnych/niejednorodnych NIE.
- **Paszke**: 85% autorów z ≥5 pracami otrzymuje k>1 przy threshold 0.15 — mają odrębne klastry tematyczne.
- **Konkluzja wspólna:** reprezentacja adaptacyjna (SINGLE vs MULTI) jest uzasadniona nie tylko koncepcyjnie ale i empirycznie.

---

## 4. Punkty niezgodności / do dyskusji

### 4.1 Różna przestrzeń embeddingowa

Rolewski testował **modele bazowe** (bez fine-tuningu) na natywnej przestrzeni każdego (z JL redukcją do 384D). Ja robiłem **fine-tuned BGE-base 768D** — jedną przestrzeń dla wszystkich eksperymentów WMI.

**Problem:** Jego diagnostyka (stabilność, Jaccard) była obliczana na 6 modelach bazowych, nie na mojej fine-tuned przestrzeni. Czy jego konkluzje dalej stoją po fine-tuningu?

**Propozycja integracji:**
1. Uruchomić jego notebooki (`pubs_num_stability`, `scientist_profile_comparison`) na **fine-tuned BGE-base** jako dodatkowym modelu — zobaczyć czy stabilność jest lepsza/gorsza.
2. Jeśli lepsza: **fine-tuning nie tylko poprawia klastrowanie (moja teza), ale także stabilność centroidów (teza wspólna)**.
3. Jeśli taka sama: warstwa diagnostyczna jest ortogonalna do wyboru modelu — dobrze dla uniwersalności.

### 4.2 Threshold silhouette 0.15 — arbitralny?

Obecnie mam twardy próg 0.15 — autor dostaje k>1 jeśli najlepszy silhouette > 0.15. To **nie** jest zdiagnozowane u Rolewskiego. On raczej sugeruje:

> *"Rozbicie autora na kilka punktów powinno być sterowane diagnozą stabilności, a nie wyłącznie lokalnym kryterium klasteryzacyjnym."*

To oznacza, że sama wysoka silhouette **nie wystarcza** — jeśli autor ma zaledwie 7 prac rozrzuconych tematycznie, silhouette może być wysokie, ale cała jego reprezentacja jest chybotliwa. Powinien dostać `LOW_CONF`, nie `MULTI`.

**Propozycja integracji:** połączyć dwa warunki:
- z Rolewskiego: `stability_score > 0.95` (centroid jest sensowny)
- ode mnie: `silhouette_k > 0.15` (jest odrębność tematyczna)

Decyzja finalna:

| `stability` | `silhouette_k>1` | Decyzja |
|-------------|------------------|---------|
| Wysoka (>0.99) | Wysoka (>0.25) | **MULTI** — stabilny i wyraźnie interdyscyplinarny |
| Wysoka (>0.99) | Niska (<0.15)  | **SINGLE** — stabilny i jednorodny |
| Średnia (0.90–0.99) | Wysoka | **MULTI z ostrzeżeniem** |
| Niska (<0.90) | dowolna | **LOW_CONF** — nawet mean pooling niestabilny |
| `n < 5`     | –                | **LOW_CONF** (brak danych) |

### 4.3 Rolewski sugeruje kategorię `AMBIGUOUS` — co to znaczy w praktyce?

W planie (s. 104): *"AMBIGUOUS — przypadek graniczny wymagający dalszej analizy"*. U mnie obecnie **nic takiego nie istnieje**. Mogłoby to oznaczać autorów, którzy:
- mają 3–5 klastrów ale wszystkie blisko siebie,
- mają wysoką silhouette dla k=2 ale równie wysoką dla k=4,
- są na granicy progu stabilności.

Wizualizacyjnie: w UI można ich wyświetlać jako **kontur z gradientem** zamiast N ostrych punktów — wskazując niepewność. To jest rozszerzenie moich obecnych wizualizacji.

---

## 5. Co muszę zrobić, żeby zgrać się z planem Rolewskiego

Idąc po `[PLACEHOLDER J. Paszke]` w `plan_pracy_rolewski_paszke.md`:

### [Rozdz. 1] "Opis końcowego modelu embeddingowego + uzasadnienie fine-tuned jako docelowej przestrzeni"

Krótki akapit (już mam treść w `opis_postepow_j_paszke.md` sekcje 2.2, 2.3):
> Docelową przestrzenią embeddingową dla finalnej reprezentacji autora jest model **BAAI/bge-base-en-v1.5** fine-tuned na 18 880 artykułach ArXiv z wykorzystaniem funkcji straty **CoSENT** i hierarchicznej odległości kategorii (3 poziomy: podkategoria → archiwum → grupa). Wybór tej przestrzeni jest uzasadniony trzema argumentami: (1) BGE-base zwyciężył porównanie 9 modeli na zbiorze ArXiv (main-category purity 70.55% przed fine-tuningiem); (2) fine-tuning poprawił variance explained w PCA 2D z 12.2% do 29.4% na ArXiv i z 33.1% do 54.7% na autorach WMI — oznacza to że ponad połowa wariancji semantycznej koduje się w dwóch pierwszych komponentach, co czyni mapy 2D wiarygodnymi; (3) hierarchia ArXiv odpowiada naturalnej strukturze dziedzin nauki, której dotyczy również WMI (matematyka, informatyka, statystyka), więc przestrzeń jest domain-appropriate mimo że treningowa.

### [Rozdz. 3] "Wariant jednopunktowy (mean pooling)"

→ Przeniesienie mojej sekcji 3.1 z `opis_postepow_j_paszke.md` (już gotowe, ~50 linii).

### [Rozdz. 3] "Algorytm multi-cluster"

→ Przeniesienie moich sekcji 3.2, 4.1–4.4 (definicja silhouette, pseudokod, edge cases, threshold experiment z tabelą progów 0.05–0.30).

### [Rozdz. 3] "Etykietowanie klastrów + wizualizacja"

→ Moja sekcja 3.2 (`get_cluster_label()`) + sekcja 8 (interaktywna wizualizacja).

### [Rozdz. 3] "Fine-tuning"

→ Moja sekcja 2.3 i cała 7 (pełna ewaluacja na ArXiv).

### [Rozdz. 3] "Ewaluacja jednopunktowego vs wielopunktowego względem zakładów"

→ Moja sekcja 6 + mogę dodać **bezpośrednie porównanie**: NMI dla mean pooling vs NMI dla multi-cluster (czy adaptacyjna polityka daje lepsze dopasowanie do zakładów niż sztywna reguła?). To jest **nowy eksperyment** — jeszcze go nie zrobiłem.

### [Rozdz. 4] "Liczby dotyczące finalnej reprezentacji w fine-tuned"

→ Już mam: 115 autorów, 298 cluster-points, 98 multi-cluster, avg 2.6 na autora, 14 zakładów, NMI 0.702.

### [Rozdz. 4 wspólny] "Progi przejścia SINGLE → MULTI"

→ To trzeba zrobić wspólnie z Rolewskim. Wymaga:
1. Uruchomienia jego notebooków na mojej fine-tuned przestrzeni (dostarczyć mu kompatybilne embeddingi).
2. Zestawienia moich silhouette/k z jego stability/jaccard.
3. Eksperymentu adaptacyjnego: porównanie baseline (wszyscy SINGLE) vs aggressive (wszyscy MULTI) vs adaptive (decyzja per autor).
4. Metryki oceny: jakość mapy (NMI), interpretowalność (% klastrów z sensownymi topics), zadowolenie użytkownika (test ekspercki? Rozdz. 5 "dalsze kierunki").

### [Rozdz. 5] "Końcowe wnioski dot. multi-cluster"

→ Mogę napisać. Kluczowe tezy:
- fine-tuning + multi-cluster ujawnia wielowymiarowość 85% autorów WMI, co byłoby niemożliwe przy mean poolingu,
- ale bez warstwy diagnostycznej Rolewskiego nie wiemy, czy te dodatkowe klastry są wiarygodne, czy artefaktem,
- **dopiero pipeline integracyjny daje pełny obraz**,
- praca adaptacyjna (SINGLE dla stabilnych, MULTI dla niestabilnych + interdyscyplinarnych) jest **praktycznie** lepsza od każdego z podejść osobno.

---

## 6. Techniczne TODO przed napisaniem rozdziałów

1. **[KRYTYCZNE]** Uruchomić `pubs_num_stability.ipynb` z fine-tuned BGE-base dodanym jako 7. model. Zobaczyć czy krzywa stabilności jest wyższa/niższa.
   - Sprawdzenie techniczne: jego notebook zakłada `data/embeddings/{type}/{model}/{openalex_id}_{type}.pkl` — mój cache jest pojedynczy plik `paper_embeddings_cosent.npy`. Trzeba napisać adapter / zregenerować embeddingi w jego formacie.

2. **[DUŻE]** Wygenerować `author_representation_policy.csv` (wspólny artefakt pośredni z Rozdz. 4):
   - input: wszyscy 115 autorów
   - kolumny: orcid, n_papers, stability@5, stability@10, avg_jaccard, dispersion, silhouette_best_k, best_k, decision (SINGLE/MULTI/LOW_CONF/AMBIGUOUS)
   - reguły decyzyjne: Tabela z sekcji 4.2 tego dokumentu

3. **[ŚREDNIE]** Eksperyment adaptacyjny vs baseline:
   - Baseline: wszyscy SINGLE → NMI, intra/inter sim
   - Aggressive: wszyscy MULTI (obecny stan mój) → NMI, intra/inter sim
   - Adaptive: wg policy.csv → NMI, intra/inter sim
   - Hipoteza: Adaptive > Aggressive > Baseline

4. **[MAŁE]** Zaktualizować `opis_postepow_j_paszke.md` / `.tex` o nawiązania do Rolewskiego i planu 5-rozdziałowego (obecnie dokument traktuje moją pracę jako samodzielną).

5. **[MAŁE]** Przygotować interaktywną wizualizację z kategorią `AMBIGUOUS` (kontur gradient zamiast N punktów).

---

## 7. Pytania otwarte do Rolewskiego

Przed dalszą integracją warto ustalić:

1. **Format embeddingów**: czy przekażesz mi swoje `.pkl` per-work dla 6 modeli, czy mam zregenerować? Prościej byłoby gdybyś udostępnił.
2. **Stability score — definicja**: średnia sim(centroid_k, centroid_all) dla jakich wartości k? 3? 5? 10? Wszystkich naraz z jakimiś wagami?
3. **Jaccard — dla jakich par modeli**: wszystkie vs wszystkie (C(6,2)=15 par) czy wybrane reprezentatywne?
4. **Testy eksperckie**: czy planujesz ewaluację manualną z promotorem/kolegami, żeby skalibrować progi SINGLE→MULTI? Bo nie wiem jaki próg stability jest "dobry" bez takiego feedbacku.
5. **Czy stability Twoich modeli bazowych vs mojego fine-tuned BGE będzie różna**: hipoteza — fine-tuning zmniejsza dispersion, więc stability powinna rosnąć. Chcesz wspólnie to zweryfikować?
6. **Kto pisze Rozdział 1 i 5 (wspólne)**: podział 50/50 każdy po jednej sekcji? Czy wspólnie iteracyjnie?

---

## 8. Podsumowanie — praca jest spójna

Plan Rolewskiego jest **bardzo dobry** — nasze prace zamiast być dwiema niezależnymi liniami badań stają się kolejnymi etapami jednego pipeline'u. On diagnozuje, ja konstruuję, razem produkujemy **adaptacyjną reprezentację naukowca**.

Moja dotychczasowa praca (multi-cluster dla wszystkich z ≥5 prac) odpowiada **wariantowi aggressive** z Rozdz. 4. Dodanie warstwy Rolewskiego daje **wariant adaptive**, który powinien być lepszy dla UX systemu SARA.

Największa rzecz do zrobienia: uruchomienie jego eksperymentów na mojej fine-tuned przestrzeni, żeby zobaczyć czy fine-tuning zmienia charakterystykę stabilności. Jeśli tak — mamy **trzeci wkład wspólny**: *"fine-tuning poprawia nie tylko klastrowanie, ale i stabilność centroidów"*.
