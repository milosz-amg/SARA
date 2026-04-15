# Wizualizacja profili badawczych naukowców WMI UAM za pomocą embeddingów tekstowych

## Embedding-based Visualisation of Researcher Profiles at WMI UAM

**Autor:** Jakub Paszke

**Wydział Matematyki i Informatyki, Uniwersytet im. Adama Mickiewicza w Poznaniu**

**Data:** marzec 2026

---

## Spis treści

1. [Wstęp i motywacja](#1-wstęp-i-motywacja)
2. [Pipeline danych](#2-pipeline-danych)
3. [Metody reprezentacji autorów](#3-metody-reprezentacji-autorów)
4. [Automatyczny dobór liczby klastrów](#4-automatyczny-dobór-liczby-klastrów)
5. [Redukcja wymiarowości: PCA](#5-redukcja-wymiarowości-pca)
6. [Ewaluacja separacji zakładów](#6-ewaluacja-separacji-zakładów)
7. [Ewaluacja fine-tuningu na zbiorze ArXiv](#7-ewaluacja-fine-tuningu-na-zbiorze-arxiv)
8. [Interaktywna wizualizacja](#8-interaktywna-wizualizacja)
9. [Wyniki i obserwacje](#9-wyniki-i-obserwacje)
10. [Wnioski](#10-wnioski)
11. [Bibliografia](#11-bibliografia)
12. [Dodatek](#12-dodatek)

---

## 1. Wstęp i motywacja

### 1.1 Kontekst projektu SARA

Niniejsza praca powstała w ramach projektu **SARA** (*Search and Research Assistant*) — systemu rekomendacji i eksploracji dorobku naukowego realizowanego na Wydziale Matematyki i Informatyki UAM. Celem projektu SARA jest zbudowanie narzędzia umożliwiającego wyszukiwanie artykułów naukowych, identyfikację potencjalnych współpracowników oraz eksplorację struktury badawczej wydziału. Jednym z kluczowych komponentów systemu jest interaktywna mapa naukowców, która wizualizuje profile badawcze pracowników WMI w przestrzeni dwuwymiarowej.

Wizualizacja autorów pełni w systemie SARA konkretną rolę: nie jest jedynie „efektownym wykresem", lecz narzędziem wspierającym decyzje. Nowy doktorant szukający promotora może na jej podstawie zidentyfikować badaczy o pokrewnych zainteresowaniach. Zespół grantowy może wykryć, że dwóch naukowców z różnych zakładów pracuje nad zbliżoną tematyką i mógłby efektywnie współpracować. Administracja wydziału może ocenić, na ile struktura organizacyjna (podział na zakłady) odzwierciedla rzeczywiste pokrewieństwo badawcze.

### 1.2 Porównanie z podejściem opartym na wizualizacji artykułów

Równolegle M. Wujec opracował komplementarny moduł systemu SARA — interaktywną wizualizację **artykułów** naukowych WMI. W jego podejściu każdy z 3440 artykułów reprezentowany jest jako punkt na mapie 2D, a użytkownik eksploruje tematyczne sąsiedztwo poszczególnych publikacji. Wujec przeprowadził obszerną analizę porównawczą ośmiu metod redukcji wymiarowości (PCA, t-SNE, UMAP, PaCMAP, Isomap, Spectral Embedding, LDA, PCA 8D+log) z wykorzystaniem siedmiu metryk jakości projekcji.

Niniejsza praca stawia inne pytanie badawcze: zamiast wizualizować poszczególne artykuły, wizualizuje **autorów** — agregując dorobek każdego naukowca w pojedynczy punkt (lub kilka punktów, w wariancie multi-cluster). Różnica w skali jest istotna: 3440 punktów/artykułów u Wujca wobec 115 punktów/autorów w niniejszej pracy. Ta zmiana skali prowadzi do odmiennych wyzwań:

- **Agregacja informacji:** Każdy punkt na mapie autorów reprezentuje średnio ~30 prac, co wymaga strategii uśredniania embeddingów (mean pooling).
- **Mniejszy zbiór:** Przy 115 punktach problem nakładania się jest mniejszy, ale za to każdy punkt jest „cięższy" — utrata informacji przy złym uśrednieniu jest bardziej dotkliwa.
- **Interpretowalność:** Na mapie artykułów użytkownik widzi tytuły i abstrakty; na mapie autorów widzi nazwiska i zakłady, co pozwala na bezpośrednie wnioski organizacyjne.
- **Wariancja danych:** Uśrednianie per autor wygładza szum pojedynczych prac, co skutkuje wyższym variance explained w PCA (54.7% na autorach vs 12.2–29.4% na artykułach — szczegóły w sekcji 5).

### 1.3 Zakres pracy

Praca obejmuje pełny pipeline — od scraping'u danych o pracownikach WMI, przez pobieranie ich dorobku naukowego z OpenAlex, generowanie embeddingów tekstowych, fine-tuning modelu embeddingowego, po końcową wizualizację interaktywną. Dokument opisuje dwa warianty wizualizacji: **mean pooling** (jeden punkt per autor) oraz **multi-cluster** (wiele punktów per autor, odpowiadających odrębnym kierunkom badawczym).

---

## 2. Pipeline danych

### 2.1 Źródła danych

Pipeline danych składa się z dwóch etapów zbierania informacji: scrapingu danych kadrowych WMI oraz pobierania dorobku naukowego z otwartych baz bibliometrycznych.

#### Scraper WMI (`collect_uam_data/`)

Pierwszy etap polega na automatycznym zebraniu listy pracowników naukowych WMI UAM. Skrypt scraping'owy przechodzi po stronach poszczególnych zakładów i pracowni wydziału, zbierając dla każdego naukowca: imię i nazwisko, afiliację (zakład/pracownia), identyfikator ORCID (jeśli podany na stronie) oraz ewentualne dodatkowe identyfikatory (Scopus Author ID, Web of Science ResearcherID). Dane zapisywane są w pliku `scientists_with_identifiers.csv`.

Na tym etapie zidentyfikowano **164 pracowników** WMI UAM, z czego 160 posiada identyfikator ORCID. Po połączeniu z danymi z OpenAlex, **115 naukowców** miało co najmniej jedną publikację w bazie. Spośród nich 92 ma jednoznaczne przypisanie do konkretnego zakładu lub pracowni (rozpoczynające się od „Zakład" lub „Pracownia"). Pozostali to doktoranci, emerytowani pracownicy lub osoby z niejednoznaczną afiliacją.

#### Pobieranie danych z OpenAlex API (`abstracts/`)

Drugi etap wykorzystuje identyfikatory ORCID do pobrania dorobku naukowego każdego autora z bazy **OpenAlex** — otwartego katalogu publikacji naukowych. Dla każdego ORCID-a pobieramy listę prac autorskich wraz z tytułami, abstraktami, tematami (topics), słowami kluczowymi (keywords), listami współautorów (co-authors), datami publikacji i identyfikatorami DOI.

Wynikowy plik `titles_with_abstracts.csv` zawiera **3440 prac** z następującymi kluczowymi kolumnami:

- `title` — tytuł artykułu,
- `abstract` — abstrakt (tekst pełny, jeśli dostępny w OpenAlex),
- `topics` — tematy przypisane przez OpenAlex (lista oddzielona średnikami),
- `keywords` — słowa kluczowe (lista oddzielona średnikami),
- `co_authors` — współautorzy (lista ORCID-ów oddzielona średnikami),
- `main_author_orcid` — ORCID autora-właściciela danego rekordu.

W naszym zbiorze praktycznie wszystkie prace posiadają abstrakt w OpenAlex — jedynie 4 z 3440 (0.1%) mają pole abstraktu wypełnione placeholder'em (np. „International audience" lub „See the abstract in the attached pdf").

#### Rozkład prac per autor

Rozkład liczby prac na autora jest silnie prawoskośny. Poniższa tabela przedstawia histogram:

| Zakres prac | Liczba autorów | Odsetek |
|-------------|---------------|---------|
| 1–4         | 9             | 7.8%    |
| 5–10        | 20            | 17.4%   |
| 11–20       | 36            | 31.3%   |
| 21–50       | 31            | 27.0%   |
| 51–100      | 14            | 12.2%   |
| >100        | 5             | 4.3%    |

Średnia liczba prac wynosi 29.9, mediana 19.0. Autorzy z mniej niż 5 pracami (9 osób) traktowani są w wizualizacji jako „low confidence" — ich embeddingi mogą być niestabilne z powodu zbyt małej próby do sensownego uśrednienia.

### 2.2 Embeddingi artykułów (Paper Embeddings)

#### Tekst wejściowy: tytuł + abstrakt

Każdy artykuł kodowany jest jako wektor 768-wymiarowy za pomocą modelu Sentence Transformer. Tekst wejściowy konstruowany jest jako konkatenacja tytułu i abstraktu: `"{title}. {abstract}"`. W przypadku braku abstraktu używany jest sam tytuł.

Decyzja o użyciu abstraktu oprócz samego tytułu opiera się na obserwacji, że abstrakt dodaje istotny kontekst semantyczny. Tytuł artykułu naukowego jest z natury zwięzły i często niejednoznaczny — np. tytuł „On spaces of operators" nie pozwala odróżnić analizy funkcjonalnej od algebry operatorowej. Abstrakt dostarcza informacji o konkretnych metodach, twierdzeniach i zastosowaniach. Oczekujemy, że dodanie abstraktu istotnie zwiększa jakość embeddingu w metrykach klastrowania, choć dokładna wielkość tego efektu nie była mierzona w ramach niniejszej pracy.

#### Wybór modelu: BGE-base-en-v1.5

Model bazowy to **BAAI/bge-base-en-v1.5** — model embeddingowy z rodziny BGE (Beijing Academy of Artificial Intelligence General Embedding), generujący wektory 768-wymiarowe. Wybór tego modelu podyktowany jest kilkoma czynnikami.

W ramach prac nad systemem SARA przetestowano 9 modeli embeddingowych na zbiorze 2 397 artykułów ArXiv z hierarchiczną strukturą kategorii (118 podkategorii). Porównanie obejmowało zarówno modele ogólnego przeznaczenia, jak i modele specjalizowane. Poniższa tabela prezentuje skrócone wyniki:

| Model | Wymiary | Main category purity (k=118) | Silhouette (k=118) |
|-------|---------|-------------------------------|---------------------|
| **BAAI/bge-base-en-v1.5** | 768 | **70.55%** | 0.346 |
| BAAI/bge-large-en-v1.5 | 1024 | 69.63% | 0.347 |
| nomic-ai/nomic-embed-text-v1.5 | 768 | 69.50% | 0.343 |
| allenai/specter | 768 | 68.88% | 0.346 |
| sentence-transformers/all-mpnet-base-v2 | 768 | 68.09% | 0.347 |
| Qwen/Qwen3-Embedding-0.6B | 1024 | 68.09% | 0.346 |
| Qwen/Qwen3-Embedding-4B | 2560 | 67.75% | 0.351 |
| Alibaba-NLP/gte-large-en-v1.5 | 1024 | 67.63% | 0.347 |
| intfloat/e5-mistral-7b-instruct | 4096 | 65.08% | 0.341 |

BGE-base osiąga najwyższą main category purity (70.55%) spośród wszystkich testowanych modeli, wyprzedzając nawet BGE-large (1024D, 69.63%). Jednocześnie BGE-base jest szybszy w inference i wymaga mniej pamięci. Warto odnotować, że największy testowany model — e5-mistral-7b-instruct (4096D) — osiąga najsłabszy wynik (65.08%), co sugeruje, że większy wymiar embeddingu nie przekłada się automatycznie na lepszą jakość klastrowania artykułów naukowych.

#### Normalizacja embeddingów

Embeddingi generowane są z włączoną normalizacją L2 (`normalize_embeddings=True`), co oznacza, że każdy wektor ma normę 1: $\|e_i\| = 1$. Dzięki temu odległość euklidesowa jest monotonicznie związana z podobieństwem kosinusowym:

$$d_{\text{euc}}(e_i, e_j) = \sqrt{2(1 - \cos(e_i, e_j))}$$

co upraszcza zarówno obliczenia (K-means z odległością euklidesową jest równoważny z metryką kosinusową), jak i interpretację.

### 2.3 Fine-tuning modelu

#### Motywacja

Model bazowy BGE-base-en-v1.5 został wytrenowany na ogólnym korpusie tekstów i nie jest specjalnie dostosowany do rozróżniania subtelnych różnic między poddziedzinami nauki. Celem fine-tuningu jest nauczenie modelu, że artykuły z tej samej podkategorii ArXiv (np. `cs.AI`) powinny mieć bardziej zbliżone embeddingi niż artykuły z różnych kategorii, przy jednoczesnym uwzględnieniu hierarchicznej natury systemu kategorii naukowych.

#### Pierwsza próba: MultipleNegativesRankingLoss (MNR)

Pierwszą próbą fine-tuningu było zastosowanie funkcji straty **MultipleNegativesRankingLoss** (MNR) z następującymi parametrami: 3 epoki, learning rate 2e-5, batch size 32. MNR traktuje problem jako binarny — para jest albo „pozytywna" (ta sama kategoria), albo „negatywna" (inna kategoria).

Podejście to prowadziło do **overfittingu**: po 2. epoce validation loss zaczął rosnąć, a model tracił zdolność generalizacji na nowych danych. Problem wynikał z dwóch czynników: braku gradacji podobieństwa (artykuł z `cs.AI` jest „bardziej podobny" do `cs.ML` niż do `math.AG`, ale MNR traktuje obie pary jako negatywne) oraz zbyt agresywnego uczenia (3 epoki z lr=2e-5).

#### Rozwiązanie: CoSENTLoss z hierarchiczną odległością kategorii

Drugie podejście zastąpiło MNR funkcją straty **CoSENTLoss**, która akceptuje ciągłe wyniki podobieństwa (float z zakresu [0.0, 1.0]) zamiast binarnych etykiet. CoSENTLoss uczy modelu *rankingu*: jeśli $\text{score}(A,B) > \text{score}(C,D)$, to model powinien nauczyć się, że $\text{sim}(A,B) > \text{sim}(C,D)$.

Wyniki podobieństwa (scores) obliczane są na podstawie **3-poziomowej hierarchii kategorii ArXiv**. System kategorii ArXiv ma strukturę drzewiastą z trzema poziomami:

- **Grupa główna** (top-level group): np. `cs` (Computer Science), `math` (Mathematics), `physics`
- **Archiwum** (archive): np. `hep-ph` (High Energy Physics - Phenomenology), `hep-th` (HEP Theory)
- **Podkategoria** (subcategory): np. `cs.AI`, `cs.ML`, `math.AG`, `math.FA`

Hierarchiczna odległość kategorii definiowana jest następująco:

$$d_{\text{hier}}(c_i, c_j) = \begin{cases} 0.0 & \text{ta sama podkategoria (np. cs.AI vs cs.AI)} \\ 0.33 & \text{to samo archiwum, inna podkategoria (np. hep-ph vs hep-th)} \\ 0.67 & \text{ta sama grupa, inne archiwum (np. astro-ph.CO vs quant-ph)} \\ 1.0 & \text{inna grupa (np. cs.AI vs math.AG)} \end{cases}$$

Ponieważ 63.8% artykułów w zbiorze treningowym ma więcej niż jedną kategorię (średnio 2.06 kategorii na artykuł), stosujemy **rozmyte etykiety (multi-label)**. Wynik podobieństwa pary artykułów $(A, B)$ obliczany jest jako:

$$\text{score}(A,B) = 0.5 \cdot J(C_A, C_B) + 0.5 \cdot \overline{d}_{\text{best}}(C_A, C_B)$$

gdzie $J(C_A, C_B)$ oznacza współczynnik Jaccarda zbiorów kategorii, a $\overline{d}_{\text{best}}$ to średnią najlepszego hierarchicznego dopasowania między kategoriami obu artykułów.

#### Stratified sampling par treningowych

Aby zapewnić równomierne pokrycie różnych poziomów hierarchii, stosujemy **stratified sampling** par treningowych. Dla każdego artykułu w zbiorze treningowym generujemy 8 par:

- 3 pary z artykułami z **tej samej podkategorii** (np. oba cs.AI),
- 2 pary z artykułami z **tego samego archiwum**, ale innej podkategorii (np. hep-ph vs hep-th),
- 2 pary z artykułami z **tej samej grupy**, ale innego archiwum (np. math.AG vs math.FA),
- 1 para z artykułem z **innej grupy** (np. cs.AI vs math.AG).

Taka strategia zapewnia, że model widzi wystarczająco dużo par na każdym poziomie trudności.

#### Hiperparametry fine-tuningu

| Parametr | Wartość |
|----------|---------|
| Model bazowy | BAAI/bge-base-en-v1.5 (768 wymiarów) |
| Funkcja straty | CoSENTLoss |
| Artykuły (trening) | 18 880 |
| Epoki | 1 |
| Learning rate | 1e-5 |
| Batch size | 64 |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| GPU | Google Colab A100 |

Kluczową zmianą względem pierwszej próby jest **1 epoka** (zamiast 3) i **niższy learning rate** (1e-5 zamiast 2e-5). Validation loss monotonicznie malał przez całą epokę, co potwierdza brak overfittingu.

---

## 3. Metody reprezentacji autorów

Przejście od embeddingów per-artykuł do embeddingów per-autor wymaga strategii agregacji. Każdy autor $a$ ma zbiór prac $\mathcal{P}_a = \{p_1, p_2, \ldots, p_{n_a}\}$, gdzie każda praca $p_i$ jest reprezentowana przez embedding $e_i \in \mathbb{R}^{768}$. Celem jest skonstruowanie reprezentacji autora w tej samej przestrzeni embeddingów.

### 3.1 Mean Pooling (jeden punkt per autor)

#### Definicja matematyczna

Embedding autora $a$ definiujemy jako znormalizowaną średnią embeddingów jego prac:

$$\bar{e}_a = \frac{1}{n_a} \sum_{i=1}^{n_a} e_i$$

$$\hat{e}_a = \frac{\bar{e}_a}{\|\bar{e}_a\|}$$

gdzie $\hat{e}_a$ jest ostatecznym, znormalizowanym embeddingiem autora.

#### Dlaczego mean pooling?

Mean pooling jest najprostszą strategią agregacji i ma kilka pożądanych właściwości. Po pierwsze, jest nieobciążony — średnia wektorów z danej podprzestrzeni leży w centroidzie tej podprzestrzeni, co jest naturalną „środkową" reprezentacją dorobku autora. Po drugie, jest odporny na szum — losowa zmienność poszczególnych embeddingów artykułów wygładza się przy uśrednianiu.

Rozważano alternatywne strategie:

- **Max pooling:** Wybranie per-wymiarowego maksimum ze wszystkich embeddingów. W praktyce produkuje to wektory, które nie odpowiadają żadnemu konkretnemu profilowi badawczemu, lecz są „superimposed" — nakładają różne tematyki w nieinterpretowalny sposób.
- **Weighted pooling (ważony liczbą cytowań):** Ważenie każdego artykułu jego impact factor'em lub liczbą cytowań. Koncepcyjnie interesujące, ale wymaga dostępu do danych o cytowaniach, które w OpenAlex są niekompletne dla mniejszych uczelni. Ponadto faworyzowałoby to starsze prace, a nam zależy na aktualnym profilu badawczym.
- **Attention pooling:** Wyuczenie mechanizmu uwagi, który automatycznie waży artykuły. Wymaga to dodatkowego modelu i danych treningowych, co jest nieproporcjonalne do skali problemu (115 autorów).

#### Wpływ normalizacji

Normalizacja centroidu ($\hat{e}_a = \bar{e}_a / \|\bar{e}_a\|$) jest istotna z dwóch powodów. Po pierwsze, bez normalizacji autor z dużą liczbą prac o spójnej tematyce miałby wektor o dużej normie (bo wektory prac „wzmacniają się" przy uśrednianiu), natomiast autor z pracami rozrzuconymi tematycznie miałby wektor o małej normie. Normalizacja eliminuje ten efekt — porównujemy jedynie *kierunki* w przestrzeni embeddingów, nie długości. Po drugie, normalizacja zapewnia spójność z embeddingami per-artykuł (które też są znormalizowane), co ułatwia porównywanie autorów z artykułami.

### 3.2 Multi-Cluster (wiele punktów per autor)

Mean pooling ma fundamentalne ograniczenie: zakłada, że profil badawczy autora jest **unimodalny** — daje się opisać jednym punktem. W rzeczywistości wielu naukowców prowadzi badania w dwóch lub więcej odrębnych kierunkach. Na przykład informatyk może publikować prace zarówno z zakresu sztucznej inteligencji, jak i kryptografii. Uśrednienie tych dwóch obszarów dałoby punkt w pustej przestrzeni „między" AI a kryptografią — niereprezentujący żadnego z rzeczywistych kierunków badawczych.

Wariant **multi-cluster** adresuje ten problem, reprezentując każdego autora jako $k_a \geq 1$ punktów, gdzie każdy punkt odpowiada jednemu kierunkowi badawczemu.

#### Algorytm

Pełny pseudokod algorytmu klasteryzacji prac autora:

```
Algorithm 1: cluster_author_papers(author_embeddings)
Input: E = {e_1, ..., e_n} — embeddingi prac autora, n = |E|
Output: (k, labels, centroids) — liczba klastrów, przypisania, centroidy

1. if n ≤ 1:
     return (1, [0], E)                          // Jeden punkt = jeden klaster

2. if n < MIN_PAPERS (=5):
     centroid ← mean(E) / ||mean(E)||
     return (1, [0,...,0], [centroid])            // Za mało danych do klasteryzacji

3. max_k ← min(MAX_CLUSTERS, n-1)               // MAX_CLUSTERS = 5
   if max_k < 2:
     centroid ← mean(E) / ||mean(E)||
     return (1, [0,...,0], [centroid])

4. best_k ← 1, best_sil ← -1, best_labels ← None

5. for k = 2, ..., max_k:
     labels ← KMeans(k).fit_predict(E)
     if |unique(labels)| < 2: continue           // Degenerate clustering
     sil ← SilhouetteScore(E, labels)
     if sil > best_sil:
       best_sil ← sil
       best_k ← k
       best_labels ← labels

6. if best_sil < THRESHOLD (=0.15) or best_labels is None:
     centroid ← mean(E) / ||mean(E)||
     return (1, [0,...,0], [centroid])            // Klasteryzacja niewystarczająca

7. centroids ← []
   for c = 0, ..., best_k - 1:
     centroid_c ← mean(E[labels == c]) / ||mean(E[labels == c])||
     centroids.append(centroid_c)

8. return (best_k, best_labels, centroids)
```

#### Przykład: autor z wieloma kierunkami badawczymi

Rozpatrzmy hipotetycznego autora z Zakładu Sztucznej Inteligencji, który opublikował 25 prac: 12 z zakresu przetwarzania języka naturalnego (NLP), 8 z zakresu widzenia komputerowego (computer vision) i 5 z zakresu teorii grafów. Algorytm multi-cluster:

1. Oblicza embeddingi dla wszystkich 25 prac.
2. Testuje K-means z k=2, 3, 4 (max_k = min(5, 24) = 5, ale efektywnie testujemy do 4–5).
3. Dla k=3 uzyskuje silhouette score 0.31, co jest powyżej progu 0.15.
4. Przypisuje prace do trzech klastrów: {NLP: 12 prac}, {CV: 8 prac}, {grafy: 5 prac}.
5. Oblicza centroid dla każdego klastra i normalizuje go.
6. Na mapie PCA autor pojawia się jako **3 punkty** połączone liniami.

W wariancie mean pooling ten sam autor byłby jednym punktem gdzieś w „środku" trójkąta NLP–CV–grafy, co jest mniej informatywne.

#### Wizualizacja: jeden klaster vs wiele klastrów

Na mapie PCA autorzy z jednym klastrem (k=1) pojawiają się jako pojedyncze punkty — tak samo jak w wariancie mean pooling. Autorzy z wieloma klastrami pojawiają się jako kilka punktów **połączonych cienkimi liniami** (krawędziami). Rozmiar każdego punktu jest proporcjonalny do pierwiastka z liczby prac w danym klastrze, co pozwala odróżnić główny kierunek badawczy od pobocznych zainteresowań.

Odległość między klastrami tego samego autora na mapie ma interpretację: im dalej od siebie leżą dwa punkty, tym bardziej odrębne są odpowiadające im kierunki badawcze. Linie łączące klastry autora ułatwiają wizualne śledzenie „zasięgu" badawczego danej osoby.

#### Generowanie etykiet per klaster

Każdy klaster autora otrzymuje automatycznie generowaną etykietę opartą na metadanych z OpenAlex. Funkcja `get_cluster_label()` zbiera tematy (`topics`) i słowa kluczowe (`keywords`) ze wszystkich prac w danym klastrze, a następnie za pomocą obiektu `Counter` identyfikuje 3 najczęstsze tematy i 5 najczęstszych słów kluczowych. Etykiety te wyświetlane są w hoverze na interaktywnej mapie.

---

## 4. Automatyczny dobór liczby klastrów

### 4.1 Silhouette Score

Kluczowym elementem wariantu multi-cluster jest automatyczny dobór liczby klastrów $k$ dla każdego autora. Stosujemy w tym celu **silhouette score** — miarę jakości klasteryzacji zaproponowaną przez Rousseeuw [10].

#### Definicja matematyczna

Dla każdego punktu $i$ przypisanego do klastra $C_i$ definiujemy dwie wielkości:

$$a(i) = \frac{1}{|C_i| - 1} \sum_{j \in C_i, j \neq i} d(i, j)$$

gdzie $a(i)$ jest średnią odległością punktu $i$ od pozostałych punktów w jego klastrze (miara **spójności**).

$$b(i) = \min_{C_k \neq C_i} \frac{1}{|C_k|} \sum_{j \in C_k} d(i, j)$$

gdzie $b(i)$ jest minimalną średnią odległością od punktu $i$ do punktów w innym klastrze (miara **separacji**).

Silhouette score dla punktu $i$ definiowany jest jako:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Wartość $s(i)$ mieści się w przedziale $[-1, 1]$:

- $s(i) \approx 1$ — punkt jest dobrze dopasowany do swojego klastra i daleko od innych klastrów,
- $s(i) \approx 0$ — punkt leży na granicy między dwoma klastrami,
- $s(i) \approx -1$ — punkt jest lepiej dopasowany do innego klastra niż do swojego.

Średni silhouette score dla całej klasteryzacji to:

$$\bar{s} = \frac{1}{n} \sum_{i=1}^{n} s(i)$$

#### Przykład intuicyjny

Rozważmy autora z 10 pracami: 3 o sztucznej inteligencji (AI), 4 o przetwarzaniu języka naturalnego (NLP) i 3 o teorii grafów. Algorytm testuje podział na 2 i 3 klastry:

- **k=2:** K-means grupuje AI i NLP razem (bo są semantycznie bliższe), a grafy oddzielnie. Ale prace AI i NLP różnią się między sobą, więc $a(i)$ jest umiarkowane. Silhouette score: 0.22.
- **k=3:** K-means rozdziela AI, NLP i grafy. Każdy klaster jest spójny ($a(i)$ małe), a klastry są wyraźnie odseparowane ($b(i)$ duże). Silhouette score: 0.38.

Algorytm wybiera k=3, ponieważ 0.38 > 0.22 i 0.38 > THRESHOLD = 0.15.

#### Dlaczego silhouette a nie inne metody?

Rozważano alternatywne kryteria automatycznego doboru $k$:

- **Elbow method** (metoda łokcia): Szuka „załamania" w krzywej inercji K-means. W praktyce dla małych zbiorów (5–100 prac per autor) krzywa jest płynna i załamanie jest niejednoznaczne.
- **Gap statistic:** Porównuje inercję rzeczywistych danych z inercją danych losowych. Wymaga wielokrotnych próbkowań referencyjnych, co jest kosztowne obliczeniowo i niestabilne dla małych $n$.
- **BIC (Bayesian Information Criterion):** Zakłada model gaussowski klastrów, co jest rozsądne dla embeddingów, ale wymaga estymacji pełnych macierzy kowariancji — niestabilnej przy 5–10 punktach w 768 wymiarach.

Silhouette score nie wymaga żadnych założeń parametrycznych, jest szybki do obliczenia i ma bezpośrednią interpretację geometryczną, co czyni go najlepszym wyborem dla naszego zastosowania.

### 4.2 Edge cases

Algorytm musi poprawnie obsługiwać przypadki brzegowe:

- **Autor z 1 pracą:** Brak możliwości klasteryzacji. Zwracany jest 1 klaster z centroidem równym embeddingowi jedynej pracy.
- **Autor z 2–4 pracami:** Poniżej progu `MIN_PAPERS_FOR_CLUSTERING = 5`. Zwracany jest 1 klaster z centroidem będącym uśrednioną wartością. Uzasadnienie: przy tak małej próbie silhouette score jest niestabilny i podział byłby arbitralny.
- **Autor z dokładnie 5 pracami:** Minimalna liczba do klasteryzacji. Dla n=5: max_k = min(5, 4) = 4, więc testowane są k=2, 3, 4. W praktyce przy 5 punktach w 768D klastry rzadko mają sensowną strukturę, więc większość autorów z 5 pracami otrzymuje k=1.

### 4.3 Złożoność obliczeniowa

Silhouette score wymaga obliczenia odległości między wszystkimi parami punktów, co daje złożoność $O(n^2)$ per autor, gdzie $n$ jest liczbą prac. Ponieważ testujemy $k$ od 2 do `MAX_CLUSTERS=5`, a K-means sam ma złożoność $O(n \cdot k \cdot d \cdot I)$ (gdzie $d=768$, $I$ = liczba iteracji), łączna złożoność per autor to $O(n^2 \cdot d + n \cdot k_{\max} \cdot d \cdot I)$.

W praktyce $n$ jest małe (maksymalnie ~100 prac dla najbardziej płodnych autorów), więc cały proces klasteryzacji dla wszystkich 115 autorów zajmuje kilka sekund na standardowym sprzęcie.

### 4.4 Dobór progu silhouette

Próg `SILHOUETTE_THRESHOLD = 0.15` decyduje, poniżej jakiego silhouette score odrzucamy klasteryzację i traktujemy autora jako jednoklastrowego. Zgodnie z klasyfikacją Rousseeuw [10]:

- $\bar{s} > 0.50$ — silna struktura klastrów,
- $0.25 < \bar{s} \leq 0.50$ — rozsądna struktura,
- $\bar{s} \leq 0.25$ — słaba lub brak struktury.

Próg 0.15 jest konserwatywny — akceptujemy klasteryzację nawet przy słabej strukturze, ponieważ naszym celem jest identyfikacja *jakichkolwiek* odrębnych kierunków badawczych, a nie idealna separacja. Poniżej przedstawiamy eksperyment z różnymi wartościami progu:

| Próg | Autorów z k>1 | % autorów z k>1 | Średnie k | Cluster-points |
|------|--------------|-----------------|-----------|----------------|
| 0.05 | 106 | 92.2% | 2.75 | 316 |
| 0.10 | 105 | 91.3% | 2.71 | 312 |
| **0.15** | **98** | **85.2%** | **2.59** | **298** |
| 0.20 | 86 | 74.8% | 2.37 | 272 |
| 0.30 | 59 | 51.3% | 1.88 | 216 |

Wyniki pokazują, że większość autorów (z ≥5 pracami) ma silhouette powyżej nawet konserwatywnych progów — ich prace faktycznie tworzą odrębne klastry tematyczne w przestrzeni 768D. Próg 0.15 daje 298 cluster-points z 115 autorów (średnio 2.59 punkty/autor), co zapewnia informatywną wizualizację bez nadmiernego zagęszczenia mapy.

---

## 5. Redukcja wymiarowości: PCA

### 5.1 Definicja matematyczna

Principal Component Analysis (PCA) jest liniową metodą redukcji wymiarowości polegającą na znalezieniu kierunków w przestrzeni danych, które maksymalizują wariancję [11].

Niech $X = \{x_1, x_2, \ldots, x_n\} \subset \mathbb{R}^d$ oznacza zbiór $n$ punktów danych ($d = 768$ w naszym przypadku). Macierz kowariancji danych definiowana jest jako:

$$\Sigma = \frac{1}{n} X^T X$$

PCA polega na znalezieniu wektorów własnych macierzy $\Sigma$. Pierwsze $k$ wektorów własnych odpowiadających największym wartościom własnym $\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_k$ definiuje przestrzeń projekcji.

Redukcja wymiarowości wykonywana jest poprzez rzutowanie danych:

$$y_i = W^T x_i$$

gdzie $W = [w_1, w_2, \ldots, w_k]$ zawiera pierwsze $k$ wektorów własnych (tzw. głównych składowych — principal components).

Proporcja wyjaśnionej wariancji (variance explained) dla $k$ komponentów wynosi:

$$\text{VarExplained}(k) = \frac{\sum_{j=1}^{k} \lambda_j}{\sum_{j=1}^{d} \lambda_j}$$

### 5.2 Variance explained: autorzy vs artykuły

Jednym z kluczowych wyników niniejszej pracy jest obserwacja, że PCA na embeddingach autorów (po mean poolingu) wyjaśnia **znacząco więcej wariancji** niż PCA na embeddingach artykułów:

| Zbiór | Model | Variance explained (2 PC) |
|-------|-------|--------------------------|
| Artykuły ArXiv (18 880) | BGE-base (oryginalny) | 12.19% |
| Artykuły ArXiv (18 880) | BGE-base (fine-tuned) | 29.37% |
| Autorzy WMI (115) | BGE-base (oryginalny) | 33.11% |
| Autorzy WMI (115) | BGE-base (fine-tuned) | **54.71%** |

Zjawisko to ma naturalne wyjaśnienie. Mean pooling per autor działa jak filtr dolnoprzepustowy — uśrednianie wielu embeddingów artykułów wygładza szum związany z poszczególnymi pracami i uwydatnia główne, dominujące tematy badawcze danego autora. Pojedynczy artykuł może mieć nieoczekiwany temat (np. praca przeglądowa, interdyscyplinarna kolaboracja), ale *średnia* dorobku autora stabilnie wskazuje na jego główne pole badawcze.

Efekt fine-tuningu jest jeszcze bardziej widoczny: model po fine-tuningu lepiej separuje tematyki naukowe, co sprawia, że centroidy autorów z różnych dziedzin są bardziej oddalone w przestrzeni 768D. Ta lepsza separacja „przenosi się" na pierwsze 2 komponenty PCA, dając 54.7% variance explained — ponad połowę informacji jest uchwycona w zaledwie 2 wymiarach.

### 5.3 Scree plot — ile wymiarów potrzeba?

Analiza wartości własnych macierzy kowariancji embeddingów autorów (model fine-tuned) pozwala ocenić, ile komponentów PCA jest potrzebnych do uchwycenia różnych proporcji informacji:

| Liczba komponentów | Variance explained (skumulowana) |
|-------------------|----------------------------------|
| 1                 | 44.26%                            |
| 2                 | 54.71%                            |
| 3                 | 63.56%                            |
| 5                 | 72.42%                            |
| 10                | 82.65%                            |
| 20                | 91.09%                            |

Przejście z 2D do 3D daje jedynie ~9 punktów procentowych dodatkowej wariancji, przy jednocześnie znacznym pogorszeniu interaktywności — wykresy 3D w przeglądarce są trudniejsze do nawigacji, rotacji i klikania. Z tego powodu zdecydowaliśmy się na projekcję 2D, tak jak Wujec w swoich wizualizacjach artykułów.

### 5.4 Dlaczego PCA a nie t-SNE/UMAP?

W pracy Wujca przeprowadzono obszerne porównanie metod redukcji wymiarowości, z którego wynika, że na zbiorze 3440 artykułów metody nieliniowe (t-SNE, PaCMAP, UMAP) osiągają composite score 0.725–0.745, podczas gdy PCA jedynie 0.576. Dlaczego więc w wizualizacji autorów stosujemy PCA?

Po pierwsze, **skala problemu jest inna.** Przy 115 punktach (autorów) zamiast 3440 (artykułów) problem redukcji jest znacznie łatwiejszy. Johnson–Lindenstrauss bound jest mniej restrykcyjny dla mniejszego $n$.

Po drugie, **variance explained jest znacznie wyższy.** 54.7% wariancji w 2 komponentach oznacza, że PCA uchwytuje ponad połowę struktury danych — w porównaniu z 12.2% na artykułach ArXiv. Mean pooling skutecznie „upraszcza" przestrzeń, sprawiając, że metoda liniowa jest wystarczająca.

Po trzecie, **interpretowalność PCA jest lepsza.** W PCA osie mają konkretny sens — PC1 i PC2 są liniowymi kombinacjami oryginalnych wymiarów. Odległości na wykresie PCA są proporcjonalne do odległości w przestrzeni embeddingów (z dokładnością do utraty informacji z odrzuconych komponentów). W t-SNE odległości między klastrami nie mają interpretacji, a wielkości klastrów są sztucznie wyrównywane — problemy szczegółowo opisane przez Wujca.

Po czwarte, **PCA jest deterministyczna.** Uruchomienie PCA na tych samych danych zawsze daje ten sam wynik, co jest pożądane w kontekście systemu produkcyjnego SARA. t-SNE i UMAP zależą od losowej inicjalizacji i parametrów (perplexity, n_neighbors), co wprowadza dodatkowy stopień swobody.

---

## 6. Ewaluacja separacji zakładów

### 6.1 Metodologia

Kluczowym pytaniem ewaluacyjnym jest: **czy embeddingi autorów odzwierciedlają strukturę organizacyjną WMI?** Innymi słowy, czy naukowcy z tego samego zakładu mają bliskie sobie embeddingi? Analizę przeprowadzono na 92 autorach z jednoznacznym przypisaniem do jednego z 14 zakładów lub pracowni, porównując model bazowy (BGE-base) z modelem po fine-tuningu (CoSENT).

### 6.2 Definicje metryk

#### Mean intra-dept cosine similarity

Średnie podobieństwo kosinusowe między wszystkimi parami autorów z tego samego zakładu:

$$\text{IntraSim} = \frac{1}{\sum_d \binom{n_d}{2}} \sum_{d} \sum_{\substack{i,j \in D_d \\ i < j}} \cos(\hat{e}_i, \hat{e}_j)$$

gdzie $D_d$ oznacza zbiór autorów z zakładu $d$, a $n_d = |D_d|$. Wyższe wartości oznaczają większą spójność tematyczną zakładu.

#### Mean inter-dept cosine similarity

Średnie podobieństwo kosinusowe między parami autorów z **różnych** zakładów. Niższe wartości oznaczają lepsze rozdzielenie.

#### Intra/Inter ratio

Stosunek $\text{IntraSim} / \text{InterSim}$. Wartości powyżej 1 oznaczają, że autorzy w ramach zakładu są do siebie bardziej podobni niż do autorów z innych zakładów. Wyższy stosunek oznacza lepszą separację.

#### NN@k dept accuracy

Dla każdego autora $i$ wyznaczamy $k$ najbliższych sąsiadów (wg cosine similarity). Sprawdzamy, czy większość tych sąsiadów należy do tego samego zakładu co autor $i$:

$$\text{NN@k} = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}\left[\text{mode}\left(\{d_j : j \in \text{NN}_k(i)\}\right) = d_i\right]$$

Metryka ta jest szczególnie ważna, bo odpowiada bezpośrednio na pytanie: „Czy moi najbliżsi sąsiedzi na mapie są z mojego zakładu?"

#### Dept purity (K-means)

Stosujemy K-means z $k$ równym liczbie zakładów (14). Dla każdego klastra identyfikujemy dominujący zakład i liczymy odsetek autorów w klastrze należących do tego zakładu:

$$\text{Purity} = \frac{1}{n} \sum_{c=1}^{k} \max_{d} |C_c \cap D_d|$$

#### Normalized Mutual Information (NMI)

NMI mierzy stopień zgodności między klasteryzacją K-means a prawdziwymi zakładami, normalizując do zakresu [0, 1]:

$$\text{NMI}(U, V) = \frac{2 \cdot I(U; V)}{H(U) + H(V)}$$

gdzie $I(U; V)$ to informacja wzajemna między podziałami $U$ (klastry K-means) i $V$ (zakłady), a $H(\cdot)$ to entropia. NMI = 0 oznacza brak zależności, NMI = 1 oznacza idealną zgodność.

#### Silhouette score (true departments)

Silhouette score obliczony z użyciem prawdziwych etykiet zakładów (zamiast etykiet K-means). Mierzy, jak dobrze embeddingi w pełnej przestrzeni 768D separują zakłady.

### 6.3 Wyniki globalne

| Metryka | Oryginalny | Fine-tuned | Diff |
|---------|-----------|------------|------|
| Mean intra-dept cos-sim | 0.8579 | 0.9665 | +0.1086 |
| Mean inter-dept cos-sim | 0.7559 | 0.9113 | +0.1555 |
| Intra/Inter ratio | 1.1350 | 1.0605 | −0.0745 |
| NN@1 dept accuracy | 0.7283 | 0.6630 | −0.0652 |
| NN@3 dept accuracy | 0.6522 | 0.5870 | −0.0652 |
| NN@5 dept accuracy | 0.6087 | 0.4783 | −0.1304 |
| Dept purity (K-means) | 0.6413 | 0.6196 | −0.0217 |
| NMI | 0.6719 | 0.7015 | +0.0296 |
| Silhouette (true depts) | 0.0643 | 0.0604 | −0.0039 |

### 6.4 Analiza wyników

Wyniki ujawniają interesujący **trade-off** między metrykami lokalnymi a globalnymi.

**Fine-tuning poprawia metryki globalne.** NMI rośnie z 0.672 do 0.702 (+0.030), co oznacza, że globalna struktura klastrów lepiej odpowiada zakładom. Zarówno intra-dept, jak i inter-dept cosine similarity rosną (o +0.109 i +0.156 odpowiednio), ponieważ fine-tuning „ściska" embeddingi — wszystkie wektory stają się bardziej do siebie podobne w sensie kosinusowym.

**Fine-tuning pogarsza metryki lokalne.** NN@1 spada z 0.728 do 0.663 (−0.065), a NN@5 z 0.609 do 0.478 (−0.130). To pozornie zaskakujący wynik, ale ma logiczne wyjaśnienie. Fine-tuning zwiększa ogólne podobieństwo wszystkich embeddingów (mean inter-dept similarity rośnie z 0.756 do 0.911), co sprawia, że **różnice między autorami stają się mniejsze w wartościach bezwzględnych**. W efekcie sąsiedztwo lokalne (kto jest „najbliższy") staje się mniej stabilne — niewielkie różnice decydują o kolejności, i autorzy z sąsiadujących tematycznie zakładów częściej „wyprzedzają" kolegów z tego samego zakładu.

Innymi słowy, fine-tuning poprawia globalne uporządkowanie (klastry są lepiej dopasowane do zakładów wg NMI), ale „ściska" embeddingi tak, że lokalne sąsiedztwo staje się szumowe. Jest to trade-off lokalny vs globalny, analogiczny do tego opisanego przez Wujca w kontekście PCA vs t-SNE.

**Intra/Inter ratio spada** z 1.135 do 1.061, co potwierdza powyższe: chociaż oba similarity rosną, inter-dept rośnie szybciej, zmniejszając względną przewagę spójności wewnątrzzakładowej.

### 6.5 Analiza per zakład

Poniższa tabela przedstawia pełne wyniki per zakład:

| Zakład | N | Intra (orig) | Intra (FT) | Diff | Najbliższy zakład (FT) | Near (orig) | Near (FT) |
|--------|---|-------------|-----------|------|----------------------|------------|----------|
| P. Algorytmiki | 4 | 0.9052 | 0.9814 | +0.076 | Z. Teorii Algorytmów i Bezpiecz. Danych | 0.8736 | 0.9762 |
| P. Logiki i Filozofii Informatyki | 2 | 0.8205 | 0.9452 | +0.125 | Z. Sztucznej Inteligencji | 0.9108 | 0.9764 |
| Z. Algebry i Teorii Liczb | 6 | 0.8706 | 0.9666 | +0.096 | Z. Analizy Nieliniowej i Topologii Stos. | 0.9449 | 0.9861 |
| Z. Analizy Funkcjonalnej | 8 | 0.9155 | 0.9854 | +0.070 | Z. Analizy Matematycznej | 0.9750 | 0.9964 |
| Z. Analizy Matematycznej | 5 | 0.8719 | 0.9712 | +0.099 | Z. Analizy Funkcjonalnej | 0.9750 | 0.9964 |
| Z. Analizy Nieliniowej i Topologii Stos. | 6 | 0.9061 | 0.9816 | +0.076 | Z. Analizy Matematycznej | 0.9731 | 0.9956 |
| Z. Arytmetycznej Geometrii Algebraicznej | 4 | 0.8603 | 0.9414 | +0.081 | Z. Geometrii Algebraicznej i Diofant. | 0.9651 | 0.9930 |
| Z. Geometrii Algebraicznej i Diofant. | 5 | 0.8423 | 0.9530 | +0.111 | Z. Arytmetycznej Geometrii Algebraicznej | 0.9651 | 0.9930 |
| Z. Matematyki Dyskretnej | 9 | 0.8794 | 0.9714 | +0.092 | Z. Teorii Algorytmów i Bezpiecz. Danych | 0.9591 | 0.9845 |
| Z. Przestrzeni Funkcyjnych i Równań Różn. | 5 | 0.9187 | 0.9894 | +0.071 | Z. Teorii Operatorów | 0.9631 | 0.9958 |
| Z. Statystyki Matematycznej i Analizy Danych | 5 | 0.8680 | 0.9269 | +0.059 | P. Logiki i Filozofii Informatyki | 0.9264 | 0.9679 |
| Z. Sztucznej Inteligencji | 21 | 0.8344 | 0.9625 | +0.128 | P. Logiki i Filozofii Informatyki | 0.9264 | 0.9764 |
| Z. Teorii Algorytmów i Bezpiecz. Danych | 5 | 0.7713 | 0.9393 | +0.168 | Z. Matematyki Dyskretnej | 0.9591 | 0.9845 |
| Z. Teorii Operatorów | 7 | 0.9316 | 0.9909 | +0.059 | Z. Przestrzeni Funkcyjnych i Równań Różn. | 0.9715 | 0.9958 |

#### Obserwacje kluczowe

**Najbardziej spójne zakłady** to Zakład Teorii Operatorów (intra-sim 0.991 po fine-tuningu) i Zakład Przestrzeni Funkcyjnych (0.989). Oba zajmują się ściśle zdefiniowanymi dziedzinami matematyki, co naturalnie skutkuje zbliżonymi embeddingami prac.

**Najmniej spójny zakład** to Zakład Statystyki Matematycznej i Analizy Danych (intra-sim 0.927 po fine-tuningu, najniższa wartość). Następny jest Zakład Sztucznej Inteligencji (0.963) — co jest zrozumiałe, ponieważ AI jest szeroką dziedziną obejmującą uczenie maszynowe, przetwarzanie języka naturalnego, widzenie komputerowe i inne poddziedziny. Z 21 członkami jest to również największy zakład, co zwiększa wewnętrzną heterogeniczność.

**Zakłady „bliźniacze"** — kilka par zakładów ma ekstremalnie wysokie wzajemne podobieństwo:

- Z. Analizy Funkcjonalnej ↔ Z. Analizy Matematycznej: **0.9964** (prawie nieodróżnialne). Oba zakłady zajmują się analizą matematyczną — różnica polega na podspecjalizacji (przestrzenie Banacha vs równania różniczkowe), ale z perspektywy embeddingów ich prace są semantycznie niemal identyczne.
- Z. Arytmetycznej Geometrii Algebraicznej ↔ Z. Geometrii Algebraicznej i Diofantycznej: **0.9930**. Ponownie, to dwa zakłady zajmujące się geometrią algebraiczną z różnych perspektyw, ale o silnie nakładającym się słownictwie i metodach.
- Z. Teorii Operatorów ↔ Z. Przestrzeni Funkcyjnych: **0.9958**. Teoria operatorów i przestrzenie funkcyjne to ściśle powiązane dziedziny analizy funkcjonalnej.

Te wyniki sugerują, że embeddingowe profile badawcze tych par zakładów są praktycznie nierozróżnialne. W kontekście systemu SARA oznacza to, że na mapie 2D odpowiadające im punkty będą się nakładać — co jest de facto prawidłowe i odzwierciedla rzeczywistą bliskość tematyczną.

**Największa poprawa po fine-tuningu** — Zakład Teorii Algorytmów i Bezpieczeństwa Danych (z 0.771 do 0.939, +0.168). Ten zakład ma relatywnie niejednorodny profil (algorytmika + kryptografia + bezpieczeństwo), więc fine-tuning, który lepiej rozpoznaje hierarchiczną strukturę tematyczną, skutkuje największą poprawą spójności.

---

## 7. Ewaluacja fine-tuningu na zbiorze ArXiv

### 7.1 Konfiguracja eksperymentu

Ewaluacja fine-tuningu przeprowadzona została na zbiorze **18 880 artykułów ArXiv** z hierarchicznym systemem kategorii (8 grup głównych, ~20 archiwów, 118 podkategorii). Zbiór ten jest niezależny od danych WMI — służy jako benchmark do oceny, czy fine-tuning faktycznie poprawia jakość embeddingów w zadaniu klastrowania i wyszukiwania podobnych artykułów.

Hiperparametry fine-tuningu opisano szczegółowo w sekcji 2.3. Kluczowa zmiana względem pierwszej próby (MNR, 3 epoki, lr=2e-5 → overfitting) to użycie CoSENTLoss z 1 epoką i lr=1e-5, co wyeliminowało problem przeuczenia.

### 7.2 Metryki klastrowania

Ewaluacja klastrowania wykorzystuje K-means na embeddingach po PCA z różnymi wartościami $k$ (8, 20, 118), porównując klastry z prawdziwymi kategoriami ArXiv.

#### Definicja fuzzy purity

Standardowa purity wymaga, aby artykuł miał dokładnie tę samą kategorię co dominująca kategoria w klastrze. **Fuzzy purity** łagodzi ten warunek: artykuł jest „poprawnie" sklasyfikowany, jeśli **dzieli jakąkolwiek kategorię** z dominującą grupą klastra. Przy 63.8% artykułów posiadających >1 kategorię (średnio 2.06), fuzzy purity lepiej oddaje rzeczywistą jakość klasteryzacji.

#### Wyniki

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

**Interpretacja.** Subcategory purity jest niska we wszystkich wariantach (9–16%) — jest to oczekiwane, ponieważ przy 118 podkategoriach i zaledwie 8 lub 20 klastrach wiele podkategorii musi dzielić klaster. Nawet przy k=118 (tyle klastrów co podkategorii) purity wynosi jedynie 16.2%, co wskazuje, że granice między podkategoriami ArXiv nie odpowiadają ostrym granicom w przestrzeni embeddingów.

Fuzzy purity (~80% po fine-tuningu) jest bardziej informatywna: oznacza, że 4 na 5 artykułów w danym klastrze dzieli przynajmniej jedną kategorię z dominującą grupą tego klastra.

### 7.3 Metryki wyszukiwania (Precision/Recall@k)

Metryki wyszukiwania mierzą, czy $k$ najbliższych sąsiadów artykułu w przestrzeni embeddingów dzieli z nim przynajmniej jedną kategorię. Obliczone na 2000 losowych zapytaniach.

| Metryka | Oryginalny | Fine-tuned | Poprawa |
|---------|-----------|------------|---------|
| P@1 | 0.740 | 0.758 | +1.80pp |
| P@3 | 0.707 | 0.730 | +2.32pp |
| P@5 | 0.684 | 0.713 | +2.87pp |
| P@10 | 0.654 | 0.690 | +3.61pp |
| P@20 | 0.625 | 0.667 | +4.29pp |
| R@1 | 0.0015 | 0.0015 | +0.00pp |
| R@3 | 0.0043 | 0.0044 | +0.01pp |
| R@5 | 0.0069 | 0.0071 | +0.02pp |
| R@10 | 0.0129 | 0.0136 | +0.06pp |
| R@20 | 0.0244 | 0.0260 | +0.16pp |

**Precision@k rośnie z k.** Poprawa wynosi +1.8pp dla P@1, ale +4.3pp dla P@20, co oznacza, że fine-tuning poprawia grupowanie artykułów w szerszym sąsiedztwie. Model po fine-tuningu nie tyle lepiej identyfikuje pojedynczego najbliższego sąsiada, co lepiej organizuje strukturę tematyczną na dalszych pozycjach rankingu.

**Recall jest niski w wartościach bezwzględnych** — R@20 wynosi zaledwie 0.026 (2.6%). Jest to oczekiwane: przy 18 880 artykułach z szerokim nakładaniem kategorii, liczba artykułów „relevant" (dzielących kategorię z zapytaniem) sięga tysięcy. R@20 = 0.026 oznacza, że w 20 sąsiadach odnajdujemy ~2.6% ze wszystkich relevant artykułów — co przy ~750 relevant artykułach na zapytanie jest rozsądnym wynikiem.

### 7.4 Variance explained (PCA)

| Model | Variance explained (2 PC) |
|-------|--------------------------|
| Oryginalny | 12.19% |
| Fine-tuned | 29.37% |

Fine-tuning ponad dwukrotnie zwiększa variance explained na artykułach ArXiv. Oznacza to, że po fine-tuningu embeddingi lepiej separują się wzdłuż głównych osi wariancji — informacja o strukturze kategorii jest bardziej skoncentrowana w pierwszych komponentach głównych.

---

## 8. Interaktywna wizualizacja

### 8.1 Wariant Mean Pooling (`pca_authors_finetuned.html`)

Wizualizacja mean pooling przedstawia 115 autorów WMI jako punkty na płaszczyźnie PCA 2D, w układzie side-by-side: lewy panel pokazuje model bazowy (variance explained 33.1%), prawy — model po fine-tuningu (54.7%).

Elementy interaktywne:

- **K-means klastry** (k=5) — kolorowanie punktów wg klastrów wyznaczonych na reliable autorach (≥5 prac). Klastry wyznaczane są niezależnie na obu panelach.
- **Wyszukiwarka** z autocomplete — wpisanie nazwiska podświetla odpowiedni punkt na obu panelach jednocześnie (powiększony marker z żółtą obramówką, pozostałe punkty wygaszone).
- **Checkbox „Show authors with <5 papers"** — domyślnie ukryci autorzy z <5 pracami (szare romby) mogą być włączeni; jeśli wyszukiwany autor jest low-confidence, checkbox włącza się automatycznie.
- **Hover** — wyświetla: imię i nazwisko, zakład/pracownia, liczbę prac, koordynaty PC1/PC2.

### 8.2 Wariant Multi-Cluster (`pca_authors_multiclusters.html`)

Wizualizacja multi-cluster wykorzystuje model fine-tuned i przedstawia autorów jako jeden lub więcej punktów (cluster-points). Kolorowanie jest wg zakładu (nie wg klastra K-means), co pozwala bezpośrednio ocenić separację organizacyjną.

Elementy interaktywne:

- **Linie łączące klastry** — cienkie białe linie łączą punkty tego samego autora, z opcją ich ukrycia checkboxem „Cluster links".
- **Podwójna wyszukiwarka A vs B** — żółte pole (Scientist A) i niebieskie pole (Scientist B) pozwalają porównać dwóch naukowców. Po wybraniu obu, wszystkie pozostałe punkty są wygaszone, a podświetlone są jedynie klastry wybranych autorów — ułatwia to wizualną ocenę, jak blisko siebie leżą ich profile badawcze.
- **Rozmiar punktów** — proporcjonalny do $\sqrt{n_{\text{cluster\_papers}}}$, co pozwala odróżnić główny kierunek badawczy (duży punkt) od pobocznych zainteresowań (mały punkt).
- **Hover** — wyświetla: imię i nazwisko, numer klastra (np. „2/3"), zakład, liczbę prac w klastrze/łącznie, top-3 tematy, top-5 słów kluczowych.

### 8.3 Szczegóły techniczne

Wizualizacje generowane są jako standalone HTML za pomocą biblioteki **Plotly** (via CDN, bez konieczności lokalnej instalacji). Interaktywność (wyszukiwarka, highlight, togglee) realizowana jest za pomocą czystego JavaScriptu wstrzyknięgo w HTML poprzez `html_content.replace("</body>", search_bar_html + "\n</body>")`.

Wydajność: przy 115 punktach (lub ~298 cluster-points w wariancie multi-cluster przy progu 0.15) wizualizacja ładuje się natychmiastowo (<100ms). Nawet na urządzeniach mobilnych interakcja jest płynna, co stanowi przewagę skali autorów nad skalą artykułów (3440 punktów u Wujca).

---

## 9. Wyniki i obserwacje

### 9.1 Mean pooling daje stabilne, interpretowalne mapy

Wizualizacja mean pooling z modelem fine-tuned osiąga 54.7% variance explained w 2 komponentach PCA — znacząco więcej niż na artykułach ArXiv (12.2% dla modelu oryginalnego, 29.4% dla fine-tuned). Oznacza to, że mapa autorów jest znacznie bardziej „wiarygodna" niż mapa artykułów: ponad połowa informacji o strukturze badawczej jest uchwycona w widocznym układzie 2D.

Efekt ten wynika z działania mean poolingu jako filtru dolnoprzepustowego. Pojedyncze artykuły mogą mieć nietypowe tematy (praca przeglądowa, interdyscyplinarna kolaboracja, zmiana zainteresowań), ale średnia z 20–50 prac stabilnie wskazuje na główne pole badawcze autora.

### 9.2 Multi-cluster ujawnia interdyscyplinarność

Wariant multi-cluster identyfikuje 85% autorów (z ≥5 pracami) jako posiadających więcej niż jeden odrębny kierunek badawczy. Szczególnie interesujące przypadki obejmują autorów z Zakładu Sztucznej Inteligencji, którzy mogą mieć odrębne klastry dla prac z NLP, computer vision i robotyki — każdy klaster z własnym zestawem tematów i słów kluczowych.

Linie łączące klastry tego samego autora tworzą na mapie wzór „konstelacji" — autor interdyscyplinarny rozpina się na dużej powierzchni mapy, podczas gdy autor jednorodny tematycznie jest skoncentrowany w jednym punkcie.

### 9.3 Struktura zakładów jest częściowo odwzorowana

NMI = 0.702 (fine-tuned) potwierdza, że istnieje istotna zgodność między klasteryzacją K-means a podziałem na zakłady, ale nie jest to zgodność idealna. Zakłady o zbliżonej tematyce (analiza funkcjonalna ↔ analiza matematyczna, geometria algebraiczna ↔ geometria arytmetyczna) są praktycznie nieodróżnialne w przestrzeni embeddingów — co jest prawidłowe z perspektywy treści badawczej.

### 9.4 Fine-tuning: poprawa globalna kosztem lokalnej

Fine-tuning na hierarchicznych kategoriach ArXiv poprawia globalne uporządkowanie (NMI +0.030, variance explained +21.6pp na autorach), ale pogarsza lokalne metryki sąsiedztwa (NN@1 −0.065). Ten trade-off jest konsekwencją „ściskania" przestrzeni embeddingów — po fine-tuningu wszystkie wektory są bardziej do siebie podobne, co zwiększa szum w lokalnym sąsiedztwie.

W kontekście wizualizacji 2D ten trade-off jest akceptowalny: na mapie interesuje nas przede wszystkim globalna struktura (kto jest blisko kogo na dużej skali), a nie dokładna kolejność 3–5 najbliższych sąsiadów.

### 9.5 Ograniczenia i przyszłe prace

Zidentyfikowaliśmy kilka ograniczeń obecnego podejścia:

**Jakość abstraktów.** Choć prawie wszystkie prace w naszym zbiorze posiadają abstrakt (99.9%), jakość ta może się różnić — niektóre abstrakty w OpenAlex mogą być niekompletne lub automatycznie wyekstrahowane. Ponadto przy rozszerzeniu na inne źródła danych (np. starsze publikacje) problem brakujących abstraktów może stać się istotny.

**Mean pooling traci informację o chronologii.** Obecna metoda traktuje prace z 2005 i 2024 równoważnie. Autor, który zmienił zainteresowania badawcze na przestrzeni 20 lat, będzie miał uśredniony profil niereprezentujący ani starych, ani nowych zainteresowań. Potencjalne rozwiązanie: **temporal embeddings** z ważeniem wykładniczym (nowsze prace mają wyższą wagę) lub okno czasowe (tylko ostatnie 5–10 lat).

**Brak ważenia prac wg impactu.** Wszystkie prace mają równą wagę, niezależnie od tego, czy jest to artykuł w Nature z 500 cytowaniami, czy raport techniczny z 0 cytowań. **Citation-weighted pooling** mógłby poprawić jakość embeddingów, ale wymaga wiarygodnych danych o cytowaniach.

**Potencjalne rozszerzenia.** Poza wymienionymi, rozważamy: collaborative filtering (wykorzystanie sieci współautorów), rozszerzenie na inne wydziały lub uczelnie, oraz integrację z mechanizmem rekomendacji artykułów w SARA.

---

## 10. Wnioski

### 10.1 Podsumowanie wyników

| Co zrobiliśmy | Wynik |
|--------------|-------|
| Fine-tuning BGE-base z CoSENTLoss | Fuzzy purity 80.9% (k=8), poprawa +6.7pp |
| Mean pooling per autor + PCA 2D | 54.7% variance explained (vs 12.2% na artykułach) |
| Ewaluacja separacji zakładów | NMI 0.702, NN@1 0.663 |
| Multi-cluster z automatycznym k (silhouette) | 85% autorów z k>1, avg 2.59 klastrów/autor, 298 cluster-points |
| Interaktywna wizualizacja (Plotly HTML) | Wyszukiwarka, porównanie A vs B, toggle low-conf |

### 10.2 Porównanie z wynikami Wujca

Niniejsza praca stanowi uzupełnienie pracy Wujca o wizualizacji artykułów:

- **Wujec:** 3440 artykułów → 8 metod redukcji wymiarowości → 7 metryk jakości → composite score → wybór UMAP/PaCMAP jako najlepszych metod.
- **Paszke:** 115 autorów (z tych samych 3440 artykułów) → mean pooling + multi-cluster → PCA (wystarczający dzięki mean poolingowi) → ewaluacja na zakładach → interaktywna mapa.

Kluczowe synergies:

- Ten sam model embeddingowy (BGE-base-en-v1.5 + CoSENT fine-tuning) jest współdzielony między wizualizacją artykułów i autorów.
- Wniosek Wujca, że PCA ma niski composite score na artykułach (0.576), nie stosuje się do autorów, gdzie mean pooling skutecznie „upraszcza" przestrzeń i PCA wyjaśnia 54.7% wariancji.
- Optymalna liczba klastrów k=9 wyznaczona przez Wujca na artykułach koreluje z naszą liczbą k=5 na autorach (mniej autorów → mniej klastrów).

### 10.3 Rekomendacje dla produkcyjnej wersji SARA

Na podstawie przeprowadzonych eksperymentów formułujemy następujące rekomendacje:

1. **Model:** Używać fine-tuned BGE-base-en-v1.5 (CoSENT) jako domyślnego modelu embeddingowego zarówno dla artykułów, jak i autorów.
2. **Wizualizacja artykułów:** Stosować UMAP lub PaCMAP (zgodnie z rekomendacjami Wujca) dla mapy artykułów.
3. **Wizualizacja autorów:** Stosować PCA z mean poolingiem (wariant prosty) lub multi-cluster (wariant zaawansowany). PCA jest wystarczający dzięki wysokiemu variance explained po mean poolingu.
4. **Caching:** Embeddingi per-artykuł powinny być cache'owane (plik .npy), ponieważ stanowią wejście zarówno dla mapy artykułów, jak i mapy autorów. Regeneracja jest potrzebna jedynie przy aktualizacji zbioru prac.
5. **Aktualizacja:** Nowo dodane artykuły powinny automatycznie odświeżać embeddingowe profile odpowiednich autorów (incremental update centroidu).

---

## 11. Bibliografia

[1] W. B. Johnson, J. Lindenstrauss, "Extensions of Lipschitz mappings into a Hilbert space," *Contemp. Math.*, vol. 26, pp. 189–206, 1984.

[2] C. C. Aggarwal, A. Hinneburg, D. A. Keim, "On the surprising behavior of distance metrics in high dimensional space," *ICDT*, pp. 420–434, 2001.

[3] L. van der Maaten, G. Hinton, "Visualizing data using t-SNE," *JMLR*, vol. 9, pp. 2579–2605, 2008.

[4] L. McInnes, J. Healy, J. Melville, "UMAP: Uniform Manifold Approximation and Projection for dimension reduction," *arXiv:1802.03426*, 2018.

[5] Y. Wang, H. Huang, C. Rudin, Y. Shaposhnik, "Understanding how dimension reduction tools work: an empirical approach to deciphering t-SNE, UMAP, TriMap, and PaCMAP," *JMLR*, vol. 22, no. 201, pp. 1–73, 2021.

[6] D. Kobak, G. C. Linderman, "Initialization is critical for preserving global data structure in both t-SNE and UMAP," *Nature Biotechnology*, vol. 39, pp. 156–157, 2021.

[7] N. Reimers, I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," *Proceedings of EMNLP-IJCNLP*, pp. 3982–3992, 2019.

[8] Z. Xiao, "C-Pack: Packaged Resources To Advance General Chinese Embedding," *arXiv:2309.07597*, 2023. (BGE models)

[9] K. Li, "CoSENT: A more efficient sentence vector scheme than Sentence-BERT," 2022. (Blogpost / implementation reference)

[10] P. J. Rousseeuw, "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis," *J. Comp. Appl. Math.*, vol. 20, pp. 53–65, 1987.

[11] J. B. Tenenbaum, V. de Silva, J. C. Langford, "A global geometric framework for nonlinear dimensionality reduction," *Science*, vol. 290, pp. 2319–2323, 2000.

[12] J. B. Kruskal, "Multidimensional scaling by optimizing goodness of fit to a nonmetric hypothesis," *Psychometrika*, vol. 29, pp. 1–27, 1964.

[13] S. P. Lloyd, "Least squares quantization in PCM," *IEEE Trans. Information Theory*, vol. 28, no. 2, pp. 129–137, 1982. (K-Means)

[14] J. Priem, H. Piwowar, R. Orr, "OpenAlex: A fully-open index of scholarly works, authors, venues, institutions, and concepts," *arXiv:2205.01833*, 2022.

[15] Plotly Technologies Inc., "Collaborative data science," Montréal, QC, 2015. https://plot.ly

[16] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *JMLR*, vol. 12, pp. 2825–2830, 2011.

---

## 12. Dodatek

### A. Pełna tabela per-department (z wmi_report.md)

Pełna tabela wyników ewaluacji per zakład przedstawiona jest w sekcji 6.5. Dane wygenerowane skryptem `evaluate_dept_separation.py` na podstawie embeddingów 92 autorów z konkretnym przypisaniem zakładowym.

### B. Pełne wyniki 9 modeli embeddingów

Tabela w sekcji 2.2 przedstawia skrócone wyniki porównania modeli. Pełne wyniki (wszystkie metryki dla wszystkich wartości k) dostępne są w pliku `ArXiv/results/finetuned_comparison/comparison.json`.

### C. Listing kluczowych fragmentów kodu

#### C.1 `cluster_author_papers()` — klasteryzacja prac autora

```python
def cluster_author_papers(author_embeddings):
    """Cluster an author's papers into research areas.
    Returns (k, labels, centroids)."""
    n = len(author_embeddings)

    if n <= 1:
        return 1, np.array([0]), author_embeddings.copy()

    if n < MIN_PAPERS_FOR_CLUSTERING:
        centroid = author_embeddings.mean(axis=0, keepdims=True)
        centroid = centroid / np.linalg.norm(centroid, axis=1, keepdims=True)
        return 1, np.zeros(n, dtype=int), centroid

    max_k = min(MAX_CLUSTERS, n - 1)
    if max_k < 2:
        centroid = author_embeddings.mean(axis=0, keepdims=True)
        centroid = centroid / np.linalg.norm(centroid, axis=1, keepdims=True)
        return 1, np.zeros(n, dtype=int), centroid

    best_k = 1
    best_sil = -1
    best_labels = None

    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(author_embeddings)
        if len(set(labels)) < 2:
            continue
        sil = silhouette_score(author_embeddings, labels)
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_labels = labels

    if best_sil < SILHOUETTE_THRESHOLD or best_labels is None:
        centroid = author_embeddings.mean(axis=0, keepdims=True)
        centroid = centroid / np.linalg.norm(centroid, axis=1, keepdims=True)
        return 1, np.zeros(n, dtype=int), centroid

    centroids = []
    for c in range(best_k):
        mask = best_labels == c
        centroid = author_embeddings[mask].mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        centroids.append(centroid)

    return best_k, best_labels, np.array(centroids)
```

#### C.2 `get_cluster_label()` — generowanie etykiet per klaster

```python
def get_cluster_label(paper_indices):
    """Extract top topics and keywords for papers in a cluster."""
    all_topics = []
    all_kw = []
    for idx in paper_indices:
        row = titles_df.iloc[idx]
        if pd.notna(row.get("topics")):
            all_topics.extend([t.strip() for t in str(row["topics"]).split(";")])
        if pd.notna(row.get("keywords")):
            all_kw.extend([k.strip() for k in str(row["keywords"]).split(";")])

    top_topics = [t for t, _ in Counter(all_topics).most_common(3)]
    top_kw = [k for k, _ in Counter(all_kw).most_common(5)]
    return top_topics, top_kw
```

#### C.3 `embed_and_pool()` — generowanie embeddingów i mean pooling

```python
def embed_and_pool(model_path, texts, paper_orcids, author_orcids):
    """Generate per-paper embeddings, then mean-pool per author."""
    model = SentenceTransformer(model_path)
    embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    author_embeddings = []
    for orcid in author_orcids:
        mask = paper_orcids == orcid
        author_emb = embeddings[mask].mean(axis=0)
        author_emb = author_emb / np.linalg.norm(author_emb)
        author_embeddings.append(author_emb)
    return np.array(author_embeddings)
```
