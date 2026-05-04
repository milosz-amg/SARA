# Plan pracy magisterskiej

## Adaptacyjna reprezentacja naukowców w systemie SARA: od diagnozy stabilności centroidu do wielopunktowej wizualizacji profili

**Miłosz Rolewski**  &nbsp;&nbsp;&nbsp;  **Jakub Paszke**

---

## Rozdział 1 — Wprowadzenie i pozyskiwanie danych (wspólny)

Rozdział stanowi punkt wyjścia całej pracy i wprowadza czytelnika w problematykę semantycznej reprezentacji **naukowców** (a nie pojedynczych publikacji) w systemie rekomendacji SARA. Omówiona jest motywacja praktyczna: aby polecać potencjalnych współpracowników lub ekspertów, system potrzebuje zwartej reprezentacji profilu badawczego każdego autora. Najprostszym podejściem jest centroid — uśredniony wektor embeddingów wszystkich prac danej osoby — ale działa on tylko dla autorów o stabilnym, jednorodnym profilu. Dla osób o szerokim dorobku lub pracujących interdyscyplinarnie centroid trafia w pustą przestrzeń „pomiędzy" obszarami badawczymi.

Opisane są źródła danych: $N = 3440$ publikacji 115 pracowników WMI UAM pozyskanych z Portalu Badawczego UAM oraz OpenAlex API na podstawie ORCID. Omówiony jest pipeline zbierania danych (scraper profili, ekstrakcja identyfikatorów, pobieranie publikacji z OpenAlex, uzupełnianie brakujących abstraktów). Następnie przedstawiony jest proces generowania embeddingów tekstowych: wybór tekstu wejściowego (tytuł, abstrakt, warianty rozszerzone) oraz normalizacja $L^2$, dzięki której odległość euklidesowa staje się monotoniczna względem podobieństwa kosinusowego. Wprowadzona jest notacja: $e_p \in \mathbb{R}^d$ to embedding pracy $p$, $\hat{e}_a = \mathrm{normalize}(\frac{1}{n_a}\sum_{p \in P_a} e_p)$ to centroid autora.

Już tutaj sygnalizowane jest napięcie badawcze motywujące strukturę pracy: centroid jest stałowymiarowy i interpretowalny, ale może niewiernie opisywać wielotorowych badaczy. Rozdział 2 (Rolewski) odpowiada za **diagnozę**, czy centroid jest wiarygodny; rozdział 3 (Paszke) — za **konstrukcję** reprezentacji adaptacyjnej (1 punkt lub N punktów). Na końcu rozdziału 1 znajduje się formalny podział zadań oraz opis wspólnego artefaktu pośredniego `author_representation_policy.csv` łączącego obie warstwy (wprowadzonego w rozdziale 3, wykorzystanego w eksperymentach w rozdziale 4).

---

## Rozdział 2 — Profilowanie badaczy i diagnoza wiarygodności centroidu (Miłosz)

Rozdział obejmuje pełną część poświęconą **ocenie jakości centroidowej reprezentacji autora**. Punktem wyjścia jest definicja profilu $\hat{e}_a = \mathrm{normalize}(\mathrm{mean}(\{e_p : p \in P_a\}))$. Omówione jest porównanie sześciu modeli embeddingowych (SPECTER, MiniLM-L6, MiniLM-L12, MPNet-Base, BGE-Small, Multilingual-MiniLM) oraz czterech wariantów tekstu wejściowego (tytuł, abstrakt, N-gramy zdań, warianty konkatenacyjne). Modele 768-wymiarowe sprowadzane są do wspólnego wymiaru 384 przez projekcję Johnsona–Lindenstraussa, umożliwiając bezpośrednie porównania i konkatenacje.

Przedstawione są trzy osie walidacji profilu: (1) **test bliskości współautorów** — czy współautorzy mają istotnie bliższe centroidy niż pary losowe (test U Manna–Whitneya); (2) **zgodność sąsiedztw top-$k$** — miara Jaccarda między konfiguracjami model × typ tekstu, mierząca stabilność semantyczną profilu; (3) **stabilność centroidu względem liczby prac** — średnie podobieństwo kosinusowe między centroidem zbudowanym z losowego podzbioru $k$ prac a centroidem pełnym, dla $k \in \{3, 5, 10, 20, 50, 100\}$. Wyniki pokazują, że stabilność osiąga ~0.95 dla $k=5$ i ~0.99 dla $k=10$ dla większości modeli, ale rozkład jest heterogeniczny — sama liczba prac nie wystarcza do oceny wiarygodności.

Rozdział wprowadza dodatkowo **miarę rozproszenia embeddingów prac autora** jako sygnał interdyscyplinarności niezależny od liczby prac. Kończy się propozycją zestawu cech diagnostycznych (`n_papers`, `stability@k`, `avg_jaccard`, `dispersion`), które stanowią podstawę decyzji w rozdziale 3 o trybie reprezentacji autora.

---

## Rozdział 3 — Reprezentacja autora i wizualizacja multi-cluster (Jakub)

Rozdział obejmuje **konstrukcję reprezentacji końcowej**. Zaczyna się od uzasadnienia wyboru docelowego modelu embeddingowego — `BAAI/bge-base-en-v1.5` zwyciężył porównanie dziewięciu modeli na zbiorze 2397 artykułów ArXiv (main-category purity 70.55% dla $k=118$). Model został **dostrojony** z wykorzystaniem funkcji straty CoSENT i hierarchicznej odległości kategorii ArXiv (3 poziomy: podkategoria → archiwum → grupa) z rozmytymi multi-label. Fine-tuning poprawia purity o 4.74 pp, fuzzy purity o 6.68 pp oraz variance explained w PCA 2D z 12.19% do 29.37% na ArXiv; na autorach WMI variance explained rośnie z 33.11% do 54.71%, co czyni mapę 2D wiarygodną.

Omówione są dwa warianty reprezentacji. **Wariant jednopunktowy** (mean pooling) definiowany jest matematycznie wraz z dyskusją własności (stałowymiarowość, utrata informacji o wielomodalności, rola normalizacji L2). **Wariant wielopunktowy** (multi-cluster) opiera się na K-means z **automatycznym doborem $k$** via silhouette score. Pseudokod algorytmu opisuje testowanie $k \in \{2, ..., 5\}$ dla autorów z $n \geq 5$ pracami, wybór $k^\star$ maksymalizującego silhouette oraz fallback do jednego klastra gdy $\max_k \text{silhouette} < 0{,}15$. Dobór progu jest uzasadniony empirycznie tabelą rozkładu decyzji dla progów 0.05–0.30. Rozdział pokazuje przykłady autorów wielokierunkowych (np. 4 klastry: decision making, robotics, fuzzy optimization, ovarian cancer) oraz mechanizm etykietowania klastrów top-topics + top-keywords z OpenAlex.

Rozdział zawiera **wspólną propozycję metodologiczną obu autorów** — schemat decyzyjny `SINGLE / MULTI / LOW_CONF / AMBIGUOUS` oraz artefakt `author_representation_policy.csv`, łączący cechy diagnostyczne z rozdziału 2 (`stability@k`, `avg_jaccard`, `dispersion`) z cechami strukturalnymi z rozdziału 3 (`best_k`, `best_silhouette`). Decyzja per autor zależy od łącznego rozpatrzenia stabilności centroidu i wyraźności struktury klastrowej — autor może dostać tryb `MULTI` tylko jeśli zarówno jego centroid jest stabilny ($\geq 0{,}95$), jak i silhouette podziału jest istotny ($\geq 0{,}15$). Rozdział kończy się **ewaluacją** obu wariantów na niezależnym zadaniu separacji 14 zakładów WMI (NMI, NN@1, per-zakład intra/inter similarity) oraz opisem interaktywnej wizualizacji (Plotly HTML z wyszukiwarką, porównaniem A vs B i toggle cluster links).

---

## Rozdział 4 — Integracja, eksperymenty i wyniki systemowe (wspólny)

Rozdział otwiera sekcja konfiguracji eksperymentów opisująca wspólne ustawienia zapewniające powtarzalność wyników: zbiór danych ($N=3440$), wymiar embeddingów (768), miara odległości kosinusowa oraz progi decyzyjne schematu policy. Pierwszy krok — **konstrukcja artefaktu `author_representation_policy.csv`** dla wszystkich 115 autorów WMI: dla każdego obliczane są cechy z rozdziału 2 i 3, następnie schemat decyzyjny przypisuje etykietę `SINGLE / MULTI / LOW_CONF / AMBIGUOUS`. Rozdział przedstawia rozkład decyzji oraz przykładowych autorów z każdej kategorii.

Kluczowym eksperymentem rozdziału jest porównanie **trzech wariantów reprezentacji**: (1) baseline — wszyscy autorzy jako pojedynczy centroid; (2) aggressive — wszyscy z $\geq 5$ pracami uruchamiają multi-cluster; (3) adaptive — decyzja per autor na podstawie policy. Dla każdego wariantu raportowane są: jakość mapy (variance explained, NMI vs zakłady, silhouette), lokalne sąsiedztwo (NN@1, NN@5), interpretowalność (odsetek klastrów z sensownymi topics) oraz kompletność (liczba punktów na mapie). Hipoteza: wariant adaptive daje lepszy kompromis między interpretowalnością a kompletnością niż sztywne reguły. Dodatkowo rozdział weryfikuje hipotezę, że fine-tuning z rozdziału 3 poprawia nie tylko klastrowanie, ale i stabilność centroidu z rozdziału 2, przez uruchomienie procedury stability analysis na fine-tuned przestrzeni jako siódmym modelu.

Rozdział kończy się tabelą zbiorczą (trzy warianty × metryki jakości) oraz tabelą przewodnika — dla jakich zastosowań w systemie SARA rekomendowany jest który wariant (mapa eksploracyjna, rekomendacja eksperta, prezentacja administracyjna).

---

## Rozdział 5 — Podsumowanie i wnioski końcowe (wspólny)

Rozdział syntetyzuje wyniki obu rozdziałów tematycznych i odpowiada na centralne pytanie pracy: **czy naukowca można wiarygodnie reprezentować pojedynczym punktem w przestrzeni embeddingowej, czy należy traktować go jako obiekt potencjalnie wielomodalny?** Kluczowym wynikiem pracy jest odejście od sztywnej reguły „jeden autor = jeden punkt" na rzecz **reprezentacji adaptacyjnej**, w której decyzja o liczbie punktów jest podejmowana per autor na podstawie łącznego sygnału diagnostycznego (rozdział 2) i strukturalnego (rozdział 3) zebranego w schemacie policy.

Rozdział podsumowuje wnioski szczegółowe — embeddingi tekstowe są sensowną podstawą profilowania, pojedynczy centroid działa dla stabilnych i jednorodnych profili, liczba publikacji nie wystarcza do oceny jakości profilu, interdyscyplinarność jest niezależnym sygnałem, fine-tuning modelu poprawia zarówno jakość klastrowania jak i stabilność centroidów — oraz wskazuje dalsze kierunki badań (ewaluacja ekspercka, fine-tuning bezpośrednio na WMI, graf cytowań, temporal embeddings, interfejs explainability).

---

*Plan ma charakter roboczy i może ulec zmianie w trakcie pisania pracy. Rozdziały 2 i 3 są rozłączne autorsko; rozdziały 1, 4 i 5 są wspólne. Schemat decyzyjny `SINGLE / MULTI / LOW_CONF / AMBIGUOUS` oraz artefakt `author_representation_policy.csv` (podsekcja w rozdziale 3) są wspólną propozycją metodologiczną obu autorów.*
