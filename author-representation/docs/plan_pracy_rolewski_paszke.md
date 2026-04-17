# Plan pracy magisterskiej

## Adaptacyjna reprezentacja naukowców w systemie SARA:
## od stabilnego centroidu do reprezentacji wieloklastrowej

**Miłosz Rolewski**

**Jakub Paszke**

---

## Rozdział 1 — Wprowadzenie, kontekst systemu SARA i pipeline danych (wspólny)

Rozdział stanowi punkt wyjścia całej pracy i wprowadza czytelnika w problematykę semantycznej eksploracji dorobku naukowego pracowników Wydziału Matematyki i Informatyki UAM. Punktem wyjścia jest obserwacja, że pojedynczy naukowiec nie zawsze daje się wiarygodnie opisać jednym punktem w przestrzeni embeddingowej: w części przypadków taki centroid jest stabilny i interpretowalny, a w części prowadzi do utraty informacji o wielotorowości zainteresowań badawczych. To napięcie motywuje wspólny cel pracy: zbudowanie schematu, który najpierw ocenia, czy profil badacza może być reprezentowany pojedynczym centroidem, a następnie — jeśli to konieczne — rozbija go na kilka subprofilów tematycznych.

W rozdziale opisane zostaną źródła danych i wspólny korpus wejściowy wykorzystywany w obu częściach pracy. Obejmuje on profile pracowników WMI pozyskane z Portalu Badawczego UAM oraz publikacje pobrane z OpenAlex na podstawie identyfikatorów ORCID. W aktualnym repozytorium odpowiadają temu przede wszystkim pliki `data/scientists_with_identifiers.csv`, `data/titles_with_abstracts.csv` oraz `data/openalex_all_results_complete.csv`. Na tym etapie zostaną przedstawione podstawowe statystyki zbioru, charakterystyka metadanych publikacji oraz procedura łączenia danych o autorach z danymi o pracach.

Wspólną warstwą metodologiczną dla obu autorów jest reprezentacja pracy naukowej za pomocą embeddingów tekstowych. W rozdziale należy opisać, że tekst wejściowy budowany jest z tytułu, abstraktu albo ich wariantów rozszerzonych, a wektory są normalizowane L2, dzięki czemu podobieństwo kosinusowe staje się naturalną miarą porównawczą zarówno dla artykułów, jak i dla profili autorów. To właśnie z tych embeddingów budowane są później profile naukowców: w części Rolewskiego jako centroidy oceniane pod kątem stabilności, a w części Paszkego jako obiekty do dalszej klasteryzacji per autor.

Rozdział powinien zakończyć się formalnym podziałem odpowiedzialności oraz krótkim opisem, jak obie części łączą się w jeden pipeline badawczy:

- część Rolewskiego odpowiada za diagnozę, czy pojedynczy centroid autora jest reprezentacją stabilną i wiarygodną,
- część Paszkego odpowiada za budowę reprezentacji końcowej: jednopunktowej albo wielopunktowej,
- wspólna część wynikowa analizuje, czy adaptacyjny wybór `1 punkt` vs `N punktów` poprawia użyteczność całego systemu SARA.

**[PLACEHOLDER J. Paszke]** Dopisać syntetyczny opis końcowego modelu embeddingowego używanego w wizualizacji autorów po fine-tuningu oraz krótko uzasadnić, dlaczego przestrzeń fine-tuned stanowi docelową przestrzeń dla finalnej reprezentacji autora.

---

## Rozdział 2 — Profilowanie badaczy i diagnoza wiarygodności centroidu (Miłosz)

Rozdział obejmuje pełną część poświęconą profilowaniu badaczy na podstawie ich publikacji oraz analizie zachowania takich profili pod wpływem wyboru modelu embeddingowego i typu tekstu wejściowego. W punkcie wyjścia profil naukowca definiowany jest jako centroid — średnia arytmetyczna embeddingów jego publikacji. Taka reprezentacja jest stałowymiarowa, łatwa do aktualizacji, obliczeniowo tania i interpretowalna geometrycznie, ale jej użyteczność zależy od tego, czy rzeczywiście streszcza dorobek autora, czy też zaciera jego wewnętrzną wielowymiarowość.

W rozdziale należy przedstawić porównanie sześciu modeli embeddingowych analizowanych w eksperymentach: `SPECTER`, `MiniLM-L6`, `MiniLM-L12`, `MPNet-Base`, `BGE-Small` oraz `Multilingual-MiniLM`. Modele 768-wymiarowe są sprowadzane do wspólnego wymiaru 384 za pomocą projekcji Johnsona-Lindenstraussa, co umożliwia bezpośrednie porównania i konkatenacje reprezentacji. Należy także opisać cztery warianty tekstu wejściowego i reprezentacji pracy: embedding samego tytułu, embedding abstraktu, embedding oparty na N-gramach zdań z abstraktu oraz warianty konkatenacyjne łączące reprezentacje różnych modeli.

Ta część pracy bazuje bezpośrednio na notebookach eksploracyjnych i porównawczych:

- `notebooks/generate_work_embeddings.ipynb` — generowanie embeddingów prac,
- `notebooks/embedding_comparison_experiment.ipynb` — porównanie modeli i typów tekstu,
- `notebooks/scientist_profile_comparison.ipynb` — porównanie konfiguracji profili naukowców,
- `notebooks/concatenate_embedings_experiment.ipynb` — eksperymenty z konkatenacją reprezentacji,
- `notebooks/pubs_num_stability.ipynb` — analiza stabilności centroidu względem liczby prac.

Rozdział powinien następnie przejść do metod walidacji profilu badacza. Pierwszym krokiem jest walidacja hipotezy, że embeddingi publikacji rzeczywiście kodują podobieństwo naukowe. W tym celu wykorzystuje się test bliskości współautorów: profile autorów, którzy współtworzyli publikacje, powinny być do siebie bardziej podobne niż profile losowo dobranych badaczy. W pracy Rolewskiego ten test wykazał silną separację dla dwóch analizowanych modeli bazowych i potwierdził, że reprezentacje tekstowe są sensowną podstawą systemu rekomendacji naukowców.

Drugą osią analizy jest zgodność sąsiedztw generowanych przez różne konfiguracje modeli i typów tekstu. Miara Jaccarda dla zbiorów top-`k` sąsiadów pozwala ocenić, czy dany autor jest opisywany spójnie w różnych przestrzeniach embeddingowych. Wysoka zgodność oznacza stabilny profil semantyczny, niski Jaccard sugeruje, że reprezentacja autora zależy silnie od wybranego modelu lub typu tekstu, co może wskazywać na wielowymiarowość zainteresowań badawczych albo na niejednorodność dorobku.

Trzecim, kluczowym elementem rozdziału jest analiza stabilności centroidu względem liczby publikacji. Dla wybranego autora oraz dla większego zbioru naukowców losowane są podzbiory prac o rozmiarze `k`, budowane są centroidy cząstkowe i porównywane z centroidem pełnym za pomocą podobieństwa kosinusowego. Ta część daje odpowiedź na pytanie praktyczne: przy ilu publikacjach pojedynczy centroid zaczyna być wiarygodny. Wyniki pokazują, że sama liczba prac nie jest wystarczająca do oceny jakości profilu — równie ważna okazuje się interdyscyplinarność autora.

Ważnym wnioskiem tej części jest zatem nie tyle wybór jednego „najlepszego” modelu, ile zbudowanie warstwy diagnostycznej dla dalszych etapów systemu. Rozdział powinien zakończyć się propozycją zestawu cech opisujących autora z perspektywy wiarygodności pojedynczego centroidu, np.:

- liczba publikacji,
- stabilność centroidu dla małych próbek (`sim(centroid_k, centroid_all)`),
- średnia zgodność Jaccarda między konfiguracjami,
- rozproszenie embeddingów prac autora,
- sygnały wskazujące na interdyscyplinarność.

Właśnie te cechy stanowią podstawę do decyzji, czy autora należy reprezentować jako:

- `SINGLE` — jeden stabilny centroid,
- `MULTI` — kilka subprofilów tematycznych,
- `LOW_CONF` — jeden punkt, ale oznaczony jako niskiej wiarygodności,
- `AMBIGUOUS` — przypadek graniczny wymagający dalszej analizy.

Rozdział 2 ma więc zakończyć się nie tylko wnioskami o zachowaniu modeli, ale także formalnym przekazaniem sygnału decyzyjnego do części Paszkego: **czy pojedynczy punkt per autor jest wystarczający, czy też należy uruchomić procedurę dekompozycji profilu na wiele punktów**.

---

## Rozdział 3 — Reprezentacja autora jako 1 punkt lub N punktów oraz wizualizacja multi-cluster (Jakub)

Rozdział opisze docelową metodę reprezentacji autora w systemie SARA. Jego głównym zadaniem będzie pokazanie, w jaki sposób autor może być reprezentowany albo jako pojedynczy punkt uzyskany przez mean pooling, albo jako kilka punktów odpowiadających różnym kierunkom badawczym.

**[PLACEHOLDER J. Paszke]** Opisać wariant jednopunktowy oparty na mean poolingu embeddingów prac autora w docelowej przestrzeni fine-tuned.

**[PLACEHOLDER J. Paszke]** Opisać algorytm rozbijania autora na wiele punktów (`multi-cluster`), w tym:

- klasteryzację embeddingów prac autora,
- automatyczny dobór liczby klastrów `k`,
- wykorzystanie silhouette score,
- obsługę przypadków brzegowych (`n < 5`, `k = 1`, brak struktury klastrowej).

**[PLACEHOLDER J. Paszke]** Opisać sposób etykietowania klastrów na podstawie `topics` i `keywords` z OpenAlex oraz sposób prezentacji wyników w interaktywnej wizualizacji.

**[PLACEHOLDER J. Paszke]** Dopisać część dotyczącą fine-tuningu modelu embeddingowego na danych ArXiv, uzasadnienie wyboru finalnego modelu oraz jego wpływ na jakość wizualizacji autorów.

**[PLACEHOLDER J. Paszke]** Dopisać ewaluację wariantu jednopunktowego i wielopunktowego względem struktury zakładów/pracowni, variance explained w PCA oraz inne metryki jakości końcowej reprezentacji.

Rozdział 3 powinien jednak od początku jasno zaznaczać, że procedura rozbijania autora na `N` punktów nie działa w próżni: jest ona uruchamiana właśnie wtedy, gdy część diagnostyczna z rozdziału 2 wskazuje, że pojedynczy centroid jest niewystarczający lub niestabilny.

---

## Rozdział 4 — Integracja obu podejść i wyniki systemowe (wspólny)

Rozdział stanowi właściwe spoiwo całej pracy. Jego celem jest przedstawienie wspólnego schematu systemowego, w którym część Rolewskiego i część Paszkego nie są dwiema niezależnymi liniami badań, lecz kolejnymi etapami jednego pipeline’u reprezentacji autora.

Proponowany schemat integracji jest następujący:

1. Z danych wejściowych (`scientists_with_identifiers.csv`, `titles_with_abstracts.csv`, `openalex_all_results_complete.csv`) budowane są embeddingi prac.
2. Moduł Rolewskiego oblicza profile centroidowe i ich metryki diagnostyczne: stabilność względem liczby prac, zgodność Jaccarda między konfiguracjami, rozproszenie reprezentacji oraz wskaźniki sugerujące interdyscyplinarność.
3. Na podstawie tych metryk każdemu autorowi przypisywana jest polityka reprezentacji (`SINGLE`, `MULTI`, `LOW_CONF`, `AMBIGUOUS`).
4. Moduł Paszkego korzysta z tej polityki i buduje finalną reprezentację:
   - `SINGLE` -> autor pozostaje jednym punktem,
   - `LOW_CONF` -> autor pozostaje jednym punktem z oznaczeniem niskiej wiarygodności,
   - `MULTI` -> uruchamiana jest dekompozycja na kilka klastrów tematycznych,
   - `AMBIGUOUS` -> stosowany jest wariant zachowawczy lub wariant rozwijalny w interfejsie.
5. Wynik końcowy trafia do wspólnej wizualizacji autora oraz może być powiązany z warstwą artykułową systemu SARA jako moduł explainability.

W praktyce rozdział powinien zaproponować wspólny artefakt pośredni, np. plik `author_representation_policy.csv` lub jego odpowiednik, zawierający dla każdego autora:

- identyfikator ORCID,
- liczbę prac,
- stabilność centroidu dla wybranych progów `k`,
- średnią zgodność Jaccarda,
- miarę rozproszenia / interdyscyplinarności,
- decyzję o trybie reprezentacji,
- uzasadnienie decyzji.

To właśnie ten rozdział odpowiada na główne pytanie integracyjne: **kiedy naukowca reprezentować jako 1 kropkę, a kiedy jako N kropek**. Część Rolewskiego dostarcza przesłanek decyzyjnych, a część Paszkego dostarcza mechanizmu konstrukcyjnego. Dzięki temu system nie rozbija każdego autora automatycznie, lecz robi to tylko tam, gdzie rzeczywiście istnieją sygnały wielomodalności profilu badawczego.

W warstwie eksperymentalnej rozdział może zestawiać co najmniej trzy warianty reprezentacji:

- baseline: wszyscy autorzy jako jeden centroid,
- wariant agresywny: wszyscy autorzy z wystarczającą liczbą prac jako kandydaci do multi-cluster,
- wariant adaptacyjny: decyzja `1 punkt` vs `N punktów` podejmowana na podstawie modułu diagnostycznego.

Takie porównanie ma bezpośrednią wartość dla systemu SARA, bo pozwala ocenić, czy adaptacyjna polityka reprezentacji poprawia interpretowalność mapy, separację struktur organizacyjnych oraz jakość rekomendacji potencjalnych współpracowników.

**[PLACEHOLDER J. Paszke]** Uzupełnić liczby i wyniki dotyczące końcowej jakości reprezentacji w przestrzeni fine-tuned, w tym liczbę autorów rozbitych na wiele klastrów, liczbę cluster-points oraz wpływ tej decyzji na końcową wizualizację.

**[PLACEHOLDER wspólny]** Dobrać ostateczne progi przejścia `SINGLE -> MULTI` na podstawie walidacji na danych WMI oraz uzasadnić ich wybór metodologicznie.

---

## Rozdział 5 — Podsumowanie i wnioski końcowe (wspólny)

Rozdział syntetyzuje wyniki obu części pracy i odpowiada na centralne pytanie badawcze: czy naukowca można wiarygodnie reprezentować pojedynczym punktem w przestrzeni embeddingowej, czy też należy traktować go jako obiekt potencjalnie wielomodalny. Kluczowym wynikiem całej pracy powinno być odejście od sztywnej reguły „jeden autor = jeden punkt” na rzecz reprezentacji adaptacyjnej, zależnej od stabilności i wewnętrznej struktury dorobku.

W części wnioskowej należy podkreślić kilka spodziewanych tez:

- embeddingi tekstowe są sensowną podstawą profilowania badaczy i rekomendacji naukowców,
- pojedynczy centroid dobrze działa dla autorów o stabilnym i jednorodnym profilu,
- liczba publikacji sama w sobie nie wystarcza do oceny jakości profilu,
- interdyscyplinarność jest niezależnym sygnałem wskazującym na potrzebę reprezentacji wielopunktowej,
- rozbicie autora na kilka punktów powinno być sterowane diagnozą stabilności, a nie wyłącznie lokalnym kryterium klasteryzacyjnym.

Rozdział może również wskazać dalsze kierunki badań:

- ewaluację ekspercką jakości rekomendacji,
- fine-tuning modeli embeddingowych bezpośrednio na danych UAM,
- integrację reprezentacji autora z mapą artykułów i warstwą explainability,
- uwzględnienie grafu cytowań i współautorstwa,
- budowę interfejsu, który pozwala płynnie przechodzić od widoku autora do widoku klastrów i powiązanych publikacji.

**[PLACEHOLDER J. Paszke]** Dopisać końcowe wnioski dotyczące jakości wariantu multi-cluster i jego praktycznej przewagi nad samym mean poolingiem.

---

Plan ma charakter roboczy i może ulec zmianie w trakcie pisania pracy. Rozdział 2 jest wypełniony na podstawie dotychczasowych materiałów Rolewskiego (`SARA_praca_magisterska_rolewski.pdf` oraz notebooki eksploracyjne), rozdział 3 zawiera miejsca do uzupełnienia przez Paszkego, a rozdziały 1, 4 i 5 stanowią wspólny szkielet integrujący oba podejścia w ramach jednego systemu SARA.
