# Embeddingi Autorów dla Systemu Rekomendacji Naukowców: Porównanie Modeli SPECTER i MiniLM

**Jakub Paszke, Miłosz Rolewski**

---

## Abstract

W niniejszym artykule badamy skuteczność embeddingów tekstowych do reprezentacji profili naukowych autorów w kontekście systemu rekomendacji potencjalnych współpracowników. Każdego autora reprezentujemy jako pojedynczy wektor - agregat embeddingów jego publikacji - a podobieństwo między autorami mierzymy kosinusową miarą odległości. Porównujemy dwa modele transformerowe: SPECTER (Cohan i in., 2020), specjalizowany w tekstach naukowych, oraz MiniLM (Reimers i Gurevych, 2019), model ogólnego przeznaczenia o zredukowanej wymiarowości. Jako kryterium walidacji stosujemy test bliskości współautorów: jeśli dwie osoby współtworzyły co najmniej jedną publikację, zakładamy, że ich profile semantyczne powinny być bliższe niż przypadkowe pary naukowców. Oba modele wykazują statystycznie istotną separację (p < 10⁻¹⁶), przy czym MiniLM osiąga większy efekt (d = 2.19 vs d = 1.51 dla SPECTER), pomimo o połowę mniejszej wymiarowości przestrzeni wektorowej.

---

## 1. Wstęp

Automatyczne wyszukiwanie podobnych naukowców na podstawie ich dorobku publikacyjnego stanowi problem o praktycznym znaczeniu dla nowoczesnych systemów zarządzania wiedzą akademicką. Potencjalne zastosowania obejmują rekomendację potencjalnych współpracowników, identyfikację ekspertów dla recenzji projektów grantowych oraz analizę struktury społeczności naukowej. Klasyczne podejścia oparte na słowach kluczowych lub kategoriach tematycznych cechują się ograniczoną zdolnością do uchwycenia semantycznego pokrewieństwa obszarów badań.

Modele transformerowe, takie jak BERT (Devlin i in., 2019) i ich warianty, umożliwiają tworzenie gęstych reprezentacji wektorowych tekstów, które kodują semantykę w geometrii przestrzeni wektorowej. Metoda Sentence-BERT (Reimers i Gurevych, 2019) rozszerza tę ideę na zdania i dokumenty, umożliwiając efektywne obliczanie podobieństwa kosinusowego między dowolnymi tekstami. Model SPECTER (Cohan i in., 2020) adaptuje to podejście do specyfiki tekstów naukowych, wykorzystując strukturę grafu cytowań jako sygnał nadzoru.

Niniejsza praca jest częścią projektu SARA realizowanego na Wydziale Matematyki i Informatyki UAM. Celem bezpośrednim jest empiryczne porównanie dwóch modeli embeddingowych i weryfikacja, czy reprezentacje oparte wyłącznie na tekstach publikacji są wystarczające do wiarygodnego profilowania autorów.

---



## 2. Architektura Modeli

Oba porównywane modele należą do rodziny enkoderów transformerowych opartych na architekturze BERT (Devlin i in., 2019). Wspólnym fundamentem jest mechanizm wielogłowicowej uwagi własnej (*multi-head self-attention*): każdy token sekwencji wejściowej rzutowany jest na trzy macierze - zapytań (Q), kluczy (K) i wartości (V) - a wynik warstwy uwagi dla głowicy *i* definiowany jest jako:

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```

gdzie d_k oznacza wymiar kluczy. Wyjścia wszystkich głowic są konkatenowane i liniowo transformowane. Każda warstwa transformerowa zawiera ponadto dwuwarstwową sieć neuronową z przejściową aktywacją GELU (*feed-forward sublayer*), a obie podwarstwy otoczone są połączeniami resztkowymi i normalizacją warstwy (*layer normalization*). Modele różnią się jednak głębokością sieci, procedurą pre-treningu i celem optymalizacji reprezentacji zdaniowej.

### 2.1 SPECTER

SPECTER (*Scientific Paper Embeddings using Citation-informed TransformERs*) to model zaproponowany przez Cohana i in. (2020), zbudowany na bazie SciBERT - wariantu BERT wstępnie wytrenowanego przez Beltagy'ego i in. (2019) na korpusie 1,14 miliona pełnotekstowych artykułów naukowych z bazy Semantic Scholar. SciBERT, podobnie jak oryginalny BERT, korzysta z architektury o 12 warstwach transformerowych, 12 głowicach uwagi, rozmiarze ukrytym 768 oraz przybliżonej liczbie 110 milionów parametrów. Pre-trening SciBERT przebiega standardowo - z wykorzystaniem zadania maskowania tokenów (*Masked Language Modeling*, MLM) - lecz na specjalistycznym słowniku naukowym (*WordPiece vocabulary* wyuczonym na korpusie naukowym zamiast ogólnojęzykowego).

SPECTER wprowadza drugi etap treningu (*fine-tuning*) ukierunkowany na reprezentacje całych dokumentów, a nie pojedynczych tokenów. Sieć jest optymalizowana metodą uczenia metrycznego z funkcją straty *triplet margin loss*:

```
L(a, p, n) = max(0, ||E(a) − E(p)||² − ||E(a) − E(n)||² + margin)
```

gdzie *a* jest artykułem zakotwiczenia (*anchor*), *p* - artykułem pozytywnym (cytowanym przez *a*), zaś *n* - artykułem negatywnym dobranym losowo. Sygnałem nadzoru nie są zatem etykiety tematyczne nadane przez człowieka, lecz istniejąca struktura grafu cytowań. Reprezentacja całego dokumentu uzyskiwana jest przez wybranie wektora specjalnego tokenu [CLS] z ostatniej warstwy. Wynikowa przestrzeń embeddingów ma wymiarowość 768.

### 2.2 MiniLM

Model `all-MiniLM-L6-v2` (Wang i in., 2020; Reimers i Gurevych, 2019) realizuje inne podejście, łącząc destylację wiedzy (*knowledge distillation*) z frameworkiem Sentence-BERT. Architektura ucznia składa się jedynie z sześciu warstw transformerowych, co stanowi połowę głębokości SPECTER, przy rozmiarze ukrytym 384 (zamiast 768). Liczba parametrów spada tym samym do około 22 milionów.

W procesie destylacji mniejszy model (uczeń) uczy się naśladować wewnętrzne reprezentacje modelu nauczyciela, konkretnie macierze uwagi z ostatniej warstwy:

```
L_KD = KL(A_teacher || A_student)
```

gdzie KL oznacza dywergencję Kullbacka–Leiblera między macierzami uwagi nauczyciela i ucznia. Takie podejście - w przeciwieństwie do distylacji jedynie wyjść modelu - pozwala przenieść wiedzę o strukturze uwagi, nie tylko o wynikach końcowych (Wang i in., 2020).

Następnie model jest dostrajany w architekturze syjamskiej (*Siamese network*) według metody Sentence-BERT: oba zdania z pary przetwarzane są niezależnie tym samym enkoderem, a wynikowe reprezentacje zdaniowe uzyskiwane są przez *mean pooling* - uśrednianie wektorów wszystkich tokenów z ostatniej warstwy (w odróżnieniu od reprezentacji tokenu [CLS] stosowanej w SPECTER). Trening prowadzony jest na ponad miliardzie par zdań z różnorodnych korpusów ogólnych z zastosowaniem funkcji straty *Multiple Negatives Ranking Loss*. Wynikowa przestrzeń embeddingów ma wymiarowość 384. Oba modele użyto bez dodatkowego fine-tuningu, w konfiguracji *out-of-the-box*.

---

## 3. Dane i Metodologia

### 3.1 Zbiór danych

Dane zebrano z dwóch źródeł. Profile naukowców (115 osób) pozyskano z Portalu Badawczego UAM, ograniczając zbiór do pracowników Wydziału Matematyki i Informatyki. Publikacje (łącznie 3 440 rekordów) pobrano za pomocą API OpenAlex (Priem i in., 2022) - otwartego indeksu prac naukowych. Wszystkie 3 440 publikacji posiadało zarówno tytuł, jak i abstrakt w języku angielskim. Na tej podstawie wyekstrahowano 57 unikalnych par współautorów spośród naukowców uwzględnionych w zbiorze. Średnia liczba wspólnych prac na parę współautorską wyniosła 4.4.

### 3.2 Reprezentacja autora

Dla każdego autora *a* jego profil semantyczny E(*a*) konstruowany jest poprzez konkatenację tytułów i abstraktów wszystkich jego publikacji w jeden ciąg tekstowy, który następnie kodowany jest przez model jako pojedynczy wektor. Takie podejście - w przeciwieństwie do uśredniania embeddingów poszczególnych publikacji - pozwala modelowi uwzględnić kontekst całego dorobku autora w obrębie jednego wywołania enkodera. Podobieństwo między dwoma autorami obliczane jest jako podobieństwo kosinusowe ich wektorów:

```
sim(a, b) = cos(E(a), E(b)) = E(a) · E(b) / (||E(a)|| · ||E(b)||)
```

### 3.3 Ewaluacja

Jako ground truth dla ewaluacji przyjęto relacje współautorstwa: jeśli dwoje naukowców ze zbioru wspólnie podpisało co najmniej jedną publikację, traktujemy ich jako semantycznie podobnych. Podejście to - stosowane powszechnie w literaturze dotyczącej systemów rekomendacji współpracowników - opiera się na założeniu, że naukowcy pracujący razem reprezentują bliskie obszary badawcze (Cohan i in., 2020).

Hipotezę o bliskości współautorów weryfikowano następująco: obliczono podobieństwo kosinusowe dla wszystkich 57 par współautorskich, a następnie pobrano 1 000 losowych par (z wyłączeniem par współautorskich) jako rozkład referencyjny. Istotność różnicy między rozkładami testowano nieparametrycznym testem Manna–Whitneya U (jednostronny, hipoteza alternatywna: współautorzy > losowe), a jako miarę wielkości efektu obliczono d Cohena (Cohen, 1988).

---

## 4. Wyniki

### 4.1 Test bliskości współautorów

Wyniki eksperymentu przedstawia Tabela 1. Oba modele wykazują wyraźną i statystycznie wysoce istotną separację między parami współautorów a losowymi parami naukowców. Współautorzy są konsekwentnie bliżej siebie w przestrzeni embeddingów niż pary dobrane losowo, co potwierdza użyteczność podejścia embeddingowego do profilowania autorów.

**Tabela 1.** Wyniki testu bliskości współautorów.

| Model   | Śr. podobieństwo (współautorzy) | Śr. podobieństwo (losowe) | Różnica | d Cohena | p-value  |
|---------|--------------------------------|--------------------------|---------|----------|----------|
| SPECTER | 0.787                          | 0.624                    | +0.163  | 1.51     | 1.82e-17 |
| MiniLM  | 0.419                          | 0.109                    | +0.310  | 2.19     | 9.11e-20 |

Wielkości efektu d = 1.51 i d = 2.19 należy klasyfikować jako bardzo duże według skali Cohena (1988), gdzie d ≥ 0.8 uznaje się za efekt duży. Obydwa wyniki wskazują, że separacja współautorów od losowych par nie jest artefaktem statystycznym, lecz odzwierciedla realną strukturę semantyczną przestrzeni embeddingów.

### 4.2 Porównanie modeli

Zaskakującym wynikiem jest wyraźna przewaga MiniLM nad SPECTER pod względem jakości separacji (d = 2.19 vs d = 1.51), mimo że SPECTER był trenowany specjalnie na tekstach naukowych. Różnica bezwzględna między grupami jest dla MiniLM prawie dwukrotnie większa (0.310 vs 0.163). Warto odnotować, że SPECTER charakteryzuje się znacząco wyższymi bezwzględnymi wartościami podobieństwa kosinusowego - zarówno dla współautorów (0.787), jak i dla losowych par (0.624) - co sugeruje, że przestrzeń embeddingów SPECTER jest bardziej „skompresowana" i mniej dyskryminatywna przy porównaniu dowolnych par autorów. MiniLM natomiast rozciąga swoją przestrzeń szerzej: losowe pary osiągają średnie podobieństwo jedynie 0.109, co pozostawia więcej miejsca na wiarygodne różnicowanie profili.

Możliwym wyjaśnieniem tej obserwacji jest różnica w zadaniu treningowym. SPECTER był optymalizowany pod kątem bliskości artykułów powiązanych cytowaniami - czyli par konkretnych dokumentów - podczas gdy nasze zadanie polega na porównywaniu zagregowanych profili całego dorobku autorów, co stanowi inny reżim semantyczny. MiniLM, trenowany na szerokich korpusach tekstów ogólnych, może generalizować lepiej do tego rodzaju długich, heterogenicznych dokumentów.

### 4.3 Zgodność modeli

Korelacja Pearsona między macierzami podobieństwa obu modeli (obliczona na wszystkich ~6 600 unikalnych parach autorów) wynosi r = 0.729 (p ≈ 0), co odpowiada umiarkowanie silnej zgodności. Modele są zatem w znaczącym stopniu zbieżne co do tego, którzy autorzy są do siebie podobni, lecz różnią się w skali i rozpiętości tych podobieństw.

### 4.4 Wpływ abstraktów na jakość embeddingów

Odrębny eksperyment ablacyjny (przeprowadzony w notebooku `abstracts_vs_noabstracts.ipynb`) wykazał, że włączenie abstraktów do agregowanego tekstu autora powoduje zauważalne przesunięcia pozycji autorów w przestrzeni PCA. Przemieszczenie to jest jednak spójne - autorzy z pokrewnych dziedzin przemieszczają się w podobnym kierunku - co sugeruje, że abstrakty wnoszą dodatkową, semantycznie spójną informację, a nie szum.

---

## 5. Dyskusja

Wyniki eksperymentów potwierdzają centralną hipotezę niniejszej pracy: embeddingi tekstowe oparte na przedtrenowanych modelach transformerowych są w stanie uchwycić podobieństwo badawcze pomiędzy naukowcami na poziomie wystarczającym do praktycznych zastosowań rekomendacyjnych. Podejście to nie wymaga ręcznego tworzenia taksonomii dziedzin ani ekstrakcji słów kluczowych - cała reprezentacja wyłania się z samego tekstu publikacji.

Przewaga MiniLM nad SPECTER stanowi nieintuicyjny wynik, który warto interpretować ostrożnie. Po pierwsze, obie architektury bazują na transformerach BERT, lecz różnią się zarówno trybem treningu domenowego, jak i głębokością sieci. SPECTER (12 warstw, 768 wymiarów) jest modelem głębszym, natomiast MiniLM (6 warstw, 384 wymiary) - zdestylowanym do połowy jego rozmiaru. Wyższy efekt separacyjny MiniLM może wynikać z bardziej „rozległej" geometrii jego przestrzeni wektorowej, a nie z lepszego rozumienia semantyki naukowej. Po drugie, nasze dane pochodzą wyłącznie z jednego wydziału jednej uczelni, co ogranicza generalność wniosków.

Istotnym ograniczeniem ewaluacji jest przyjęcie współautorstwa jako proxy dla podobieństwa semantycznego. Naukowcy mogą być współautorami z powodów organizacyjnych lub administracyjnych, niekoniecznie z powodu bliskości obszarów badań. Z drugiej strony, brak lepszych, skalowalnych danych referencyjnych (np. ocen ekspertów domenowych) sprawia, że współautorstwo pozostaje najrozsądniejszym dostępnym kryterium, powszechnie stosowanym w literaturze (Cohan i in., 2020).

---

## 6. Wnioski

Niniejsza praca wykazuje, że proste podejście oparte na agregacji tekstów publikacji i ich embeddowaniu za pomocą przedtrenowanych modeli transformerowych skutecznie koduje podobieństwo badawcze naukowców. Oba testowane modele - SPECTER i MiniLM - umożliwiają statystycznie istotną separację par współautorów od losowych par, przy czym MiniLM osiąga lepszą jakość separacji przy dwukrotnie niższej wymiarowości wektora. Wynik ten sugeruje, że dla zadania profilowania autorów model ogólny może być praktycznym wyborem: jest szybszy, zajmuje mniej pamięci i generuje bardziej dyskryminatywne reprezentacje na tym konkretnym zadaniu.

Przyszłe prace powinny obejmować: (1) rozszerzenie reprezentacji o embedding grafu współautorstwa, który uwzględniałby strukturę sieci powiązań, (2) uwzględnienie danych o grantach i projektach badawczych, (3) fine-tuning modeli na danych ze środowiska UAM oraz (4) ewaluację przez ekspertów domenowych w celu weryfikacji trafności rekomendacji.

---

## Literatura

1. Beltagy, I., Lo, K., & Cohan, A. (2019). SciBERT: A Pretrained Language Model for Scientific Text. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP 2019)*, 3615–3620.

2. Cohan, A., Feldman, S., Beltagy, I., Downey, D., & Weld, D. S. (2020). SPECTER: Document-level Representation Learning using Citation-informed Transformers. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020)*, 2270–2282.

3. Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.

4. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of NAACL-HLT 2019*, 4171–4186.

5. Priem, J., Piwowar, H., & Orr, R. (2022). OpenAlex: A fully-open index of scholarly works, authors, venues, institutions, and concepts. *arXiv preprint arXiv:2205.01833*.

6. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP 2019)*, 3980–3990.

7. Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. *Advances in Neural Information Processing Systems (NeurIPS 2020)*, 33, 5776–5788.

---
## Wkład poszczególnych osób (dla prac zespołowych)


| Osoba | Wkład |
|-------|-------|
| Jakub Paszke | Analiza modelu i eksperymenty Specter |
| Jakub Paszke | Budowa embedingów na podstawie tytułów |
| Miłosz Rolewski | Analiza modelu i eksperymenty MiniLM |
| Miłosz Rolewski | Budowa embedingów na podstawie abstraktów |
| Jakub Paszke | Ewaluacja, analiza statystyczna |
| Miłosz Rolewski | Wizualizacje, raport |

---

## Załączniki

### A. Kod źródłowy
- `evaluation_and_comparison.ipynb` - główny notebook ewaluacyjny
- `abstracts_vs_noabstracts.ipynb` - analiza wpływu abstraktów
- `inference.py` - skrypt do inferencji

### B. Dane
- Źródło: OpenAlex API (https://openalex.org/)
- Format: CSV
- Lokalizacja: `../abstracts/data/`

