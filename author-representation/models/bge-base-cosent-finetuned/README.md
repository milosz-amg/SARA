# Model BGE-base fine-tuned (CoSENTLoss)

Model fine-tuned nie jest commitowany do repo — **`model.safetensors` ma 418 MB** (limit GitHub: 100 MB).

## Jak odtworzyć model

### Opcja A — wytrenować samodzielnie (Google Colab A100)

```bash
# 1. Przygotowanie danych treningowych (pary + tryplety)
python scripts/05_prepare_finetune_data.py
# → produkuje data/finetune/ (~230 MB)

# 2. Zzipuj do upload'u na Colab
cd data && zip -r finetune.zip finetune/ && cd ..

# 3. Otwórz notebooks/finetune_bge_colab.ipynb w Colab
#    - Wybierz A100 (wymagane ~40 GB VRAM przy batch 64)
#    - Upload finetune.zip (przez górny przycisk upload w Colab)
#    - Uruchom wszystkie komórki (~1–1.5 h na A100)
#    - Pobierz bge-base-arxiv-finetuned.zip

# 4. Rozpakuj do models/
unzip bge-base-arxiv-finetuned.zip -d models/bge-base-cosent-finetuned/
```

**Koszt:** ~$12 (pay-as-you-go Colab, 90 dni ważności compute units).

**Hiperparametry:**
- Model bazowy: `BAAI/bge-base-en-v1.5` (768-dim, 110M params)
- Loss: `CoSENTLoss` z ciągłymi similarity scores [0.0, 1.0]
- Labels: hierarchiczna odległość kategorii ArXiv (3 poziomy) + rozmyte multi-label (Jaccard)
- 1 epoka, lr=1e-5, batch=64, weight_decay=0.01, warmup=0.1
- 18 880 paperów treningowych, 62 310 par, ~5844 kroków

### Opcja B — ściągnąć wcześniej wytrenowany model

Jeśli dostałeś model osobno (np. na dysku zewnętrznym / przez chmurę), rozpakuj go tak żeby struktura była:

```
models/bge-base-cosent-finetuned/
└── final/
    ├── config.json
    ├── model.safetensors       ← 418 MB
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    ├── vocab.txt
    └── sentence_bert_config.json
```

## Weryfikacja

Po umieszczeniu modelu uruchom:

```bash
python scripts/07_evaluate_finetuned.py
```

Powinno odtworzyć wyniki z `results/finetuning/wyniki.md`:
- main category purity (k=8): 70.49 %
- fuzzy purity (k=8): 80.86 %
- P@20: 0.667
