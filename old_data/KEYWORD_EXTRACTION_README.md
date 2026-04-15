# Keyword Extraction from LLM Narratives

This document explains how to extract keywords from the LLM-generated author narratives using the Qwen3 model.

## Overview

The keyword extraction pipeline uses the same Qwen3-8B model to analyze the `story_text` column and extract 5-10 relevant scientific keywords for each researcher. These keywords are stored in the `keywords` column as JSONB.

## Phase 2 of LLM Enhancement Plan

This implements **Phase 2: Multi-Source Keyword Extraction** from [LLM_ENHANCED_EMBEDDINGS_PLAN.md](LLM_ENHANCED_EMBEDDINGS_PLAN.md), which aims for:
- 15-25% improvement in search performance
- Enhanced hybrid search capabilities (vector + keyword matching)

## Database Schema

The script automatically creates:
```sql
-- Keywords column (JSONB array)
ALTER TABLE authors ADD COLUMN keywords JSONB;

-- GIN index for fast keyword queries
CREATE INDEX idx_authors_keywords ON authors USING GIN (keywords);
```

**Keyword format:**
```json
["oncology", "cancer immunotherapy", "melanoma", "clinical trials", "tumor biology"]
```

## Scripts

### 1. Test Keyword Extraction (test_keyword_extraction.py)

Test on 5 sample authors to verify the model works correctly:

```bash
cd /home/jakub/Projekty/SARA/OpenAlex/scripts
source ../../venv/bin/activate
python test_keyword_extraction.py
```

**Output:**
- Shows narrative for each author
- Displays extracted keywords
- Verifies LLM is working properly

### 2. Fast Keyword Extraction (extract_keywords_fast.py)

Production script using GPU batch processing for maximum efficiency.

**Basic usage:**
```bash
# Process all authors with narratives
python extract_keywords_fast.py

# Process first 1000 authors (testing)
python extract_keywords_fast.py --limit 1000

# Custom GPU batch size
python extract_keywords_fast.py --gpu-batch-size 64
```

**Performance:**
- **GPU Batch Size:** 32 (processes 32 authors in parallel)
- **Expected Speed:** 3-5 authors/sec
- **Estimated Time:** 6-10 hours for ~95k authors with narratives

**Configuration:**
```python
GPU_BATCH_SIZE = 32   # Process 32 in parallel on GPU
DB_BATCH_SIZE = 256   # Fetch 256 from database at a time
MIN_KEYWORDS = 3      # Minimum keywords required
MAX_KEYWORDS = 15     # Maximum keywords to extract
```

## How It Works

### 1. Prompt Template

```
Extract 5-10 key research keywords from this researcher profile. Focus on:
- Research domains (e.g., "oncology", "machine learning")
- Techniques/methods (e.g., "immunotherapy", "neural networks")
- Key topics and expertise areas

Return ONLY a JSON array of lowercase keywords. Format: ["keyword1", "keyword2", ...]

Profile:
[story_text content]

Keywords JSON:
```

### 2. Batch Processing

The script uses **GPU batch generation** (same approach as `generate_author_narratives_fast.py`):

1. Fetch 256 authors from database
2. Split into GPU batches of 32
3. Generate all 32 keyword responses in **one GPU forward pass**
4. Parse JSON arrays from responses
5. Save to database in bulk
6. Repeat until all authors processed

### 3. Response Parsing

The LLM returns a JSON array which is parsed and validated:
- Must be valid JSON
- Must contain 3+ keywords
- Keywords are normalized to lowercase
- Limited to 15 keywords max

## Example Output

### Sample Author: Dr. Piotr Rutkowski

**Narrative:**
> Dr. Piotr Rutkowski is a highly distinguished oncology researcher at the Maria Sklodowska-Curie National Research Institute in Poland, specializing in melanoma treatment and cancer immunotherapy. With an exceptional h-index of 108 and over 81,000 citations across 456 publications, they are recognized as a leading international expert in tumor immunology and oncological medicine.

**Extracted Keywords:**
```json
[
  "oncology",
  "melanoma",
  "cancer immunotherapy",
  "tumor immunology",
  "clinical oncology",
  "skin cancer",
  "cancer treatment",
  "immunotherapy"
]
```

## Progress Tracking

The script provides real-time progress updates:

```
Progress: 5,120/95,141 (5.4%) | ✅ 4,892 | ❌ 228
Rate: 4.23 authors/sec | ETA: 5:53:12
```

**Metrics:**
- ✅ Successful: Keywords extracted (3+ keywords)
- ❌ Failed: Could not extract sufficient keywords (saved as empty array to avoid reprocessing)

## Database Queries

### Query authors by keyword

```sql
-- Find authors researching "immunotherapy"
SELECT display_name, keywords
FROM authors
WHERE keywords @> '["immunotherapy"]'::jsonb
LIMIT 10;

-- Find authors with multiple keywords
SELECT display_name, keywords
FROM authors
WHERE keywords @> '["machine learning", "neural networks"]'::jsonb;

-- Count authors by keyword presence
SELECT jsonb_array_elements_text(keywords) as keyword, COUNT(*) as count
FROM authors
WHERE keywords IS NOT NULL
GROUP BY keyword
ORDER BY count DESC
LIMIT 20;
```

### Statistics

```sql
-- Authors with keywords
SELECT COUNT(*) as authors_with_keywords
FROM authors
WHERE keywords IS NOT NULL AND keywords != '[]'::jsonb;

-- Average number of keywords per author
SELECT AVG(jsonb_array_length(keywords)) as avg_keywords
FROM authors
WHERE keywords IS NOT NULL AND keywords != '[]'::jsonb;

-- Authors pending keyword extraction
SELECT COUNT(*) as pending
FROM authors
WHERE story_text IS NOT NULL
  AND story_text != ''
  AND (keywords IS NULL OR keywords = '[]'::jsonb);
```

## Resumability

The script is fully resumable:
- Only processes authors with `story_text` but no `keywords`
- Can be interrupted (Ctrl+C) and restarted without losing progress
- Failed extractions are marked with empty array to avoid reprocessing

## Logging

All activity is logged to:
```
keyword_extraction_fast_YYYYMMDD_HHMMSS.log
```

**Log includes:**
- Model loading status
- Batch processing times
- Success/failure counts
- Progress and ETA
- Final statistics

## Next Steps

After keyword extraction is complete:

1. **Verify quality:**
   ```sql
   -- Sample keywords
   SELECT display_name, keywords
   FROM authors
   WHERE keywords IS NOT NULL AND keywords != '[]'::jsonb
   LIMIT 20;
   ```

2. **Implement hybrid search** (Phase 2):
   - Combine vector similarity with keyword matching
   - Re-rank results using keyword overlap
   - Test optimal weights (semantic: 0.7, keyword: 0.3)

3. **Evaluate improvement:**
   - Run validation queries with keyword boost
   - Measure concept match rate improvement
   - Compare with vector-only search

## Troubleshooting

### Issue: Model not found
```
Error: Model Qwen/Qwen3-8B not found
```
**Solution:** The model will be downloaded automatically on first run (requires ~6GB download)

### Issue: CUDA out of memory
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce GPU batch size:
```bash
python extract_keywords_fast.py --gpu-batch-size 16
```

### Issue: Database connection failed
```
psycopg2.OperationalError: connection refused
```
**Solution:** Check database is running and credentials in `llm_generator.py` are correct

### Issue: JSON parsing errors
The script handles parsing errors gracefully:
- Logs warnings for unparseable responses
- Marks as failed (empty array)
- Continues processing

## Performance Optimization

### Faster processing:
1. Increase GPU batch size (if VRAM permits):
   ```bash
   python extract_keywords_fast.py --gpu-batch-size 64
   ```

2. Run on dedicated GPU without other processes

3. Use faster storage (SSD) for database

### Lower VRAM usage:
1. Decrease GPU batch size:
   ```bash
   python extract_keywords_fast.py --gpu-batch-size 16
   ```

## References

- [LLM Enhanced Embeddings Plan](LLM_ENHANCED_EMBEDDINGS_PLAN.md)
- [Milestone 1 Completion](MILESTONE_1_COMPLETED.md)
- Phase 2 Implementation: This document
