#!/bin/bash
set -euo pipefail
echo "SARA - UAM WMiI Research Data Pipeline"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

DATA_DIR="./data"
SRC_DIR="./src"

ok()   { echo -e "${GREEN}✓${NC} $1"; }
err()  { echo -e "${RED}✗${NC} $1"; }
info() { echo -e "${BLUE}i${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }

mkdir -p "$DATA_DIR"

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate && ok "Virtual environment activated"
else
    warn "No venv found - using system Python"
fi

# ── Step 1: Scrape portal ────────────────────────────────────────────────────
echo
echo "==  STEP 1: Scrape Research Portal  =="
python "$SRC_DIR/research_portal_scraper.py"

[ -f "$DATA_DIR/scientists_data.csv" ] || { err "scientists_data.csv not created"; exit 1; }
COUNT=$(( $(wc -l < "$DATA_DIR/scientists_data.csv") - 1 ))
ok "Step 1 done - $COUNT scientists scraped"

# ── Step 2: Extract identifiers ─────────────────────────────────────────────
echo
echo "==  STEP 2: Extract Identifiers  =="
python "$SRC_DIR/extract_identifiers.py"

[ -f "$DATA_DIR/scientists_with_identifiers.csv" ] || { err "scientists_with_identifiers.csv not created"; exit 1; }
ORCID_COUNT=$(awk -F',' 'NR>1 && $10!="" {c++} END {print c+0}' "$DATA_DIR/scientists_with_identifiers.csv")
ok "Step 2 done - $ORCID_COUNT scientists with ORCID"

# ── Step 3: Filter OpenAlex datasets (optional - requires large input files) ─
echo
echo "==  STEP 3: Filter OpenAlex Authors & Works  =="

FILTER_ARGS="--csv $DATA_DIR/scientists_with_identifiers.csv --out-dir $DATA_DIR"
SKIP_STEP3=true

if [ -f "$DATA_DIR/uam_authors.json" ]; then
    FILTER_ARGS="$FILTER_ARGS --authors $DATA_DIR/uam_authors.json"
    SKIP_STEP3=false
else
    warn "uam_authors.json not found - skipping authors filter"
fi

if [ -f "$DATA_DIR/uam_works.json" ]; then
    FILTER_ARGS="$FILTER_ARGS --works $DATA_DIR/uam_works.json"
    SKIP_STEP3=false
else
    warn "uam_works.json not found - skipping works filter"
fi

if [ "$SKIP_STEP3" = true ]; then
    warn "Step 3 skipped - no OpenAlex dump files found in data/"
    info "Download from: https://uam-my.sharepoint.com/:f:/r/personal/jakpas3_st_amu_edu_pl/Documents/SARA"
else
    python "$SRC_DIR/filter_data.py" $FILTER_ARGS
    ok "Step 3 done"
fi

# ── Step 4: Fetch abstracts from OpenAlex API ────────────────────────────────
echo
echo "==  STEP 4: Fetch Abstracts from OpenAlex API  =="
python "$SRC_DIR/fetch_abstracts.py"

[ -f "$DATA_DIR/wmii_publications.csv" ] || { err "wmii_publications.csv not created"; exit 1; }
PUB_COUNT=$(( $(wc -l < "$DATA_DIR/wmii_publications.csv") - 1 ))
ok "Step 4 done - $PUB_COUNT publications with abstracts"

# ── Summary ──────────────────────────────────────────────────────────────────
echo
echo "════════════════════════════════════════════"
echo "  PIPELINE COMPLETE"
echo "════════════════════════════════════════════"
echo
echo "  Key output files:"
[ -f "$DATA_DIR/scientists_with_identifiers.csv" ] && echo "  ✓ data/scientists_with_identifiers.csv  - scientist profiles + ORCIDs"
[ -f "$DATA_DIR/wmii_publications.csv" ]           && echo "  ✓ data/wmii_publications.csv            - publications with abstracts"
echo
echo "  Optional (if OpenAlex dumps were available):"
[ -f "$DATA_DIR/wmii_authors.json" ]               && echo "  ✓ data/wmii_authors.json"
[ -f "$DATA_DIR/wmii_works.json" ]                 && echo "  ✓ data/wmii_works.json"
echo
