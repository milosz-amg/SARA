#!/bin/bash
#
# Usage:
#   ./run_visualisation.sh               # auto k, all methods
#   ./run_visualisation.sh --k 9         # skip k-search, use k=9 directly
#   ./run_visualisation.sh --no-tsne     # skip t-SNE
#   ./run_visualisation.sh --k-min 4 --k-max 16

set -euo pipefail

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

ok()   { echo -e "${GREEN}✓${NC} $1"; }
err()  { echo -e "${RED}✗${NC} $1"; }
info() { echo -e "${BLUE}i${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }

# Defaults
DATA_DIR="../wmii-data-collection/data"
OUT_DIR="./output"
K_MIN=4
K_MAX=12
FIXED_K=""
NO_TSNE=""
K_METRICS="$OUT_DIR/k_metrics.json"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --k)       FIXED_K="$2";  shift 2 ;;
        --k-min)   K_MIN="$2";    shift 2 ;;
        --k-max)   K_MAX="$2";    shift 2 ;;
        --no-tsne) NO_TSNE="--no-tsne"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUT_DIR"

echo
echo "========================================"
echo "  Publications Visualisation Pipeline"
echo "========================================"

for f in "$DATA_DIR/embeddings.npy" "$DATA_DIR/embeddings_metadata.csv"; do
    if [ ! -f "$f" ]; then
        err "Missing required file: $f"
        err "Place embeddings.npy and embeddings_metadata.csv in $DATA_DIR/"
        exit 1
    fi
done
ok "Input files found"

# Activate venv if present
if [ -f "../venv/bin/activate" ]; then
    source ../venv/bin/activate && ok "venv activated"
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate && ok "venv activated"
else
    warn "No venv found — using system Python"
fi

# Step 1: find optimal k (skip if --k supplied)
echo
echo "==  STEP 1: Find optimal k  =="

if [ -n "$FIXED_K" ]; then
    warn "Skipping k-search — using fixed k=$FIXED_K"
    CHOSEN_K=$FIXED_K
else
    info "Running find_optimal_k.py  (k=$K_MIN..$K_MAX)..."
    python src/find_optimal_k.py \
        --embeddings "$DATA_DIR/embeddings.npy" \
        --output     "$K_METRICS" \
        --k-min      "$K_MIN" \
        --k-max      "$K_MAX"

    [ -f "$K_METRICS" ] || { err "k_metrics.json not created"; exit 1; }

    CHOSEN_K=$(python3 -c "import json; print(json.load(open('$K_METRICS'))['consensus_k'])")
    ok "Step 1 done — consensus k = $CHOSEN_K"
fi

# Step 2: project & evaluate
echo
echo "==  STEP 2: Project embeddings (k=$CHOSEN_K)  =="

ANALYSE_ARGS="--embeddings $DATA_DIR/embeddings.npy \
              --metadata   $DATA_DIR/embeddings_metadata.csv \
              --clusters   $CHOSEN_K \
              --output     $OUT_DIR/vis_methods.json"

[ -f "$DATA_DIR/scientists_with_identifiers.csv" ] && \
    ANALYSE_ARGS="$ANALYSE_ARGS --scientists $DATA_DIR/scientists_with_identifiers.csv"

[ -f "$K_METRICS" ] && \
    ANALYSE_ARGS="$ANALYSE_ARGS --k-analysis $K_METRICS"

[ -n "$NO_TSNE" ] && ANALYSE_ARGS="$ANALYSE_ARGS --no-tsne"

python src/analyse.py $ANALYSE_ARGS

[ -f "$OUT_DIR/vis_methods.json" ] || { err "vis_methods.json not created"; exit 1; }
ok "Step 2 done"

# Summary
echo
echo "========================================"
echo "  DONE"
echo "========================================"
[ -f "$OUT_DIR/k_metrics.json" ]   && echo "  ✓ wyniki/k_metrics.json"
[ -f "$OUT_DIR/vis_methods.json" ] && echo "  ✓ wyniki/vis_methods.json"
echo
info "Open output/vis_methods.html in a browser"
echo