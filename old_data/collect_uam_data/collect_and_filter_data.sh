#!/bin/bash
echo "Complete Data Collection Pipeline for Research Portal"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

DATA_DIR="./data"
VENV_ACTIVATE="venv/bin/activate"

print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}i${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

check_file() {
    if [ -f "$1" ]; then
        print_status "Found: $1"
        return 0
    else
        print_error "Missing: $1"
        return 1
    fi
}

mkdir -p "$DATA_DIR"

if [ -f "$VENV_ACTIVATE" ]; then
    print_info "Activating virtual environment..."
    source "$VENV_ACTIVATE"
    print_status "Virtual environment activated"
else
    print_warning "No virtual environment found at $VENV_ACTIVATE"
    print_info "Continuing with system Python..."
fi

echo ""
echo "========================================================================"
echo "  STEP 1: Scrape Scientists from Research Portal"
echo "========================================================================"
echo ""
print_info "Running research_portal_scraper.py..."
echo ""

python research_portal_scraper.py

if [ $? -eq 0 ]; then
    print_status "Step 1 completed successfully!"
    echo ""
    
    if check_file "./data/scientists_data.csv"; then
        SCIENTIST_COUNT=$(($(wc -l < scientists_data.csv) - 1))
        print_status "Extracted ${SCIENTIST_COUNT} scientists"
    else
        print_error "Output files from Step 1 were not created"
        exit 1
    fi
else
    print_error "Step 1 failed"
    exit 1
fi

echo ""
echo "========================================================================"
echo "  STEP 2: Extract Identifiers from Profiles"
echo "========================================================================"
echo ""
print_info "Running extract_identifiers.py..."
echo ""

python extract_identifiers.py

if [ $? -eq 0 ]; then
    print_status "Step 2 completed successfully!"
    echo ""
    
    if check_file "./data/scientists_with_identifiers.csv"; then
        ORCID_COUNT=$(awk -F',' 'NR>1 && $10 != "" {count++} END {print count}' ./data/scientists_with_identifiers.csv)
        print_status "Found ORCID for ${ORCID_COUNT} scientists"
    else
        print_error "Output files from Step 2 were not created"
        exit 1
    fi
else
    print_error "Step 2 failed"
    exit 1
fi

echo ""
echo "========================================================================"
echo "  STEP 3: Filter UAM Authors Data"
echo "========================================================================"
echo ""

if [ ! -f "$DATA_DIR/uam_authors.json" ]; then
    print_warning "UAM authors file not found at $DATA_DIR/uam_authors.json"
    print_info "Skipping Step 3..."
else
    print_info "Running filter_authors.py..."
    echo ""
    
    python filter_authors.py --csv "$DATA_DIR/scientists_with_identifiers.csv" \
                             --json "$DATA_DIR/uam_authors.json" \
                             --output "$DATA_DIR/wmii_authors.json"
    
    if [ $? -eq 0 ]; then
        print_status "Step 3 completed successfully!"
        echo ""
        
        if check_file "$DATA_DIR/wmii_authors.json"; then
            print_status "Filtered authors saved"
        fi
    else
        print_error "Step 3 failed"
        exit 1
    fi
fi

echo ""
echo "========================================================================"
echo "  STEP 4: Filter UAM Works Data"
echo "========================================================================"
echo ""

# Check if works data file exists
if [ ! -f "$DATA_DIR/uam_works.json" ]; then
    print_warning "UAM works file not found at $DATA_DIR/uam_works.json"
    print_info "Skipping Step 4..."
else
    print_info "Running filter_works.py..."
    echo ""
    
    python filter_works.py --csv "$DATA_DIR/scientists_with_identifiers.csv" \
                          --json "$DATA_DIR/uam_works.json" \
                          --output "$DATA_DIR/wmii_works.json"
    
    if [ $? -eq 0 ]; then
        print_status "Step 4 completed successfully!"
        echo ""
        
        if check_file "$DATA_DIR/wmii_works.json"; then
            print_status "Filtered works saved"
        fi
    else
        print_error "Step 4 failed"
        exit 1
    fi
fi

echo ""
echo "========================================================================"
echo "  PIPELINE COMPLETED SUCCESSFULLY!"
echo "========================================================================"
echo ""
print_status "All data collection steps completed!"
echo ""

echo "Output Files Summary:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "1. Scraped Data:"
if [ -f "$DATA_DIR/scientists_data.csv" ]; then
    echo "   ✓ scientists_data.csv"
fi

echo ""
echo "2. Data with Identifiers:"
if [ -f "$DATA_DIR/scientists_with_identifiers.csv" ]; then
    echo "   ✓ scientists_with_identifiers.csv"
fi

echo ""
echo "3. Filtered UAM Data:"
if [ -f "$DATA_DIR/wmii_authors.json" ]; then
    echo "   ✓ $DATA_DIR/wmii_authors.json"
fi
if [ -f "$DATA_DIR/wmii_works.json" ]; then
    echo "   ✓ $DATA_DIR/wmii_works.json"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -f "$DATA_DIR/scientists_with_identifiers.csv" ]; then
    TOTAL_SCIENTISTS=$(($(wc -l < $DATA_DIR/scientists_with_identifiers.csv) - 1))
    ORCID_COUNT=$(awk -F',' 'NR>1 && $10 != "" {count++} END {print count}' $DATA_DIR/scientists_with_identifiers.csv)
    
    echo "Statistics:"
    echo "  • Total scientists: $TOTAL_SCIENTISTS"
    echo "  • With ORCID: $ORCID_COUNT"
    
    if [ -f "$DATA_DIR/wmii_works.json" ]; then
        WORKS_COUNT=$(python3 -c "import json; print(len(json.load(open('$DATA_DIR/wmii_works.json'))))" 2>/dev/null || echo "N/A")
        echo "  • Total works: $WORKS_COUNT"
    fi
fi

echo ""
print_status "Data collection complete! Ready for analysis."
echo ""