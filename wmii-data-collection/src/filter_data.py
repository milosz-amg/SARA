import csv
import json
import sys
import argparse
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def load_orcids(csv_path: Path) -> set:
    print(f"Reading ORCIDs from {csv_path}...")
    orcids = set()

    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = (row.get("orcid") or "").strip()
            if not raw:
                continue
            normalized = raw.replace("https://orcid.org/", "").replace("http://orcid.org/", "")
            orcids.add(normalized)
            orcids.add(f"https://orcid.org/{normalized}")

    print(f"Found {len(orcids) // 2} unique ORCIDs")
    return orcids


def load_json(path: Path) -> list:
    print(f"Loading {path}...")
    with open(path, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("  Retrying as JSONL...")
            f.seek(0)
            data = [json.loads(line) for line in f if line.strip()]

    if isinstance(data, dict):
        data = (
            data.get("results")
            or data.get("data")
            or data.get("authors")
            or data.get("works")
            or []
        )

    print(f"  Loaded {len(data)} records")
    return data


def normalize_orcid(orcid):
    if not orcid:
        return None
    return orcid.strip().replace("https://orcid.org/", "").replace("http://orcid.org/", "")


def filter_authors(authors: list, orcids: set) -> list:
    print("\nFiltering authors...")
    matched = []
    for author in authors:
        raw = author.get("orcid") or (author.get("ids") or {}).get("orcid")
        n = normalize_orcid(raw)
        if n and (n in orcids or f"https://orcid.org/{n}" in orcids):
            matched.append(author)
            print(f"  Matched: {author.get('display_name', '?')} ({n})")
    print(f"Matched {len(matched)} authors")
    return matched


def filter_works(works: list, orcids: set) -> list:
    print("\nFiltering works...")
    matched = []
    for idx, work in enumerate(works):
        if (idx + 1) % 5000 == 0:
            print(f"  Processed {idx + 1}/{len(works)} works...")
        for authorship in work.get("authorships", []):
            n = normalize_orcid((authorship.get("author") or {}).get("orcid"))
            if n and (n in orcids or f"https://orcid.org/{n}" in orcids):
                matched.append(work)
                break
    print(f"Matched {len(matched)} works")
    return matched


def save_json(data: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} records → {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter OpenAlex authors and works by faculty ORCIDs"
    )
    parser.add_argument("--csv",     default=DATA_DIR / "scientists_with_identifiers.csv",
                        help="CSV with ORCIDs (default: data/scientists_with_identifiers.csv)")
    parser.add_argument("--authors", default=DATA_DIR / "uam_authors.json",
                        help="OpenAlex authors JSON (default: data/uam_authors.json)")
    parser.add_argument("--works",   default=DATA_DIR / "uam_works.json",
                        help="OpenAlex works JSON (default: data/uam_works.json)")
    parser.add_argument("--out-dir", default=DATA_DIR,
                        help="Output directory (default: data/)")
    args = parser.parse_args()

    csv_path   = Path(args.csv)
    authors_in = Path(args.authors)
    works_in   = Path(args.works)
    out_dir    = Path(args.out_dir)

    orcids = load_orcids(csv_path)
    if not orcids:
        print("No ORCIDs found — aborting")
        sys.exit(1)

    if authors_in.exists():
        authors = load_json(authors_in)
        save_json(filter_authors(authors, orcids), out_dir / "wmii_authors.json")
    else:
        print(f"\nSkipping authors — {authors_in} not found")

    if works_in.exists():
        works = load_json(works_in)
        save_json(filter_works(works, orcids), out_dir / "wmii_works.json")
    else:
        print(f"\nSkipping works — {works_in} not found")

    print("\nDone.")


if __name__ == "__main__":
    main()
