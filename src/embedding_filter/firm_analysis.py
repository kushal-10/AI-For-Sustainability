#!/usr/bin/env python3
import os, json
from pathlib import Path
from typing import Dict, List, Set
import pandas as pd
from tqdm import tqdm

# ---------------- Config ----------------
CSV_DIR = Path("data/scores_csv")      # directory with all csv files
AI_JSON = Path("kw_data/ai.json")      # filtered AI keywords (dict: EN -> DE or list/string values)
THRESHOLDS = [0.5]

# ---------------- Helpers ----------------
def collect_ai_terms(ai_json_path: Path) -> Set[str]:
    """
    ai.json may look like: { "english": "deutsch", "term": ["syn1","syn2"], ... }
    Return all keys + all string/list values as terms (case-sensitive, exact match to CSV columns).
    """
    with open(ai_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    terms: Set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.strip():
                terms.add(k.strip())
            if isinstance(v, str) and v.strip():
                terms.add(v.strip())
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, str) and item.strip():
                        terms.add(item.strip())
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, str) and item.strip():
                terms.add(item.strip())
    return terms

def parse_company_year(csv_path: Path):
    """
    Expect path like data/scores_csv/COMPANY/YEAR/filename.csv
    Returns (COMPANY, YEAR or None).
    """
    # .../scores_csv/<company>/<year>/<file.csv>
    parts = csv_path.parts
    try:
        idx = parts.index("scores_csv")
        company = parts[idx + 1]
        year = parts[idx + 2] if idx + 2 < len(parts) else None
        return company, year
    except ValueError:
        # fallback: best effort using parent dirs
        return csv_path.parent.parent.name, csv_path.parent.name

# ---------------- SDG columns ----------------
SDG_COLS = [f"sdg_{i}" for i in range(1, 18)]

def main():
    # Load AI terms (allowing EN+DE)
    ai_terms = collect_ai_terms(AI_JSON)

    # Gather all CSV files
    csv_files: List[Path] = []
    for dirpath, _, filenames in os.walk(CSV_DIR):
        for filename in filenames:
            if filename.endswith(".csv"):
                csv_files.append(Path(dirpath) / filename)

    if not csv_files:
        print(f"No CSV files found under {CSV_DIR}")
        return

    # Global counters
    total_rows_global = 0
    selected_global = {thr: 0 for thr in THRESHOLDS}

    # Per-company aggregation
    # company_stats[company] = {
    #   "total_rows": int,
    #   "selected": {thr: int}
    # }
    company_stats: Dict[str, Dict] = {}

    for csv_file in tqdm(csv_files, desc="Scanning CSVs"):
        company, year = parse_company_year(csv_file)
        df = pd.read_csv(csv_file)

        # Relevant columns = SDGs + AI terms actually present in this CSV
        # (skip the id column if present—you didn't specify its name; we assume first col is id)
        cols_in_df = set(df.columns)
        # drop id-like first column if it's clearly an integer id column; safer to just filter by names we want:
        relevant_cols = [c for c in df.columns if (c in SDG_COLS or c in ai_terms)]

        # If no relevant columns, skip this file
        if not relevant_cols:
            continue

        data = df[relevant_cols]
        n_rows = len(data)
        total_rows_global += n_rows

        # init company bucket
        cstat = company_stats.setdefault(company, {
            "total_rows": 0,
            "selected": {thr: 0 for thr in THRESHOLDS}
        })
        cstat["total_rows"] += n_rows

        # Count per threshold for this file
        for thr in THRESHOLDS:
            mask = (data >= thr).any(axis=1)
            count = int(mask.sum())
            selected_global[thr] += count
            cstat["selected"][thr] += count

    # ----------- Print global summary -----------
    print(f"\nTotal sentences across CSVs: {total_rows_global}")
    for thr in THRESHOLDS:
        print(f"Sentences with ≥ {thr} in SDG+AI columns: {selected_global[thr]}")

    # ----------- Firms entirely filtered out -----------
    # "entirely filtered out" := zero selected for a threshold while having rows
    companies = sorted(company_stats.keys())
    print("\nCompany coverage summary:")
    print(f"Companies found: {len(companies)}")

    for thr in THRESHOLDS:
        filtered_out = sorted([
            c for c, stats in company_stats.items()
            if stats["total_rows"] > 0 and stats["selected"].get(thr, 0) == 0
        ])
        print(f"\nFirms with 0 sentences ≥ {thr}: {len(filtered_out)}")
        if filtered_out:
            # print a compact, comma-separated list
            print(", ".join(filtered_out))

    # (Optional) show a few with highest counts per thr
    for thr in THRESHOLDS:
        top = sorted(
            ((c, s["selected"][thr]) for c, s in company_stats.items()),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        print(f"\nTop 10 companies by sentences ≥ {thr}:")
        for c, cnt in top:
            print(f"  {c}: {cnt}")

if __name__ == "__main__":
    main()
