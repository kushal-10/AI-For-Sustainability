# src/build_ai_counts_csv.py
import os
import json
import csv
from collections import defaultdict

import duckdb

# ----- Hardcoded config -----
ROOT_DIR = "data/jsons"                 # discovers ALL company/year combos here
DB_PATH = "data/matches.duckdb"         # your existing DB with matched_sentences
TABLE = "matched_sentences"
TECH_JSON = "data/keywords_tech.json"
OUT_CSV = "data/ai_counts_by_company_year.csv"

# The four stacks (must match keys in keywords_tech.json)
STACKS = ["ai_ml", "cloud_computing", "big_data_blockchain", "applications_practice"]


def find_all_company_years(root: str):
    """Return sorted list of (company, year) from folder structure."""
    pairs = []
    if not os.path.isdir(root):
        return pairs
    for company in os.listdir(root):
        cdir = os.path.join(root, company)
        if not os.path.isdir(cdir):
            continue
        for year in os.listdir(cdir):
            ydir = os.path.join(cdir, year)
            if not os.path.isdir(ydir):
                continue
            sp = os.path.join(ydir, "splits_semantic.json")
            if os.path.isfile(sp):
                pairs.append((company, year))
    # Sort by company, then year
    pairs.sort(key=lambda x: (x[0].lower(), x[1]))
    return pairs


def load_stack_terms(path):
    """Load keywords_tech.json and build lowercase term sets per stack."""
    with open(path, "r", encoding="utf-8") as f:
        tech = json.load(f)
    stack2terms = {k: set() for k in STACKS}
    for k, v in tech.items():
        if isinstance(v, list) and k in stack2terms:
            stack2terms[k] = {t.lower().strip() for t in v}
    return stack2terms


def main():
    os.makedirs("data", exist_ok=True)

    # 1) Discover ALL company-year combos from the filesystem
    all_pairs = find_all_company_years(ROOT_DIR)
    if not all_pairs:
        print(f"[ERROR] No reports found under {ROOT_DIR}")
        return
    print(f"[INFO] Discovered {len(all_pairs)} company-year reports from {ROOT_DIR}")

    # 2) Prepare zero-initialized counts for EVERY company-year
    counts = {(c, y): {s: 0 for s in STACKS} for (c, y) in all_pairs}

    # 3) Load stacks and their terms
    stack2terms = load_stack_terms(TECH_JSON)

    # 4) Read AI hits from DuckDB (may be missing for zero-hit reports)
    con = duckdb.connect(DB_PATH)
    # only need company, year, ai_keywords
    rows = con.execute(f"SELECT company, year, ai_keywords FROM {TABLE}").fetchall()
    con.close()

    # 5) Aggregate sentence-level AI mentions into company-year counts
    for company, year, ai_json in rows:
        if not ai_json or ai_json == "null":
            continue
        try:
            hits = json.loads(ai_json)
        except Exception:
            continue
        if not isinstance(hits, list):
            continue

        hitset = {str(h).lower().strip() for h in hits}
        # One increment per stack IF any term from that stack appears in the sentence
        d = counts.get((company, year))
        if d is None:
            # In case company/year exists in DB but not in FS (rare), add it so it's not lost
            d = {s: 0 for s in STACKS}
            counts[(company, year)] = d
        for stack in STACKS:
            if stack2terms[stack] & hitset:
                d[stack] += 1

    # 6) Write complete CSV with ALL company-year rows (including zero rows)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["company", "year"] + [f"{s}_count" for s in STACKS]
        w.writerow(header)
        for (company, year) in sorted(counts.keys(), key=lambda x: (x[0].lower(), x[1])):
            d = counts[(company, year)]
            w.writerow([company, year] + [d[s] for s in STACKS])

    # 7) Print which company-years are all-zero across the four stacks
    zero_pairs = [(c, y) for (c, y), d in counts.items() if all(d[s] == 0 for s in STACKS)]
    nonzero = len(counts) - len(zero_pairs)

    print(f"[OK] Wrote {len(counts)} rows to {OUT_CSV}")
    print(f"[INFO] Non-zero company-years: {nonzero} | All-zero company-years: {len(zero_pairs)}")

    if zero_pairs:
        print("\n[ZERO SCORE COMPANY-YEARS]")
        for c, y in zero_pairs:
            print(f"- {c} — {y}")


if __name__ == "__main__":
    main()
