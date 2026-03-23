#!/usr/bin/env python3
"""
Create a single CSV aggregated by Company Name + Year:

Columns:
- Company Name, Year
- SDG 1 Substantive ... SDG 17 Substantive
- SDG 1 Symbolic ... SDG 17 Symbolic
- AI-ML Substantive, AI-ML Symbolic
- Cloud Computing Substantive, Cloud Computing Symbolic
- Big Data/Blockchain Substantive, Big Data/Blockchain Symbolic
- Applications Practice Substantive, Applications Practice Symbolic

For each company+year (a report), values are the total frequency across all
passages in that report. Only 'symbolic' and 'substantive' are counted;
'unclassified' is ignored.

Usage:
  python3 src/postprocessing/build_company_year_summary.py
  python3 src/postprocessing/build_company_year_summary.py --out data/exports/company_year_summary.csv
"""

import os
import re
import json
import argparse
from typing import Any, Dict, List, Tuple, DefaultDict
from collections import defaultdict

import duckdb
import pandas as pd

# ---------- Defaults ----------
SDG_DB   = "data/dbs/sdg_hits_classified.duckdb"
SDG_TBL  = "sdg_hits_classified"
TECH_DB  = "data/dbs/tech_hits_classified.duckdb"
TECH_TBL = "tech_hits_classified"

OUT_CSV  = "data/exports/company_year_summary.csv"

VALID_LABELS = {"symbolic", "substantive"}

TECH_PREFIXES = {
    "AI-ML": "hits_ai_ml",
    "Cloud Computing": "hits_cloud_computing",
    "Big Data/Blockchain": "hits_big_data_blockchain",
    "Applications Practice": "hits_applications_practice",
}

# ---------- IO ----------
def get_df(db_path: str, table: str) -> pd.DataFrame:
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute(f"SELECT * FROM {table}").fetchdf()
    con.close()
    return df

# ---------- Column detection ----------
def detect_sdg_cols(cols: List[str]) -> Dict[int, List[str]]:
    """
    Return mapping: {sdg_number -> [column names]} for columns like hits_sdg1 ... hits_sdg17
    Case-insensitive; will collect any columns that contain digits 1..17 after 'hits_sdg'.
    """
    sdg_map: Dict[int, List[str]] = defaultdict(list)
    for c in cols:
        if not isinstance(c, str):
            continue
        lc = c.lower()
        if not lc.startswith("hits_sdg"):
            continue
        # extract the first number after hits_sdg
        m = re.search(r"hits_sdg\s*([0-9]{1,2})", lc)
        if not m:
            continue
        n = int(m.group(1))
        if 1 <= n <= 17:
            sdg_map[n].append(c)
    return sdg_map

def detect_tech_cols(cols: List[str]) -> Dict[str, List[str]]:
    """
    Return mapping: {friendly_name -> [column names]} for known tech categories.
    Case-insensitive startswith for the known prefixes.
    """
    out: Dict[str, List[str]] = {k: [] for k in TECH_PREFIXES.keys()}
    lowers = {c: c.lower() for c in cols if isinstance(c, str)}
    for friendly, pref in TECH_PREFIXES.items():
        p = pref.lower()
        for orig, low in lowers.items():
            if low.startswith(p):
                out[friendly].append(orig)
    return out

# ---------- Parsing ----------
def parse_classified_cell(cell: Any) -> Dict[str, str]:
    """
    Cells are dicts or JSON strings of {pattern: label}.
    Return {pattern: label} with labels lowercased; invalid cells -> {}.
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return {}
    if isinstance(cell, dict):
        return {str(k): (str(v).strip().lower() if v is not None else "")
                for k, v in cell.items() if isinstance(k, str)}
    if isinstance(cell, list):
        # should not happen in *classified* tables; ignore / count nothing
        return {}
    if isinstance(cell, str):
        s = cell.strip()
        if not s:
            return {}
        # JSON parse, with salvage of invalid escapes (\s, \w, etc.)
        try:
            obj = json.loads(s)
        except Exception:
            try:
                s2 = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)
                obj = json.loads(s2)
            except Exception:
                return {}
        if isinstance(obj, dict):
            return {str(k): (str(v).strip().lower() if v is not None else "")
                    for k, v in obj.items() if isinstance(k, str)}
        return {}
    return {}

# ---------- Aggregation ----------
def ensure_row(agg: DefaultDict[Tuple[str, str], Dict[str, int]],
               company: str, year: str,
               sdg_sub_cols: List[str], sdg_sym_cols: List[str],
               tech_cols: List[str]):
    key = (company, year)
    if key in agg:
        return
    # init all columns to 0
    row: Dict[str, int] = {}
    for col in sdg_sub_cols + sdg_sym_cols + tech_cols:
        row[col] = 0
    agg[key] = row

def main():
    ap = argparse.ArgumentParser(description="Build Company-Year summary CSV from classified DuckDBs.")
    ap.add_argument("--sdg-db", default=SDG_DB);   ap.add_argument("--sdg-table", default=SDG_TBL)
    ap.add_argument("--tech-db", default=TECH_DB); ap.add_argument("--tech-table", default=TECH_TBL)
    ap.add_argument("--out", default=OUT_CSV)
    args = ap.parse_args()

    # Load dataframes
    df_sdg = get_df(args.sdg_db, args.sdg_table)
    df_tech = get_df(args.tech_db, args.tech_table)

    # Basic column checks
    for name, df in [("SDG", df_sdg), ("TECH", df_tech)]:
        for needed in ("company", "year"):
            if needed not in df.columns:
                raise SystemExit(f"[ERR] {name} table missing '{needed}' column.")

    # Detect columns
    sdg_cols_map = detect_sdg_cols(df_sdg.columns.tolist())  # {1:[...], 2:[...], ...}
    tech_cols_map = detect_tech_cols(df_tech.columns.tolist())  # {"AI-ML":[...], ...}

    # Build final column list
    sdg_sub_cols = [f"SDG {i} Substantive" for i in range(1, 18)]
    sdg_sym_cols = [f"SDG {i} Symbolic" for i in range(1, 18)]

    tech_pairs = [
        ("AI-ML", "AI-ML Substantive", "AI-ML Symbolic"),
        ("Cloud Computing", "Cloud Computing Substantive", "Cloud Computing Symbolic"),
        ("Big Data/Blockchain", "Big Data/Blockchain Substantive", "Big Data/Blockchain Symbolic"),
        ("Applications Practice", "Applications Practice Substantive", "Applications Practice Symbolic"),
    ]
    tech_all_cols = [p for _, a, b in tech_pairs for p in (a, b)]

    # Aggregator
    agg: DefaultDict[Tuple[str, str], Dict[str, int]] = defaultdict(dict)

    # --------- Accumulate SDG counts ---------
    for _, row in df_sdg.iterrows():
        company = str(row["company"])
        year = str(row["year"])
        ensure_row(agg, company, year, sdg_sub_cols, sdg_sym_cols, tech_all_cols)

        # For each SDG number, accumulate counts from all its columns in this row
        for sdg_num, cols in sdg_cols_map.items():
            if not cols:
                continue
            # Sum labels across all columns for that SDG number
            symbols = 0
            subs = 0
            for c in cols:
                d = parse_classified_cell(row.get(c))
                for _, v in d.items():
                    if v == "symbolic":
                        symbols += 1
                    elif v == "substantive":
                        subs += 1
            if symbols or subs:
                agg[(company, year)][f"SDG {sdg_num} Symbolic"] += symbols
                agg[(company, year)][f"SDG {sdg_num} Substantive"] += subs

    # --------- Accumulate TECH counts ---------
    for _, row in df_tech.iterrows():
        company = str(row["company"])
        year = str(row["year"])
        ensure_row(agg, company, year, sdg_sub_cols, sdg_sym_cols, tech_all_cols)

        for friendly, cols in tech_cols_map.items():
            if not cols:
                continue
            symbols = 0
            subs = 0
            for c in cols:
                d = parse_classified_cell(row.get(c))
                for _, v in d.items():
                    if v == "symbolic":
                        symbols += 1
                    elif v == "substantive":
                        subs += 1
            # find output column names for this category
            for (name, sub_col, sym_col) in tech_pairs:
                if name == friendly:
                    agg[(company, year)][sub_col] += subs
                    agg[(company, year)][sym_col] += symbols
                    break

    # --------- Build final DataFrame ---------
    records: List[Dict[str, Any]] = []
    for (company, year), counts in agg.items():
        rec = {"Company Name": company, "Year": year}
        # fill all columns explicitly with zeros if missing
        for col in sdg_sub_cols + sdg_sym_cols + tech_all_cols:
            rec[col] = int(counts.get(col, 0))
        records.append(rec)

    out_df = pd.DataFrame(records)

    # Consistent sort by company/year
    if not out_df.empty:
        out_df = out_df.sort_values(by=["Company Name", "Year"], kind="stable")

    # Ensure output dirs and write CSV
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"[OK] Wrote CSV -> {args.out}  (rows: {len(out_df)})")

    # Quick sanity print (avoid boolean index mismatch)
    metric_cols = [c for c in out_df.columns if c not in ("Company Name", "Year")]
    nonzero_cols = []
    for c in metric_cols:
        col_sum = pd.to_numeric(out_df[c], errors="coerce").fillna(0).astype(float).sum()
        if col_sum > 0:
            nonzero_cols.append(c)

    print(f"[INFO] Non-zero columns (excluding Company/Year): "
          f"{', '.join(nonzero_cols) if nonzero_cols else '(none)'}")


if __name__ == "__main__":
    main()
