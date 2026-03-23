#!/usr/bin/env python3
"""
Update CSV language from DBs and verify (Year, Company) linkage to JSONs.

- Path convention for reports:
    data/jsons/COMPANY_NAME/YEAR/splits_semantic.json

- DB global_id convention:
    global_id = <YYYY><company_normalized><sentence_id>
    where company_normalized = company name lowercased with all non [a-z0-9] removed.

What this script does:
1) Verifies that each (year, normalized_company) seen in DB global_ids can be matched
   to a JSON path under data/jsons (by normalizing the folder company name the same way),
   and vice versa; prints a summary.
2) Updates the CSV language column (3rd position) from DBs:
   - Build language per (year, normalized company) via majority vote across SDG+Tech rows.
   - Fill CSV 'language' when empty for rows whose normalized (Company Name) + Year matches.
3) Blanks all metric columns for rows whose total metrics sum to zero (leaves as empty strings).

Usage:
  python3 src/postprocessing/update_language_and_verify.py
  # options:
  #   --csv data/exports/company_year_summary.csv
  #   --json-root data/jsons
  #   --sdg-db data/dbs/sdg_hits_classified.duckdb --sdg-table sdg_hits_classified
  #   --tech-db data/dbs/tech_hits_classified.duckdb --tech-table tech_hits_classified
  #   --out data/exports/company_year_summary.csv   (defaults to overwrite input)
"""

import os
import re
import json
import argparse
from typing import Dict, Tuple, List, Set
from collections import Counter, defaultdict

import duckdb
import pandas as pd

# -------- Defaults --------
CSV_PATH   = "data/exports/company_year_summary.csv"
JSON_ROOT  = "data/jsons"
SDG_DB     = "data/dbs/sdg_hits_classified.duckdb"
SDG_TABLE  = "sdg_hits_classified"
TECH_DB    = "data/dbs/tech_hits_classified.duckdb"
TECH_TABLE = "tech_hits_classified"

# -------- Normalization / Parsing --------
COMP_RETAIN = re.compile(r"[a-z0-9]")

def normalize_company(name: str) -> str:
    """Lowercase, keep only [a-z0-9]."""
    if not isinstance(name, str):
        return ""
    s = name.lower()
    return "".join(ch for ch in s if COMP_RETAIN.match(ch))

GID_RE = re.compile(r"^(?P<year>\d{4})(?P<company>[a-z0-9]+?)(?P<sent>\d+)$")

def parse_global_id(gid: str) -> Tuple[str, str]:
    """
    Parse global_id into (year, normalized_company). Returns ("","") if not matched.
    Example: '2016siltronicag61' -> ('2016','siltronicag')
    """
    if not isinstance(gid, str):
        return "", ""
    m = GID_RE.match(gid.strip().lower())
    if not m:
        return "", ""
    return m.group("year"), m.group("company")

# -------- IO helpers --------
def df_from_db(db_path: str, table: str, cols=("global_id","language")) -> pd.DataFrame:
    if not os.path.exists(db_path):
        return pd.DataFrame(columns=list(cols))
    con = duckdb.connect(db_path, read_only=True)
    try:
        qcols = ", ".join(cols)
        return con.execute(f"SELECT {qcols} FROM {table}").fetchdf()
    finally:
        con.close()

def iter_json_reports(json_root: str) -> List[Tuple[str, str, str]]:
    """
    Return list of (company_folder_name, year, json_path) for all splits_semantic.json found
    under json_root/COMPANY_NAME/YEAR/splits_semantic.json
    """
    out = []
    for root, _, files in os.walk(json_root):
        for fn in files:
            if fn != "splits_semantic.json":
                continue
            full = os.path.join(root, fn)
            rel = os.path.relpath(os.path.abspath(full), os.path.abspath(json_root))
            parts = rel.split(os.sep)
            if len(parts) >= 3:
                company = parts[-3].strip()
                year = parts[-2].strip()
                out.append((company, year, full))
    return sorted(out)

# -------- Build language map from DBs --------
def build_language_map_from_dbs(sdg_df: pd.DataFrame, tech_df: pd.DataFrame) -> Dict[Tuple[str,str], str]:
    """
    Majority vote language for each (year, normalized_company) across both DBs.
    """
    counters: Dict[Tuple[str,str], Counter] = defaultdict(Counter)

    for df in (sdg_df, tech_df):
        if df.empty:
            continue
        for _, r in df.iterrows():
            gid = str(r.get("global_id") or "")
            lang = str(r.get("language") or "").strip()
            if not gid or not lang:
                continue
            year, comp = parse_global_id(gid)
            if year and comp:
                counters[(year, comp)][lang] += 1

    lang_map: Dict[Tuple[str,str], str] = {}
    for key, cnt in counters.items():
        lang_map[key] = cnt.most_common(1)[0][0] if cnt else ""
    return lang_map

# -------- Verification --------
def verify_linkage(json_root: str, sdg_df: pd.DataFrame, tech_df: pd.DataFrame) -> Dict[str, int]:
    """
    Verify that (year, normalized_company) from DB global_ids can be linked to JSON paths.
    - JSON side: normalize each COMPANY_NAME folder and pair with its YEAR.
    - DB side : parse from global_id.
    Prints a short report; returns counts.
    """
    # DB pairs
    db_pairs: Set[Tuple[str,str]] = set()
    bad_gid = 0
    for df in (sdg_df, tech_df):
        if df.empty: continue
        for gid in df["global_id"].astype(str).tolist():
            y, c = parse_global_id(gid)
            if y and c:
                db_pairs.add((y, c))
            else:
                bad_gid += 1

    # JSON pairs
    json_pairs: Set[Tuple[str,str]] = set()
    json_pair_to_company: Dict[Tuple[str,str], str] = {}
    for comp, year, _ in iter_json_reports(json_root):
        norm = normalize_company(comp)
        if year and norm:
            json_pairs.add((year, norm))
            json_pair_to_company[(year, norm)] = comp  # keep original folder name

    missing_in_json = sorted([p for p in db_pairs if p not in json_pairs])[:20]
    missing_in_db   = sorted([p for p in json_pairs if p not in db_pairs])[:20]

    print("\n=== Verification: DB ↔ JSON linkage ===")
    print(f"DB pairs              : {len(db_pairs)} (bad global_id parses: {bad_gid})")
    print(f"JSON pairs            : {len(json_pairs)}")
    print(f"DB→JSON missing pairs : {len([p for p in db_pairs if p not in json_pairs])}")
    if missing_in_json:
        print("  Examples:", missing_in_json[:5])
    print(f"JSON→DB missing pairs : {len([p for p in json_pairs if p not in db_pairs])}")
    if missing_in_db:
        print("  Examples:", missing_in_db[:5])

    return {
        "db_pairs": len(db_pairs),
        "json_pairs": len(json_pairs),
        "db_to_json_missing": len([p for p in db_pairs if p not in json_pairs]),
        "json_to_db_missing": len([p for p in json_pairs if p not in db_pairs]),
        "bad_global_ids": bad_gid,
    }

# -------- Update CSV language & blank all-zero metrics --------
def update_csv_language_and_blank_zeros(
    csv_path: str,
    out_path: str,
    json_root: str,
    sdg_df: pd.DataFrame,
    tech_df: pd.DataFrame
):
    if not os.path.exists(csv_path):
        raise SystemExit(f"[ERR] CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str)

    # Ensure base cols
    for base in ("Company Name", "Year"):
        if base not in df.columns:
            raise SystemExit(f"[ERR] CSV must contain '{base}' column.")

    # Ensure 'language' at 3rd col
    if "language" not in df.columns:
        df.insert(2, "language", "")
    else:
        if df.columns.get_loc("language") != 2:
            cols = df.columns.tolist()
            cols.insert(2, cols.pop(cols.index("language")))
            df = df[cols]

    # Keep tokens wherever it is; we won't modify it.
    if "tokens" not in df.columns:
        df["tokens"] = 0

    # Build language map per (Year, normalized_company) from DBs
    lang_map = build_language_map_from_dbs(sdg_df, tech_df)

    # Build a mapping from JSON structure to original Company casing, to align with CSV
    json_pairs = {}
    for comp, year, _ in iter_json_reports(json_root):
        json_pairs[(year, normalize_company(comp))] = comp  # COMPANY_NAME as on disk

    # Fill language if blank, matching by normalized company and Year
    filled = 0
    for idx, row in df.iterrows():
        comp_csv = str(row["Company Name"])
        year_csv = str(row["Year"])
        key_norm = (year_csv, normalize_company(comp_csv))
        lang = lang_map.get(key_norm, "")
        if lang and (not isinstance(row["language"], str) or not row["language"].strip()):
            df.at[idx, "language"] = lang
            filled += 1

    print(f"\n[LANG] Filled language for {filled} CSV rows using DBs.")

    # Blank all-zero metrics (leave tokens/language intact)
    metric_cols = [c for c in df.columns if c not in ("Company Name", "Year", "language", "tokens")]
    if metric_cols:
        # Compute numeric sums per row
        numeric_df = df[metric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        zero_mask = (numeric_df.sum(axis=1) == 0)
        blanked_rows = int(zero_mask.sum())
        # Set those metric cells to empty string
        for c in metric_cols:
            df.loc[zero_mask, c] = ""
        print(f"[BLANK] Set metric cells blank for {blanked_rows} all-zero rows.")

    # Write out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[OK] Wrote updated CSV -> {out_path}")

# -------- CLI --------
def main():
    ap = argparse.ArgumentParser(description="Verify DB↔JSON linkage and update CSV language; blank all-zero metric rows.")
    ap.add_argument("--csv", default=CSV_PATH, help="Input CSV to update")
    ap.add_argument("--out", default=CSV_PATH, help="Output CSV path (default: overwrite input)")
    ap.add_argument("--json-root", default=JSON_ROOT)
    ap.add_argument("--sdg-db", default=SDG_DB)
    ap.add_argument("--sdg-table", default=SDG_TABLE)
    ap.add_argument("--tech-db", default=TECH_DB)
    ap.add_argument("--tech-table", default=TECH_TABLE)
    args = ap.parse_args()

    # Load DB subsets
    sdg_df = df_from_db(args.sdg_db, args.sdg_table, cols=("global_id","language"))
    tech_df = df_from_db(args.tech_db, args.tech_table, cols=("global_id","language"))

    # 1) Verify linkage
    verify_linkage(args.json_root, sdg_df, tech_df)

    # 2) Update language & blank zeros
    update_csv_language_and_blank_zeros(args.csv, args.out, args.json_root, sdg_df, tech_df)

if __name__ == "__main__":
    main()
