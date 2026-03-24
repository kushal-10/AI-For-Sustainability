#!/usr/bin/env python3
"""
Augment company_year_summary.csv based strictly on folder paths:

Expected path pattern:
  data/jsons/COMPANY_NAME/YEAR/splits_semantic.json

For each such JSON:
  - Derive (Company Name, Year) from PATH (no lowercasing)
  - Compute 'tokens' = total tokens across all STRING values inside the JSON
  - Ensure a CSV row exists; if missing, add an ALL-ZEROS row for all metric columns
  - Fill 'language' (3rd column) from DuckDBs using most common non-null per (Company, Year)
  - Ensure 'tokens' is the LAST column (create/update)

Usage:
  python3 src/postprocessing/augment_summary_with_jsons.py
  python3 src/postprocessing/augment_summary_with_jsons.py --out data/exports/company_year_summary_with_tokens.csv
"""

import os
import json
import argparse
from typing import Any, Dict, Tuple, List
from collections import Counter, defaultdict

import duckdb
import pandas as pd
from tqdm import tqdm

# -------- Paths / Defaults --------
JSON_ROOT = "data/jsons"
CSV_PATH  = "data/exports/company_year_summary.csv"

SDG_DB   = "data/dbs/sdg_hits_classified.duckdb"
SDG_TBL  = "sdg_hits_classified"
TECH_DB  = "data/dbs/tech_hits_classified.duckdb"
TECH_TBL = "tech_hits_classified"

# -------- Token counter (tiktoken optional) --------
class TokenCounter:
    def __init__(self):
        self._tok = None
        self._use = False
        try:
            import tiktoken  # type: ignore
            try:
                self._tok = tiktoken.get_encoding("o200k_base")
            except Exception:
                self._tok = tiktoken.get_encoding("cl100k_base")
            self._use = True
        except Exception:
            self._use = False

    def count(self, s: str) -> int:
        if not s:
            return 0
        if self._use and self._tok:
            try:
                return len(self._tok.encode(str(s)))
            except Exception:
                pass
        # heuristic fallback ~4 chars/token
        s = str(s)
        return max(1, int(len(s) / 4.0))

def json_token_count(obj: Any, tc: TokenCounter) -> int:
    """Sum tokens of all STRING values in a JSON-like structure."""
    if obj is None:
        return 0
    if isinstance(obj, str):
        return tc.count(obj)
    if isinstance(obj, (int, float, bool)):
        return 0
    if isinstance(obj, dict):
        return sum(json_token_count(v, tc) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(json_token_count(x, tc) for x in obj)
    return 0

# -------- DB helpers --------
def df_from_db(db_path: str, table: str) -> pd.DataFrame:
    if not os.path.exists(db_path):
        return pd.DataFrame(columns=["company", "year", "language"])
    con = duckdb.connect(db_path, read_only=True)
    try:
        df = con.execute(f"SELECT company, year, language FROM {table}").fetchdf()
    finally:
        con.close()
    return df

def build_language_map() -> Dict[Tuple[str, str], str]:
    """
    Mode of non-null languages across SDG/Tech for each (company, year).
    """
    from collections import Counter, defaultdict
    lang_counter: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    for db, tbl in [(SDG_DB, SDG_TBL), (TECH_DB, TECH_TBL)]:
        df = df_from_db(db, tbl)
        if df.empty:
            continue
        for _, r in df.iterrows():
            comp = str(r.get("company") or "").strip()
            year = str(r.get("year") or "").strip()
            lang = str(r.get("language") or "").strip()
            if comp and year and lang:
                lang_counter[(comp, year)][lang] += 1
    out = {}
    for k, c in lang_counter.items():
        out[k] = c.most_common(1)[0][0] if c else ""
    return out

# -------- JSON scanners --------
def find_splits_files(json_root: str) -> List[str]:
    hits = []
    for root, _, files in os.walk(json_root):
        for fn in files:
            if fn == "splits_semantic.json":
                hits.append(os.path.join(root, fn))
    return sorted(hits)

def company_year_from_path(path: str, json_root: str) -> Tuple[str, str]:
    """
    Expect: <json_root>/<COMPANY_NAME>/<YEAR>/splits_semantic.json
    Returns (Company Name, Year) exactly as in path (no lowercasing).
    """
    rel = os.path.relpath(os.path.abspath(path), os.path.abspath(json_root))
    parts = rel.split(os.sep)
    # Expect [..., COMPANY_NAME, YEAR, 'splits_semantic.json']
    if len(parts) < 3 or parts[-1] != "splits_semantic.json":
        return "", ""
    company = parts[-3].strip()
    year = parts[-2].strip()
    return company, year

# -------- Main logic --------
def main():
    ap = argparse.ArgumentParser(description="Ensure CSV has rows for every report path; add language and tokens.")
    ap.add_argument("--json-root", default=JSON_ROOT)
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--out", default=CSV_PATH, help="Output CSV path (default: overwrite input)")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"[ERR] CSV not found: {args.csv}")

    # Load CSV (as strings), convert metrics to ints later
    df = pd.read_csv(args.csv, dtype=str)

    # Ensure base columns present
    if "Company Name" not in df.columns or "Year" not in df.columns:
        raise SystemExit("[ERR] CSV must contain 'Company Name' and 'Year' columns.")

    # Insert/move 'language' to 3rd position
    if "language" not in df.columns:
        df.insert(2, "language", "")
    else:
        if df.columns.get_loc("language") != 2:
            cols = df.columns.tolist()
            cols.insert(2, cols.pop(cols.index("language")))
            df = df[cols]

    # Append/move 'tokens' to last position
    if "tokens" not in df.columns:
        df["tokens"] = 0
    else:
        if df.columns[-1] != "tokens":
            cols = df.columns.tolist()
            cols.append(cols.pop(cols.index("tokens")))
            df = df[cols]

    # Identify metric columns to zero-fill for new rows
    metric_cols = [c for c in df.columns if c not in ("Company Name", "Year", "language", "tokens")]

    # Coerce metric columns to ints
    for c in metric_cols + ["tokens"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # Build language map from DBs
    lang_map = build_language_map()

    # Scan JSONs
    tc = TokenCounter()
    json_paths = find_splits_files(args.json_root)

    added, updated, skipped = 0, 0, 0

    # Use a (Company Name, Year) index for fast updates
    df.set_index(["Company Name", "Year"], inplace=True, drop=False)

    for jp in tqdm(json_paths):
        company, year = company_year_from_path(jp, args.json_root)
        if not company or not year:
            skipped += 1
            continue

        # Load JSON to compute tokens
        try:
            with open(jp, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            skipped += 1
            continue

        tokens = json_token_count(payload, tc)
        key = (company, year)

        if key in df.index:
            # update tokens; fill language if empty
            df.at[key, "tokens"] = int(tokens)
            lang = lang_map.get(key, "")
            if lang and not str(df.at[key, "language"]).strip():
                df.at[key, "language"] = lang
            updated += 1
        else:
            # add an all-zero row
            row = {"Company Name": company, "Year": year, "language": lang_map.get(key, "")}
            for c in metric_cols:
                row[c] = 0
            row["tokens"] = int(tokens)
            df.loc[key, :] = row
            added += 1

    # Restore normal columns
    df = df.reset_index(drop=True)

    # Sort deterministically
    if not df.empty:
        try:
            df["__Y__"] = pd.to_numeric(df["Year"], errors="coerce")
            df.sort_values(by=["Company Name", "__Y__", "Year"], inplace=True, kind="stable")
            df.drop(columns="__Y__", inplace=True)
        except Exception:
            df.sort_values(by=["Company Name", "Year"], inplace=True, kind="stable")

    # Reassert column order: language 3rd, tokens last
    metric_cols = [c for c in df.columns if c not in ("Company Name", "Year", "language", "tokens")]
    cols_final = ["Company Name", "Year", "language"] + metric_cols + ["tokens"]
    df = df[cols_final]

    # Ensure numeric types for metrics/tokens
    for c in metric_cols + ["tokens"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # Write out
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[OK] Updated CSV -> {args.out}")
    print(f"   splits_semantic.json files found : {len(json_paths)}")
    print(f"   Rows added (all-zero)            : {added}")
    print(f"   Rows updated                     : {updated}")
    if skipped:
        print(f"   Skipped (bad path/read)          : {skipped}")

if __name__ == "__main__":
    main()
