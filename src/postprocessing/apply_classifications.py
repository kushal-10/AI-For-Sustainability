#!/usr/bin/env python3
import os
import json
import argparse
from typing import Dict, Any, List, Tuple

import duckdb
import pandas as pd

DEFAULT_RESULTS_ALL = "data/batches/results/results_all_map.json"
DEFAULT_RESULTS_MAP = "data/batches/results/results_map.json"

SDG_DB_IN   = "data/dbs/sdg_hits.duckdb"
SDG_TBL_IN  = "sdg_hits"
SDG_DB_OUT  = "data/dbs/sdg_hits_classified.duckdb"
SDG_TBL_OUT = "sdg_hits_classified"

TECH_DB_IN   = "data/dbs/tech_hits.duckdb"
TECH_TBL_IN  = "tech_hits"
TECH_DB_OUT  = "data/dbs/tech_hits_classified.duckdb"
TECH_TBL_OUT = "tech_hits_classified"

UNCLASSIFIED = "unclassified"

def load_results_map(path_all: str, path_map: str) -> Dict[str, Dict[str, str]]:
    """
    Returns: { "sdg||<global_id>": {pattern: label, ...}, "tech||<global_id>": {...}, ... }
    """
    if os.path.exists(path_all):
        with open(path_all, "r", encoding="utf-8") as f:
            return json.load(f)
    if os.path.exists(path_map):
        with open(path_map, "r", encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(f"No results map found. Tried:\n  {path_all}\n  {path_map}")

def guess_hit_cols_sdg(cols: List[str]) -> List[str]:
    return [c for c in cols if c.startswith("hits_sdg")]

def guess_hit_cols_tech(cols: List[str]) -> List[str]:
    prefixes = ("hits_ai_ml", "hits_cloud_computing", "hits_big_data_blockchain", "hits_applications_practice")
    return [c for c in cols if c.startswith(prefixes)]

def guess_passage_col(cols: List[str]) -> str:
    for cand in ("passage", "sentence", "text", "content"):
        if cand in cols:
            return cand
    return "passage"  # best-effort

def ensure_dir(p: str):
    os.makedirs(os.path.dirname(p), exist_ok=True)

def to_json_str(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)

def classify_row_hits(row: pd.Series, mode: str, results_map: Dict[str, Dict[str, str]], hit_cols: List[str]) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """
    For a single row:
      - Build key: f"{mode}||{global_id}"
      - For each hits column (list of patterns), produce a dict {pattern: label_or_unclassified}
      - Track coverage counters.
    Returns:
      coverage: {"total_patterns": int, "classified_patterns": int}
      updated_values: {col_name: dict_json_ready, ...}
    """
    gid = str(row["global_id"])
    key = f"{mode}||{gid}"
    row_result = results_map.get(key, {})  # dict {pattern: label}
    coverage = {"total_patterns": 0, "classified_patterns": 0}
    updated: Dict[str, Any] = {}

    for col in hit_cols:
        raw = row[col]
        # Normalize: we expect a list, but tolerate JSON-strings
        patterns = None
        if isinstance(raw, list):
            patterns = raw
        elif isinstance(raw, str):
            s = raw.strip()
            if (s.startswith("[") and s.endswith("]")):
                try:
                    patterns = json.loads(s)
                except Exception:
                    patterns = []
            else:
                patterns = []
        else:
            patterns = []

        if not patterns:
            # Replace empty list with empty dict
            updated[col] = {}
            continue

        out_dict = {}
        for pat in patterns:
            if not isinstance(pat, str) or not pat.strip():
                continue
            coverage["total_patterns"] += 1
            label = row_result.get(pat)
            if label in ("symbolic", "substantive"):
                coverage["classified_patterns"] += 1
                out_dict[pat] = label
            else:
                out_dict[pat] = UNCLASSIFIED  # keep key, mark as unclassified

        updated[col] = out_dict

    return coverage, updated

def process_table(
    db_in: str, tbl_in: str, db_out: str, tbl_out: str,
    mode: str, results_map: Dict[str, Dict[str, str]]
):
    """
    mode: "sdg" or "tech"
    """
    print(f"[INFO] Loading {db_in}:{tbl_in} ...")
    con = duckdb.connect(db_in, read_only=True)
    df = con.execute(f"SELECT * FROM {tbl_in}").fetchdf()
    cols = list(df.columns)
    if "global_id" not in cols:
        raise ValueError(f"'global_id' column missing in {tbl_in}")

    if mode == "sdg":
        hit_cols = guess_hit_cols_sdg(cols)
    else:
        hit_cols = guess_hit_cols_tech(cols)

    if not hit_cols:
        raise ValueError(f"No hit columns found in {tbl_in} for mode={mode}")

    passage_col = guess_passage_col(cols)
    print(f"[INFO] Found {len(hit_cols)} hit columns: {hit_cols}")
    print(f"[INFO] Passage column: {passage_col}")

    total_rows = len(df)
    total_patterns = 0
    total_classified = 0
    total_unclassified_rows = 0

    # Build updated columns as JSON strings
    new_df = df.copy()
    for idx, row in df.iterrows():
        coverage, updated = classify_row_hits(row, mode, results_map, hit_cols)
        total_patterns += coverage["total_patterns"]
        total_classified += coverage["classified_patterns"]
        if coverage["total_patterns"] > 0 and coverage["classified_patterns"] < coverage["total_patterns"]:
            total_unclassified_rows += 1

        # Write JSON strings into new_df
        for col, d in updated.items():
            new_df.at[idx, col] = to_json_str(d)  # store dict as JSON string

    # Write out to a new DuckDB
    ensure_dir(db_out)
    cout = duckdb.connect(db_out)
    cout.execute(f"DROP TABLE IF EXISTS {tbl_out}")
    cout.register("df_tmp", new_df)
    cout.execute(f"CREATE TABLE {tbl_out} AS SELECT * FROM df_tmp")
    cout.unregister("df_tmp")
    cout.close()

    print(f"[OK] Wrote {db_out}:{tbl_out}  (rows: {total_rows})")
    if total_patterns > 0:
        pct = 100.0 * total_classified / total_patterns
        print(f"[REPORT] Patterns: {total_classified}/{total_patterns} classified ({pct:.2f}%).")
        print(f"[REPORT] Rows with any unclassified patterns: {total_unclassified_rows}")

def main():
    ap = argparse.ArgumentParser(description="Apply classification results to DuckDB hit tables (list -> dict) and save new DBs.")
    ap.add_argument("--results-all", default=DEFAULT_RESULTS_ALL, help="Path to results_all_map.json (preferred).")
    ap.add_argument("--results-map", default=DEFAULT_RESULTS_MAP, help="Fallback results_map.json.")
    ap.add_argument("--skip_sdg", action="store_true", help="Skip SDG table.")
    ap.add_argument("--skip_tech", action="store_true", help="Skip TECH table.")
    args = ap.parse_args()

    results_map = load_results_map(args.results_all, args.results_map)

    if not args.skip_sdg:
        process_table(SDG_DB_IN, SDG_TBL_IN, SDG_DB_OUT, SDG_TBL_OUT, "sdg", results_map)

    if not args.skip_tech:
        process_table(TECH_DB_IN, TECH_TBL_IN, TECH_DB_OUT, TECH_TBL_OUT, "tech", results_map)

if __name__ == "__main__":
    main()
