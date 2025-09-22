#!/usr/bin/env python3
"""
Scan classified DuckDB tables (loaded as pandas DataFrames) and report
'unclassified' labels in hits columns.

- Robustly parses each cell (dict / JSON string / list / literal 'unclassified' / empty)
- Counts per-column totals and unclassified counts
- Reports rows with ANY unclassified (key-level) and cells that are literally 'unclassified'
- Shows sample rows and (optionally) exports a CSV

Usage:
  python3 src/postprocessing/check_unclassified_df.py
  python3 src/postprocessing/check_unclassified_df.py --n-samples 20 --export-csv data/batches/unclassified_df.csv --show-passages
  python3 src/postprocessing/check_unclassified_df.py --skip-sdg
  python3 src/postprocessing/check_unclassified_df.py --skip-tech
"""

import os
import re
import json
import argparse
from typing import Any, Dict, List, Tuple

import duckdb
import pandas as pd

# Defaults
SDG_DB  = "data/dbs/sdg_hits_classified.duckdb"
SDG_TBL = "sdg_hits_classified"
TECH_DB = "data/dbs/tech_hits_classified.duckdb"
TECH_TBL= "tech_hits_classified"

def get_df(db_path: str, table: str) -> pd.DataFrame:
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute(f"SELECT * FROM {table}").fetchdf()
    con.close()
    return df

def guess_hit_cols_sdg(cols: List[str]) -> List[str]:
    lower = {c: c.lower() for c in cols if isinstance(c, str)}
    hits = [c for c in cols if isinstance(c, str) and lower[c].startswith("hits_sdg")]
    if hits: return hits
    hits = [c for c in cols if isinstance(c, str) and lower[c].startswith("hits") and "sdg" in lower[c]]
    if hits: return hits
    return [c for c in cols if isinstance(c, str) and lower[c].startswith("hits")]

def guess_hit_cols_tech(cols: List[str]) -> List[str]:
    lower = {c: c.lower() for c in cols if isinstance(c, str)}
    prefixes = ("hits_ai_ml", "hits_cloud_computing", "hits_big_data_blockchain", "hits_applications_practice")
    hits = [c for c in cols if isinstance(c, str) and any(lower[c].startswith(p) for p in prefixes)]
    if hits: return hits
    return [c for c in cols if isinstance(c, str) and lower[c].startswith("hits") and "sdg" not in lower[c]]

def parse_cell_to_dict(cell: Any) -> Tuple[Dict[str, str], bool]:
    """
    Return (dict, cell_is_literal_unclassified)
    dict maps pattern -> label (symbolic|substantive|unclassified)
    """
    # None / NaN
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return {}, False

    # Already a dict
    if isinstance(cell, dict):
        out = {}
        for k, v in cell.items():
            if isinstance(k, str):
                val = (str(v).strip().lower() if v is not None else "")
                if val in ("symbolic", "substantive", "unclassified"):
                    out[k] = val
                else:
                    out[k] = "unclassified" if val else "unclassified"
        return out, False

    # List of patterns -> treat as all unclassified (rare in *classified* tables, but safe)
    if isinstance(cell, list):
        out = {}
        for item in cell:
            if isinstance(item, str) and item.strip():
                out[item] = "unclassified"
        return out, False

    # Strings
    if isinstance(cell, str):
        s = cell.strip()
        if not s:
            return {}, False
        sl = s.lower()
        if sl in ("unclassified", '"unclassified"'):
            # whole cell is literally 'unclassified'
            return {"__cell__": "unclassified"}, True
        # JSON-ish?
        if s[0] in "{[" and s[-1] in "}]" and len(s) >= 2:
            try:
                obj = json.loads(s)
                # recurse on parsed object
                return parse_cell_to_dict(obj)
            except Exception:
                # if it's a JSON-like text with invalid escapes, try to salvage by doubling backslashes
                try:
                    s2 = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)
                    obj = json.loads(s2)
                    return parse_cell_to_dict(obj)
                except Exception:
                    return {}, False
        # Non-JSON arbitrary string -> ignore
        return {}, False

    # Anything else -> ignore
    return {}, False

def analyze_df(df: pd.DataFrame, table_name: str, hit_cols: List[str], n_samples: int, show_passages: bool):
    if not hit_cols:
        print(f"\n=== {table_name} ===")
        print(f"Columns ({len(df.columns)}): {df.columns.tolist()}")
        print("!! No hit columns auto-detected.")
        return

    has_company = "company" in df.columns
    has_year = "year" in df.columns
    has_passage = "passage" in df.columns
    has_gid = "global_id" in df.columns

    print(f"\n=== {table_name} ===")
    print(f"Rows: {len(df)}")
    print(f"Detected hit columns: {hit_cols}")

    tot_keys = 0
    tot_uncl = 0
    rows_with_any_uncl = 0
    rows_with_literal_cell_uncl = 0

    samples: List[Dict[str, Any]] = []

    # Iterate rows once, accumulate per-column stats
    per_col = {c: {"total_keys": 0, "unclassified_keys": 0, "rows_with_unclassified": 0} for c in hit_cols}

    for idx, row in df.iterrows():
        row_any_uncl = False
        row_cell_uncl = False
        for col in hit_cols:
            cell_dict, is_literal_uncl = parse_cell_to_dict(row[col] if col in row else None)

            if is_literal_uncl:
                row_cell_uncl = True

            # count keys
            keys_in_cell = len(cell_dict)
            if keys_in_cell:
                per_col[col]["total_keys"] += keys_in_cell
                tot_keys += keys_in_cell

            # count unclassified values
            this_uncl = 0
            for k, v in cell_dict.items():
                if v == "unclassified":
                    this_uncl += 1
                    tot_uncl += 1
                    # collect sample until we hit n_samples per column (soft)
                    if per_col[col].get("sampled", 0) < n_samples:
                        sample = {
                            "table": table_name,
                            "column": col,
                            "global_id": row["global_id"] if has_gid else None,
                            "pattern": k,
                            "label": v
                        }
                        if has_company: sample["company"] = row["company"]
                        if has_year:    sample["year"] = row["year"]
                        if show_passages and has_passage: sample["passage"] = row["passage"]
                        samples.append(sample)
                        per_col[col]["sampled"] = per_col[col].get("sampled", 0) + 1

            if this_uncl > 0:
                row_any_uncl = True
                per_col[col]["rows_with_unclassified"] += 1

        if row_any_uncl:
            rows_with_any_uncl += 1
        if row_cell_uncl:
            rows_with_literal_cell_uncl += 1

    # Print summary
    print(f"Total keys: {tot_keys} | Unclassified keys: {tot_uncl} ({(100.0*tot_uncl/tot_keys if tot_keys else 0):.2f}%)")
    print(f"Rows with ANY unclassified keys: {rows_with_any_uncl}")
    print(f"Rows with literal cell == 'unclassified': {rows_with_literal_cell_uncl}")

    print("\nBy column:")
    for col, stats in per_col.items():
        tk = stats["total_keys"]
        uk = stats["unclassified_keys"] if "unclassified_keys" in stats else None  # not used (we used tot_uncl aggregated); compute per col now
        # compute per-col unclassified keys accurately
        # Recompute uk from samples + second pass is costly; we compute directly here:
        # For accuracy, we already tallied per_col[col]["rows_with_unclassified"]; compute unclassified_keys by querying again quickly:
        # To avoid another full pass, we stored only totals and rows_with_unclassified; we'll compute unclassified_keys via a lightweight pass:
        uk = 0
        for idx, row in df.iterrows():
            cell_dict, _ = parse_cell_to_dict(row[col] if col in row else None)
            uk += sum(1 for v in cell_dict.values() if v == "unclassified")
        rate = (100.0 * uk / tk) if tk else 0.0
        print(f" - {col}: unclassified {uk}/{tk} ({rate:.2f}%), rows_with_unclassified={per_col[col]['rows_with_unclassified']}")

    # Show samples
    if samples:
        print(f"\nSample unclassified entries (up to {n_samples} per column):")
        samp_df = pd.DataFrame(samples)
        # keep display concise
        print(samp_df.to_string(index=False, max_rows=min(len(samp_df), n_samples * len(hit_cols)), max_cols=0))
    else:
        print("\n(no unclassified samples to display)")

    return samples

def main():
    ap = argparse.ArgumentParser(description="Report unclassified hit labels using pandas (no JSON SQL).")
    ap.add_argument("--sdg-db", default=SDG_DB)
    ap.add_argument("--sdg-table", default=SDG_TBL)
    ap.add_argument("--tech-db", default=TECH_DB)
    ap.add_argument("--tech-table", default=TECH_TBL)
    ap.add_argument("--skip-sdg", action="store_true")
    ap.add_argument("--skip-tech", action="store_true")
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--export-csv", default=None)
    ap.add_argument("--show-passages", action="store_true")
    args = ap.parse_args()

    all_samples: List[Dict[str, Any]] = []

    if not args.skip_sdg:
        df_sdg = get_df(args.sdg_db, args.sdg_table)
        hit_cols_sdg = guess_hit_cols_sdg(df_sdg.columns.tolist())
        samples_sdg = analyze_df(df_sdg, f"{args.sdg_db}:{args.sdg_table}", hit_cols_sdg, args.n_samples, args.show_passages)
        if samples_sdg: all_samples.extend(samples_sdg)

    if not args.skip_tech:
        df_tech = get_df(args.tech_db, args.tech_table)
        hit_cols_tech = guess_hit_cols_tech(df_tech.columns.tolist())
        samples_tech = analyze_df(df_tech, f"{args.tech_db}:{args.tech_table}", hit_cols_tech, args.n_samples, args.show_passages)
        if samples_tech: all_samples.extend(samples_tech)

    if args.export_csv and all_samples:
        os.makedirs(os.path.dirname(args.export_csv), exist_ok=True)
        pd.DataFrame(all_samples).to_csv(args.export_csv, index=False)
        print(f"\n[OK] Exported samples -> {args.export_csv} (rows: {len(all_samples)})")

if __name__ == "__main__":
    main()
