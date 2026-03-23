#!/usr/bin/env python3
"""
Verify that every classification from results_map exists in the original hit lists
and has a valid label ("symbolic"|"substantive"). Also report missing/extraneous/invalid.

Inputs (defaults):
- Results map JSON: data/batches/results/results_all_map.json (fallback to results_map.json)
- Original DBs (pre-classified):
    SDG  : data/dbs/sdg_hits.duckdb (table: sdg_hits)
    TECH : data/dbs/tech_hits.duckdb (table: tech_hits)

Usage:
  python3 src/postprocessing/verify_classifications.py
  python3 src/postprocessing/verify_classifications.py --normalize
  python3 src/postprocessing/verify_classifications.py --export-csv data/batches/results/verification_issues.csv
  python3 src/postprocessing/verify_classifications.py --skip-sdg
  python3 src/postprocessing/verify_classifications.py --skip-tech
"""

import os
import re
import json
import argparse
from typing import Any, Dict, List, Tuple, Set

import duckdb
import pandas as pd

# ---------- Defaults ----------
RESULTS_ALL = "data/batches/results/results_all_map.json"
RESULTS_MAP = "data/batches/results/results_map.json"

SDG_DB  = "data/dbs/sdg_hits.duckdb"
SDG_TBL = "sdg_hits"
TECH_DB = "data/dbs/tech_hits.duckdb"
TECH_TBL= "tech_hits"

VALID_LABELS = {"symbolic", "substantive"}

# ---------- IO ----------
def load_results_map() -> Dict[str, Dict[str, str]]:
    if os.path.exists(RESULTS_ALL):
        with open(RESULTS_ALL, "r", encoding="utf-8") as f:
            return json.load(f)
    if os.path.exists(RESULTS_MAP):
        with open(RESULTS_MAP, "r", encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(f"Results map not found. Tried:\n  {RESULTS_ALL}\n  {RESULTS_MAP}")

def get_df(db_path: str, table: str) -> pd.DataFrame:
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute(f"SELECT * FROM {table}").fetchdf()
    con.close()
    return df

# ---------- Helpers ----------
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

def parse_hits_cell_to_list(cell: Any) -> List[str]:
    """
    Turn a cell from original hits columns into a list of pattern strings.
    Accepts list/JSON-string/dict (rare)/other.
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    if isinstance(cell, list):
        return [s for s in cell if isinstance(s, str) and s.strip()]
    if isinstance(cell, dict):
        # original should be list; but if dict appears, use keys as patterns
        return [k for k in cell.keys() if isinstance(k, str) and k.strip()]
    if isinstance(cell, str):
        s = cell.strip()
        if not s:
            return []
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            # try parse JSON
            try:
                obj = json.loads(s)
            except Exception:
                # salvage invalid escapes by doubling backslashes
                try:
                    s2 = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)
                    obj = json.loads(s2)
                except Exception:
                    return []
            # recurse if needed
            if isinstance(obj, list):
                return [x for x in obj if isinstance(x, str) and x.strip()]
            if isinstance(obj, dict):
                return [k for k in obj.keys() if isinstance(k, str) and k.strip()]
        # non-JSON plain string -> not a list of patterns
        return []
    return []

def normalize_key(s: str) -> str:
    """Normalize pattern string for tolerant matching (collapses backslashes)."""
    if s is None: return ""
    out = s
    # collapse multiple backslashes to single (repeat until stable)
    while "\\\\" in out:
        out = out.replace("\\\\", "\\")
    # strip trivial whitespace around
    return out.strip()

def build_row_patterns_set(row: pd.Series, hit_cols: List[str], normalize: bool) -> Set[str]:
    pats: Set[str] = set()
    for col in hit_cols:
        for p in parse_hits_cell_to_list(row.get(col)):
            pats.add(normalize_key(p) if normalize else p)
    return pats

def lower_label(v: Any) -> str:
    return str(v).strip().lower() if v is not None else ""

# ---------- Core verification ----------
def verify_mode(
    mode: str,
    df: pd.DataFrame,
    hit_cols: List[str],
    results_map: Dict[str, Dict[str, str]],
    normalize: bool,
    show_passages: bool,
    n_samples: int
) -> Dict[str, Any]:
    assert mode in ("sdg", "tech")
    if "global_id" not in df.columns:
        raise ValueError(f"{mode}: missing 'global_id' in dataframe")

    has_company = "company" in df.columns
    has_year = "year" in df.columns
    has_passage = "passage" in df.columns

    issues: List[Dict[str, Any]] = []

    total_rows = len(df)
    total_patterns = 0
    cnt_missing_result_for_row = 0
    cnt_missing_class = 0
    cnt_invalid_label = 0
    cnt_extraneous = 0

    for _, row in df.iterrows():
        gid = str(row["global_id"])
        key = f"{mode}||{gid}"

        row_patterns = build_row_patterns_set(row, hit_cols, normalize)
        total_patterns += len(row_patterns)

        res = results_map.get(key)

        if res is None:
            cnt_missing_result_for_row += 1
            issues.append({
                "type": "missing_result_for_row",
                "mode": mode,
                "global_id": gid,
                "row_patterns": list(sorted(row_patterns))[:50],
                "company": row["company"] if has_company else None,
                "year": row["year"] if has_year else None,
                "passage": row["passage"] if (show_passages and has_passage) else None
            })
            # Still check nothing else for this row
            continue

        # Build label map (normalized or raw)
        res_map = {}
        for k, v in res.items():
            kk = normalize_key(k) if normalize else k
            res_map[kk] = lower_label(v)

        # 1) Missing classifications: patterns in row not present in res_map
        for p in row_patterns:
            if p not in res_map:
                cnt_missing_class += 1
                issues.append({
                    "type": "missing_classification",
                    "mode": mode,
                    "global_id": gid,
                    "pattern": p,
                    "label": None,
                    "company": row["company"] if has_company else None,
                    "year": row["year"] if has_year else None,
                    "passage": row["passage"] if (show_passages and has_passage) else None
                })
            else:
                lbl = res_map[p]
                if lbl not in VALID_LABELS:
                    cnt_invalid_label += 1
                    issues.append({
                        "type": "invalid_label",
                        "mode": mode,
                        "global_id": gid,
                        "pattern": p,
                        "label": lbl,
                        "company": row["company"] if has_company else None,
                        "year": row["year"] if has_year else None,
                        "passage": row["passage"] if (show_passages and has_passage) else None
                    })

        # 2) Extraneous classifications: keys in res_map not present in row
        for k_res, lbl in res_map.items():
            if k_res not in row_patterns:
                cnt_extraneous += 1
                issues.append({
                    "type": "extraneous_classification",
                    "mode": mode,
                    "global_id": gid,
                    "pattern": k_res,
                    "label": lbl,
                    "company": row["company"] if has_company else None,
                    "year": row["year"] if has_year else None,
                    "passage": row["passage"] if (show_passages and has_passage) else None
                })

    summary = {
        "mode": mode,
        "rows": total_rows,
        "hit_columns": hit_cols,
        "total_patterns_in_rows": total_patterns,
        "missing_result_for_row": cnt_missing_result_for_row,
        "missing_classifications": cnt_missing_class,
        "invalid_labels": cnt_invalid_label,
        "extraneous_classifications": cnt_extraneous,
    }

    # Print concise summary
    print(f"\n=== VERIFY {mode.upper()} ===")
    print(f"Rows: {total_rows} | Hit cols: {hit_cols}")
    print(f"Patterns in rows: {total_patterns}")
    print(f"- Rows missing any result entry : {cnt_missing_result_for_row}")
    print(f"- Missing classifications       : {cnt_missing_class}")
    print(f"- Invalid labels                : {cnt_invalid_label}")
    print(f"- Extraneous classifications    : {cnt_extraneous}")

    # Sample print
    if issues:
        samp = issues[:min(len(issues), n_samples)]
        print(f"\nSample issues (up to {n_samples}):")
        print(pd.DataFrame(samp).to_string(index=False, max_rows=n_samples, max_cols=0))
    else:
        print("\nNo issues found. ✅")

    return {"summary": summary, "issues": issues}

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Verify classifications exist and are valid (symbolic/substantive).")
    ap.add_argument("--normalize", action="store_true", help="Tolerant matching: collapse '\\\\' to '\\' in keys.")
    ap.add_argument("--skip-sdg", action="store_true")
    ap.add_argument("--skip-tech", action="store_true")
    ap.add_argument("--show-passages", action="store_true")
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--export-csv", default=None, help="Optional path to write all issues as CSV")
    ap.add_argument("--out-json", default="data/batches/results/verification_report.json")
    args = ap.parse_args()

    results_map = load_results_map()

    reports = {}
    all_issues: List[Dict[str, Any]] = []

    if not args.skip_sdg:
        df_sdg = get_df(SDG_DB, SDG_TBL)
        hit_cols_sdg = guess_hit_cols_sdg(df_sdg.columns.tolist())
        rep_sdg = verify_mode("sdg", df_sdg, hit_cols_sdg, results_map, args.normalize, args.show_passages, args.n_samples)
        reports["sdg"] = rep_sdg["summary"]
        all_issues.extend(rep_sdg["issues"])

    if not args.skip_tech:
        df_tech = get_df(TECH_DB, TECH_TBL)
        hit_cols_tech = guess_hit_cols_tech(df_tech.columns.tolist())
        rep_tech = verify_mode("tech", df_tech, hit_cols_tech, results_map, args.normalize, args.show_passages, args.n_samples)
        reports["tech"] = rep_tech["summary"]
        all_issues.extend(rep_tech["issues"])

    # Write JSON report
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"reports": reports, "issues": all_issues}, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Wrote JSON report -> {args.out_json}")

    # Optional CSV
    if args.export_csv:
        os.makedirs(os.path.dirname(args.export_csv), exist_ok=True)
        pd.DataFrame(all_issues).to_csv(args.export_csv, index=False)
        print(f"[OK] Wrote CSV issues -> {args.export_csv} (rows: {len(all_issues)})")

if __name__ == "__main__":
    main()
