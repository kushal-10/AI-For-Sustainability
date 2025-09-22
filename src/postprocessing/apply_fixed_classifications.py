#!/usr/bin/env python3
"""
Apply FIX results (symbolic/substantive) into the classified DuckDBs.

- Reads:  data/batches/fix/results/results_map.json
          custom_id format: "sdg||<global_id>||fix" or "tech||<global_id>||fix"
- Updates:
    data/dbs/sdg_hits_classified.duckdb : table sdg_hits_classified
    data/dbs/tech_hits_classified.duckdb: table tech_hits_classified
- Only updates existing keys in the classified dicts; ignores extraneous keys.
- Creates a timestamped backup of each DB before writing (can disable with --no-backup).
- Optional tolerant matching (--normalize) collapses double backslashes to reduce regex-escape mismatches.

Usage:
  python3 src/postprocessing/apply_fix_into_classified.py
  python3 src/postprocessing/apply_fix_into_classified.py --normalize
  python3 src/postprocessing/apply_fix_into_classified.py --no-backup --dry-run
"""

import os
import re
import json
import argparse
import shutil
import datetime
from typing import Any, Dict, List, Tuple

import duckdb
import pandas as pd

# ---------- Defaults ----------
FIX_RESULTS_PATH = "data/batches/fix/results/results_map.json"

SDG_DB   = "data/dbs/sdg_hits_classified.duckdb"
SDG_TBL  = "sdg_hits_classified"
TECH_DB  = "data/dbs/tech_hits_classified.duckdb"
TECH_TBL = "tech_hits_classified"

VALID_LABELS = {"symbolic", "substantive"}

# ---------- Helpers ----------
def ts() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_df(db_path: str, table: str) -> pd.DataFrame:
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute(f"SELECT * FROM {table}").fetchdf()
    con.close()
    return df

def backup_db(path: str) -> str:
    dst = f"{path}.{ts()}.bak"
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(path, dst)
    return dst

def write_df_to_db(df: pd.DataFrame, db_path: str, table: str):
    con = duckdb.connect(db_path)
    con.execute(f"DROP TABLE IF EXISTS {table}")
    con.register("df_tmp", df)
    con.execute(f"CREATE TABLE {table} AS SELECT * FROM df_tmp")
    con.unregister("df_tmp")
    con.close()

def guess_hit_cols_sdg(cols: List[str]) -> List[str]:
    scols = [c for c in cols if isinstance(c, str)]
    lower = {c: c.lower() for c in scols}
    hits = [c for c in scols if lower[c].startswith("hits_sdg")]
    if hits: return hits
    hits = [c for c in scols if lower[c].startswith("hits") and "sdg" in lower[c]]
    if hits: return hits
    return [c for c in scols if lower[c].startswith("hits")]

def guess_hit_cols_tech(cols: List[str]) -> List[str]:
    scols = [c for c in cols if isinstance(c, str)]
    lower = {c: c.lower() for c in scols}
    prefixes = (
        "hits_ai_ml",
        "hits_cloud_computing",
        "hits_big_data_blockchain",
        "hits_applications_practice",
    )
    hits = [c for c in scols if any(lower[c].startswith(p) for p in prefixes)]
    if hits: return hits
    return [c for c in scols if lower[c].startswith("hits") and "sdg" not in lower[c]]

def normalize_key(s: str) -> str:
    if s is None: return ""
    out = s
    while "\\\\" in out:
        out = out.replace("\\\\", "\\")
    return out.strip()

def parse_classified_cell(cell: Any) -> Dict[str, str]:
    """
    Classified cells should be dicts or JSON strings like {"\\bfoo\\b": "symbolic", ...}.
    Return a dict; invalid cells become {}.
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return {}
    if isinstance(cell, dict):
        # normalize values to lower-case strings
        return {str(k): (str(v).strip().lower() if v is not None else "") for k, v in cell.items() if isinstance(k, str)}
    if isinstance(cell, str):
        s = cell.strip()
        if not s:
            return {}
        # Try JSON parse
        try:
            obj = json.loads(s)
        except Exception:
            # salvage invalid escapes like \s, \w, then parse
            try:
                s2 = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)
                obj = json.loads(s2)
            except Exception:
                return {}
        if isinstance(obj, dict):
            return {str(k): (str(v).strip().lower() if v is not None else "") for k, v in obj.items() if isinstance(k, str)}
        # if it's a list, convert to dict of unclassified
        if isinstance(obj, list):
            return {str(x): "unclassified" for x in obj if isinstance(x, str) and x.strip()}
        return {}
    # anything else
    return {}

def dump_classified_cell(d: Dict[str, str]) -> str:
    return json.dumps(d, ensure_ascii=False)

def split_fix_results_by_mode(results_map: Dict[str, Dict[str, str]], normalize: bool) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    """
    Returns (sdg_map, tech_map) keyed by global_id -> {pattern: label}
    """
    sdg_map: Dict[str, Dict[str, str]] = {}
    tech_map: Dict[str, Dict[str, str]] = {}
    for cid, mapping in results_map.items():
        parts = str(cid).split("||")
        if len(parts) < 2:
            continue
        mode = parts[0]
        gid = parts[1]
        clean_map = {}
        for k, v in (mapping or {}).items():
            kk = normalize_key(k) if normalize else str(k)
            vv = str(v).strip().lower()
            if vv in VALID_LABELS:
                clean_map[kk] = vv
        if mode == "sdg":
            sdg_map[gid] = clean_map
        elif mode == "tech":
            tech_map[gid] = clean_map
    return sdg_map, tech_map

def apply_fix_to_df(df: pd.DataFrame, mode: str, fix_map_by_gid: Dict[str, Dict[str, str]], normalize: bool) -> Dict[str, int]:
    """
    Update classified dict-cells with labels from fix_map_by_gid (only keys that already exist in the cell).
    Returns counters.
    """
    cols = df.columns.tolist()
    hit_cols = guess_hit_cols_sdg(cols) if mode == "sdg" else guess_hit_cols_tech(cols)
    counters = {
        "rows_considered": 0,
        "rows_updated": 0,
        "keys_updated": 0,
        "rows_skipped_no_fix": 0,
    }
    if "global_id" not in df.columns:
        raise ValueError(f"{mode}: missing 'global_id' column")

    for idx, row in df.iterrows():
        gid = str(row["global_id"])
        fix_map = fix_map_by_gid.get(gid)
        if not fix_map:
            counters["rows_skipped_no_fix"] += 1
            continue

        counters["rows_considered"] += 1
        row_changed = False

        for col in hit_cols:
            cell_dict = parse_classified_cell(row[col] if col in row else None)
            if not cell_dict:
                continue

            # Build a lookup for tolerant matching if enabled
            if normalize:
                norm_to_real = {}
                for k in list(cell_dict.keys()):
                    norm_to_real[normalize_key(k)] = k

            # For each key in the existing cell dict, if present in fix_map -> update
            for existing_key in list(cell_dict.keys()):
                match_key = normalize_key(existing_key) if normalize else existing_key
                new_label = fix_map.get(match_key)
                if new_label in VALID_LABELS and cell_dict.get(existing_key) != new_label:
                    cell_dict[existing_key] = new_label
                    counters["keys_updated"] += 1
                    row_changed = True

            # write back if changed
            if row_changed:
                df.at[idx, col] = dump_classified_cell(cell_dict)

        if row_changed:
            counters["rows_updated"] += 1

    return counters

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Update classified DuckDBs with FIX batch results.")
    ap.add_argument("--results", default=FIX_RESULTS_PATH, help="Path to data/batches/fix/results/results_map.json")
    ap.add_argument("--sdg-db", default=SDG_DB);   ap.add_argument("--sdg-table", default=SDG_TBL)
    ap.add_argument("--tech-db", default=TECH_DB); ap.add_argument("--tech-table", default=TECH_TBL)
    ap.add_argument("--normalize", action="store_true", help="Collapse '\\\\' -> '\\' for tolerant regex key matching")
    ap.add_argument("--no-backup", action="store_true", help="Do not create .bak copies before writing")
    ap.add_argument("--dry-run", action="store_true", help="Compute updates but do not write DBs")
    args = ap.parse_args()

    # Load fix results
    if not os.path.exists(args.results):
        raise SystemExit(f"[ERR] Fix results not found: {args.results}")
    fix_results = load_json(args.results)
    sdg_fix_map, tech_fix_map = split_fix_results_by_mode(fix_results, normalize=args.normalize)

    # SDG
    if os.path.exists(args.sdg_db):
        df_sdg = get_df(args.sdg_db, args.sdg_table)
        sdg_counts = apply_fix_to_df(df_sdg, "sdg", sdg_fix_map, normalize=args.normalize)
        print(f"\n[SDG] rows_considered={sdg_counts['rows_considered']}, rows_updated={sdg_counts['rows_updated']}, keys_updated={sdg_counts['keys_updated']}, rows_skipped_no_fix={sdg_counts['rows_skipped_no_fix']}")
        if not args.dry_run and sdg_counts["rows_updated"] > 0:
            if not args.no_backup and os.path.exists(args.sdg_db):
                bak = backup_db(args.sdg_db)
                print(f"[SDG] Backup created -> {bak}")
            write_df_to_db(df_sdg, args.sdg_db, args.sdg_table)
            print(f"[SDG] Updated DB/table -> {args.sdg_db}:{args.sdg_table}")
    else:
        print(f"[WARN] SDG DB not found: {args.sdg_db}")

    # TECH
    if os.path.exists(args.tech_db):
        df_tech = get_df(args.tech_db, args.tech_table)
        tech_counts = apply_fix_to_df(df_tech, "tech", tech_fix_map, normalize=args.normalize)
        print(f"\n[TECH] rows_considered={tech_counts['rows_considered']}, rows_updated={tech_counts['rows_updated']}, keys_updated={tech_counts['keys_updated']}, rows_skipped_no_fix={tech_counts['rows_skipped_no_fix']}")
        if not args.dry_run and tech_counts["rows_updated"] > 0:
            if not args.no_backup and os.path.exists(args.tech_db):
                bak = backup_db(args.tech_db)
                print(f"[TECH] Backup created -> {bak}")
            write_df_to_db(df_tech, args.tech_db, args.tech_table)
            print(f"[TECH] Updated DB/table -> {args.tech_db}:{args.tech_table}")
    else:
        print(f"[WARN] TECH DB not found: {args.tech_db}")

    print("\n[DONE] If keys_updated is low but you still see 'unclassified', try rerunning with --normalize.")

if __name__ == "__main__":
    main()
