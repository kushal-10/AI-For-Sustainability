#!/usr/bin/env python3

import os
import re
import json
import glob
import argparse
import datetime
from typing import Any, Dict, List, Set, Tuple

import duckdb
import pandas as pd
from openai import OpenAI

# ---- import the existing builders (prompts & object creation) ----
try:
    from src.gpt_classifier.objects import create_batch_object_sdg, create_batch_object_tech
except Exception as e:
    raise SystemExit("Could not import create_batch_object_* from src/batching/batch_objects.py") from e

# ---- Defaults ----
VERIF_REPORT      = "data/batches/results/verification_report.json"

SDG_DB            = "data/dbs/sdg_hits.duckdb"
SDG_TABLE         = "sdg_hits"
TECH_DB           = "data/dbs/tech_hits.duckdb"
TECH_TABLE        = "tech_hits"

OUT_DIR_FIX       = "data/batches/fix"
BATCH_LIMIT       = 1_000  # 1k as requested
GLOB_FIX_JSONL    = "*.jsonl"

# ---- Helpers ----

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def now_stamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def read_report(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

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
    # fallback: any hits_* that does NOT contain 'sdg'
    return [c for c in cols if isinstance(c, str) and lower[c].startswith("hits") and "sdg" not in lower[c]]

def parse_hits_cell_to_list(cell: Any) -> List[str]:
    """Turn a hits cell into a list of regex patterns."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    if isinstance(cell, list):
        return [s for s in cell if isinstance(s, str) and s.strip()]
    if isinstance(cell, dict):
        # rare in original hits; if dict present, use keys
        return [k for k in cell.keys() if isinstance(k, str) and k.strip()]
    if isinstance(cell, str):
        s = cell.strip()
        if not s:
            return []
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                obj = json.loads(s)
            except Exception:
                # salvage invalid escapes: \s, \w, etc.
                try:
                    s2 = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)
                    obj = json.loads(s2)
                except Exception:
                    return []
            if isinstance(obj, list):
                return [x for x in obj if isinstance(x, str) and x.strip()]
            if isinstance(obj, dict):
                return [k for k in obj.keys() if isinstance(k, str) and k.strip()]
        # non-JSON string is not a hits list
        return []
    return []

def build_hits_dict_for_row(row: pd.Series, hit_cols: List[str]) -> Dict[str, List[str]]:
    """Build TECH_HITS/SDG_HITS dict for the row (original patterns only)."""
    out: Dict[str, List[str]] = {}
    for col in hit_cols:
        pats = parse_hits_cell_to_list(row.get(col))
        if pats:
            out[col] = pats
    return out

def chunk_write(objs: List[dict], out_dir: str, prefix: str, limit: int = BATCH_LIMIT) -> List[str]:
    ensure_dir(out_dir)
    ts = now_stamp()
    paths = []
    for i in range(0, len(objs), limit):
        shard = i // limit + 1
        path = os.path.join(out_dir, f"{prefix}_{ts}_part{shard:04d}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for obj in objs[i:i+limit]:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        paths.append(path)
    return paths

# ---- Core: build fix batches ----

def collect_problem_rows(report: Dict[str, Any]) -> Tuple[Set[str], Set[str]]:
    """
    Return sets of global_ids for sdg and tech that need fixing
    (types: 'missing_classification' or 'extraneous_classification').
    """
    issues = report.get("issues", [])
    sdg_ids: Set[str] = set()
    tech_ids: Set[str] = set()
    for it in issues:
        t = (it.get("type") or "").lower()
        if t not in ("missing_classification", "extraneous_classification"):
            continue
        mode = it.get("mode")
        gid = str(it.get("global_id"))
        if mode == "sdg":
            sdg_ids.add(gid)
        elif mode == "tech":
            tech_ids.add(gid)
    return sdg_ids, tech_ids

def build_fix_for_mode(mode: str, ids: Set[str], db_path: str, table: str, out_dir: str) -> List[str]:
    if not ids:
        print(f"[INFO] {mode.upper()}: no rows to fix.")
        return []
    df = get_df(db_path, table)
    cols = df.columns.tolist()
    if "global_id" not in cols:
        raise ValueError(f"{mode}: 'global_id' column missing in {table}")

    hit_cols = guess_hit_cols_sdg(cols) if mode == "sdg" else guess_hit_cols_tech(cols)
    if not hit_cols:
        print(f"[WARN] {mode.upper()}: no hit columns detected in {table}. Skipping.")
        return []

    # filter relevant rows
    df_sub = df[df["global_id"].astype(str).isin(ids)].copy()
    if df_sub.empty:
        print(f"[WARN] {mode.upper()}: none of the target global_ids found in {table}.")
        return []

    objs: List[dict] = []
    count_skipped_no_hits = 0

    for _, row in df_sub.iterrows():
        passage = str(row.get("passage", ""))  # safe
        hits_dict = build_hits_dict_for_row(row, hit_cols)
        if not hits_dict:
            count_skipped_no_hits += 1
            continue

        gid = str(row["global_id"])
        custom_id = f"{mode}||{gid}||fix"

        if mode == "sdg":
            obj = create_batch_object_sdg(passage, custom_id, hits_dict)  # pass custom_id as "global_id" field in builder
            # builder expects (passage, global_id, hits_dict); we pass our composite for uniqueness
            # and it will set custom_id=f"sdg||{global_id}"
            # -> we want the final custom_id to carry ||fix
            obj["custom_id"] = custom_id
        else:
            obj = create_batch_object_tech(passage, custom_id, hits_dict)
            obj["custom_id"] = custom_id

        objs.append(obj)

    if count_skipped_no_hits:
        print(f"[INFO] {mode.upper()}: skipped {count_skipped_no_hits} rows with no hits.")

    if not objs:
        print(f"[INFO] {mode.upper()}: no batch requests to write.")
        return []

    prefix = f"fix_{mode}_batch"
    paths = chunk_write(objs, out_dir, prefix, limit=BATCH_LIMIT)
    print(f"[OK] {mode.upper()}: wrote {len(objs)} requests into {len(paths)} file(s).")
    for p in paths: print("   ", p)
    return paths

# ---- Submit all fix batches ----

def submit_all_fix_batches(root_dir: str = OUT_DIR_FIX, glob_pat: str = GLOB_FIX_JSONL, window: str = "24h"):
    files = sorted(glob.glob(os.path.join(root_dir, glob_pat)))
    if not files:
        print("[INFO] No fix JSONL files found to submit.")
        return

    client = OpenAI()  # requires OPENAI_API_KEY
    for fp in files:
        with open(fp, "rb") as f:
            file_obj = client.files.create(file=f, purpose="batch")
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window=window,
        )
        print(f"[OK] Submitted: {os.path.basename(fp)} -> batch_id={batch.id} status={batch.status}")

# ---- CLI ----

def main():
    ap = argparse.ArgumentParser(description="Build/submit fix batches for rows with missing/extraneous classifications.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build", action="store_true", help="Build fix batches JSONL under data/batches/fix/")
    g.add_argument("--submit", action="store_true", help="Submit all fix batch JSONLs under data/batches/fix/")
    ap.add_argument("--report", default=VERIF_REPORT, help="Path to verification_report.json")
    ap.add_argument("--outdir", default=OUT_DIR_FIX, help="Output directory for fix batches")
    ap.add_argument("--sdg-db", default=SDG_DB);   ap.add_argument("--sdg-table", default=SDG_TABLE)
    ap.add_argument("--tech-db", default=TECH_DB); ap.add_argument("--tech-table", default=TECH_TABLE)
    ap.add_argument("--window", default="24h", help="Batch completion window for submission")
    args = ap.parse_args()

    if args.build:
        rep = read_report(args.report)
        sdg_ids, tech_ids = collect_problem_rows(rep)
        ensure_dir(args.outdir)
        build_fix_for_mode("sdg", sdg_ids, args.sdg_db, args.sdg_table, args.outdir)
        build_fix_for_mode("tech", tech_ids, args.tech_db, args.tech_table, args.outdir)

    elif args.submit:
        submit_all_fix_batches(args.outdir, GLOB_FIX_JSONL, args.window)

if __name__ == "__main__":
    main()
