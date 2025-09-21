#!/usr/bin/env python3
import os, json, glob, argparse, datetime
from typing import List, Dict, Any
import duckdb
from openai import OpenAI

from src.gpt_classifier.objects import (
    create_batch_object_sdg,
    create_batch_object_tech,
)

# -------- Defaults (no CLI args for these) --------
SDG_DB_PATH   = "data/dbs/sdg_hits.duckdb"
SDG_TABLE     = "sdg_hits"
TECH_DB_PATH  = "data/dbs/tech_hits.duckdb"
TECH_TABLE    = "tech_hits"

OUT_DIR_SDG   = "data/batches/sdgs"
OUT_DIR_TECH  = "data/batches/tech"
BATCH_LIMIT   = 10_000
BATCH_GLOB    = "*.jsonl"
COMPLETION_WINDOW = "24h"
ENDPOINT = "/v1/chat/completions"

# -------- Helpers --------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _maybe_parse_json(v: Any):
    if v is None:
        return None
    if isinstance(v, (list, dict)):
        return v
    if isinstance(v, str):
        s = v.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                return json.loads(s)
            except Exception:
                return None
    return None

def guess_passage_col(cols: List[str]) -> str:
    for cand in ("passage", "sentence", "text", "content"):
        if cand in cols:
            return cand
    raise ValueError("Could not find passage column (tried: passage/sentence/text/content).")

def guess_global_id_col(cols: List[str]) -> str:
    for cand in ("global_id", "Global_ID", "GLOBAL_ID", "id", "uid", "row_id"):
        if cand in cols:
            return cand
    raise ValueError("global_id column not found (tried: global_id/Global_ID/GLOBAL_ID/id/uid/row_id).")

def guess_hit_cols_sdg(cols: List[str]) -> List[str]:
    return [c for c in cols if c.startswith("hits_sdg")]

def guess_hit_cols_tech(cols: List[str]) -> List[str]:
    prefixes = ("hits_ai_ml", "hits_cloud_computing", "hits_big_data_blockchain", "hits_applications_practice")
    return [c for c in cols if c.startswith(prefixes)]

def build_hits_dict(row: dict, hit_cols: List[str]) -> Dict[str, List[str]]:
    out = {}
    for c in hit_cols:
        parsed = _maybe_parse_json(row.get(c))
        if isinstance(parsed, list) and parsed:
            patterns = [p for p in parsed if isinstance(p, str) and p.strip()]
            if patterns:
                out[c] = patterns
    return out

def chunk_write(jsonl_objs: List[dict], out_dir: str, prefix: str, limit: int = BATCH_LIMIT) -> List[str]:
    ensure_dir(out_dir)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = []
    for i in range(0, len(jsonl_objs), limit):
        shard = i // limit + 1
        path = os.path.join(out_dir, f"{prefix}_{ts}_part{shard:04d}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for obj in jsonl_objs[i:i+limit]:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        paths.append(path)
    return paths

# -------- Build (DuckDB -> JSONL) --------

def build_batches_from_duckdb(db_path: str, table: str, mode: str, out_dir: str) -> List[str]:
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute(f"SELECT * FROM {table}").fetchdf()
    cols = list(df.columns)

    id_col = guess_global_id_col(cols)
    passage_col = guess_passage_col(cols)

    if mode == "sdg":
        hit_cols = guess_hit_cols_sdg(cols)
        if not hit_cols:
            raise ValueError("No SDG hit columns found (expected columns starting with 'hits_sdg').")
    else:
        hit_cols = guess_hit_cols_tech(cols)
        if not hit_cols:
            raise ValueError("No TECH hit columns found (expected columns starting with hits_ai_ml/cloud_computing/big_data_blockchain/applications_practice).")

    objs = []
    for _, r in df.iterrows():
        row = r.to_dict()
        hits_dict = build_hits_dict(row, hit_cols)
        if not hits_dict:
            continue
        gid = str(row[id_col])
        passage = str(row[passage_col])
        if mode == "sdg":
            obj = create_batch_object_sdg(passage, gid, hits_dict)
        else:
            obj = create_batch_object_tech(passage, gid, hits_dict)
        objs.append(obj)

    prefix = "sdg_batch" if mode == "sdg" else "tech_batch"
    paths = chunk_write(objs, out_dir, prefix, limit=BATCH_LIMIT)
    print(f"[OK] {mode.upper()}: built {len(objs)} requests into {len(paths)} file(s):")
    for p in paths: print("   ", p)
    return paths

# -------- Submit (JSONL -> OpenAI Batch) --------

def submit_jsonl_batches(dirs: List[str]):
    files = []
    for d in dirs:
        files.extend(sorted(glob.glob(os.path.join(d, BATCH_GLOB))))
    if not files:
        print("[INFO] No JSONL batch files found.")
        return

    client = OpenAI()  # uses OPENAI_API_KEY
    manifest = []
    for path in files:
        with open(path, "rb") as f:
            file_obj = client.files.create(file=f, purpose="batch")
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint=ENDPOINT,
            completion_window=COMPLETION_WINDOW,
        )
        print(f"[OK] Submitted {os.path.basename(path)} -> batch_id={batch.id} status={batch.status}")
        manifest.append({"jsonl_path": path, "file_id": file_obj.id, "batch_id": batch.id, "status": batch.status})

    # Optional: write a small manifest next to each dir
    for d in set(os.path.dirname(p) for p in files):
        mpath = os.path.join(d, "manifest.json")
        prev = []
        if os.path.exists(mpath):
            try:
                with open(mpath, "r", encoding="utf-8") as f:
                    prev = json.load(f)
            except Exception:
                prev = []
        prev.extend([m for m in manifest if os.path.dirname(m["jsonl_path"]) == d])
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(prev, f, ensure_ascii=False, indent=2)

# -------- CLI (exactly two options) --------

def main():
    ap = argparse.ArgumentParser(description="Build or submit OpenAI batch JSONL files.")
    # Exactly two flags only; everything else uses defaults above.
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build", action="store_true", help="Build batches (DuckDB -> JSONL).")
    g.add_argument("--submit", action="store_true", help="Submit all JSONL batches.")
    args = ap.parse_args()

    if args.build:
        build_batches_from_duckdb(SDG_DB_PATH, SDG_TABLE, "sdg", OUT_DIR_SDG)
        build_batches_from_duckdb(TECH_DB_PATH, TECH_TABLE, "tech", OUT_DIR_TECH)
    elif args.submit:
        submit_jsonl_batches([OUT_DIR_SDG, OUT_DIR_TECH])

if __name__ == "__main__":
    main()
