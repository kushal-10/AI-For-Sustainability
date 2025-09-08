#!/usr/bin/env python3
import argparse
from pathlib import Path
import duckdb
import random
import json

TEXTS_ROOT = Path("data/texts")  # where splits.json live: data/texts/{COMPANY}/{YEAR}/splits.json
_SPLITS_CACHE = {}

def parse_custom_id(custom_id: str):
    """
    Expected: task||{sentence_id}||{COMPANY}||{YEAR}
    """
    try:
        parts = custom_id.split("||")
        if len(parts) < 4:
            return None, None, None
        _, sid, company, year = parts[:4]
        return sid, company, year
    except Exception:
        return None, None, None

def load_sentence(company: str, year: str, sentence_id: str):
    if not (company and year and sentence_id):
        return None
    p = TEXTS_ROOT / company / year / "splits.json"
    if not p.exists():
        return None
    if p not in _SPLITS_CACHE:
        try:
            with open(p, "r", encoding="utf-8") as f:
                _SPLITS_CACHE[p] = json.load(f)
        except Exception:
            _SPLITS_CACHE[p] = {}
    try:
        return _SPLITS_CACHE[p].get(str(int(sentence_id)))
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser(description="Analyse results and print samples for [0, False], not [0, False], and True entries.")
    ap.add_argument("--db", default="data/outputs_merged/classifications.duckdb",
                    help="Path to DuckDB database file.")
    ap.add_argument("--sample", type=int, default=10,
                    help="How many sentences to sample from each group.")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB file not found: {db_path}")

    con = duckdb.connect(db_path.as_posix())
    try:
        total = con.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        zero_false_count = con.execute("SELECT COUNT(*) FROM results WHERE raw = '[0, False]'").fetchone()[0]
        non_zero_false_count = total - zero_false_count
        true_count = con.execute("SELECT COUNT(*) FROM results WHERE is_ai = TRUE").fetchone()[0]

        zero_false_ids = con.execute("SELECT custom_id, raw FROM results WHERE raw = '[0, False]'").fetchall()
        non_zero_false_ids = con.execute("SELECT custom_id, raw FROM results WHERE raw != '[0, False]'").fetchall()
        true_ids = con.execute("SELECT custom_id, raw FROM results WHERE is_ai = TRUE").fetchall()
    finally:
        con.close()

    print(f"Total rows: {total}")
    print(f"[0, False]: {zero_false_count} ({zero_false_count/total*100:.2f}%)")
    print(f"Not [0, False]: {non_zero_false_count} ({non_zero_false_count/total*100:.2f}%)")
    print(f"AI=True: {true_count} ({true_count/total*100:.2f}%)")
    print("*" * 150)

    random.seed(42)
    zf_sample = random.sample(zero_false_ids, min(args.sample, len(zero_false_ids)))
    nzf_sample = random.sample(non_zero_false_ids, min(args.sample, len(non_zero_false_ids)))
    true_sample = random.sample(true_ids, min(args.sample, len(true_ids)))

    print(f"Sample {len(zf_sample)} sentences with [0, False]:")
    for cid, raw in zf_sample:
        sid, comp, year = parse_custom_id(cid)
        sent = load_sentence(comp, year, sid) or "<sentence not found>"
        print(f"- {sent} || {cid} || {raw}")
    print("*" * 150)

    print(f"Sample {len(nzf_sample)} sentences NOT [0, False]:")
    for cid, raw in nzf_sample:
        sid, comp, year = parse_custom_id(cid)
        sent = load_sentence(comp, year, sid) or "<sentence not found>"
        print(f"- {sent} || {cid} || {raw}")
    print("*" * 150)

    print(f"Sample {len(true_sample)} sentences classified as AI=True:")
    for cid, raw in true_sample:
        sid, comp, year = parse_custom_id(cid)
        sent = load_sentence(comp, year, sid) or "<sentence not found>"
        print(f"- {sent} || {cid} || {raw}")
    print("*" * 150)

if __name__ == "__main__":
    main()
