""" AI vs sdg counts"""

import os
import json
import duckdb
from collections import defaultdict

# ---------- Hardcoded config ----------
DB_PATH = "data/matches.duckdb"
TABLE   = "matched_sentences"

SDG_JSON_EN = "data/keywords_sdg.json"
SDG_JSON_DE = "data/keywords_sdg_de.json"  # optional
TECH_JSON_EN = "data/keywords_tech.json"
TECH_JSON_DE = "data/keywords_tech_de.json"  # optional

EXTRA_SDG_BUCKETS = ["sdg", "gc", "gri", "int"]  # “extra sdg stuff”
TECH_STACKS = ["ai_ml", "cloud_computing", "big_data_blockchain", "applications_practice"]

# ---------- Helpers ----------
def parse_json_array_text(text):
    try:
        arr = json.loads(text) if text and text != "null" else []
        if not isinstance(arr, list):
            arr = [str(arr)]
        return [str(x).strip().lower() for x in arr if str(x).strip()]
    except Exception:
        return []

def load_sdg_maps(en_path, de_path=None):
    """
    Returns:
      term2num: term -> set({1..17})
      term2extra: term -> set({'sdg'|'gc'|'gri'|'int'})
    """
    term2num, term2extra = {}, {}
    def ingest(path):
        if not path or not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, terms in data.items():
            if not isinstance(terms, list):  # skip 'meta' dicts here; we'll handle keys explicitly
                continue
            key = k.strip().lower()
            # SDG 01..17
            if key.startswith("sdg") and key != "sdg":
                try:
                    n = int(key[3:])
                except Exception:
                    n = None
                if n is not None and 1 <= n <= 17:
                    for t in terms:
                        tt = (t or "").strip().lower()
                        if not tt: continue
                        term2num.setdefault(tt, set()).add(n)
                    continue
            # Extra buckets: sdg/gc/gri/int
            if key in set(EXTRA_SDG_BUCKETS):
                for t in terms:
                    tt = (t or "").strip().lower()
                    if not tt: continue
                    term2extra.setdefault(tt, set()).add(key)

    ingest(en_path)
    ingest(de_path)
    return term2num, term2extra

def load_tech_map(en_path, de_path=None):
    """
    Returns:
      term2stack: term -> set({'ai_ml'|'cloud_computing'|'big_data_blockchain'|'applications_practice'})
    """
    term2stack = {}
    def ingest(path):
        if not path or not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            tech = json.load(f)
        for stack, terms in tech.items():
            if stack not in TECH_STACKS or not isinstance(terms, list):
                continue
            for t in terms:
                tt = (t or "").strip().lower()
                if not tt: continue
                term2stack.setdefault(tt, set()).add(stack)
    ingest(en_path)
    ingest(de_path)
    return term2stack

def main():
    # Load mappings
    term2num, term2extra = load_sdg_maps(SDG_JSON_EN, SDG_JSON_DE)
    term2stack = load_tech_map(TECH_JSON_EN, TECH_JSON_DE)

    if not os.path.exists(DB_PATH):
        print(f"[ERROR] Missing DuckDB: {DB_PATH}")
        return

    con = duckdb.connect(DB_PATH)
    rows = con.execute(f"SELECT sdg_keywords, ai_keywords FROM {TABLE}").fetchall()
    con.close()

    # Counters (sentence-level, 1 per category per sentence)
    sdg_num_counts = {n: 0 for n in range(1, 18)}
    extra_counts = {b: 0 for b in EXTRA_SDG_BUCKETS}
    tech_counts = {s: 0 for s in TECH_STACKS}

    total_rows = 0
    for sdg_json, ai_json in rows:
        total_rows += 1

        sdg_terms = set(parse_json_array_text(sdg_json))
        ai_terms  = set(parse_json_array_text(ai_json))

        # SDG numbers 1..17
        hit_nums = set()
        for t in sdg_terms:
            nums = term2num.get(t)
            if nums:
                hit_nums.update(nums)
        for n in hit_nums:
            sdg_num_counts[n] += 1

        # Extra SDG buckets
        hit_extra = set()
        for t in sdg_terms:
            xs = term2extra.get(t)
            if xs:
                hit_extra.update(xs)
        for b in hit_extra:
            extra_counts[b] += 1

        # Tech stacks
        hit_stacks = set()
        for t in ai_terms:
            ss = term2stack.get(t)
            if ss:
                hit_stacks.update(ss)
        for s in hit_stacks:
            tech_counts[s] += 1

    # ----- Print summary -----
    print("\n=== Sentence-level hit counts ===")
    print(f"Total sentences scanned: {total_rows}")

    print("\nSDG 1–17 (sentences with ≥1 term of that SDG):")
    for n in range(1, 18):
        print(f"  SDG {n:02d}: {sdg_num_counts[n]}")

    print("\nExtra SDG buckets (meta):")
    for b in EXTRA_SDG_BUCKETS:
        print(f"  {b.upper():>3}: {extra_counts[b]}")

    print("\nAI stacks:")
    for s in TECH_STACKS:
        print(f"  {s}: {tech_counts[s]}")

if __name__ == "__main__":
    main()
