#!/usr/bin/env python3
"""
sdg_backfill_enumerate.py

Recompute exact SDG keyword hits for existing rows in DuckDB and update in place.
Useful when your table has placeholder ["__hit__"] values.

Usage:
  python3 src/filtering/sdg_backfill_enumerate.py \
    --db data/dbs/sdg_hits.duckdb \
    --table sdg_hits \
    --kw_en kw_data/keywords_sdg.json \
    --kw_de kw_data/keywords_sdg_de.json \
    --wildcard

Notes:
- Uses the row's `language` column: if 'de' -> merge EN+DE keywords; else EN only.
- Updates columns: hits_sdg1 ... hits_sdg17 (JSON strings).
"""

import argparse, json, re
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
from tqdm import tqdm

# ---------- wildcard-aware token -> regex ----------
def _is_alnum(c: str) -> bool:
    return bool(re.match(r"[A-Za-z0-9]", c))

def _token_to_regex(token: str, star_is_wildcard: bool) -> str:
    raw = token.strip()
    if not raw:
        return ""
    parts: List[str] = []
    for i, ch in enumerate(raw):
        if star_is_wildcard and ch == "*":
            parts.append(r"\w*" if i == len(raw) - 1 else r".*")
        elif star_is_wildcard and ch == " ":
            parts.append(r"\s+")
        else:
            parts.append(re.escape(ch))
    pat = "".join(parts)
    first_vis = next((c for c in raw if c != " "), raw[:1])
    last_vis = next((c for c in reversed(raw) if c != "*"), "")
    if _is_alnum(first_vis):
        pat = r"\b" + pat
    if _is_alnum(last_vis):
        pat = pat + r"\b"
    return pat

def _compile_patterns_list(tokens: List[str], wildcard: bool) -> List[re.Pattern]:
    out = []
    for t in tokens:
        p = _token_to_regex(t, wildcard)
        if p:
            out.append(re.compile(p, re.IGNORECASE))
    return out

def _flatten_keyword_dict_list(dct: Dict[str, List[str]], wildcard: bool) -> Dict[str, List[re.Pattern]]:
    return {cat: _compile_patterns_list(words, wildcard) for cat, words in dct.items()}

def _find_hits_enumerate(text: str, per_token: Dict[str, List[re.Pattern]]) -> Dict[str, List[str]]:
    hits: Dict[str, List[str]] = {}
    for cat, pats in per_token.items():
        found = []
        for p in pats:
            if p.search(text):
                found.append(p.pattern)
        # dedup, keep order
        seen = set(); uniq = []
        for f in found:
            k = f.lower()
            if k not in seen:
                seen.add(k); uniq.append(f)
        hits[cat] = uniq
    return hits

def _snake(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

def main():
    ap = argparse.ArgumentParser(description="Backfill exact SDG hits in DuckDB table.")
    ap.add_argument("--db", required=True, help="DuckDB file, e.g., data/dbs/sdg_hits.duckdb")
    ap.add_argument("--table", default="sdg_hits", help="Table name (default: sdg_hits)")
    ap.add_argument("--kw_en", required=True, help="keywords_sdg.json")
    ap.add_argument("--kw_de", required=True, help="keywords_sdg_de.json")
    ap.add_argument("--wildcard", action="store_true", help="Enable '*' wildcard expansion")
    ap.add_argument("--where", default=None, help="Optional WHERE filter, e.g. \"year='2023'\"")
    ap.add_argument("--batch", type=int, default=5000, help="UPDATE batch size (default 5000)")
    args = ap.parse_args()

    # Load keyword dicts
    kw_en = json.loads(Path(args.kw_en).read_text(encoding="utf-8"))
    kw_de = json.loads(Path(args.kw_de).read_text(encoding="utf-8"))

    # Expect categories sdg1..sdg17
    cats = [f"sdg{i}" for i in range(1, 18)]
    missing = [c for c in cats if c not in kw_en]
    if missing:
        raise SystemExit(f"Missing categories in EN keywords: {missing}")

    # Compile per-token regex lists
    compiled_en = _flatten_keyword_dict_list(kw_en, args.wildcard)
    compiled_de = _flatten_keyword_dict_list(kw_de, args.wildcard)

    # Connect DB
    con = duckdb.connect(args.db)

    # Index for speed (safe if already exists)
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_{args.table}_gid ON {args.table}(global_id);")

    # Columns we need
    base_cols = "global_id, passage, language"
    where_clause = f"WHERE {args.where}" if args.where else ""
    total = con.execute(f"SELECT COUNT(*) FROM {args.table} {where_clause}").fetchone()[0]
    print(f"[INFO] Backfilling {total} rows from {args.table}...")

    # Stream rows in chunks
    offset = 0
    while offset < total:
        rows = con.execute(
            f"SELECT {base_cols} FROM {args.table} {where_clause} LIMIT {args.batch} OFFSET {offset}"
        ).fetchall()
        if not rows:
            break

        updates: List[Tuple] = []
        for gid, passage, lang in tqdm(rows, unit="row", leave=False):
            text = (passage or "").strip()
            if not text:
                continue

            # choose EN-only vs EN+DE by language
            per_token = {}
            if str(lang).lower() == "de":
                for c in cats:
                    per_token[c] = compiled_en.get(c, []) + compiled_de.get(c, [])
            else:
                for c in cats:
                    per_token[c] = compiled_en.get(c, [])

            hits = _find_hits_enumerate(text, per_token)

            # Build JSON strings in order sdg1..sdg17
            hit_jsons = [json.dumps(hits.get(c, []), ensure_ascii=False) for c in cats]

            updates.append((*hit_jsons, gid))

        if updates:
            set_clause = ", ".join([f"hits_{_snake(c)}=?" for c in cats])
            placeholders = ",".join(["?"] * (len(cats)+1))
            con.executemany(
                f"UPDATE {args.table} SET {set_clause} WHERE global_id=?",
                updates
            )

        offset += args.batch

    con.close()
    print("[OK] Backfill complete.")

if __name__ == "__main__":
    main()

"""
python3 src/utils/fix_sdg_db.py \
  --db data/dbs/sdg_hits.duckdb \
  --table sdg_hits \
  --kw_en kw_data/keywords_sdg.json \
  --kw_de kw_data/keywords_sdg_de.json \
  --wildcard
"""