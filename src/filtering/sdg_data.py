# src/build_sdg_sentences_17.py
import os
import json
import duckdb
from tqdm import tqdm
import csv

# ===== Hardcoded config =====
SRC_DB      = "data/matches.duckdb"
SRC_TABLE   = "matched_sentences"   # columns: company, year, sentence, sdg_keywords (JSON text), ...
SDG_JSON_EN = "data/keywords_sdg.json"
SDG_JSON_DE = "data/keywords_sdg_de.json"  # optional; used if exists
OUT_DB      = "data/sdg_sentences.duckdb"
OUT_TABLE   = "sdg_sentences_17"
OUT_CSV     = "data/sdg_sentences_17.csv"

# ===== Helpers =====
def build_term_to_sdgs(*json_paths):
    """
    Build a mapping: normalized_term -> set({1..17})
    Only keys sdg01..sdg17 are considered; 'meta' keys are ignored.
    """
    term2nums = {}
    for path in json_paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, terms in data.items():
            if not isinstance(terms, list):
                continue
            if not k.lower().startswith("sdg") or k.lower() == "sdg":  # skip 'sdg' meta bucket
                # e.g., skip meta: 'sdg', 'gc', 'gri', 'int'
                if k.lower() not in {f"sdg{n:02d}" for n in range(1, 18)}:
                    continue
            # Extract number if this is sdg01..sdg17
            if k.lower().startswith("sdg") and k.lower() != "sdg":
                try:
                    num = int(k[3:])
                except Exception:
                    num = None
            else:
                num = None

            if num is None or not (1 <= num <= 17):
                continue

            for term in terms:
                t = (term or "").strip().lower()
                if not t:
                    continue
                term2nums.setdefault(t, set()).add(num)
    return term2nums

def parse_json_array_text(text):
    try:
        arr = json.loads(text) if text and text != "null" else []
        if not isinstance(arr, list):
            arr = [str(arr)]
        return [str(x) for x in arr]
    except Exception:
        return []

def main():
    os.makedirs("data", exist_ok=True)

    # 1) Build term->SDG numbers mapping from EN (+DE if present)
    term2nums = build_term_to_sdgs(SDG_JSON_EN, SDG_JSON_DE)
    if not term2nums:
        print("[ERROR] No SDG term mapping built. Check keywords_sdg*.json files.")
        return

    # 2) Read source sentences that have any sdg_keywords payload
    if not os.path.exists(SRC_DB):
        print(f"[ERROR] Missing source DB: {SRC_DB}")
        return

    src = duckdb.connect(SRC_DB)
    try:
        rows = src.execute(f"""
            SELECT company, year, sentence, sdg_keywords
            FROM {SRC_TABLE}
            WHERE sdg_keywords IS NOT NULL
              AND sdg_keywords <> 'null'
              AND sdg_keywords <> '[]'
        """).fetchall()
    except Exception as e:
        print(f"[ERROR] Querying source table failed: {e}")
        src.close()
        return
    src.close()

    # 3) Transform rows -> keep only sentences with SDG 1..17 hits; compute numbers list
    out_rows = []   # tuples ready for DuckDB/CSV
    kept = 0
    for company, year, sentence, sdg_json in tqdm(rows, desc="Filtering to SDG 1–17"):
        terms = parse_json_array_text(sdg_json)
        if not terms:
            continue
        # Collect only those terms that map to SDG numbers 1..17
        terms_norm = [t.strip().lower() for t in terms if t and t.strip()]
        sdg_nums = set()
        kept_terms = []
        for t in terms_norm:
            nums = term2nums.get(t)
            if nums:
                sdg_nums.update(nums)
                kept_terms.append(t)
        if not sdg_nums:
            continue  # sentence had only meta/other; skip

        sdg_nums_sorted = sorted(sdg_nums)
        # We keep original sentence as-is; store sdg_terms (filtered) and sdg_numbers (ints)
        out_rows.append((
            company,
            year,
            sentence or "",
            json.dumps(kept_terms, ensure_ascii=False),        # sdg_terms (filtered to 1..17)
            json.dumps(sdg_nums_sorted, ensure_ascii=False),   # sdg_numbers (ints)
            len(sdg_nums_sorted)                               # sdg_count (distinct)
        ))
        kept += 1

    print(f"[INFO] Kept {kept} sentence rows with SDG 1–17 hits.")

    # 4) Write DuckDB
    dst = duckdb.connect(OUT_DB)
    dst.execute(f"""
        CREATE TABLE IF NOT EXISTS {OUT_TABLE} (
            company TEXT,
            year TEXT,
            sentence TEXT,
            sdg_terms TEXT,     -- JSON array of matched terms (only those in sdg01..sdg17)
            sdg_numbers TEXT,   -- JSON array of SDG ints (e.g., [7,13])
            sdg_count INTEGER   -- number of distinct SDGs hit in the sentence
        )
    """)
    dst.execute(f"DELETE FROM {OUT_TABLE}")
    if out_rows:
        dst.executemany(
            f"INSERT INTO {OUT_TABLE} VALUES (?, ?, ?, ?, ?, ?)",
            out_rows
        )
    total_duck = dst.execute(f"SELECT COUNT(*) FROM {OUT_TABLE}").fetchone()[0]
    dst.close()
    print(f"[OK] Wrote {total_duck} rows to {OUT_DB} (table: {OUT_TABLE})")

    # 5) Also write CSV (Python writer with explicit quoting/escape)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(
            f,
            delimiter=",",
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
            escapechar="\\",
            lineterminator="\n",
            doublequote=True,
        )
        w.writerow(["company", "year", "sentence", "sdg_terms", "sdg_numbers", "sdg_count"])
        w.writerows(out_rows)
    print(f"[OK] Wrote CSV: {OUT_CSV}")

    # 6) Quick sanity print
    # Count sentences per SDG number
    try:
        dst2 = duckdb.connect(OUT_DB)
        dst2.execute("INSTALL json"); dst2.execute("LOAD json")
        dist = dst2.execute(f"""
            SELECT x AS sdg_no, COUNT(*) AS sentences
            FROM (
                SELECT UNNEST(CAST(sdg_numbers AS JSON))::INTEGER AS x
                FROM {OUT_TABLE}
            )
            GROUP BY 1
            ORDER BY 1
        """).fetchall()
        dst2.close()
        if dist:
            print("\n[Distribution] Sentences per SDG:")
            for sdg_no, cnt in dist:
                print(f"  SDG {int(sdg_no):2d}: {cnt}")
    except Exception:
        pass

if __name__ == "__main__":
    main()
