# src/analytics/langdetect_ai_quadrant_summary.py
# Requires: pip install langdetect
import os
import csv
import json
import re
from langdetect import detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# ----- Hardcoded config -----
CSV_PATH = "data/ai_counts_by_company_year.csv"   # built earlier
ROOT_DIR = "data/jsons"                            # data/jsons/<Company>/<YEAR>/splits_semantic.json
LANG_SAMPLE_CHARS = 10_000
DE_MIN_PROB = 0.70  # German if P(de) >= threshold OR top language == 'de'

STACK_COLS = [
    "ai_ml_count",
    "cloud_computing_count",
    "big_data_blockchain_count",
    "applications_practice_count",
]

DetectorFactory.seed = 0  # deterministic langdetect

def sample_text_from_json_file(path: str, limit: int = LANG_SAMPLE_CHARS) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)  # { "0": "...", "1": "...", ... }
    out, total = [], 0
    for _, v in data.items():  # preserves JSON order in Py3.7+
        s = str(v or "")
        out.append(s)
        total += len(s)
        if total >= limit:
            break
    txt = " ".join(out)[:limit]
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def is_german_langdetect(text: str, p_threshold: float = DE_MIN_PROB) -> bool:
    if not text or len(text) < 50:
        return False
    try:
        langs = detect_langs(text)  # e.g., [en:0.72, de:0.26, ...]
    except LangDetectException:
        return False
    if not langs:
        return False
    p_de = 0.0
    top_lang = langs[0].lang
    for lp in langs:
        if lp.lang == "de":
            p_de = max(p_de, float(lp.prob))
    return (p_de >= p_threshold) or (top_lang == "de")

def main():
    if not os.path.isfile(CSV_PATH):
        print(f"[ERROR] Missing CSV: {CSV_PATH}")
        return

    # Quadrant counters
    de_nonzero = de_zero = en_nonzero = en_zero = 0
    unknown = 0  # missing/failed JSON

    # Optional: totals per stack by language (for your own sanity checks)
    de_stack_sums  = {k: 0 for k in STACK_COLS}
    en_stack_sums  = {k: 0 for k in STACK_COLS}

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        company = row["company"]
        year    = row["year"]

        # Parse counts
        counts = []
        for col in STACK_COLS:
            try:
                counts.append(int(row.get(col, 0)))
            except Exception:
                counts.append(0)

        any_positive = any(v > 0 for v in counts)
        all_zero     = all(v == 0 for v in counts)

        # Load underlying text for langdetect
        json_path = os.path.join(ROOT_DIR, company, year, "splits_semantic.json")
        if not os.path.isfile(json_path):
            unknown += 1
            continue

        try:
            sample = sample_text_from_json_file(json_path, limit=LANG_SAMPLE_CHARS)
        except Exception:
            unknown += 1
            continue

        is_de = is_german_langdetect(sample, DE_MIN_PROB)

        if is_de:
            if any_positive: de_nonzero += 1
            if all_zero:     de_zero    += 1
            # sum stacks
            for col, val in zip(STACK_COLS, counts):
                de_stack_sums[col] += val
        else:
            if any_positive: en_nonzero += 1
            if all_zero:     en_zero    += 1
            for col, val in zip(STACK_COLS, counts):
                en_stack_sums[col] += val

    # Print the four requested counts
    print("=== AI stack presence by language (company-year level; langdetect) ===")
    print(f"German  & ≥1 non-zero : {de_nonzero}")
    print(f"German  & all zeros   : {de_zero}")
    print(f"English & ≥1 non-zero : {en_nonzero}")
    print(f"English & all zeros   : {en_zero}")

    total_seen = de_nonzero + de_zero + en_nonzero + en_zero
    if unknown:
        print(f"\n[Note] {unknown} company-year(s) skipped due to missing/failed JSON for language detection.")
    print(f"Total rows accounted in quadrants: {total_seen} (CSV rows: {len(rows)})")

    # Optional: quick stack totals per language
    print("\n[Optional] Sum of counts per AI stack by language")
    print("German :", ", ".join(f"{k.replace('_count','')}: {v}" for k, v in de_stack_sums.items()))
    print("English:", ", ".join(f"{k.replace('_count','')}: {v}" for k, v in en_stack_sums.items()))

if __name__ == "__main__":
    main()

"""
German  & ≥1 non-zero : 44
German  & all zeros   : 14
English & ≥1 non-zero : 1047
English & all zeros   : 315

Sum of counts per AI stack by language
German : ai_ml: 12, cloud_computing: 70, big_data_blockchain: 9, applications_practice: 525
English: ai_ml: 2041, cloud_computing: 1331, big_data_blockchain: 908, applications_practice: 5540

"""