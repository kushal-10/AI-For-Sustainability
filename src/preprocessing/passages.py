#!/usr/bin/env python3
"""
Count sentences that mention any SDG keyword across all splits.json files.

- Loads keywords from kw_data/sdg_keys.json
- Scans data/texts/**/splits.json
- Each splits.json maps string indices -> sentence strings
- If a sentence matches >=1 keyword (with '*' treated as a word-wildcard), it counts as a hit
- Prints total sentence count and total hits (and %)

Usage:
    python scripts/count_sdg_sentence_hits.py
"""

import json
import re
from pathlib import Path

from tqdm import tqdm

SDG_JSON = Path("kw_data/sdg_keys.json")
SPLITS_GLOB = "data/texts/**/splits.json"

def load_keywords_union():
    """Load SDG keywords and build a single alternation regex for ANY keyword."""
    with open(SDG_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Flatten all keyword lists into one list
    all_keywords = []
    for _, keywords in data.items():
        if isinstance(keywords, list):
            all_keywords.extend(keywords)

    # Prepare regex parts:
    # - escape everything
    # - turn '*' into \w* (word wildcard)
    parts = []
    for kw in all_keywords:
        if not kw:
            continue
        pat = re.escape(kw).replace(r"\*", r"\w*")
        # word boundaries; allow keywords with spaces too
        parts.append(r"\b" + pat + r"\b")

    if not parts:
        # Fallback that matches nothing
        return re.compile(r"a^")

    # One big alternation, case-insensitive
    combined = "(?:" + "|".join(parts) + ")"
    return re.compile(combined, flags=re.IGNORECASE)

def main():
    matcher = load_keywords_union()

    files = list(Path(".").glob(SPLITS_GLOB))
    if not files:
        print(f"⚠️  No splits.json files found under {SPLITS_GLOB}")
        return

    total_sentences = 0
    hit_sentences = 0
    total_files = 0

    for fp in tqdm(files):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as e:
            print(f"⚠️  Skipping {fp} (failed to read/parse): {e}")
            continue

        # Expect dict of {"0": "...", "1": "...", ...}
        # If it's a list, handle gracefully.
        if isinstance(obj, dict):
            sentences = list(obj.values())
        elif isinstance(obj, list):
            sentences = obj
        else:
            # Unknown shape
            continue

        total_files += 1
        for s in sentences:
            if not isinstance(s, str):
                continue
            total_sentences += 1
            if matcher.search(s):
                hit_sentences += 1

    pct = (hit_sentences / total_sentences * 100) if total_sentences else 0.0

    print("===== SDG Sentence Hit Summary =====")
    print(f"Files scanned:            {total_files}")
    print(f"Total sentences:          {total_sentences}")
    print(f"Sentences with any SDG:   {hit_sentences}")
    print(f"Hit rate:                 {pct:.2f}%")

if __name__ == "__main__":
    main()
