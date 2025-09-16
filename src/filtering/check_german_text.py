# src/analytics/detect_german_langdetect.py
# Requires: pip install langdetect
import os
import json
import re
from langdetect import detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# ----- Hardcoded config -----
ROOT_DIR = "data/jsons"        # expects data/jsons/<Company>/<YEAR>/splits_semantic.json
LANG_SAMPLE_CHARS = 10_000     # use first ~10k chars
DE_MIN_PROB = 0.70             # consider "German" if P(de) >= 0.70 (tune if needed)

# Make langdetect deterministic across runs
DetectorFactory.seed = 0

def find_all_company_years(root: str):
    pairs = []
    if not os.path.isdir(root):
        return pairs
    for company in os.listdir(root):
        cdir = os.path.join(root, company)
        if not os.path.isdir(cdir):
            continue
        for year in os.listdir(cdir):
            ydir = os.path.join(cdir, year)
            if not os.path.isdir(ydir):
                continue
            sp = os.path.join(ydir, "splits_semantic.json")
            if os.path.isfile(sp):
                pairs.append((company, year, sp))
    pairs.sort(key=lambda x: (x[0].lower(), x[1]))
    return pairs

def sample_text_from_json_file(path: str, limit: int = LANG_SAMPLE_CHARS) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)  # { "0": "...", "1": "...", ... }
    out, total = [], 0
    # Preserve insertion order (Python 3.7+ dicts keep JSON order)
    for _, v in data.items():
        s = str(v or "")
        out.append(s)
        total += len(s)
        if total >= limit:
            break
    # Simple whitespace collapse to help detector
    txt = " ".join(out)[:limit]
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def is_german_langdetect(text: str, p_threshold: float = DE_MIN_PROB):
    """
    Return (is_german: bool, p_de: float, top_lang: str, top_prob: float, langs_raw: str)
    Uses langdetect.detect_langs to get a probability distribution.
    """
    if not text or len(text) < 50:
        return (False, 0.0, None, None, "")
    try:
        langs = detect_langs(text)  # e.g., [en:0.72, de:0.26, ...]
    except LangDetectException:
        return (False, 0.0, None, None, "")
    if not langs:
        return (False, 0.0, None, None, "")

    # Find de prob, and the top language
    p_de = 0.0
    top_lang, top_prob = langs[0].lang, float(langs[0].prob)
    for lp in langs:
        if lp.lang == "de":
            p_de = max(p_de, float(lp.prob))

    # Heuristic: German if P(de) >= threshold OR 'de' is the top language
    is_de = (p_de >= p_threshold) or (top_lang == "de")
    langs_raw = ", ".join(f"{lp.lang}:{lp.prob:.2f}" for lp in langs)
    return (is_de, p_de, top_lang, top_prob, langs_raw)

def main():
    pairs = find_all_company_years(ROOT_DIR)
    if not pairs:
        print(f"[ERROR] No reports found under {ROOT_DIR}")
        return

    german = []
    english_or_other = []
    unknown = []

    for company, year, path in pairs:
        try:
            sample = sample_text_from_json_file(path, limit=LANG_SAMPLE_CHARS)
            is_de, p_de, top_lang, top_prob, langs_raw = is_german_langdetect(sample, DE_MIN_PROB)
        except Exception as e:
            unknown.append((company, year, f"error:{e}"))
            continue

        if top_lang is None:
            unknown.append((company, year, "no_top_lang"))
            continue

        if is_de:
            german.append((company, year, p_de, top_lang, top_prob, langs_raw))
        else:
            english_or_other.append((company, year, p_de, top_lang, top_prob, langs_raw))

    # Print results
    print("\n=== German-detected reports (langdetect) ===")
    for company, year, p_de, top_lang, top_prob, langs_raw in sorted(german, key=lambda x: (x[0].lower(), x[1])):
        print(f"- {company} — {year} | P(de)={p_de:.2f} | top={top_lang}:{top_prob:.2f} | {langs_raw}")
    print(f"\nTotal German-detected: {len(german)}")

    if unknown:
        print("\n[WARN] Unknown/failed language detection for:")
        for company, year, reason in sorted(unknown, key=lambda x: (x[0].lower(), x[1])):
            print(f"- {company} — {year} | {reason}")

    print(f"\nGrand total scanned: {len(pairs)} | German: {len(german)} | Non-German: {len(english_or_other)} | Unknown: {len(unknown)}")
    print(f"(Threshold P(de) >= {DE_MIN_PROB:.2f} OR top language == 'de')")

if __name__ == "__main__":
    main()
