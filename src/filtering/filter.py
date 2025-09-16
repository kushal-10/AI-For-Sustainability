# src/extract_matched_sentences.py
import os
import re
import json
import unicodedata
from typing import Dict, List, Tuple, Iterable

import duckdb
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# NEW: langdetect
from langdetect import detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
DetectorFactory.seed = 0  # deterministic

# ====== HARD-CODED PATHS / SETTINGS ======
ROOT_DIR = "data/jsons"                      # expects data/jsons/<Company>/<YEAR>/splits_semantic.json
SDG_JSON_EN = "data/keywords_sdg.json"
TECH_JSON_EN = "data/keywords_tech.json"
SDG_JSON_DE = "data/keywords_sdg_de.json"
TECH_JSON_DE = "data/keywords_tech_de.json"

DUCKDB_PATH = "data/matches.duckdb"
TABLE_NAME = "matched_sentences"

MAX_WORKERS = max(1, (os.cpu_count() or 4) - 1)  # leave 1 core free on M3
BATCH_FLUSH_ROWS = 25000                         # flush to DuckDB every ~25k rows
LANG_SAMPLE_CHARS = 10_000                       # look at first ~10k chars for language detection
DE_MIN_PROB = 0.70                               # consider German if P(de) >= 0.70 or top==de

# ====== Helpers ======
def norm_text(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")

def _escape_except_wildcard(term: str) -> str:
    parts = []
    for ch in term:
        if ch == "*":
            parts.append("*")
        else:
            parts.append(re.escape(ch))
    return "".join(parts)

def _word_boundary_if_needed(pattern_core: str) -> Tuple[str, str]:
    start = r"\b" if re.match(r"^\w", pattern_core, flags=re.UNICODE | re.IGNORECASE) else ""
    end = r"\b" if re.search(r"\w$", pattern_core, flags=re.UNICODE | re.IGNORECASE) else ""
    return start, end

def build_keyword_regex(term: str) -> re.Pattern:
    t = term.strip()
    if not t:
        return re.compile(r"(?!x)x", flags=re.IGNORECASE | re.UNICODE)
    t = _escape_except_wildcard(t)              # keep '*' raw for now
    t = t.replace("*", r"\w*")                  # '*' -> word-suffix wildcard
    t = re.sub(r"\s+", lambda _: r"[\s-]+", t)  # flexible space/hyphen between words
    start_b, end_b = _word_boundary_if_needed(t)
    pattern = f"{start_b}{t}{end_b}"
    return re.compile(pattern, flags=re.IGNORECASE | re.UNICODE)

def flatten_keyword_map(d: Dict[str, Iterable[str]], prefix: str = "") -> Dict[str, List[str]]:
    term2cats: Dict[str, List[str]] = {}
    for cat, terms in d.items():
        if not isinstance(terms, list):
            continue
        cat_name = f"{prefix}{cat}" if prefix else cat
        for term in terms:
            term_norm = term.strip().lower()
            if not term_norm:
                continue
            term2cats.setdefault(term_norm, []).append(cat_name)
    return term2cats

def _load_terms_from_json(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    main = {k: v for k, v in raw.items() if k != "meta"}
    meta = raw.get("meta", {})
    term2cats = flatten_keyword_map(main)
    term2cats.update(flatten_keyword_map(meta, prefix="meta."))
    return term2cats

def _load_tech_terms_from_json(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        tech_raw = json.load(f)
    # top-level lists are categories
    term2cats: Dict[str, List[str]] = {}
    for cat, terms in tech_raw.items():
        if isinstance(terms, list):
            for term in terms:
                tt = term.strip().lower()
                if not tt:
                    continue
                term2cats.setdefault(tt, []).append(cat)
    return term2cats

def load_keywords(sdg_en_path: str, tech_en_path: str,
                  sdg_de_path: str = None, tech_de_path: str = None):
    # English
    sdg_en_term2cats = _load_terms_from_json(sdg_en_path)
    tech_en_term2cats = _load_tech_terms_from_json(tech_en_path)
    # German (optional)
    sdg_de_term2cats = _load_terms_from_json(sdg_de_path) if sdg_de_path and os.path.exists(sdg_de_path) else {}
    tech_de_term2cats = _load_tech_terms_from_json(tech_de_path) if tech_de_path and os.path.exists(tech_de_path) else {}

    def to_pairs(term2cats: Dict[str, List[str]]):
        return [(term, cats) for term, cats in term2cats.items()]

    return (
        to_pairs(sdg_en_term2cats),
        to_pairs(tech_en_term2cats),
        to_pairs(sdg_de_term2cats),
        to_pairs(tech_de_term2cats),
    )

def find_splits_files(root: str) -> List[Tuple[str, str, str]]:
    results = []
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
                results.append((company, year, sp))
    return results

def sample_text_from_json(data: Dict[str, str], limit: int = LANG_SAMPLE_CHARS) -> str:
    # Concatenate values until we reach ~limit chars
    out = []
    total = 0
    for _, v in data.items():
        s = str(v or "")
        out.append(s)
        total += len(s)
        if total >= limit:
            break
    txt = " ".join(out)[:limit]
    # light cleanup helps the detector
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

# NEW: langdetect-based German detection
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

# ====== Worker globals & functions (for multiprocessing) ======
_sdg_en = None
_tech_en = None
_sdg_de = None
_tech_de = None

def _init_worker(sdg_en_pairs, tech_en_pairs, sdg_de_pairs, tech_de_pairs):
    """Initializer: compile regex ONCE per worker (macOS uses spawn)."""
    global _sdg_en, _tech_en, _sdg_de, _tech_de
    def compile_pairs(pairs):
        return [(term, cats, build_keyword_regex(term)) for term, cats in pairs]
    _sdg_en  = compile_pairs(sdg_en_pairs)
    _tech_en = compile_pairs(tech_en_pairs)
    _sdg_de  = compile_pairs(sdg_de_pairs) if sdg_de_pairs else []
    _tech_de = compile_pairs(tech_de_pairs) if tech_de_pairs else []

def match_terms(text: str, compiled_terms: List[Tuple[str, List[str], re.Pattern]]) -> List[str]:
    hits = []
    for term, _cats, rx in compiled_terms:
        if rx.search(text):
            hits.append(term)
    return hits

def process_file(company: str, year: str, path: str) -> List[Dict]:
    # Load once
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)  # { "0": "sentence or chunk", ... }

    # Language probe on first ~10k chars via langdetect
    probe = sample_text_from_json(data, limit=LANG_SAMPLE_CHARS)
    germanish = is_german_langdetect(probe, DE_MIN_PROB)

    # Choose term sets
    if germanish:
        sdg_terms  = _sdg_en + _sdg_de
        tech_terms = _tech_en + _tech_de
    else:
        sdg_terms  = _sdg_en
        tech_terms = _tech_en

    rows = []
    for sid, sentence in data.items():
        sent_orig = sentence if isinstance(sentence, str) else str(sentence)
        sent_norm = norm_text(sent_orig).casefold()

        sdg_hits = match_terms(sent_norm, sdg_terms)
        ai_hits  = match_terms(sent_norm, tech_terms)

        if not sdg_hits and not ai_hits:
            continue

        all_hits = sorted(set(sdg_hits + ai_hits))
        rows.append({
            "company": company,
            "year": year,
            "sentence_id": sid,
            "sentence": sent_orig.strip(),
            "sdg_keywords": json.dumps(sorted(sdg_hits), ensure_ascii=False),
            "ai_keywords": json.dumps(sorted(ai_hits), ensure_ascii=False),
            "all_keywords": json.dumps(all_hits, ensure_ascii=False)
        })
    return rows

def _worker_process(args: Tuple[str, str, str]) -> List[Dict]:
    company, year, spath = args
    return process_file(company, year, spath)

# ====== DuckDB I/O ======
def ensure_table(con: duckdb.DuckDBPyConnection, table: str):
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            company TEXT,
            year TEXT,
            sentence_id TEXT,
            sentence TEXT,
            sdg_keywords TEXT,
            ai_keywords TEXT,
            all_keywords TEXT
        )
    """)

def insert_rows(con: duckdb.DuckDBPyConnection, table: str, rows: List[Dict]):
    if not rows:
        return
    data = [
        (
            r["company"],
            r["year"],
            r["sentence_id"],
            r["sentence"],
            r["sdg_keywords"],
            r["ai_keywords"],
            r["all_keywords"],
        )
        for r in rows
    ]
    con.executemany(f"INSERT INTO {table} VALUES (?, ?, ?, ?, ?, ?, ?)", data)

# ====== Main ======
def main():
    # Prepare keyword pairs (EN + optional DE). Regex compiled per worker.
    sdg_en_pairs, tech_en_pairs, sdg_de_pairs, tech_de_pairs = load_keywords(
        SDG_JSON_EN, TECH_JSON_EN, SDG_JSON_DE, TECH_JSON_DE
    )

    # Discover all files
    targets = find_splits_files(ROOT_DIR)
    if not targets:
        print(f"[ERROR] No splits found under {ROOT_DIR}")
        return

    # Build sample list: first 15 unique companies
    seen = set()
    ordered_companies = []
    for company, _, _ in targets:
        if company not in seen:
            seen.add(company)
            ordered_companies.append(company)
    sample_companies = set(ordered_companies[:15])
    sample_targets = [t for t in targets if t[0] in sample_companies]

    # Open DuckDB and ensure table
    con = duckdb.connect(DUCKDB_PATH)
    ensure_table(con, TABLE_NAME)

    # ---- SAMPLE RUN (parallel over 15 companies) ----
    sample_rows: List[Dict] = []
    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=_init_worker,
        initargs=(sdg_en_pairs, tech_en_pairs, sdg_de_pairs, tech_de_pairs),
    ) as ex:
        for rows in tqdm(
            ex.map(_worker_process, sample_targets, chunksize=8),
            total=len(sample_targets),
            desc="Sample (15 companies)",
            unit="file",
        ):
            sample_rows.extend(rows)

    # Replace table content with sample (so you can quickly inspect)
    con.execute(f"DELETE FROM {TABLE_NAME}")
    insert_rows(con, TABLE_NAME, sample_rows)
    cnt_after_sample = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
    print(f"[INFO] Sample write OK. Rows in {TABLE_NAME}: {cnt_after_sample}")

    if cnt_after_sample == 0:
        print("[ERROR] Sample resulted in 0 rows; aborting full run.")
        con.close()
        return

    # ---- FULL RUN (parallel over all companies, streaming flush) ----
    # Clear sample, then stream-insert batches to keep memory low.
    con.execute(f"DELETE FROM {TABLE_NAME}")

    buffer: List[Dict] = []
    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=_init_worker,
        initargs=(sdg_en_pairs, tech_en_pairs, sdg_de_pairs, tech_de_pairs),
    ) as ex:
        for rows in tqdm(
            ex.map(_worker_process, targets, chunksize=8),
            total=len(targets),
            desc="Full run (all companies)",
            unit="file",
        ):
            if rows:
                buffer.extend(rows)
            if len(buffer) >= BATCH_FLUSH_ROWS:
                insert_rows(con, TABLE_NAME, buffer)
                buffer.clear()

    # Final flush
    if buffer:
        insert_rows(con, TABLE_NAME, buffer)
        buffer.clear()

    cnt_after_full = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
    con.close()
    print(f"[INFO] Full write OK. Rows in {TABLE_NAME}: {cnt_after_full}")
    print(f"[DONE] DuckDB at: {DUCKDB_PATH} / table: {TABLE_NAME} / workers: {MAX_WORKERS}")

if __name__ == "__main__":
    main()
