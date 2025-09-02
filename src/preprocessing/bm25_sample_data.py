#!/usr/bin/env python3
"""
BM25 pipeline on sample_data using a fast recursive rule-based sentence splitter.
- No spaCy required for splitting/tokenization (faster startup and runtime).
- Language detection per sentence (langdetect).
- BM25 over AI keywords, SDG definitions, and SDG detailed (EN/DE switch).
Outputs per doc:
  - sentences.parquet  (canonical truth: all sentences)
  - matches.parquet    (only matched rows, sparse)
"""

import re
import json
from pathlib import Path
from typing import List, Tuple, Dict
from collections import Counter, defaultdict
import math

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from langdetect import detect_langs
from tqdm import tqdm

# ---------- Config ----------
KW_DIR = Path("kw_data")
DATA_ROOT = Path("data/sample_texts")

BM25_K1 = 1.2
BM25_B = 0.75
BM25_TOPK = 5
BM25_THRESHOLD = 3.0
DE_CONF_THRESHOLD = 0.70

# Regex for tokenization (letters only, includes German umlauts/ß)
TOKEN_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+", re.UNICODE)

# ---------- Lightweight BM25 ----------
class BM25Okapi:
    def __init__(self, documents: List[List[str]], k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.N = len(documents)
        self.avgdl = sum(len(doc) for doc in documents) / (self.N or 1)
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        df = defaultdict(int)
        for doc in documents:
            freqs = Counter(doc)
            self.doc_freqs.append(freqs)
            L = len(doc)
            self.doc_len.append(L)
            for t in freqs.keys():
                df[t] += 1
        for term, f in df.items():
            # standard BM25 idf with +1 smoothing
            self.idf[term] = math.log(1 + (self.N - f + 0.5) / (f + 0.5))

    def get_scores(self, query: List[str]) -> List[float]:
        scores = [0.0] * self.N
        if not query:
            return scores
        qf = Counter(query)
        for term, _ in qf.items():
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i, freqs in enumerate(self.doc_freqs):
                f = freqs.get(term, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * (self.doc_len[i] / (self.avgdl or 1.0)))
                scores[i] += idf * (f * (self.k1 + 1)) / (denom or 1.0)
        return scores

    def get_top_n(self, query: List[str], n: int = 5) -> List[Tuple[int, float]]:
        scores = self.get_scores(query)
        return sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:n]

# ---------- Utilities ----------
def to_tokens(text: str) -> List[str]:
    # Lowercase, keep alphabetic tokens (English+German letters)
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]

def detect_lang(sentence: str) -> Tuple[str, float]:
    try:
        cands = detect_langs(sentence)
        prob_de = 0.0
        prob_en = 0.0
        for c in cands:
            if c.lang == "de":
                prob_de = c.prob
            elif c.lang == "en":
                prob_en = c.prob
        if prob_de >= DE_CONF_THRESHOLD:
            return "de", prob_de
        best = max(cands, key=lambda x: x.prob) if cands else None
        conf = prob_en or (best.prob if best else 0.0)
        return "en", conf
    except Exception:
        return "en", 0.0

# ---------- Recursive rule-based sentence splitter ----------
# Strategy:
#  1) Split on paragraph boundaries (blank lines).
#  2) Within each paragraph, split on strong end punctuation (., !, ?)
#     with safeguards for common abbreviations (Mr., Dr., etc.) and decimals.
#  3) If a chunk remains too long (> max_len), recursively split on semicolons, then commas.
#  4) As a last resort, hard-wrap by max_len characters to avoid giant “sentences”.
ABBR = set([
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs",
    "z.b", "bzw", "usw", "sog", "vgl", "ca", "d.h", "u.a", "etc"
])

END_RE = re.compile(r"([.!?])(\s+|$)")
SEMICOL_RE = re.compile(r";\s+")
COMMA_RE = re.compile(r",\s+")
PARA_RE = re.compile(r"\n\s*\n+")

def split_on_end_punct(paragraph: str) -> List[str]:
    # Greedy scan to keep abbreviations and decimals together
    out = []
    start = 0
    for m in END_RE.finditer(paragraph):
        end = m.end()
        chunk = paragraph[start:end].strip()
        if not chunk:
            start = end
            continue
        # Heuristics: don't cut if likely abbreviation
        tail = chunk.rsplit(" ", 1)[-1].lower()
        tail = tail.rstrip(".")
        if tail in ABBR:
            # keep going; don't split here
            continue
        # Decimal number like 3.14 or 1.000.000
        if re.search(r"\d\.\d", chunk):
            continue
        out.append(chunk)
        start = end
    # Remainder
    rest = paragraph[start:].strip()
    if rest:
        out.append(rest)
    return out

def force_wrap(text: str, max_len: int) -> List[str]:
    return [text[i:i+max_len].strip() for i in range(0, len(text), max_len)]

def recursive_split(text: str, max_len: int = 600) -> List[str]:
    sentences = []
    paragraphs = PARA_RE.split(text)
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        chunks = split_on_end_punct(para)
        for ch in chunks:
            ch = ch.strip()
            if not ch:
                continue
            if len(ch) <= max_len:
                sentences.append(ch)
                continue
            # Try semicolons
            parts = SEMICOL_RE.split(ch)
            if all(len(p) <= max_len for p in parts) and len(parts) > 1:
                sentences.extend([p.strip() for p in parts if p.strip()])
                continue
            # Try commas
            parts = COMMA_RE.split(ch)
            if all(len(p) <= max_len for p in parts) and len(parts) > 1:
                sentences.extend([p.strip() for p in parts if p.strip()])
                continue
            # Last resort: hard wrap
            sentences.extend(force_wrap(ch, max_len))
    return sentences

# ---------- Load targets and build BM25 indices ----------
def load_assets() -> Dict[str, Dict]:
    with open(KW_DIR / "nist_airc.json", "r", encoding="utf-8") as f:
        kw_map = json.load(f)  # en -> de

    with open(KW_DIR / "sdgs.json", "r", encoding="utf-8") as f:
        sdg_en = json.load(f)
    with open(KW_DIR / "sdgs_de.json", "r", encoding="utf-8") as f:
        sdg_de = json.load(f)

    with open(KW_DIR / "sdgs_detailed.json", "r", encoding="utf-8") as f:
        sdgdet_en = json.load(f)
    with open(KW_DIR / "sdgs_detailed_de.json", "r", encoding="utf-8") as f:
        sdgdet_de = json.load(f)

    ai_terms_en = sorted(set(kw_map.keys()))
    ai_terms_de = sorted(set([v for v in kw_map.values() if isinstance(v, str) and v.strip()]))

    sdg_defs_en = [(sid, txt) for sid, txt in sdg_en.items()]
    sdg_defs_de = [(sid, txt) for sid, txt in sdg_de.items()]

    def flatten_detailed(d: Dict[str, List[str]]) -> List[Tuple[str, str]]:
        out = []
        for sid, items in d.items():
            for i, q in enumerate(items):
                out.append((f"{sid}.q{i+1}", q))
        return out

    sdg_detailed_en = flatten_detailed(sdgdet_en)
    sdg_detailed_de = flatten_detailed(sdgdet_de)

    # Tokenize targets once (fast)
    ai_en_docs = [to_tokens(t) for t in ai_terms_en]
    ai_de_docs = [to_tokens(t) for t in ai_terms_de]
    sdg_en_docs = [to_tokens(txt) for (_sid, txt) in sdg_defs_en]
    sdg_de_docs = [to_tokens(txt) for (_sid, txt) in sdg_defs_de]
    sdgdet_en_docs = [to_tokens(txt) for (_tid, txt) in sdg_detailed_en]
    sdgdet_de_docs = [to_tokens(txt) for (_tid, txt) in sdg_detailed_de]

    indices = {
        "ai_en": (BM25Okapi(ai_en_docs, k1=BM25_K1, b=BM25_B), ai_terms_en),
        "ai_de": (BM25Okapi(ai_de_docs, k1=BM25_K1, b=BM25_B), ai_terms_de),
        "sdg_en": (BM25Okapi(sdg_en_docs, k1=BM25_K1, b=BM25_B), sdg_defs_en),
        "sdg_de": (BM25Okapi(sdg_de_docs, k1=BM25_K1, b=BM25_B), sdg_defs_de),
        "sdgdet_en": (BM25Okapi(sdgdet_en_docs, k1=BM25_K1, b=BM25_B), sdg_detailed_en),
        "sdgdet_de": (BM25Okapi(sdgdet_de_docs, k1=BM25_K1, b=BM25_B), sdg_detailed_de),
    }
    return indices

# ---------- I/O ----------
def write_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)

# ---------- Per-document processing ----------
def process_doc(txt_path: Path, indices):
    company = txt_path.parent.parent.name
    year = int(txt_path.parent.name)

    out_sentences = txt_path.parent / "sentences.parquet"
    out_matches = txt_path.parent / "matches.parquet"

    raw = txt_path.read_text(encoding="utf-8", errors="ignore")
    # Normalize a bit: remove soft hyphen & collapse spaces
    raw = raw.replace("\u00AD", "")
    raw = re.sub(r"[ \t]+", " ", raw)

    # Fast sentence split
    sentences = recursive_split(raw, max_len=600)

    sentences_rows = []
    matches_rows = []

    for i, sent in enumerate(tqdm(sentences, desc=f"{company}/{year}", leave=False)):
        if not sent:
            continue
        lang, lang_conf = detect_lang(sent)
        toks = to_tokens(sent)
        n_tokens = len(toks)

        sent_id = f"{company}_{year}_{i:06d}"
        sentences_rows.append({
            "company": company,
            "year": year,
            "sent_id": sent_id,
            "sentence": sent,
            "language": lang,
            "lang_confidence": float(lang_conf),
            "n_tokens": int(n_tokens),
        })

        if n_tokens == 0:
            continue

        # Pick language-specific indices
        ai_idx, ai_targets = indices[f"ai_{lang}"]
        sdg_idx, sdg_targets = indices[f"sdg_{lang}"]
        sdgdet_idx, sdgdet_targets = indices[f"sdgdet_{lang}"]

        def push_matches(matcher_name: str, bm25_idx, targets, get_id_text):
            ranked = bm25_idx.get_top_n(toks, n=BM25_TOPK)
            for rank_pos, (t_ix, score) in enumerate(ranked, start=1):
                if score < BM25_THRESHOLD:
                    continue
                target_id, _ = get_id_text(t_ix)
                matches_rows.append({
                    "company": company,
                    "year": year,
                    "sent_id": sent_id,
                    "matcher": matcher_name,
                    "target_id": target_id,
                    "bm25_score": float(score),
                    "rank": int(rank_pos),
                    "threshold_pass": True,
                })

        # AI keywords: str list
        push_matches("ai_kw", ai_idx, ai_targets, lambda ix: (ai_targets[ix], ai_targets[ix]))
        # SDG defs: (sid, text)
        push_matches("sdg_def", sdg_idx, sdg_targets, lambda ix: (sdg_targets[ix][0], sdg_targets[ix][1]))
        # SDG detailed: (tid, text)
        push_matches("sdg_detailed", sdgdet_idx, sdgdet_targets, lambda ix: (sdgdet_targets[ix][0], sdgdet_targets[ix][1]))

    # Write outputs
    df_s = pd.DataFrame(sentences_rows)
    write_parquet(df_s, out_sentences)

    df_m = pd.DataFrame(matches_rows)
    if df_m.empty:
        df_m = pd.DataFrame(columns=[
            "company","year","sent_id","matcher","target_id","bm25_score","rank","threshold_pass"
        ])
    write_parquet(df_m, out_matches)

# ---------- Main ----------
def main():
    indices = load_assets()
    companies = [p for p in DATA_ROOT.iterdir() if p.is_dir()]

    for cdir in tqdm(companies, desc="Companies"):
        for ydir in sorted([p for p in cdir.iterdir() if p.is_dir()]):
            txt_path = ydir / "results.txt"
            if txt_path.exists():
                process_doc(txt_path, indices)

    print("All done ✅")

if __name__ == "__main__":
    main()
