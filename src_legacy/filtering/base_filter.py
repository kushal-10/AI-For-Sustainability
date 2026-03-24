#!/usr/bin/env python3
"""
base_filter.py

Extensible filtering framework (fast path).
Optimizations:
- Optional wildcard expansion (--wildcard)
- Combined regex per category (1 search/category/passage)
- Fast language detection via pycld3 (fallback to langdetect)
- Single DuckDB connection with Appender
- Progress bar; only errors printed
- Optional exact-hit enumeration (--enumerate-hits) [slower]

Subclasses must implement:
  - category_names(self) -> List[str]
  - table_name(self) -> str
  - extra_columns(self) -> List[Tuple[col_name, duckdb_type]]
  - make_row(self, *, global_id, passage, company, year, language, hits_by_cat) -> Tuple[Any,...]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional

import duckdb
from tqdm import tqdm

# ---- Language ID: prefer pycld3, fallback to langdetect ----
try:
    import pycld3  # type: ignore
    _USE_CLD3 = True
except Exception:
    _USE_CLD3 = False
    from langdetect import detect, DetectorFactory  # type: ignore
    DetectorFactory.seed = 42  # deterministic


# --------------------------- Small utils ---------------------------

def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _clean_company_for_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())

def _first_n_passages(data: Dict[str, str], n: int = 5) -> List[str]:
    items = sorted(((int(k), v) for k, v in data.items()), key=lambda x: x[0])
    return [v for _, v in items[:n]]

def _detect_is_german(passages: List[str]) -> Tuple[bool, str]:
    """Return (is_german, best_label) using fast CLD3 if available."""
    labels: List[str] = []
    for p in passages:
        t = (p or "").strip()
        if not t:
            continue
        if _USE_CLD3:
            res = pycld3.get_language(t[:2000])
            if res and res.is_reliable:
                labels.append(res.language)  # 'de', 'en', ...
        else:
            try:
                labels.append(detect(t))
            except Exception:
                pass
    if not labels:
        return (False, "unknown")
    is_de = sum(1 for l in labels if l == "de") >= 2
    best = max(set(labels), key=labels.count)
    return (is_de, best)

def _is_alnum(c: str) -> bool:
    return bool(re.match(r"[A-Za-z0-9]", c))

def _token_to_regex(token: str, star_is_wildcard: bool) -> str:
    """
    Build a regex pattern from a keyword token.
    - If star_is_wildcard:
        * trailing '*' after an alnum stem -> '\\w*'  (tight suffix)
        * '*' elsewhere -> '.*'                        (loose span)
        * spaces -> '\\s+'                             (robust whitespace)
      Otherwise, '*' is literal.
    - Add '\\b' at start/end when the visible ends are alphanumeric.
    """
    raw = token.strip()
    if not raw:
        return ""

    pieces: List[str] = []
    for i, ch in enumerate(raw):
        if star_is_wildcard and ch == "*":
            if i == len(raw) - 1:
                pieces.append(r"\w*")
            else:
                pieces.append(r".*")
        elif star_is_wildcard and ch == " ":
            pieces.append(r"\s+")
        else:
            pieces.append(re.escape(ch))

    pat_body = "".join(pieces)

    # Word boundaries if starts/ends with alnum (ignore trailing '*')
    first_vis = next((c for c in raw if c != " "), raw[:1])
    last_vis = next((c for c in reversed(raw) if c != "*"), "")
    if _is_alnum(first_vis):
        pat_body = r"\b" + pat_body
    if _is_alnum(last_vis):
        pat_body = pat_body + r"\b"

    return pat_body

# --------- COMBINED REGEX (fast path) + optional per-token list (for enumeration) ---------

def _compile_one(token: str, star_is_wildcard: bool) -> str:
    return _token_to_regex(token, star_is_wildcard)

def _combine_category_regex(tokens: List[str], star_is_wildcard: bool) -> Optional[re.Pattern]:
    alts: List[str] = []
    for t in tokens:
        pat = _compile_one(t, star_is_wildcard)
        if pat:
            alts.append(pat)
    if not alts:
        return None
    combined = "(?:" + "|".join(alts) + ")"
    return re.compile(combined, re.IGNORECASE)

def _compile_patterns_list(tokens: Iterable[str], star_is_wildcard: bool) -> List[re.Pattern]:
    out: List[re.Pattern] = []
    for t in tokens:
        pat = _compile_one(t, star_is_wildcard)
        if pat:
            out.append(re.compile(pat, re.IGNORECASE))
    return out

def _flatten_keyword_dict_combined(dct: Dict[str, List[str]], star_is_wildcard: bool) -> Dict[str, re.Pattern]:
    out: Dict[str, re.Pattern] = {}
    for cat, toks in dct.items():
        cp = _combine_category_regex(toks, star_is_wildcard)
        if cp is not None:
            out[cat] = cp
    return out

def _flatten_keyword_dict_list(dct: Dict[str, List[str]], star_is_wildcard: bool) -> Dict[str, List[re.Pattern]]:
    return {cat: _compile_patterns_list(words, star_is_wildcard) for cat, words in dct.items()}

def _find_hits_fast(text: str, combined: Dict[str, re.Pattern]) -> Dict[str, List[str]]:
    """Fast category-level detector. Returns ['__hit__'] when category matches."""
    hits: Dict[str, List[str]] = {}
    for cat, pat in combined.items():
        hits[cat] = ["__hit__"] if pat.search(text) else []
    return hits

def _find_hits_enumerate(text: str, per_token: Dict[str, List[re.Pattern]]) -> Dict[str, List[str]]:
    """Exact token enumeration (slower). Returns regex patterns that matched."""
    hits: Dict[str, List[str]] = {}
    for cat, pats in per_token.items():
        found: List[str] = []
        for p in pats:
            if p.search(text):
                found.append(p.pattern)
        # dedup preserving order
        seen = set(); uniq = []
        for f in found:
            k = f.lower()
            if k not in seen:
                seen.add(k); uniq.append(f)
        hits[cat] = uniq
    return hits

def _any_hits(h: Dict[str, List[str]]) -> bool:
    return any(h[cat] for cat in h)

def _snake(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

def _json_str(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False)


# --------------------------- Base class ---------------------------

class Filter:
    """Base class. Subclasses configure categories, table name, extras, and row-building."""

    BASE_COLUMNS: List[Tuple[str, str]] = [
        ("global_id", "TEXT"),
        ("passage", "TEXT"),
        ("company", "TEXT"),
        ("year", "TEXT"),
        ("language", "TEXT"),
    ]

    def __init__(
        self,
        *,
        root_path: str,
        kw_en_path: str,
        kw_de_path: str,
        out_db: str,
        table: Optional[str] = None,
        use_de_when_detected: bool = True,
        star_is_wildcard: bool = False,
        enumerate_hits: bool = False,
    ):
        self.root = Path(root_path)
        self.star_is_wildcard = star_is_wildcard
        self.enumerate_hits = enumerate_hits

        self.kw_en_raw = _load_json(Path(kw_en_path))
        self.kw_de_raw = _load_json(Path(kw_de_path))

        # Fast combined regex (default path)
        self.compiled_en_combined = _flatten_keyword_dict_combined(self.kw_en_raw, self.star_is_wildcard)
        self.compiled_de_combined = _flatten_keyword_dict_combined(self.kw_de_raw, self.star_is_wildcard)

        # Optional per-token lists (only used if enumerate_hits is True)
        self.compiled_en_list = _flatten_keyword_dict_list(self.kw_en_raw, self.star_is_wildcard) if enumerate_hits else {}
        self.compiled_de_list = _flatten_keyword_dict_list(self.kw_de_raw, self.star_is_wildcard) if enumerate_hits else {}

        self.out_db = Path(out_db)
        self._table_override = table
        self.use_de_when_detected = use_de_when_detected

        self._validate_categories()

        # Ensure DB dir and table; keep ONE connection open
        self.out_db.parent.mkdir(parents=True, exist_ok=True)
        cols = self.BASE_COLUMNS + self.hit_columns + self.extra_columns()
        col_defs = ",\n                ".join(f"{name} {dtype}" for name, dtype in cols)
        self._con = duckdb.connect(self.out_db.as_posix())
        self._con.execute(f"CREATE TABLE IF NOT EXISTS {self.table} ({col_defs});")

    # ---------- Hooks for subclasses ----------

    def category_names(self) -> List[str]:
        raise NotImplementedError

    def table_name(self) -> str:
        raise NotImplementedError

    def extra_columns(self) -> List[Tuple[str, str]]:
        return []

    def make_row(
        self,
        *,
        global_id: str,
        passage: str,
        company: str,
        year: str,
        language: str,
        hits_by_cat: Dict[str, List[str]],
    ) -> Tuple[Any, ...]:
        raise NotImplementedError

    # ---------- Internals ----------

    @property
    def table(self) -> str:
        return self._table_override or self.table_name()

    @property
    def hit_columns(self) -> List[Tuple[str, str]]:
        # TEXT columns holding JSON arrays, named by category
        return [(f"hits_{_snake(cat)}", "TEXT") for cat in self.category_names()]

    def _validate_categories(self) -> None:
        cats = set(self.category_names())
        en_cats = set(self.kw_en_raw.keys())
        de_cats = set(self.kw_de_raw.keys())
        missing_en = cats - en_cats
        missing_de = cats - de_cats
        if missing_en:
            raise ValueError(f"Missing EN keyword categories: {sorted(missing_en)}")
        if missing_de and self.use_de_when_detected:
            raise ValueError(f"Missing DE keyword categories: {sorted(missing_de)}")

    # --- in base_filter.py ---

    def _append_rows(self, rows: list[tuple]) -> None:
        """
        Fast-path append using DuckDB appender; falls back to executemany() if not available.
        """
        if not rows:
            return

        # total columns = base + dynamic hit cols + any extras
        ncols = len(self.BASE_COLUMNS) + len(self.hit_columns) + len(self.extra_columns())
        placeholders = ",".join(["?"] * ncols)

        try:
            # Newer DuckDBs
            with self._con.appender(self.table) as app:
                for r in rows:
                    app.append(r)
        except AttributeError:
            # Older DuckDBs (no .appender)
            self._con.executemany(f"INSERT INTO {self.table} VALUES ({placeholders})", rows)


    def _iter_files(self) -> List[Tuple[str, str, Path]]:
        out: List[Tuple[str, str, Path]] = []
        for company_dir in self.root.iterdir():
            if not company_dir.is_dir():
                continue
            company = company_dir.name
            for year_dir in company_dir.iterdir():
                if not year_dir.is_dir():
                    continue
                year = year_dir.name
                fpath = year_dir / "splits_semantic.json"
                if fpath.exists():
                    out.append((company, year, fpath))
        return out

    def _merge_combined(self, use_german: bool) -> Dict[str, re.Pattern]:
        if use_german and self.use_de_when_detected:
            return {c: self.compiled_en_combined.get(c, None) or self.compiled_de_combined.get(c, None)
                    for c in self.category_names()
                    if (self.compiled_en_combined.get(c) or self.compiled_de_combined.get(c))}
        else:
            return {c: self.compiled_en_combined[c] for c in self.category_names() if c in self.compiled_en_combined}

    def _merge_list(self, use_german: bool) -> Dict[str, List[re.Pattern]]:
        if not self.enumerate_hits:
            return {}
        if use_german and self.use_de_when_detected:
            merged: Dict[str, List[re.Pattern]] = {}
            for c in self.category_names():
                merged[c] = self.compiled_en_list.get(c, []) + self.compiled_de_list.get(c, [])
            return merged
        else:
            return {c: self.compiled_en_list.get(c, []) for c in self.category_names()}

    # ---------- Per-file processing ----------

    def _rows_for_file(self, company: str, year: str, json_path: Path) -> List[Tuple[Any, ...]]:
        data = _load_json(json_path)
        passages_first5 = _first_n_passages(data, 5)
        is_german, best_label = _detect_is_german(passages_first5)

        combined = self._merge_combined(use_german=is_german)
        per_token = self._merge_list(use_german=is_german) if self.enumerate_hits else {}

        company_clean = _clean_company_for_id(company)
        rows: List[Tuple[Any, ...]] = []

        # Iterate passages in numeric order
        for sid, passage in sorted(((int(k), v) for k, v in data.items()), key=lambda x: x[0]):
            text = (passage or "").strip()
            if not text:
                continue

            if self.enumerate_hits:
                hits = _find_hits_enumerate(text, per_token)
            else:
                hits = _find_hits_fast(text, combined)

            if not _any_hits(hits):
                continue

            global_id = f"{year}{company_clean}{sid}"
            language = "de" if is_german else best_label

            row = self.make_row(
                global_id=global_id,
                passage=text,
                company=company,
                year=year,
                language=language,
                hits_by_cat=hits,
            )
            rows.append(row)

        return rows

    # ---------- Public pipeline ----------

    def run(self) -> None:
        files = self._iter_files()
        if not files:
            print(f"[WARN] No files found under {self.root}")
            return

        total_rows = 0
        errors: List[str] = []

        for company, year, fpath in tqdm(files, desc=f"Filtering -> {self.table}", unit="file"):
            try:
                rows = self._rows_for_file(company, year, fpath)
                self._append_rows(rows)
                total_rows += len(rows)
            except Exception as e:
                errors.append(f"[ERROR] {company}/{year} ({fpath}): {e}")

        for line in errors:
            print(line)
        print(f"Done. Files: {len(files)} | Rows inserted: {total_rows} -> {self.table}")
        self._con.close()


# -------- CLI helper (used by subclasses) --------

def build_cli(parser_desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=parser_desc)
    ap.add_argument("--root", required=True, help="Root with data/jsons/<COMPANY>/<YEAR>/splits_semantic.json")
    ap.add_argument("--kw_en", required=True, help="Path to English keyword JSON")
    ap.add_argument("--kw_de", required=True, help="Path to German keyword JSON")
    ap.add_argument("--out_db", required=True, help="Output DuckDB file (e.g., data/dbs/xxx.duckdb)")
    ap.add_argument("--table", default=None, help="Override table name (optional)")
    ap.add_argument("--wildcard", action="store_true",
                    help="Treat '*' as wildcard: trailing '*' -> \\w*, internal '*' -> .* ; spaces -> \\s+.")
    ap.add_argument("--enumerate-hits", action="store_true",
                    help="Return exact matched tokens (slower). Default stores ['__hit__'] per hit category.")
    return ap
