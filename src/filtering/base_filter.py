#!/usr/bin/env python3
"""
base_filter.py

Extensible filtering framework with optional wildcard support.

- Walks data/jsons/<COMPANY>/<YEAR>/splits_semantic.json
- Detects language from first 5 passages
- Matches EN / (optionally) EN+DE keyword sets
- Writes only *hit* passages to DuckDB
- Progress bar; only errors printed

Subclasses must implement:
  - category_names(self) -> List[str]
  - table_name(self) -> str
  - extra_columns(self) -> List[Tuple[col_name, duckdb_type]]
  - make_row(self, *, global_id, passage, company, year, language, hits_by_cat) -> Tuple[Any,...]
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional

import duckdb
from langdetect import detect, DetectorFactory
from tqdm import tqdm

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
    labels: List[str] = []
    for p in passages:
        t = (p or "").strip()
        if not t:
            continue
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
        * trailing '*' after an alnum stem -> '\\w*'
        * '*' elsewhere -> '.*'
        * spaces -> '\\s+'
      Otherwise, '*' is treated literally.
    - Add '\\b' at start/end when the visible ends are alphanumeric.
    """
    raw = token.strip()
    if not raw:
        return ""

    # Build escaped pattern character by character to control '*' and space behavior.
    pieces: List[str] = []
    for i, ch in enumerate(raw):
        if star_is_wildcard and ch == "*":
            # trailing '*' -> \w*, otherwise .* (looser, can bridge across)
            if i == len(raw) - 1:
                pieces.append(r"\w*")
            else:
                pieces.append(r".*")
        elif star_is_wildcard and ch == " ":
            pieces.append(r"\s+")
        else:
            # normal escape
            pieces.append(re.escape(ch))

    pat_body = "".join(pieces)

    # Word-boundaries if starts/ends with alnum (ignoring trailing '*')
    first_vis = next((c for c in raw if c != " "), raw[:1])
    last_vis = next((c for c in reversed(raw) if c != "*"), "")
    if _is_alnum(first_vis):
        pat_body = r"\b" + pat_body
    if _is_alnum(last_vis):
        pat_body = pat_body + r"\b"

    return pat_body

def _compile_patterns(keywords: Iterable[str], star_is_wildcard: bool = False) -> List[re.Pattern]:
    pats: List[re.Pattern] = []
    for kw in keywords:
        pat = _token_to_regex(kw, star_is_wildcard)
        if pat:
            pats.append(re.compile(pat, re.IGNORECASE))
    return pats

def _flatten_keyword_dict(dct: Dict[str, List[str]], star_is_wildcard: bool = False) -> Dict[str, List[re.Pattern]]:
    return {cat: _compile_patterns(words, star_is_wildcard) for cat, words in dct.items()}

def _find_hits(text: str, compiled_kw: Dict[str, List[re.Pattern]]) -> Dict[str, List[str]]:
    hits: Dict[str, List[str]] = {}
    for cat, pats in compiled_kw.items():
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
    ):
        self.root = Path(root_path)
        self.star_is_wildcard = star_is_wildcard

        self.kw_en_raw = _load_json(Path(kw_en_path))
        self.kw_de_raw = _load_json(Path(kw_de_path))

        self.compiled_en = _flatten_keyword_dict(self.kw_en_raw, self.star_is_wildcard)
        self.compiled_de = _flatten_keyword_dict(self.kw_de_raw, self.star_is_wildcard)

        self.out_db = Path(out_db)
        self._table_override = table
        self.use_de_when_detected = use_de_when_detected

        self._validate_categories()
        self._init_db()

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

    def _init_db(self) -> None:
        # Ensure folder exists (e.g., data/dbs/)
        self.out_db.parent.mkdir(parents=True, exist_ok=True)
        cols = self.BASE_COLUMNS + self.hit_columns + self.extra_columns()
        col_defs = ",\n                ".join(f"{name} {dtype}" for name, dtype in cols)
        con = duckdb.connect(self.out_db.as_posix())
        con.execute(f"CREATE TABLE IF NOT EXISTS {self.table} ({col_defs});")
        con.close()

    def _insert_rows(self, rows: List[Tuple[Any, ...]]) -> None:
        if not rows:
            return
        cols = self.BASE_COLUMNS + self.hit_columns + self.extra_columns()
        placeholders = ", ".join(["?"] * len(cols))
        con = duckdb.connect(self.out_db.as_posix())
        con.executemany(f"INSERT INTO {self.table} VALUES ({placeholders})", rows)
        con.close()

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

    def _merge_compiled(self, use_german: bool) -> Dict[str, List[re.Pattern]]:
        if use_german and self.use_de_when_detected:
            merged: Dict[str, List[re.Pattern]] = {}
            for c in self.category_names():
                merged[c] = self.compiled_en.get(c, []) + self.compiled_de.get(c, [])
            return merged
        else:
            return {c: self.compiled_en.get(c, []) for c in self.category_names()}

    # ---------- Public pipeline ----------

    def process_one(self, company: str, year: str, json_path: Path) -> int:
        data = _load_json(json_path)
        passages_first5 = _first_n_passages(data, 5)
        is_german, best_label = _detect_is_german(passages_first5)
        compiled = self._merge_compiled(use_german=is_german)

        company_clean = _clean_company_for_id(company)
        rows: List[Tuple[Any, ...]] = []

        # Iterate passages in numeric order
        for sid, passage in sorted(((int(k), v) for k, v in data.items()), key=lambda x: x[0]):
            text = (passage or "").strip()
            if not text:
                continue

            hits = _find_hits(text, compiled)
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

        self._insert_rows(rows)
        return len(rows)

    def run(self) -> None:
        files = self._iter_files()
        if not files:
            print(f"[WARN] No files found under {self.root}")
            return

        total_rows = 0
        errors: List[str] = []

        for company, year, fpath in tqdm(files, desc=f"Filtering -> {self.table}", unit="file"):
            try:
                inserted = self.process_one(company, year, fpath)
                total_rows += inserted
            except Exception as e:
                errors.append(f"[ERROR] {company}/{year} ({fpath}): {e}")

        for line in errors:
            print(line)
        print(f"Done. Files: {len(files)} | Rows inserted: {total_rows} -> {self.table}")


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
    return ap
