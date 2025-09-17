#!/usr/bin/env python3
"""
extract_sdg_keywords.py

- Reads:
    1) All .RData files under --root (default: kw_data/)
    2) Pre-exported JSONs under --json-dir (default: kw_data/_export_tmp/sdsn_queries)

- For each row (must include at least 'sdg' and one of 'query' or 'keyword'):
    • Collect `keyword` column values (if present)
    • Parse/expand `query` strings into keywords/phrases:
        - Handles quotes "…", angle <…>, braces {…}, parentheses (…)
        - Boolean AND/OR, proximity W/n (treated as AND)
        - Wildcards '*'/'%' kept ( '%' → '*' , '_' → '?' )
        - Generates combinations for AND of OR-groups (e.g., A AND (B OR C) → A B, A C)
    • Map SDG-XX → sdgN
    • Deduplicate

- Writes:
    data/dbs/sdg_keywords.json
    (optional) data/dbs/sdg_keywords.csv

Usage:
  python3 src/filtering/extract_sdg_keywords.py \
      --root kw_data \
      --json-dir kw_data/_export_tmp/sdsn_queries \
      --out data/dbs/sdg_keywords.json \
      --csv data/dbs/sdg_keywords.csv
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Set
from itertools import product

import pandas as pd

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="kw_data", help="Folder to scan for .RData files")
    ap.add_argument("--json-dir", type=str, default="kw_data/_export_tmp/sdsn_queries", help="Folder of pre-exported JSONs (e.g., sdsn_queries.json)")
    ap.add_argument("--out", type=str, default="data/dbs/sdg_keywords.json", help="Output JSON path")
    ap.add_argument("--csv", type=str, default=None, help="Optional CSV output (sdg, keyword)")
    ap.add_argument("--max_expand", type=int, default=2000, help="Max combinations per query to avoid explosion")
    return ap.parse_args()

# ---------------- Utils ----------------
SDG_DIRECT_RE = re.compile(r"\bsdg[-\s_]?([0-9]{1,2})\b", re.I)

def sdg_key(s: str) -> str:
    """Map 'SDG-01' → 'sdg1'."""
    if not isinstance(s, str):
        return "misc"
    m = SDG_DIRECT_RE.search(s)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 17:
            return f"sdg{n}"
    return "misc"

def percent_to_star(s: str) -> str:
    return s.replace("%", "*").replace("_", "?")

def normalize_phrase(s: str) -> str:
    s = s.strip().strip('"').strip("'").strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def wrap_norm(s: str) -> str:
    """Normalize wrappers: {…} and <…> → quotes, W/n → AND."""
    if not isinstance(s, str):
        return ""
    # Normalize wrappers to quotes
    s = re.sub(r"\{([^{}]*)\}", r'"\1"', s)
    s = re.sub(r"<([^<>]*)>", r'"\1"', s)
    # Proximity W/n -> treat as AND
    s = re.sub(r"\bW/\d+\b", " AND ", s, flags=re.IGNORECASE)
    # Normalize SQL wildcards to our standard
    s = percent_to_star(s)
    # Collapse whitespace around operators
    s = re.sub(r"\s+", " ", s).strip()
    return s

def strip_outer_parens(s: str) -> str:
    s = s.strip()
    while s.startswith("(") and s.endswith(")"):
        depth = 0
        balanced = True
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    balanced = False
                    break
        if balanced:
            s = s[1:-1].strip()
        else:
            break
    return s

def split_top_level(s: str, sep: str) -> List[str]:
    """Split by top-level ' sep ' ignoring parentheses and quotes."""
    out, buf = [], []
    depth = 0
    in_quote = None
    i, n = 0, len(s)
    sep_len = len(sep)
    while i < n:
        ch = s[i]
        if in_quote:
            buf.append(ch)
            if ch == in_quote:
                in_quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            in_quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch == "(":
            depth += 1
            buf.append(ch)
            i += 1
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
            i += 1
            continue
        # try separator match only at depth 0 and not inside quotes
        if depth == 0 and s[i:i+sep_len].lower() == sep.lower():
            part = "".join(buf).strip()
            if part:
                out.append(part)
            buf = []
            i += sep_len
            continue
        buf.append(ch)
        i += 1
    part = "".join(buf).strip()
    if part:
        out.append(part)
    return out

def parse_expr_to_phrases(s: str, max_expand: int) -> List[str]:
    """Recursively parse boolean expression into phrases (cartesian for AND, union for OR)."""
    s = wrap_norm(s)
    s = strip_outer_parens(s)
    if not s:
        return []

    # OR at top level → union
    ors = split_top_level(s, " OR ")
    if len(ors) > 1:
        phrases: List[str] = []
        for part in ors:
            phrases.extend(parse_expr_to_phrases(part, max_expand))
        # dedupe preserve order
        seen, out = set(), []
        for p in phrases:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    # AND at top level → cartesian product
    ands = split_top_level(s, " AND ")
    if len(ands) > 1:
        groups: List[List[str]] = []
        for part in ands:
            g = parse_expr_to_phrases(part, max_expand)
            if not g:
                # if no phrases came out, fall back to tokens from the segment
                g = fallback_tokens(part)
            groups.append(g)
        # guard against explosion
        total = 1
        for g in groups:
            total *= max(1, len(g))
            if total > max_expand:
                # Fallback: flatten all unique terms instead of cartesian
                uniq = []
                seen = set()
                for gg in groups:
                    for t in gg:
                        if t not in seen:
                            seen.add(t)
                            uniq.append(t)
                return uniq
        combos: List[str] = []
        for tup in product(*groups):
            combos.append(normalize_phrase(" ".join(tup)))
        # dedupe
        seen, out = set(), []
        for p in combos:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    # No top-level AND/OR → base case
    t = s.strip()
    t = strip_outer_parens(t)
    # quoted phrase?
    m = re.match(r'^["\'](.*)["\']$', t)
    if m:
        return [normalize_phrase(m.group(1))]
    # still has inner ORs? split them
    inner_ors = split_top_level(t, " OR ")
    if len(inner_ors) > 1:
        out: List[str] = []
        for part in inner_ors:
            out.extend(parse_expr_to_phrases(part, max_expand))
        # dedupe
        seen, uniq = set(), []
        for p in out:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return uniq
    # otherwise, return tokens fallback as phrase(s)
    toks = fallback_tokens(t)
    return [normalize_phrase(x) for x in toks] if toks else []

def fallback_tokens(s: str) -> List[str]:
    """Extract simple tokens preserving wildcards; drop boolean noise."""
    s = wrap_norm(s)
    # strip parentheses but keep inside
    s = s.replace("(", " ").replace(")", " ")
    # remove commas/semicolons etc
    s = re.sub(r"[.,;:|`]+", " ", s)
    # remove boolean operators
    s_low = re.sub(r"\b(AND|OR|NOT)\b", " ", s, flags=re.IGNORECASE)
    # tokens: allow letters, digits, +, -, *, ?, and internal hyphens
    toks = re.findall(r"[A-Za-z0-9+\-*?]+", s_low)
    # filter tiny/meaningless tokens
    toks = [t.lower() for t in toks if t and t.lower() not in {"and","or","not"} and len(t) > 1]
    # dedupe preserving order
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def rows_from_json_file(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # dict-of-lists or list-of-dicts → DataFrame
    if isinstance(obj, dict) and obj and all(isinstance(v, list) for v in obj.values()):
        return pd.DataFrame(obj)
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    # unknown → empty
    return pd.DataFrame()

def find_rdata_files(root: str) -> List[Path]:
    out: List[Path] = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.endswith(".RData"):
                out.append(Path(dp) / fn)
    return sorted(out)

def read_rdata_tables(p: Path) -> List[pd.DataFrame]:
    """Load .RData and return any pandas DataFrames found."""
    tables: List[pd.DataFrame] = []
    # pyreadr
    try:
        import pyreadr  # type: ignore
        try:
            res = pyreadr.read_r(str(p))
            for _, obj in res.items():
                if isinstance(obj, pd.DataFrame):
                    tables.append(obj)
        except Exception:
            pass
    except Exception:
        pass
    # rdata fallback (best-effort to DataFrame)
    if not tables:
        try:
            import rdata  # type: ignore
            try:
                parsed = rdata.parser.parse_file(str(p))
                converted = rdata.conversion.convert(parsed)
                if isinstance(converted, dict):
                    for _, v in converted.items():
                        try:
                            df = pd.DataFrame(v)
                            if not df.empty:
                                tables.append(df)
                        except Exception:
                            continue
                else:
                    try:
                        df = pd.DataFrame(converted)
                        if not df.empty:
                            tables.append(df)
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            pass
    return tables

# ---------------- Core ----------------
def collect_keywords_from_df(df: pd.DataFrame, max_expand: int) -> Dict[str, Set[str]]:
    """
    From a DataFrame with columns including:
      - 'sdg' (e.g., 'SDG-01')
      - 'query' (string)
      - optional: 'keyword', 'extra'
    Build {sdgN: set(keywords)}.
    """
    out: Dict[str, Set[str]] = {}
    cols = {c.lower(): c for c in df.columns}
    c_sdg = cols.get("sdg")
    if not c_sdg:
        return out

    # identify other possible columns
    c_query = cols.get("query")
    c_keyword = cols.get("keyword")
    c_extra = cols.get("extra")

    for _, row in df.iterrows():
        sdg = sdg_key(str(row[c_sdg]))
        out.setdefault(sdg, set())

        # 1) keyword column (if present)
        if c_keyword and pd.notna(row[c_keyword]):
            kw = normalize_phrase(str(row[c_keyword]))
            if kw:
                out[sdg].add(kw)

        # 2) extra column may contain OR lists or quoted phrases
        if c_extra and pd.notna(row[c_extra]):
            extra = str(row[c_extra])
            for ph in parse_expr_to_phrases(extra, max_expand):
                if ph:
                    out[sdg].add(ph)

        # 3) query column
        if c_query and pd.notna(row[c_query]):
            q = str(row[c_query])
            for ph in parse_expr_to_phrases(q, max_expand):
                if ph:
                    out[sdg].add(ph)

    return out

def merge_buckets(dst: Dict[str, Set[str]], src: Dict[str, Set[str]]):
    for k, vs in src.items():
        dst.setdefault(k, set()).update(vs)

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Final buckets
    buckets: Dict[str, Set[str]] = {}

    # 1) Pre-exported JSONs (e.g., sdsn_queries.json)
    json_dir = Path(args.json_dir)
    if json_dir.exists():
        for p in sorted(json_dir.glob("*.json")):
            if p.name.lower() in {"args.json", "infile.json", "outdir.json", "safelabel.json", "to_utf8.json"}:
                continue  # skip meta
            df = rows_from_json_file(p)
            if not df.empty:
                kb = collect_keywords_from_df(df, args.max_expand)
                merge_buckets(buckets, kb)

    # 2) .RData files
    for p in find_rdata_files(args.root):
        for df in read_rdata_tables(p):
            kb = collect_keywords_from_df(df, args.max_expand)
            merge_buckets(buckets, kb)

    # Sort & dump
    out_json = {k: sorted(v) for k, v in buckets.items() if v}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote {args.out}")
    for k in sorted(out_json.keys(), key=lambda x: (x != 'misc', int(x[3:]) if x.startswith('sdg') else 99)):
        print(f"  {k}: {len(out_json[k])} keywords")

    # Optional CSV
    if args.csv:
        rows = [(sdg, kw) for sdg, kws in out_json.items() for kw in kws]
        pd.DataFrame(rows, columns=["sdg", "keyword"]).to_csv(args.csv, index=False)
        print(f"[OK] Wrote {args.csv}")

if __name__ == "__main__":
    main()


"""
python3 src/filtering/scan_rdata.py \
  --root kw_data \
  --json-dir kw_data/_export_tmp/sdsn_queries \
  --out data/dbs/sdg_keywords.json \
  --csv data/dbs/sdg_keywords.csv

"""