#!/usr/bin/env python3
import argparse, ast, json, re, sys, random
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from collections import Counter, defaultdict

try:
    import duckdb  # optional if using --from-duckdb
except Exception:
    duckdb = None

TEXTS_ROOT = Path("data/texts")  # data/texts/{COMPANY}/{YEAR}/splits.json

# ---------- helpers ----------
def extract_content_from_batch_line(obj: dict):
    """Return (custom_id, assistant content) if status_code==200, else (custom_id, None)."""
    try:
        cid = obj.get("custom_id")
        resp = obj.get("response", {})
        if resp.get("status_code") != 200:
            return cid, None
        body = resp.get("body", {})
        choices = body.get("choices") or []
        if not choices:
            return cid, None
        msg = choices[0].get("message", {})
        content = msg.get("content")
        if isinstance(content, str):
            return cid, content
        if isinstance(content, list) and content and isinstance(content[0], dict):
            t = content[0].get("text")
            if isinstance(t, str):
                return cid, t
        return cid, None
    except Exception:
        return obj.get("custom_id"), None

def parse_list_literal(text: str):
    """Parse '[1,12,False]' (or text containing it) into a Python list, else None."""
    if text is None:
        return None
    s = text.strip()
    try:
        return ast.literal_eval(s)
    except Exception:
        m = re.search(r"\[(?:.|\n)*\]", s)
        if not m:
            return None
        try:
            return ast.literal_eval(m.group(0))
        except Exception:
            return None

def validate_item(lst) -> Tuple[bool, str]:
    """
    Valid if:
      - list with >=2 items; last is boolean and the only boolean;
      - earlier items all ints in {1..17} or {0};
      - if 0 appears, it must be alone.
    """
    if not isinstance(lst, list):
        return False, "invalid list format"
    if len(lst) < 2:
        return False, "too few elements"

    tail = lst[-1]
    if not isinstance(tail, bool):
        return False, "last element not boolean"

    if any(isinstance(x, bool) for x in lst[:-1]):
        return False, "multiple boolean classifications"

    ints = []
    for x in lst[:-1]:
        if isinstance(x, bool):
            return False, "multiple boolean classifications"
        try:
            xi = int(x)
        except Exception:
            return False, "could not parse list literal"
        ints.append(xi)

    for xi in ints:
        if xi != 0 and not (1 <= xi <= 17):
            return False, f"SDG out of range: {xi}"

    if 0 in ints and len(ints) > 1:
        return False, "0 with other numbers"

    return True, "ok"

def parse_custom_id(custom_id: Optional[str]):
    """
    Expected: 'task||{sentence_id}||{COMPANY}||{YEAR}'
    Returns (sid, company, year) or (None,None,None).
    """
    if not isinstance(custom_id, str):
        return None, None, None
    parts = custom_id.split("||")
    if len(parts) < 4:
        return None, None, None
    _, sid, company, year = parts[:4]
    return sid, company, year

# cache splits.json files
_SPLITS_CACHE: Dict[Path, Dict[str, str]] = {}
def load_sentence(company: str, year: str, sentence_id: str) -> Optional[str]:
    """Load sentence text from data/texts/{company}/{year}/splits.json using string(sentence_id) key."""
    if not (company and year and sentence_id):
        return None
    p = TEXTS_ROOT / company / year / "splits.json"
    if not p.exists():
        return None
    try:
        data = _SPLITS_CACHE.get(p)
        if data is None:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            _SPLITS_CACHE[p] = data
        key = str(int(sentence_id))
        return data.get(key)
    except Exception:
        return None

# ---------- iterators ----------
def iter_from_duckdb(db_path: Path):
    if duckdb is None:
        raise RuntimeError("duckdb not installed. `pip install duckdb`")
    con = duckdb.connect(db_path.as_posix())
    try:
        for cid, raw in con.execute("SELECT custom_id, raw FROM results").fetchall():
            yield cid, raw
    finally:
        con.close()

def iter_from_batch_dir(batch_dir: Path):
    for path in sorted(batch_dir.glob("batch_*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                cid, content = extract_content_from_batch_line(obj)
                yield cid, content

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Validate and print invalids as 'Sentence||Customid||Reason||Output'.")
    ap.add_argument("--from-duckdb", type=str, default=None,
                    help="Path to DuckDB with table results(custom_id, raw).")
    ap.add_argument("--from-batch-dir", type=str, default=None,
                    help="Directory with batch_*.jsonl files.")
    ap.add_argument("--limit-first", type=int, default=5,
                    help="How many random items to print for '0 with other numbers'.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    args = ap.parse_args()

    if not args.from_duckdb and not args.from_batch_dir:
        print("Provide --from-duckdb or --from-batch-dir", file=sys.stderr)
        sys.exit(1)

    if args.from_duckdb:
        it = iter_from_duckdb(Path(args.from_duckdb))
        source = f"duckdb={args.from_duckdb}"
    else:
        it = iter_from_batch_dir(Path(args.from_batch_dir))
        source = f"batch_dir={args.from_batch_dir}"

    counts = Counter()
    buckets: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)  # reason -> [(sentence, custom_id, output)]

    for custom_id, content in it:
        lst = parse_list_literal(content)
        if lst is None:
            reason = "could not parse list literal"
        else:
            ok, reason = validate_item(lst)
            if ok:
                continue
        sid, comp, year = parse_custom_id(custom_id)
        sentence = load_sentence(comp, year, sid) or "<sentence not found>"
        buckets[reason].append((sentence, custom_id or "-", str(content)))
        counts[reason] += 1

    # Breakdown
    print("Invalid breakdown by reason:")
    for r, c in counts.most_common():
        print(f" - {r}: {c}")
    print()

    # Print in required format
    random.seed(args.seed)
    first_reason = "0 with other numbers"
    if first_reason in buckets and buckets[first_reason]:
        sample = buckets[first_reason]
        if len(sample) > args.limit_first:
            sample = random.sample(sample, args.limit_first)
        for sentence, cid, out in sample:
            print(f"{sentence}||{cid}||{first_reason}||{out}")

    for reason, rows in buckets.items():
        if reason == first_reason:
            continue
        for sentence, cid, out in rows:
            print(f"{sentence}||{cid}||{reason}||{out}")

if __name__ == "__main__":
    main()
