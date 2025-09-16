# src/preview/pretty_print_duckdb.py
import duckdb
import json
import re
import textwrap

# ===== Config =====
DUCKDB_PATH = "data/matches.duckdb"
TABLE_NAME  = "matched_sentences"
SAMPLE_SIZE = 20
WRAP_WIDTH  = 110
HIGHLIGHT   = True  # turn off if your terminal doesn't like ANSI colors

# Optional filters (set to None to ignore)
FILTER_COMPANY  = None   # e.g., "BASF SE"
FILTER_YEAR     = None   # e.g., "2021"
ONLY_WITH_SDG   = False  # show only rows where sdg_keywords not empty
ONLY_WITH_AI    = False  # show only rows where ai_keywords not empty

# ===== ANSI colors =====
ANSI_RESET = "\033[0m"
ANSI_DIM   = "\033[2m"
ANSI_BOLD  = "\033[1m"
ANSI_SDG   = "\033[32m"  # green
ANSI_AI    = "\033[36m"  # cyan

def parse_hits(hits_json):
    try:
        arr = json.loads(hits_json) if hits_json else []
        if not isinstance(arr, list):
            arr = [str(arr)]
        return [str(x) for x in arr]
    except Exception:
        return []

def build_term_regex(term: str) -> re.Pattern:
    """
    Build a robust, case-insensitive regex for a keyword/phrase:
      - '*' becomes '\w*' (suffix wildcard)
      - spaces become '[\\s-]+' (space(s) or hyphen)
      - word boundaries when appropriate
    """
    t = term.strip()
    if not t:
        return re.compile(r"(?!x)x", re.I)
    # escape everything but '*' first
    esc = []
    for ch in t:
        esc.append("*" if ch == "*" else re.escape(ch))
    t = "".join(esc)
    t = t.replace("*", r"\w*")
    t = re.sub(r"\s+", lambda _: r"[\s-]+", t)

    # add word boundaries when helpful
    start = r"\b" if re.match(r"^\w", t, flags=re.U) else ""
    end   = r"\b" if re.search(r"\w$", t, flags=re.U) else ""
    patt  = f"{start}{t}{end}"
    return re.compile(patt, flags=re.IGNORECASE | re.UNICODE)

def highlight_sentence(sentence: str, sdg_terms, ai_terms) -> str:
    if not HIGHLIGHT or (not sdg_terms and not ai_terms):
        return sentence

    # Compile patterns (longer first to avoid nested/partial highlights)
    sdg_patts = sorted({build_term_regex(t) for t in sdg_terms}, key=lambda p: -len(p.pattern))
    ai_patts  = sorted({build_term_regex(t) for t in ai_terms},  key=lambda p: -len(p.pattern))

    out = sentence

    # Apply AI first, then SDG (arbitrary but consistent)
    for rx in ai_patts:
        out = rx.sub(lambda m: f"{ANSI_AI}{m.group(0)}{ANSI_RESET}", out)
    for rx in sdg_patts:
        out = rx.sub(lambda m: f"{ANSI_SDG}{m.group(0)}{ANSI_RESET}", out)

    return out

def fmt_hits(arr, label, color):
    if not arr:
        return f"{ANSI_DIM}{label}: —{ANSI_RESET}"
    shown = ", ".join(arr[:6])
    more  = f" … +{len(arr)-6}" if len(arr) > 6 else ""
    return f"{color}{label}: {shown}{more}{ANSI_RESET}"

def main():
    con = duckdb.connect(DUCKDB_PATH)
    where = []
    params = []

    if FILTER_COMPANY:
        where.append("company = ?")
        params.append(FILTER_COMPANY)
    if FILTER_YEAR:
        where.append("year = ?")
        params.append(FILTER_YEAR)
    if ONLY_WITH_SDG:
        where.append("sdg_keywords IS NOT NULL AND sdg_keywords <> 'null' AND sdg_keywords <> '[]'")
    if ONLY_WITH_AI:
        where.append("ai_keywords IS NOT NULL AND ai_keywords <> 'null' AND ai_keywords <> '[]'")

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    sql = f"""
        SELECT company, year, sentence_id, sentence, sdg_keywords, ai_keywords
        FROM {TABLE_NAME}
        {where_sql}
        ORDER BY random()
        LIMIT {SAMPLE_SIZE}
    """
    rows = con.execute(sql, params).fetchall()

    total = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
    con.close()

    if not rows:
        print("[INFO] No rows matched the filters.")
        return

    print(f"\n{ANSI_BOLD}Showing {len(rows)} example(s){ANSI_RESET} (table has {total} total rows)\n" + "-"*90)

    for i, (company, year, sid, sentence, sdg_json, ai_json) in enumerate(rows, 1):
        sdg_hits = parse_hits(sdg_json)
        ai_hits  = parse_hits(ai_json)

        header = f"#{i}  {ANSI_BOLD}{company}{ANSI_RESET}  (Year: {year}, ID: {sid})"
        tags   = f"{fmt_hits(sdg_hits, 'SDG hits', ANSI_SDG)}    {fmt_hits(ai_hits, 'AI hits', ANSI_AI)}"

        sent = sentence or ""
        sent = re.sub(r"\s+", " ", sent).strip()
        sent_h = highlight_sentence(sent, sdg_hits, ai_hits)

        print(header)
        print(tags)
        print("-"*90)
        for line in textwrap.wrap(sent_h, width=WRAP_WIDTH):
            print(line)
        print()

    print("-"*90)
    print("Tip: tweak FILTER_* at the top (company/year) or ONLY_WITH_SDG / ONLY_WITH_AI. "
          "Set HIGHLIGHT=False to disable colors.")

if __name__ == "__main__":
    main()
