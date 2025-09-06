#!/usr/bin/env python3
import argparse, ast, json, re, sys
from pathlib import Path
import duckdb
import pandas as pd
import re

def parse_list_literal(text: str):
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

def classify_reason(lst):
    """
    Return reason string for invalid items (same labels as checker).
    If valid, return None.
    """
    ok, reason = validate_item(lst)
    return None if ok else reason

def validate_item(lst):
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


def salvage_list(text: str):
    """
    Repair common malformed outputs and return a Python list.
    Strictly operates on the FIRST [...] block in the string, discarding any extra text.
    Handles:
      - 'True/False'  -> 'False'
      - missing comma before False (e.g., '0 False]' -> '0, False]')
      - missing trailing boolean -> append ', False]'
    Returns list or None.
    """
    if not isinstance(text, str):
        return None

    s = text.strip()

    # 0) Work ONLY on the first [...] occurrence; drop trailing commentary
    m = re.search(r'\[.*?\]', s, flags=re.DOTALL)
    if not m:
        return None
    sub = m.group(0)

    # 1) Replace True/False with False (case-insensitive)
    sub = re.sub(r'(?i)\btrue\s*/\s*false\b', 'False', sub)

    # 2) Insert missing comma before 'False' at the end of the list
    #    e.g., "[8, 0 False]" -> "[8, 0, False]"
    sub = re.sub(r'(\d)\s+False\s*\]', r'\1, False]', sub)

    # 3) Ensure a trailing boolean exists; if not, append ", False]"
    if not re.search(r',\s*(True|False)\s*\]$', sub):
        # Only append if it's a bracketed list
        if re.fullmatch(r'\[\s*.*\s*\]', sub, flags=re.DOTALL):
            sub = sub[:-1].rstrip() + ', False]'

    # 4) Try to parse the fixed bracket-only string
    try:
        lst = ast.literal_eval(sub)
    except Exception:
        return None

    return lst if isinstance(lst, list) else None


def fix_record(raw: str):
    """
    Returns (new_raw:str|None, new_sdgs:list|None, new_ai:bool|None, changed:bool).
    Now attempts to salvage 'could not parse list literal' cases.
    """
    lst = parse_list_literal(raw)

    salvaged = False
    if lst is None:
        # Try to salvage malformed cases like '[13, True/False]', '[8, 0 False]', etc.
        lst = salvage_list(raw)
        if lst is None:
            # Leave as-is for truly unparseable text
            return None, None, None, False
        salvaged = True

    ok, reason = validate_item(lst)
    if ok and not salvaged:
        # Already valid & not salvaged → no change needed
        return None, None, None, False

    # Helper to stringify canonical form
    def canon(sdgs, ai_bool):
        # If sdgs empty → canonical '[0, False/True]'
        if not sdgs:
            return f"[0, {str(bool(ai_bool))}]"
        inside = ", ".join(str(x) for x in sdgs)
        return f"[{inside}, {str(bool(ai_bool))}]"

    # Extract ai & sdgs robustly
    core = lst[:]
    ai = None
    if len(core) >= 1 and isinstance(core[-1], bool):
        ai = core.pop()
    # Coerce core to ints best-effort
    sdgs = []
    for x in core:
        try:
            sdgs.append(int(x))
        except Exception:
            # leave for rule-based handling
            pass

    # Apply rules

    # Rule A) 0 with other numbers => [0, False]
    # (Also applies after salvage or any other path)
    if 0 in sdgs and len(sdgs) > 1:
        return "[0, False]", [], False, True

    # If reason was last element not boolean (from original validate), fix:
    if reason == "last element not boolean":
        # If last token was numeric, we already added it above as SDG; now force ai=False
        ai = False
        # After appending False, re-apply zero rule just in case (handled above already)

        return canon(
            [] if 0 in sdgs else sdgs,
            ai if ai is not None else False
        ), ([] if 0 in sdgs else [x for x in sdgs if 1 <= x <= 17]), (ai if ai is not None else False), True

    # If we got here due to "could not parse list literal" but salvage succeeded:
    if salvaged:
        # Ensure ai present
        if ai is None:
            ai = False
        # Re-apply zero rule
        if 0 in sdgs and len(sdgs) > 1:
            return "[0, False]", [], False, True
        # Clamp to 1..17, or [] if 0 present
        if 0 in sdgs:
            sdgs = []
        else:
            sdgs = [x for x in sdgs if 1 <= x <= 17]
        return canon(sdgs, ai), sdgs, ai, True

    # SDG out of range => [0, False]
    if isinstance(reason, str) and reason.startswith("SDG out of range"):
        return "[0, False]", [], False, True

    # Otherwise: no change
    return None, None, None, False

def main():
    ap = argparse.ArgumentParser(description="Fix invalid rows in DuckDB according to specified rules.")
    ap.add_argument("--db", default="data/outputs_merged/classifications.duckdb")
    args = ap.parse_args()

    db_path = Path(args.db)
    con = duckdb.connect(db_path.as_posix())

    rows = con.execute("SELECT custom_id, raw FROM results").fetchall()

    updates = []
    changed = 0
    for cid, raw in rows:
        new_raw, new_sdgs, new_ai, did = fix_record(raw)
        if did:
            # Compute sdgs array + is_ai
            if new_raw is None:
                continue
            # parse back to populate columns
            lst = parse_list_literal(new_raw)
            # safe defaults
            sdgs = []
            is_ai = False
            if isinstance(lst, list) and len(lst) >= 2:
                if isinstance(lst[-1], bool):
                    is_ai = lst[-1]
                # sdgs inferred: if 0 present -> no sdgs; else in 1..17
                acc = []
                for x in lst[:-1]:
                    try:
                        xi = int(x)
                        acc.append(xi)
                    except Exception:
                        pass
                if 0 in acc:
                    acc = []
                sdgs = [x for x in acc if 1 <= x <= 17]
            updates.append((new_raw, sdgs, is_ai, cid))
            changed += 1

    if updates:
        df = pd.DataFrame(updates, columns=["new_raw", "new_sdgs", "new_ai", "custom_id"])

        # Start a transaction, register DF, and update via join
        con.execute("BEGIN")
        con.register("upd", df)
        con.execute("""
            UPDATE results AS r
            SET raw   = u.new_raw,
                sdgs  = u.new_sdgs,
                is_ai = u.new_ai
            FROM upd AS u
            WHERE r.custom_id = u.custom_id;
        """)
        # optional: cleanup registration (not strictly required)
        con.unregister("upd")
        con.execute("COMMIT")
        con.execute("VACUUM")
    else:
        print("No rows to fix.")

    con.close()
    print(f"Fixed rows: {changed}")

if __name__ == "__main__":
    main()
