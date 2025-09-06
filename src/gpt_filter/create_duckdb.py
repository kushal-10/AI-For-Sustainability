#!/usr/bin/env python3
import json, ast, re, os
from pathlib import Path
from typing import List, Optional, Tuple

import duckdb
import pandas as pd

IN_DIR = Path("data/batch_outputs")                 # input batch outputs
OUT_DIR = Path("data/outputs_merged")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = OUT_DIR / "classifications.duckdb"

# ---------- parsing helpers ----------
def parse_classification(text: str) -> Tuple[List[int], Optional[bool]]:
    """
    Accepts content like "[1, 5, True]" or "[0, False]".
    Returns (sdgs, is_ai).
    """
    text = (text or "").strip()
    if not text:
        return [], None
    try:
        data = ast.literal_eval(text)
    except Exception:
        m = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if not m:
            return [], None
        try:
            data = ast.literal_eval(m.group(0))
        except Exception:
            return [], None

    if not isinstance(data, (list, tuple)):
        return [], None

    is_ai = None
    if len(data) >= 1 and isinstance(data[-1], bool):
        is_ai = data[-1]
        data = data[:-1]

    sdgs: List[int] = []
    for x in data:
        try:
            xi = int(x)
            sdgs.append(xi)
        except Exception:
            continue

    # 0 means "no SDG"
    if 0 in sdgs:
        sdgs = []

    # keep 1..17
    sdgs = [s for s in sdgs if 1 <= s <= 17]
    return sdgs, is_ai

def extract_content(obj: dict) -> Optional[str]:
    """
    From a batch output object, return assistant content if status_code == 200.
    """
    try:
        resp = obj.get("response", {})
        if resp.get("status_code") != 200:
            return None
        body = resp.get("body", {})
        choices = body.get("choices") or []
        if not choices:
            return None
        msg = choices[0].get("message", {})
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list) and content and isinstance(content[0], dict):
            t = content[0].get("text")
            if isinstance(t, str):
                return t
        return None
    except Exception:
        return None

# ---------- ETL ----------
def main():
    # --- open connection as before ---
    con = duckdb.connect(DB_PATH.as_posix())

    # Table with a list/array column (DuckDB syntax: INTEGER[])
    con.execute("""
    CREATE TABLE IF NOT EXISTS results (
        custom_id TEXT PRIMARY KEY,
        sdgs      INTEGER[],   -- list of ints
        is_ai     BOOLEAN,
        raw       TEXT,
        batch_id  TEXT
    );
    """)

    # Portable UNNEST usage
    con.execute("""
    CREATE OR REPLACE VIEW results_exploded AS
    SELECT
        r.custom_id,
        sdg,
        r.is_ai,
        r.raw,
        r.batch_id
    FROM results AS r, UNNEST(r.sdgs) AS sdg;
    """)

    buffer = []
    BATCH_SIZE = 50000  # big, but safe for 124k rows
    files = sorted(IN_DIR.glob("batch_*.jsonl"))
    if not files:
        print(f"No files found in {IN_DIR}")
        return

    # optional: speed boost
    con.execute("PRAGMA threads=4;")

    total_rows = 0
    for path in files:
        batch_id = path.stem  # e.g., batch_68bb...
        with path.open("r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                custom_id = obj.get("custom_id")
                if not custom_id:
                    continue

                content = extract_content(obj)
                if content is None:
                    continue

                sdgs, is_ai = parse_classification(content)
                buffer.append({
                    "custom_id": custom_id,
                    "sdgs": sdgs,
                    "is_ai": is_ai,
                    "raw": content,
                    "batch_id": batch_id,
                })

                if len(buffer) >= BATCH_SIZE:
                    df = pd.DataFrame(buffer)
                    con.register("buf", df)
                    con.execute("""
                        INSERT INTO results
                        SELECT * FROM buf;
                    """)
                    con.unregister("buf")
                    total_rows += len(buffer)
                    buffer.clear()
                    print(f"Inserted {total_rows} rows so far...")

    # final flush
    if buffer:
        df = pd.DataFrame(buffer)
        con.register("buf", df)
        con.execute("INSERT INTO results SELECT * FROM buf;")
        con.unregister("buf")
        total_rows += len(buffer)
        buffer.clear()

    # housekeeping: compact db
    con.execute("VACUUM;")
    con.close()
    print(f"Done. Inserted {total_rows} rows into {DB_PATH}")

if __name__ == "__main__":
    main()
