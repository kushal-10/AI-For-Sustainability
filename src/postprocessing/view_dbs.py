#!/usr/bin/env python3
# src/batching/peek_classified_dbs.py
import duckdb
import argparse
import pandas as pd

SDG_DB  = "data/dbs/sdg_hits_classified.duckdb"
SDG_TBL = "sdg_hits_classified"
TECH_DB = "data/dbs/tech_hits_classified.duckdb"
TECH_TBL= "tech_hits_classified"

def peek(db_path: str, table: str, n: int = 5):
    con = duckdb.connect(db_path, read_only=True)
    total = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    head_df = con.execute(f"SELECT * FROM {table} LIMIT {n}").fetchdf()
    tail_df = con.execute(
        f"""
        SELECT * FROM {table}
        OFFSET (SELECT CASE WHEN COUNT(*) > {n} THEN COUNT(*) - {n} ELSE 0 END FROM {table})
        """
    ).fetchdf()

    print(f"\n=== {db_path}:{table} ===")
    print(f"Rows: {total} | Columns: {len(head_df.columns)} -> {list(head_df.columns)}")

    print(f"\n-- HEAD ({n}) --")
    print(head_df.to_string(max_rows=n, max_cols=0))

    print(f"\n-- TAIL ({n}) --")
    print(tail_df.to_string(max_rows=n, max_cols=0))

def main():
    ap = argparse.ArgumentParser(description="Print head & tail of classified DuckDB tables.")
    ap.add_argument("--n", type=int, default=5, help="Rows to show for head/tail (default: 5)")
    ap.add_argument("--skip-sdg", action="store_true", help="Skip SDG table")
    ap.add_argument("--skip-tech", action="store_true", help="Skip TECH table")
    args = ap.parse_args()

    if not args.skip_sdg:
        peek(SDG_DB, SDG_TBL, args.n)
    if not args.skip_tech:
        peek(TECH_DB, TECH_TBL, args.n)

if __name__ == "__main__":
    main()
