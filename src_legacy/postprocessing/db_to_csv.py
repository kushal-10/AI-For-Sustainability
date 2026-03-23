#!/usr/bin/env python3
"""
Export two DuckDB tables to CSV files.

Usage:
  python3 export_to_csv.py
"""

import duckdb
import pandas as pd
import os

# Paths and table names
SDG_DB  = "data/dbs/sdg_hits_classified.duckdb"
SDG_TBL = "sdg_hits_classified"
TECH_DB = "data/dbs/tech_hits_classified.duckdb"
TECH_TBL= "tech_hits_classified"

EXPORT_DIR = "data/exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

def export_table(db_path, table_name, output_name):
    con = duckdb.connect(db_path)
    out_path = os.path.join(EXPORT_DIR, f"{output_name}.csv")

    print(f"→ Exporting {table_name} from {db_path} to {out_path} ...")

    # Use DuckDB's built-in COPY for fast, memory-efficient export
    con.execute(f"""
        COPY (SELECT * FROM {table_name})
        TO '{out_path}' (HEADER, DELIMITER ',');
    """)

    con.close()
    print(f"✅ Done: {out_path}")

if __name__ == "__main__":
    export_table(SDG_DB, SDG_TBL, "sdg_hits_classified")
    export_table(TECH_DB, TECH_TBL, "tech_hits_classified")
