#!/usr/bin/env python3
import duckdb
import pandas as pd
import json
from collections import Counter

TECH_DB  = "data/dbs/tech_hits_classified.duckdb"
TECH_TBL = "tech_hits_classified"

def get_df(db_path, table):
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute(f"SELECT * FROM {table}").fetchdf()
    con.close()
    return df

def check_unclassified_with_passages(df, name):
    print(f"\n=== Checking {name} ===")
    hit_cols = [c for c in df.columns if c.lower().startswith("hits")]
    if not hit_cols:
        print("No hit columns detected.")
        return

    label_counter = Counter()
    total_items = 0
    bad_rows = []

    for _, row in df.iterrows():
        for col in hit_cols:
            cell = row[col]
            if pd.isna(cell):
                continue

            # Parse if JSON or dict
            if isinstance(cell, str) and cell.strip().startswith("{"):
                try:
                    cell = json.loads(cell)
                except Exception:
                    continue
            elif not isinstance(cell, dict):
                continue

            for label in cell.values():
                total_items += 1
                lbl = str(label).strip().lower()
                if lbl not in ("substantive", "symbolic"):
                    label_counter[lbl] += 1
                    bad_rows.append({
                        "column": col,
                        "label": lbl,
                        "company": row.get("company"),
                        "year": row.get("year"),
                        "passage": row.get("passage", "")[:300].replace("\n", " ")
                    })

    print(f"Total classified items: {total_items}")
    print(f"Non-substantive/symbolic labels found: {sum(label_counter.values())}")
    print("Labels found instead:")
    for lbl, count in label_counter.most_common():
        print(f"  {lbl!r}: {count}")

    if bad_rows:
        print("\n=== Example passages with non-substantive/symbolic labels ===")
        for r in bad_rows[:30]:  # limit to 30 examples for readability
            print(f"[{r['company']}, {r['year']}] ({r['column']}) -> {r['label']}")
            print(f"  {r['passage']}\n")

def main():
    df_tech = get_df(TECH_DB, TECH_TBL)
    check_unclassified_with_passages(df_tech, "TECH")

if __name__ == "__main__":
    main()
