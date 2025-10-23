#!/usr/bin/env python3
"""
Build a company-year CSV from two DuckDB tables:
- Aggregates duplicate rows and merges SDG + TECH counts
- Guarantees unique (company, year) rows (language collapsed to 'mixed' if needed)
- Counts hits as the length of the list (0 if empty/None)

Output: data/exports/company_year_hits.csv
"""

import os
import json
import duckdb
import pandas as pd

SDG_DB  = "data/dbs/sdg_hits.duckdb"
SDG_TBL = "sdg_hits"
TECH_DB = "data/dbs/tech_hits.duckdb"
TECH_TBL= "tech_hits"

OUT_DIR = "data/exports"
OUT_CSV = os.path.join(OUT_DIR, "company_year_hits.csv")


def _count_hits(v):
    """Return length of hits list robustly across types."""
    if v is None:
        return 0
    if isinstance(v, (list, tuple)):
        return len(v)
    if isinstance(v, str):
        s = v.strip()
        if not s or s.lower() == "none" or s == "[]":
            return 0
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return len(parsed)
            return 1  # non-list JSON (rare) → treat as one hit
        except Exception:
            return 1  # non-JSON, non-empty string → one hit
    return 0


def _normalize_keys(df):
    """Light normalization: strip whitespace and standardize types."""
    df["company"]  = df["company"].astype(str).str.strip()
    df["language"] = df["language"].astype(str).str.strip()
    # Coerce year to int if possible
    try:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    except Exception:
        pass
    return df


def load_sdg_df():
    con = duckdb.connect(SDG_DB, read_only=True)
    df = con.execute(f"SELECT * FROM {SDG_TBL}").fetchdf()
    con.close()

    sdg_cols = [f"hits_sdg{i}" for i in range(1, 18) if f"hits_sdg{i}" in df.columns]
    keep = ["company", "year", "language"] + sdg_cols
    df = df[keep].copy()
    df = _normalize_keys(df)

    for c in sdg_cols:
        df[c] = df[c].apply(_count_hits)

    # collapse duplicates at (company, year, language)
    df = df.groupby(["company", "year", "language"], as_index=False)[sdg_cols].sum()

    # Rename to SDG1..SDG17
    rename_map = {f"hits_sdg{i}": f"SDG{i}" for i in range(1, 18) if f"hits_sdg{i}" in df.columns}
    df = df.rename(columns=rename_map)
    return df


def load_tech_df():
    con = duckdb.connect(TECH_DB, read_only=True)
    df = con.execute(f"SELECT * FROM {TECH_TBL}").fetchdf()
    con.close()

    tech_cols = [c for c in df.columns if c.startswith("hits_")]
    keep = ["company", "year", "language"] + tech_cols
    df = df[keep].copy()
    df = _normalize_keys(df)

    for c in tech_cols:
        df[c] = df[c].apply(_count_hits)

    df = df.groupby(["company", "year", "language"], as_index=False)[tech_cols].sum()

    rename_map = {
        "hits_ai_ml": "AI_ML",
        "hits_cloud_computing": "Cloud_Computing",
        "hits_big_data_blockchain": "Big_Data_Blockchain",
        "hits_applications_practice": "Applications_Practice",
    }
    df = df.rename(columns=rename_map)
    # Ensure all expected tech cols exist even if absent
    for col in ["AI_ML", "Cloud_Computing", "Big_Data_Blockchain", "Applications_Practice"]:
        if col not in df.columns:
            df[col] = 0
    return df


def collapse_to_company_year(df, count_cols):
    """
    From (company, year, language) rows → one row per (company, year).
    - Sum counts across languages.
    - language column → 'mixed' if multiple, else that single language.
    """
    # Build a language marker per company-year
    langs = (
        df.groupby(["company", "year"])["language"]
          .agg(lambda s: list(pd.unique([x for x in s if isinstance(x, str) and x])) )
          .reset_index()
    )
    langs["language"] = langs["language"].apply(lambda L: L[0] if len(L) == 1 else "mixed")

    agg = df.groupby(["company", "year"], as_index=False)[count_cols].sum()
    out = pd.merge(agg, langs, on=["company", "year"], how="left")
    # Reorder to company, year, language, counts...
    out = out[["company", "year", "language"] + count_cols]
    return out


def main():
    sdg_df  = load_sdg_df()
    tech_df = load_tech_df()

    merged = pd.merge(
        sdg_df,
        tech_df,
        on=["company", "year", "language"],
        how="outer",
        validate="m:m"
    )

    # Fill missing counts with 0 and cast to int
    sdg_cols = [c for c in merged.columns if c.startswith("SDG")]
    tech_cols = ["AI_ML", "Cloud_Computing", "Big_Data_Blockchain", "Applications_Practice"]
    for col in sdg_cols + tech_cols:
        if col not in merged.columns:
            merged[col] = 0
    merged[sdg_cols + tech_cols] = merged[sdg_cols + tech_cols].fillna(0).astype(int)

    # Collapse to one row per (company, year)
    collapsed = collapse_to_company_year(merged, sdg_cols + tech_cols)

    # Order and save
    ordered_cols = (
        ["company", "year", "language"] +
        [f"SDG{i}" for i in range(1, 18)] +
        ["AI_ML", "Cloud_Computing", "Big_Data_Blockchain", "Applications_Practice"]
    )
    # Ensure all expected columns exist
    for col in ordered_cols:
        if col not in collapsed.columns:
            collapsed[col] = 0 if col not in ("company", "year", "language") else ""
    collapsed = collapsed[ordered_cols].sort_values(["company", "year"]).reset_index(drop=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    collapsed.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(collapsed):,} rows to {OUT_CSV}")

if __name__ == "__main__":
    main()
