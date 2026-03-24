"""
Exploration script for sdg_hits_classified.duckdb and tech_hits_classified.duckdb.

Each hits column stores a JSON object mapping matched regex patterns to their
classification: {"<pattern>": "symbolic"|"substantive", ...}
An empty dict "{}" means no keyword from that category was found in the passage.

Usage:
    python3 src/analysis/explore_dbs.py
"""

import json
import duckdb
import pandas as pd

DB_DIR = "data/dbs"
SDG_DB  = f"{DB_DIR}/sdg_hits.duckdb"
TECH_DB = f"{DB_DIR}/tech_hits.duckdb"

SDG_CATEGORIES  = [f"hits_sdg{i}" for i in range(1, 18)]
TECH_CATEGORIES = ["hits_ai_ml", "hits_cloud_computing", "hits_big_data_blockchain", "hits_applications_practice"]

SEP = "-" * 70


def parse_hits(cell: str) -> dict:
    """Parse a hits cell (JSON string) into a dict. Returns {} on failure."""
    try:
        return json.loads(cell) if cell else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def classification_counts(df: pd.DataFrame, hit_cols: list[str]) -> pd.DataFrame:
    """
    For each hits column, count symbolic and substantive classifications
    across all non-empty cells.
    Returns a DataFrame with columns: category, symbolic, substantive, total_hits, passages_with_hits.
    """
    rows = []
    for col in hit_cols:
        symbolic = substantive = total_hits = passages_with_hits = 0
        for cell in df[col]:
            hits = parse_hits(cell)
            if hits:
                passages_with_hits += 1
                for label in hits.values():
                    total_hits += 1
                    if label == "symbolic":
                        symbolic += 1
                    elif label == "substantive":
                        substantive += 1
        rows.append({
            "category":           col.removeprefix("hits_"),
            "symbolic":           symbolic,
            "substantive":        substantive,
            "total_hits":         total_hits,
            "passages_with_hits": passages_with_hits,
        })
    return pd.DataFrame(rows)


def explore(db_path: str, table: str, hit_cols: list[str]) -> None:
    print(SEP)
    print(f"DB:    {db_path}")
    print(f"Table: {table}")
    print(SEP)

    con = duckdb.connect(db_path, read_only=True)
    df  = con.execute(f"SELECT * FROM {table}").df()
    con.close()

    # ── Column names ──────────────────────────────────────────────────────────
    print(f"Columns: {list(df.columns)}")

    # ── Basic counts ──────────────────────────────────────────────────────────
    print(f"Total rows  : {len(df):,}")
    print(f"Companies   : {df['company'].nunique():,}")
    print(f"Years       : {sorted(df['year'].unique())}")
    print(f"Languages   : {df['language'].value_counts().to_dict()}")

    # ── Passages per company (top 10) ─────────────────────────────────────────
    print("\nTop 10 companies by passage count:")
    print(df["company"].value_counts().head(10).to_string())

    # ── Passages per year ─────────────────────────────────────────────────────
    print("\nPassages per year:")
    print(df["year"].value_counts().sort_index().to_string())

    # ── Classification breakdown per category ─────────────────────────────────
    print("\nClassification counts per category:")
    counts = classification_counts(df, hit_cols)
    print(counts.to_string(index=False))

    # ── Overall symbolic vs substantive totals ────────────────────────────────
    total_sym  = counts["symbolic"].sum()
    total_sub  = counts["substantive"].sum()
    total_hits = counts["total_hits"].sum()
    print(f"\nOverall — symbolic: {total_sym:,} | substantive: {total_sub:,} | total hits: {total_hits:,}")
    if total_hits:
        print(f"Substantive share: {total_sub / total_hits:.1%}")

    # ── Passages that have at least one hit ───────────────────────────────────
    has_any_hit = df[hit_cols].apply(
        lambda col: col.map(lambda c: bool(parse_hits(c)))
    ).any(axis=1)
    print(f"\nPassages with ≥1 hit : {has_any_hit.sum():,} / {len(df):,} ({has_any_hit.mean():.1%})")

    # ── Multi-category passages ───────────────────────────────────────────────
    hit_count_per_row = df[hit_cols].apply(
        lambda col: col.map(lambda c: bool(parse_hits(c)))
    ).sum(axis=1)
    multi = (hit_count_per_row > 1).sum()
    print(f"Passages with hits in >1 category: {multi:,}")

    # ── Sample passages ───────────────────────────────────────────────────────
    print("\nSample passages (first 2 rows with at least one hit):")
    sample = df[has_any_hit].head(2)
    for _, row in sample.iterrows():
        print(f"\n  global_id : {row['global_id']}")
        print(f"  company   : {row['company']}  year: {row['year']}  lang: {row['language']}")
        print(f"  passage   : {row['passage'][:200].strip().replace(chr(10), ' ')}...")
        for col in hit_cols:
            hits = parse_hits(row[col])
            if hits:
                print(f"  {col.removeprefix('hits_'):30s}: {hits}")

    print()


if __name__ == "__main__":
    explore(SDG_DB,  "sdg_hits_classified",  SDG_CATEGORIES)
    explore(TECH_DB, "tech_hits_classified", TECH_CATEGORIES)
