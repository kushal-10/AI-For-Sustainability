#!/usr/bin/env python3
"""
generate_sym_sub_results.py — Aggregate classified DuckDBs into company×year CSVs.

For each model config, reads:
  data/dbs/<model_config>/sdg_hits_classified.duckdb
  data/dbs/<model_config>/tech_hits_classified.duckdb

and writes:
  data/exports/<model_config>_sym_sub.csv

Each CSV row = one company+year combination with passage-level counts:
  sdg1_symbolic, sdg1_substantive, ..., sdg17_symbolic, sdg17_substantive,
  ai_ml_symbolic, ai_ml_substantive, cloud_computing_symbolic, ...

Counting rule: for each (column × label), count the number of passages where
that column contains at least one pattern classified with that label.
A passage with both symbolic and substantive patterns in the same column
increments both counters.

Usage:
    python3 src/postprocessing/generate_sym_sub_results.py
    python3 src/postprocessing/generate_sym_sub_results.py --model gpt-5.2__high__cot
    python3 src/postprocessing/generate_sym_sub_results.py --dbs-base data/backup_dbs --out data/exports/backup_sym_sub.csv
"""

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────

DBS_BASE    = "data/dbs"
EXPORTS_DIR = "data/exports"

SDG_TABLE  = "sdg_hits_classified"
TECH_TABLE = "tech_hits_classified"

# ── Column definitions ─────────────────────────────────────────────────────────

SDG_HIT_COLS  = [f"hits_sdg{i}" for i in range(1, 18)]   # hits_sdg1 … hits_sdg17
TECH_HIT_COLS = [
    "hits_ai_ml",
    "hits_cloud_computing",
    "hits_big_data_blockchain",
    "hits_applications_practice",
]

# Output column name stems (maps hit_col → csv stem)
SDG_STEMS = {f"hits_sdg{i}": f"sdg{i}" for i in range(1, 18)}
TECH_STEMS = {
    "hits_ai_ml":                   "ai_ml",
    "hits_cloud_computing":         "cloud_computing",
    "hits_big_data_blockchain":     "big_data_blockchain",
    "hits_applications_practice":   "applications_practice",
}

LABELS = ("symbolic", "substantive")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_hit(val) -> set[str]:
    """Parse a hit column value and return the set of labels present."""
    if not val or val in ("{}", "[]", "null"):
        return set()
    try:
        d = json.loads(val) if isinstance(val, str) else val
        if isinstance(d, dict):
            return {v for v in d.values() if isinstance(v, str)}
        if isinstance(d, list):
            return set()  # unclassified list format — no labels
    except Exception:
        pass
    return set()


def compute_counts(df: pd.DataFrame, hit_cols: list[str], stems: dict[str, str]) -> pd.DataFrame:
    """
    Add binary indicator columns for each (hit_col × label) pair, then
    group by company+year and sum.

    Returns a DataFrame indexed by (company, year) with count columns.
    """
    indicator_cols: list[str] = []

    for col in hit_cols:
        if col not in df.columns:
            continue
        stem = stems[col]
        for label in LABELS:
            out_col = f"{stem}_{label}"
            df[out_col] = df[col].apply(lambda v, lbl=label: int(lbl in _parse_hit(v)))
            indicator_cols.append(out_col)

    grouped = (
        df.groupby(["company", "year"], sort=True)[indicator_cols]
        .sum()
        .reset_index()
    )
    return grouped


def load_table(db_path: str, table: str) -> pd.DataFrame | None:
    if not Path(db_path).exists():
        return None
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute(f"SELECT * FROM {table}").fetchdf()
    con.close()
    return df


# ── Per-model builder ──────────────────────────────────────────────────────────

def build_for_model(
    model_id: str,
    sdg_db:   str,
    tech_db:  str,
    out_path: Path,
) -> bool:
    """
    Build the company×year CSV for one model config.
    Returns True on success.
    """
    print(f"── {model_id}")

    # ── SDG ───────────────────────────────────────────────────────────────────
    sdg_df = load_table(sdg_db, SDG_TABLE)
    if sdg_df is None:
        print(f"  [SKIP] SDG DB not found: {sdg_db}")
        sdg_counts = None
    else:
        print(f"  SDG:  {len(sdg_df):,} passages, "
              f"{sdg_df[['company','year']].drop_duplicates().shape[0]:,} company×year combos")
        sdg_counts = compute_counts(sdg_df, SDG_HIT_COLS, SDG_STEMS)

    # ── Tech ──────────────────────────────────────────────────────────────────
    tech_df = load_table(tech_db, TECH_TABLE)
    if tech_df is None:
        print(f"  [SKIP] Tech DB not found: {tech_db}")
        tech_counts = None
    else:
        print(f"  Tech: {len(tech_df):,} passages, "
              f"{tech_df[['company','year']].drop_duplicates().shape[0]:,} company×year combos")
        tech_counts = compute_counts(tech_df, TECH_HIT_COLS, TECH_STEMS)

    if sdg_counts is None and tech_counts is None:
        print(f"  [SKIP] No data available — skipping {model_id}\n")
        return False

    # ── Merge SDG + Tech on company×year ──────────────────────────────────────
    if sdg_counts is not None and tech_counts is not None:
        merged = pd.merge(sdg_counts, tech_counts, on=["company", "year"], how="outer")
    elif sdg_counts is not None:
        merged = sdg_counts
    else:
        merged = tech_counts

    # Fill missing counts with 0 (company×year present in one DB but not the other)
    count_cols = [c for c in merged.columns if c not in ("company", "year")]
    merged[count_cols] = merged[count_cols].fillna(0).astype(int)

    # ── Column order: SDG first (sdg1_sym, sdg1_sub, sdg2_sym, ...), then tech ─
    ordered_cols = ["company", "year"]
    for i in range(1, 18):
        for label in LABELS:
            col = f"sdg{i}_{label}"
            if col in merged.columns:
                ordered_cols.append(col)
    for stem in ("ai_ml", "cloud_computing", "big_data_blockchain", "applications_practice"):
        for label in LABELS:
            col = f"{stem}_{label}"
            if col in merged.columns:
                ordered_cols.append(col)

    merged = merged[ordered_cols].sort_values(["company", "year"]).reset_index(drop=True)

    # ── Write ──────────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"  [OK] {len(merged):,} rows × {len(merged.columns)} cols → {out_path}\n")
    return True


# ── Main ───────────────────────────────────────────────────────────────────────

def build_all(
    dbs_base:     str = DBS_BASE,
    exports_dir:  str = EXPORTS_DIR,
    filter_model: str | None = None,
) -> None:
    dbs_path = Path(dbs_base)

    # Each subdirectory of dbs_base that has at least one of the two DB files
    model_dirs = sorted(
        d for d in dbs_path.iterdir()
        if d.is_dir()
        and (
            (d / "sdg_hits_classified.duckdb").exists()
            or (d / "tech_hits_classified.duckdb").exists()
        )
    )

    if filter_model:
        model_dirs = [d for d in model_dirs if filter_model.lower() in d.name.lower()]

    if not model_dirs:
        print(f"No model config DB folders found under {dbs_base}")
        return

    print(f"Found {len(model_dirs)} model config(s): {[d.name for d in model_dirs]}\n")

    for model_dir in model_dirs:
        model_id = model_dir.name
        build_for_model(
            model_id = model_id,
            sdg_db   = str(model_dir / "sdg_hits_classified.duckdb"),
            tech_db  = str(model_dir / "tech_hits_classified.duckdb"),
            out_path = Path(exports_dir) / f"{model_id}_sym_sub.csv",
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate classified DuckDBs into per-model company×year CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--model",       default=None, metavar="MODEL_ID",
                    help="Scope to one model config, e.g. gpt-5.2__high__cot")
    ap.add_argument("--dbs-base",    default=DBS_BASE, metavar="PATH",
                    help=f"Base dir containing <model_id>/ subdirs (default: {DBS_BASE})")
    ap.add_argument("--exports-dir", default=EXPORTS_DIR, metavar="PATH",
                    help=f"Output directory for CSVs (default: {EXPORTS_DIR})")

    # Convenience: point directly at a single model's DB folder
    ap.add_argument("--sdg-db",  default=None, metavar="PATH",
                    help="Direct path to sdg_hits_classified.duckdb (overrides --dbs-base)")
    ap.add_argument("--tech-db", default=None, metavar="PATH",
                    help="Direct path to tech_hits_classified.duckdb (overrides --dbs-base)")
    ap.add_argument("--out",     default=None, metavar="PATH",
                    help="Direct output CSV path (used with --sdg-db / --tech-db)")
    args = ap.parse_args()

    if args.sdg_db or args.tech_db:
        # Single-DB mode
        model_id = args.model or "custom"
        out_path = Path(args.out) if args.out else Path(args.exports_dir) / f"{model_id}_sym_sub.csv"
        build_for_model(
            model_id = model_id,
            sdg_db   = args.sdg_db  or "",
            tech_db  = args.tech_db or "",
            out_path = out_path,
        )
    else:
        build_all(
            dbs_base     = args.dbs_base,
            exports_dir  = args.exports_dir,
            filter_model = args.model,
        )


if __name__ == "__main__":
    main()
