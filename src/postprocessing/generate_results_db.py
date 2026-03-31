#!/usr/bin/env python3
"""
generate_results_db.py — Convert classification results JSON → DuckDB.

For each model config folder under RESULTS_BASE, builds two DuckDB files:
  data/dbs/<model_config>/sdg_hits_classified.duckdb
  data/dbs/<model_config>/tech_hits_classified.duckdb

Each DB has the same schema as the source DB (global_id, passage, company, year,
language, hit columns), but hit columns are updated with new labels from results.

Hit column format (classified):
  {"<pattern>": "symbolic" | "substantive", ...}
  Empty dict {} means no match for that column.

Usage:
    python3 src/classifications/generate_results_db.py
    python3 src/classifications/generate_results_db.py --model gpt-4o__tot
    python3 src/classifications/generate_results_db.py --dry-run
"""

import argparse
import json
from pathlib import Path

import duckdb

# ── Paths ──────────────────────────────────────────────────────────────────────

RESULTS_BASE = "data/classifications/results"
SDG_SOURCE   = "data/dbs/sdg_hits.duckdb"
TECH_SOURCE  = "data/dbs/tech_hits.duckdb"
OUT_BASE     = "data/dbs"

SDG_SOURCE_TABLE  = "sdg_hits_classified"
TECH_SOURCE_TABLE = "tech_hits_classified"

# ── Domain → column mappings ───────────────────────────────────────────────────

SDG_DOMAINS: dict[str, list[str]] = {
    "sdg_a": [f"hits_sdg{i}" for i in range(1, 10)],   # sdg1–sdg9
    "sdg_b": [f"hits_sdg{i}" for i in range(10, 14)],  # sdg10–sdg13
    "sdg_c": [f"hits_sdg{i}" for i in range(14, 18)],  # sdg14–sdg17
}
SDG_HIT_COLS = [col for cols in SDG_DOMAINS.values() for col in cols]

TECH_DOMAINS: dict[str, list[str]] = {
    "tech": [
        "hits_ai_ml",
        "hits_cloud_computing",
        "hits_big_data_blockchain",
        "hits_applications_practice",
    ]
}
TECH_HIT_COLS = TECH_DOMAINS["tech"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_hit_dict(val) -> dict:
    """Parse a hit column value (JSON string or dict) into a plain dict."""
    if val is None:
        return {}
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        s = val.strip()
        if s in ("", "[]", "null"):
            return {}
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                # pre-classification list format: ["pattern1", "pattern2"]
                return {p: "" for p in parsed if isinstance(p, str)}
        except Exception:
            pass
    return {}


def load_results(results_dir: Path, domains: list[str]) -> dict[str, dict[str, str]]:
    """
    Load result JSONs for the given domains and merge into a single lookup:
      { global_id: { pattern: label } }
    Keys in the result files are "<domain>||<global_id>".
    """
    combined: dict[str, dict[str, str]] = {}
    for domain in domains:
        path = results_dir / f"{domain}_results.json"
        if not path.exists():
            print(f"  [WARN] Missing results file: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            data: dict = json.load(f)
        n_before = sum(len(v) for v in combined.values())
        for key, labels in data.items():
            global_id = key.split("||", 1)[1] if "||" in key else key
            if global_id not in combined:
                combined[global_id] = {}
            combined[global_id].update(labels)
        n_after = sum(len(v) for v in combined.values())
        print(f"  [LOAD] {path.name}: {len(data):,} rows → +{n_after - n_before:,} pattern labels")
    return combined


def apply_results(
    source_db: str,
    source_table: str,
    results_lookup: dict[str, dict[str, str]],
    hit_cols: list[str],
    out_path: Path,
    dry_run: bool = False,
) -> dict:
    """
    Load source DB, replace hit column labels with new results, write to out_path.
    Returns stats dict.
    """
    con = duckdb.connect(source_db, read_only=True)
    df = con.execute(f"SELECT * FROM {source_table}").fetchdf()
    con.close()

    total_patterns = 0
    matched_patterns = 0
    missing_rows: list[str] = []

    for col in hit_cols:
        if col not in df.columns:
            continue
        new_vals: list[str] = []
        for _, row in df.iterrows():
            global_id = str(row["global_id"])
            old_dict  = _parse_hit_dict(row[col])
            patterns  = list(old_dict.keys())

            if not patterns:
                new_vals.append("{}")
                continue

            row_results = results_lookup.get(global_id, {})
            if not row_results and global_id not in results_lookup:
                missing_rows.append(global_id)

            new_dict: dict[str, str] = {}
            for pattern in patterns:
                total_patterns += 1
                label = row_results.get(pattern)
                if label:
                    new_dict[pattern] = label
                    matched_patterns += 1
                # Patterns with no result are omitted (left unclassified)

            new_vals.append(json.dumps(new_dict, ensure_ascii=False))

        df[col] = new_vals

    stats = {
        "rows": len(df),
        "total_patterns": total_patterns,
        "matched_patterns": matched_patterns,
        "coverage_pct": round(100 * matched_patterns / total_patterns, 1) if total_patterns else 0.0,
        "missing_result_rows": len(set(missing_rows)),
    }

    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_con = duckdb.connect(str(out_path))
        out_con.execute(f"CREATE OR REPLACE TABLE {source_table} AS SELECT * FROM df")
        out_con.close()
        print(f"  [WRITE] {out_path}  ({stats['rows']:,} rows, "
              f"{stats['matched_patterns']:,}/{stats['total_patterns']:,} patterns classified "
              f"[{stats['coverage_pct']}%])")
    else:
        print(f"  [DRY-RUN] Would write {out_path}  ({stats['rows']:,} rows, "
              f"{stats['matched_patterns']:,}/{stats['total_patterns']:,} patterns classified "
              f"[{stats['coverage_pct']}%])")

    if stats["missing_result_rows"]:
        print(f"  [WARN] {stats['missing_result_rows']:,} rows had no results at all")

    return stats


# ── Main ───────────────────────────────────────────────────────────────────────

def build_dbs(
    results_base: str = RESULTS_BASE,
    sdg_source: str   = SDG_SOURCE,
    tech_source: str  = TECH_SOURCE,
    out_base: str     = OUT_BASE,
    filter_model: str | None = None,
    dry_run: bool     = False,
) -> None:
    results_dir = Path(results_base)
    model_dirs  = sorted(d for d in results_dir.iterdir() if d.is_dir())

    if not model_dirs:
        print(f"No model config folders found in {results_base}")
        return

    if filter_model:
        model_dirs = [d for d in model_dirs if filter_model.lower() in d.name.lower()]
        if not model_dirs:
            print(f"No model config matched filter '{filter_model}'")
            return

    print(f"Found {len(model_dirs)} model config(s): {[d.name for d in model_dirs]}\n")

    for model_dir in model_dirs:
        model_id = model_dir.name
        print(f"── {model_id} ──────────────────────────────")

        # ── SDG ───────────────────────────────────────────────────────────────
        print("  Loading SDG results (sdg_a, sdg_b, sdg_c)...")
        sdg_lookup = load_results(model_dir, list(SDG_DOMAINS.keys()))
        if sdg_lookup:
            out_sdg = Path(out_base) / model_id / "sdg_hits_classified.duckdb"
            apply_results(
                source_db      = sdg_source,
                source_table   = SDG_SOURCE_TABLE,
                results_lookup = sdg_lookup,
                hit_cols       = SDG_HIT_COLS,
                out_path       = out_sdg,
                dry_run        = dry_run,
            )
        else:
            print("  [SKIP] No SDG results found")

        # ── Tech ──────────────────────────────────────────────────────────────
        print("  Loading Tech results...")
        tech_lookup = load_results(model_dir, list(TECH_DOMAINS.keys()))
        if tech_lookup:
            out_tech = Path(out_base) / model_id / "tech_hits_classified.duckdb"
            apply_results(
                source_db      = tech_source,
                source_table   = TECH_SOURCE_TABLE,
                results_lookup = tech_lookup,
                hit_cols       = TECH_HIT_COLS,
                out_path       = out_tech,
                dry_run        = dry_run,
            )
        else:
            print("  [SKIP] No Tech results found")

        print()


def main():
    ap = argparse.ArgumentParser(
        description="Convert classification results JSON → DuckDB per model config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--model",   default=None, metavar="MODEL_ID",
                    help="Filter to one model config, e.g. gpt-4o__tot")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print stats without writing any files")
    ap.add_argument("--results-base", default=RESULTS_BASE, metavar="PATH",
                    help=f"Results directory (default: {RESULTS_BASE})")
    ap.add_argument("--sdg-source",   default=SDG_SOURCE,   metavar="PATH",
                    help=f"Source SDG DuckDB (default: {SDG_SOURCE})")
    ap.add_argument("--tech-source",  default=TECH_SOURCE,  metavar="PATH",
                    help=f"Source Tech DuckDB (default: {TECH_SOURCE})")
    ap.add_argument("--out-base",     default=OUT_BASE,     metavar="PATH",
                    help=f"Output base dir (default: {OUT_BASE})")
    args = ap.parse_args()

    build_dbs(
        results_base  = args.results_base,
        sdg_source    = args.sdg_source,
        tech_source   = args.tech_source,
        out_base      = args.out_base,
        filter_model  = args.model,
        dry_run       = args.dry_run,
    )


if __name__ == "__main__":
    main()
