#!/usr/bin/env python3
"""
fix_results.py — Analyze classification result JSONs and fix malformed rows.

Two types of anomalies are detected:

  (1) Column-keyed rows — the model returned { column_name: label } instead of
      { pattern: label }.  Fix: expand the column-level label to every individual
      pattern that belongs to that column for this row (from the source DB).

  (2) Completely missing rows — a row with keyword hits was never classified.
      These cannot be fixed locally; they are reported so you can decide whether
      to resubmit them.

Usage:
    # Report anomalies across all model configs (no writes)
    python3 src/postprocessing/fix_results.py --analyze

    # Preview fixes (no writes)
    python3 src/postprocessing/fix_results.py --fix --dry-run

    # Apply fixes in-place
    python3 src/postprocessing/fix_results.py --fix

    # Scope to one model config
    python3 src/postprocessing/fix_results.py --fix --model gpt-5.2__low__zero_shot
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import duckdb

# ── Paths ──────────────────────────────────────────────────────────────────────

RESULTS_BASE = "data/classifications/results"
SDG_SOURCE   = "data/dbs/sdg_hits.duckdb"
TECH_SOURCE  = "data/dbs/tech_hits.duckdb"

SDG_SOURCE_TABLE  = "sdg_hits_classified"
TECH_SOURCE_TABLE = "tech_hits_classified"

# ── Domain / column mappings ───────────────────────────────────────────────────

SDG_HIT_COLS: dict[str, list[str]] = {
    "sdg_a": [f"hits_sdg{i}" for i in range(1, 10)],   # sdg1–sdg9
    "sdg_b": [f"hits_sdg{i}" for i in range(10, 14)],  # sdg10–sdg13
    "sdg_c": [f"hits_sdg{i}" for i in range(14, 18)],  # sdg14–sdg17
}
TECH_HIT_COLS: dict[str, list[str]] = {
    "tech": [
        "hits_ai_ml",
        "hits_cloud_computing",
        "hits_big_data_blockchain",
        "hits_applications_practice",
    ]
}

# All column names (used to detect column-keyed anomalies)
ALL_COL_NAMES: set[str] = {
    c for cols in {**SDG_HIT_COLS, **TECH_HIT_COLS}.values() for c in cols
}


# ── Source DB helpers ──────────────────────────────────────────────────────────

def _parse_hit_col(val) -> dict:
    """Parse a hit column value into {pattern: label_or_empty}."""
    if not val or val in ("{}", "[]", "null"):
        return {}
    if isinstance(val, dict):
        return val
    try:
        parsed = json.loads(val)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {p: "" for p in parsed if isinstance(p, str)}
    except Exception:
        pass
    return {}


def load_source_index(
    db_path: str,
    table: str,
    hit_cols: list[str],
) -> dict[str, dict[str, list[str]]]:
    """
    Build a lookup:  { global_id: { column_name: [pattern, ...] } }

    Only columns with at least one pattern are included.
    """
    con = duckdb.connect(db_path, read_only=True)
    cols_to_select = ["global_id"] + [c for c in hit_cols if c]
    df = con.execute(f"SELECT {', '.join(cols_to_select)} FROM {table}").fetchdf()
    con.close()

    index: dict[str, dict[str, list[str]]] = {}
    for _, row in df.iterrows():
        gid = str(row["global_id"])
        col_patterns: dict[str, list[str]] = {}
        for col in hit_cols:
            if col not in df.columns:
                continue
            patterns = list(_parse_hit_col(row[col]).keys())
            if patterns:
                col_patterns[col] = patterns
        if col_patterns:
            index[gid] = col_patterns
    return index


# ── Anomaly detection ──────────────────────────────────────────────────────────

def find_anomalies(
    results: dict[str, dict],
    source_index: dict[str, dict[str, list[str]]],
    domain_cols: list[str],
    scope_cols: list[str] | None = None,
) -> tuple[list[dict], list[str]]:
    """
    Returns:
      column_keyed  — list of { key, global_id, result_val, fixable_expansion }
      missing_rows  — global_ids present in source (with hits) but absent from results
    """
    result_gids = {
        (k.split("||", 1)[1] if "||" in k else k): k
        for k in results
    }

    # Scope the source index to rows that have hits in the domain-specific columns.
    # e.g. for sdg_a, only check rows that actually have sdg1–sdg9 hits.
    scoped_cols = set(scope_cols) if scope_cols else set(domain_cols)
    scoped_index = {
        gid: col_pats
        for gid, col_pats in source_index.items()
        if any(c in scoped_cols for c in col_pats)
    }

    column_keyed: list[dict] = []
    for key, val in results.items():
        if not isinstance(val, dict) or not val:
            continue
        # Anomaly: any result key is a known column name
        col_keys = [k for k in val if k in ALL_COL_NAMES]
        if not col_keys:
            continue
        global_id = key.split("||", 1)[1] if "||" in key else key
        # Build the expansion: for each bad column key, map each source pattern to the label
        row_source = source_index.get(global_id, {})
        expansion: dict[str, str] = {}
        for col_key in col_keys:
            label = val[col_key]
            patterns = row_source.get(col_key, [])
            if patterns:
                for p in patterns:
                    expansion[p] = label
            else:
                # Column key present but no patterns found in source — keep as-is
                expansion[col_key] = label

        # Also carry over any well-formed (non-column) keys
        for k, v in val.items():
            if k not in ALL_COL_NAMES:
                expansion[k] = v

        column_keyed.append({
            "key":       key,
            "global_id": global_id,
            "original":  val,
            "fixed":     expansion,
        })

    missing_rows = [
        gid for gid in scoped_index
        if gid not in result_gids
    ]

    return column_keyed, missing_rows


# ── Main analysis ──────────────────────────────────────────────────────────────

def analyze_all(
    results_base: str = RESULTS_BASE,
    sdg_source:   str = SDG_SOURCE,
    tech_source:  str = TECH_SOURCE,
    filter_model: str | None = None,
) -> dict[str, dict]:
    """
    Scan all model configs and domains. Return anomaly report dict.
    """
    results_dir = Path(results_base)
    model_dirs  = sorted(d for d in results_dir.iterdir() if d.is_dir())
    if filter_model:
        model_dirs = [d for d in model_dirs if filter_model.lower() in d.name.lower()]

    # Lazily load source indexes (shared across models, heavy to build)
    _sdg_index_cache:  dict[str, dict] | None = None
    _tech_index_cache: dict[str, dict] | None = None

    def get_sdg_index(cols: list[str]) -> dict:
        nonlocal _sdg_index_cache
        if _sdg_index_cache is None:
            print("  [load] Building SDG source index…")
            _sdg_index_cache = load_source_index(sdg_source, SDG_SOURCE_TABLE, cols)
        return _sdg_index_cache

    def get_tech_index(cols: list[str]) -> dict:
        nonlocal _tech_index_cache
        if _tech_index_cache is None:
            print("  [load] Building Tech source index…")
            _tech_index_cache = load_source_index(tech_source, TECH_SOURCE_TABLE, cols)
        return _tech_index_cache

    all_sdg_cols  = [c for cols in SDG_HIT_COLS.values()  for c in cols]
    all_tech_cols = TECH_HIT_COLS["tech"]

    report: dict[str, dict] = {}   # model_id -> { domain -> {column_keyed, missing} }

    for model_dir in model_dirs:
        model_id = model_dir.name
        report[model_id] = {}

        domains_to_check = [
            ("sdg_a",  SDG_HIT_COLS["sdg_a"],   get_sdg_index,  all_sdg_cols),
            ("sdg_b",  SDG_HIT_COLS["sdg_b"],   get_sdg_index,  all_sdg_cols),
            ("sdg_c",  SDG_HIT_COLS["sdg_c"],   get_sdg_index,  all_sdg_cols),
            ("tech",   all_tech_cols,            get_tech_index, all_tech_cols),
        ]

        for domain, domain_cols, index_fn, index_cols in domains_to_check:
            res_path = model_dir / f"{domain}_results.json"
            if not res_path.exists():
                continue
            with open(res_path, encoding="utf-8") as f:
                results = json.load(f)

            source_index = index_fn(index_cols)
            col_keyed, missing = find_anomalies(
                results, source_index, domain_cols, scope_cols=domain_cols
            )
            report[model_id][domain] = {
                "results_path":  str(res_path),
                "total_rows":    len(results),
                "column_keyed":  col_keyed,
                "missing_rows":  missing,
            }

    return report


def print_report(report: dict[str, dict]) -> None:
    any_issue = False
    for model_id, domains in report.items():
        model_printed = False
        for domain, info in domains.items():
            ck  = info["column_keyed"]
            mis = info["missing_rows"]
            if not ck and not mis:
                continue
            if not model_printed:
                print(f"\n── {model_id} ──────────────────────────────")
                model_printed = True
            any_issue = True
            print(f"  {domain} ({info['total_rows']:,} rows total):")
            if ck:
                print(f"    column-keyed anomalies: {len(ck)}")
                for item in ck[:5]:
                    print(f"      {item['key']}")
                    print(f"        original : {item['original']}")
                    print(f"        fixed    : {item['fixed']}")
                if len(ck) > 5:
                    print(f"      … and {len(ck) - 5} more")
            if mis:
                print(f"    completely missing rows: {len(mis)}")
                for gid in mis[:5]:
                    print(f"      {gid}")
                if len(mis) > 5:
                    print(f"      … and {len(mis) - 5} more")
    if not any_issue:
        print("\nNo anomalies found across all model configs.")


# ── Fix ────────────────────────────────────────────────────────────────────────

def apply_fixes(
    report: dict[str, dict],
    dry_run: bool = False,
) -> None:
    """
    For each domain with column-keyed anomalies, rewrite the results JSON
    in-place (after taking a timestamped backup).

    Missing rows are logged but not touched — they need re-submission.
    """
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")

    for model_id, domains in report.items():
        for domain, info in domains.items():
            ck  = info["column_keyed"]
            mis = info["missing_rows"]

            if mis:
                print(f"[INFO] {model_id}/{domain}: {len(mis)} rows missing from results "
                      f"(need re-submission, not fixable here)")

            if not ck:
                continue

            res_path = Path(info["results_path"])

            with open(res_path, encoding="utf-8") as f:
                results: dict = json.load(f)

            # Build fix map: key -> fixed dict
            fix_map = {item["key"]: item["fixed"] for item in ck}
            fixed_results = {k: fix_map.get(k, v) for k, v in results.items()}

            n_changed = sum(1 for k in fix_map if k in results)

            if dry_run:
                print(f"[DRY-RUN] {model_id}/{domain}: would fix {n_changed} rows → {res_path}")
                for item in ck[:3]:
                    print(f"  {item['key']}")
                    print(f"    before: {item['original']}")
                    print(f"    after:  {item['fixed']}")
                if len(ck) > 3:
                    print(f"  … and {len(ck) - 3} more")
                continue

            # Backup original
            backup_path = res_path.with_name(f"{res_path.stem}_backup_{ts}.json")
            shutil.copy2(res_path, backup_path)
            print(f"[BACKUP] {backup_path.name}")

            # Write fixed results
            res_path.write_text(
                json.dumps(fixed_results, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[FIXED]  {model_id}/{domain}: {n_changed} rows updated → {res_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyze and fix malformed classification result rows.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--analyze",  action="store_true",
                    help="Scan all results and print anomaly report")
    ap.add_argument("--fix",      action="store_true",
                    help="Fix column-keyed anomalies in-place (backs up originals)")
    ap.add_argument("--dry-run",  action="store_true",
                    help="With --fix: preview changes without writing")
    ap.add_argument("--model",    default=None, metavar="MODEL_ID",
                    help="Scope to one model config, e.g. gpt-5.2__low__zero_shot")
    ap.add_argument("--results-base", default=RESULTS_BASE, metavar="PATH",
                    help=f"Results directory (default: {RESULTS_BASE})")
    ap.add_argument("--sdg-source",   default=SDG_SOURCE,   metavar="PATH")
    ap.add_argument("--tech-source",  default=TECH_SOURCE,  metavar="PATH")
    args = ap.parse_args()

    if not args.analyze and not args.fix:
        ap.error("Specify --analyze and/or --fix")

    print("Building anomaly report…")
    report = analyze_all(
        results_base = args.results_base,
        sdg_source   = args.sdg_source,
        tech_source  = args.tech_source,
        filter_model = args.model,
    )

    if args.analyze or (args.fix and args.dry_run):
        print_report(report)

    if args.fix:
        print()
        apply_fixes(report, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
