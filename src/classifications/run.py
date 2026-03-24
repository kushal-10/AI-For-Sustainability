#!/usr/bin/env python3
"""
run.py — CLI to build, push, check, and/or cancel JSONL batch files.

Usage:
    python3 src/classifications/run.py --build
    python3 src/classifications/run.py --push
    python3 src/classifications/run.py --build --push
    python3 src/classifications/run.py --check
    python3 src/classifications/run.py --check --watch 30
    python3 src/classifications/run.py --cancel
    python3 src/classifications/run.py --build --config src/classifications/config.json \\
                                               --sdg-db data/dbs/sdg_hits.duckdb \\
                                               --tech-db data/dbs/tech_hits.duckdb

Filtering (--push and --build only):
    --entry SUBSTR   Substring match on config entry id, e.g. "4o-tot", "high-cot", "low"
    --domain DOMAIN  Exact domain to target, e.g. "sdg-a", "sdg-b", "sdg-c", "tech"

    python3 src/classifications/run.py --push --entry 4o-tot --domain sdg-a
    python3 src/classifications/run.py --push --entry 5.2-high
    python3 src/classifications/run.py --push --domain tech
    python3 src/classifications/run.py --build --push --entry low --domain sdg-c
"""

import argparse
from openai import OpenAI
from src.classifications.batch_builder import build_all, load_config, save_config
from src.classifications.push_batches import push_all
from src.classifications.poll_batches import poll_all

CONFIG   = "src/classifications/config.json"
SDG_DB   = "data/dbs/sdg_hits.duckdb"
TECH_DB  = "data/dbs/tech_hits.duckdb"
OUT_BASE = "data/classifications/batches"

CANCELLABLE = {"validating", "in_progress"}


def cancel_all(config_path: str = CONFIG) -> None:
    """Cancel all in-progress/validating batches in config and clear their IDs."""
    config  = load_config(config_path)
    client  = OpenAI()
    cleared = 0

    for i, entry in enumerate(config):
        for domain, id_key in (
            ("sdg_a", "batch_id_sdg_a"),
            ("sdg_b", "batch_id_sdg_b"),
            ("sdg_c", "batch_id_sdg_c"),
            ("tech",  "batch_id_tech"),
        ):
            bid = entry.get(id_key, "")
            if not bid:
                continue
            try:
                b      = client.batches.retrieve(bid).model_dump()
                status = (b.get("status") or "").lower()
            except Exception as e:
                print(f"[ERR]  {entry['id']}/{domain} — could not retrieve {bid}: {e}")
                continue

            if status in CANCELLABLE:
                try:
                    client.batches.cancel(bid)
                    print(f"[CANCELLED] {entry['id']}/{domain} — {bid}  (was: {status})")
                except Exception as e:
                    print(f"[ERR]  {entry['id']}/{domain} — cancel failed: {e}")
                    continue
                entry[id_key] = ""
                cleared += 1
            else:
                print(f"[SKIP] {entry['id']}/{domain} — {bid}  status={status} (not cancellable)")

        config[i] = entry

    save_config(config_path, config)
    print(f"\nCancelled {cleared} batch(es). Cleared IDs written to {config_path}.")


def main():
    ap = argparse.ArgumentParser(
        description="Build, push, check, and/or cancel JSONL batch files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ── Actions ────────────────────────────────────────────────────────────────
    ap.add_argument("--build",      action="store_true", help="Build JSONL batch files from DuckDB")
    ap.add_argument("--push",       action="store_true", help="Push JSONL batch files to OpenAI Batch API")
    ap.add_argument("--check",      action="store_true", help="Check status of submitted batches")
    ap.add_argument("--cancel",     action="store_true", help="Cancel all in-progress/validating batches and clear their IDs")
    # ── Filters (--build / --push) ─────────────────────────────────────────────
    ap.add_argument("--entry",      default=None, metavar="SUBSTR",
                                    help="Filter by config entry (substring match, e.g. '4o-tot', 'high-cot', 'low')")
    ap.add_argument("--domain",     default=None, metavar="DOMAIN",
                                    help="Filter by domain (e.g. 'sdg-a', 'sdg-b', 'sdg-c', 'tech')")
    # ── Misc ───────────────────────────────────────────────────────────────────
    ap.add_argument("--watch",      type=int, default=0, metavar="SEC",
                                    help="Auto-refresh --check every SEC seconds (0 = once)")
    ap.add_argument("--config",     default=CONFIG,   help="Path to config.json")
    ap.add_argument("--sdg-db",     default=SDG_DB,   help="Path to sdg_hits.duckdb")
    ap.add_argument("--tech-db",    default=TECH_DB,  help="Path to tech_hits.duckdb")
    ap.add_argument("--out-base",   default=OUT_BASE, help="Base output dir for JSONL files")
    ap.add_argument("--sdg-table",  default="sdg_hits_classified")
    ap.add_argument("--tech-table", default="tech_hits_classified")
    args = ap.parse_args()

    if not args.build and not args.push and not args.check and not args.cancel:
        ap.error("Specify at least one action: --build, --push, --check, and/or --cancel")

    if args.build:
        build_all(
            config_path   = args.config,
            sdg_db        = args.sdg_db,
            tech_db       = args.tech_db,
            out_base      = args.out_base,
            sdg_table     = args.sdg_table,
            tech_table    = args.tech_table,
            filter_entry  = args.entry,
            filter_domain = args.domain,
        )

    if args.push:
        push_all(
            config_path   = args.config,
            out_base      = args.out_base,
            filter_entry  = args.entry,
            filter_domain = args.domain,
        )

    if args.cancel:
        cancel_all(config_path = args.config)

    if args.check:
        poll_all(
            config_path = args.config,
            watch       = args.watch,
        )


if __name__ == "__main__":
    main()
