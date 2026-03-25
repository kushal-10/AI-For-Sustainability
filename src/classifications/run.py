#!/usr/bin/env python3
"""
run.py — CLI to build, push, check, collect, and cancel JSONL batch files.

Usage:
    python3 src/classifications/run.py --build
    python3 src/classifications/run.py --push
    python3 src/classifications/run.py --push --file data/classifications/batches/entry/sdg_a_part0001.jsonl
    python3 src/classifications/run.py --push --model gpt-4o__tot --batch sdg_a_part0001.jsonl
    python3 src/classifications/run.py --list
    python3 src/classifications/run.py --check
    python3 src/classifications/run.py --check --watch 30
    python3 src/classifications/run.py --cancel

Tier 1 limits (900,000 token batch queue):
    --build splits each domain into numbered part files.
    --list  shows all parts with token counts and submission status.
    Submit one part at a time with --push --model <entry-id> --batch <filename>
    or with --push --file <full-path>.

Filtering (--push and --build only):
    --entry SUBSTR   Substring match on config entry id
    --domain DOMAIN  Exact domain: "sdg-a", "sdg-b", "sdg-c", "tech"

    python3 src/classifications/run.py --push --entry 4o-tot --domain sdg-a
    python3 src/classifications/run.py --build --push --entry low --domain sdg-c
"""

import argparse
from openai import OpenAI
from src.classifications.batch_builder import build_all, load_config, save_config
from src.classifications.push_batches import push_all, push_file, list_batches
from src.classifications.poll_batches import poll_all
from src.classifications.collect_results import collect_all

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
            ("sdg_a", "batch_ids_sdg_a"),
            ("sdg_b", "batch_ids_sdg_b"),
            ("sdg_c", "batch_ids_sdg_c"),
            ("tech",  "batch_ids_tech"),
        ):
            batch_map = entry.get(id_key) or {}
            for file_path, bid in list(batch_map.items()):
                if not bid:
                    continue
                try:
                    b      = client.batches.retrieve(bid).model_dump()
                    status = (b.get("status") or "").lower()
                except Exception as e:
                    print(f"[ERR]  {entry['id']}/{domain} — could not retrieve {bid}: {e}")
                    continue

                import os
                part_name = os.path.basename(file_path)
                if status in CANCELLABLE:
                    try:
                        client.batches.cancel(bid)
                        print(f"[CANCELLED] {entry['id']}/{domain}/{part_name} — {bid}  (was: {status})")
                    except Exception as e:
                        print(f"[ERR]  {entry['id']}/{domain}/{part_name} — cancel failed: {e}")
                        continue
                    batch_map[file_path] = ""
                    cleared += 1
                else:
                    print(f"[SKIP] {entry['id']}/{domain}/{part_name} — {bid}  status={status} (not cancellable)")

            config[i][id_key] = batch_map

    save_config(config_path, config)
    print(f"\nCancelled {cleared} batch(es). Cleared IDs written to {config_path}.")


def main():
    ap = argparse.ArgumentParser(
        description="Build, push, list, check, and/or cancel JSONL batch files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ── Actions ────────────────────────────────────────────────────────────────
    ap.add_argument("--build",    action="store_true", help="Build JSONL part files from DuckDB")
    ap.add_argument("--push",     action="store_true", help="Push JSONL part files to OpenAI Batch API")
    ap.add_argument("--list",     action="store_true", help="List part files with token counts and submission status")
    ap.add_argument("--check",    action="store_true", help="Check status of submitted batches")
    ap.add_argument("--collect",  action="store_true", help="Download completed batch results")
    ap.add_argument("--cancel",   action="store_true", help="Cancel all in-progress/validating batches")
    # ── Per-file push ──────────────────────────────────────────────────────────
    ap.add_argument("--file",     default=None, metavar="PATH",
                                  help="With --push: submit one specific part file (full path)")
    ap.add_argument("--model",    default=None, metavar="ENTRY_ID",
                                  help="With --push --batch: entry id, e.g. gpt-4o__tot")
    ap.add_argument("--batch",    default=None, metavar="FILENAME",
                                  help="With --push --model: batch filename, e.g. sdg_a_part0001.jsonl")
    # ── Filters (--build / --push) ─────────────────────────────────────────────
    ap.add_argument("--entry",    default=None, metavar="SUBSTR",
                                  help="Filter by config entry id (substring match)")
    ap.add_argument("--domain",   default=None, metavar="DOMAIN",
                                  help="Filter by domain: 'sdg-a', 'sdg-b', 'sdg-c', 'tech'")
    # ── Misc ───────────────────────────────────────────────────────────────────
    ap.add_argument("--watch",    type=int, default=0, metavar="SEC",
                                  help="Auto-refresh --check every SEC seconds (0 = once)")
    ap.add_argument("--force",    action="store_true",
                                  help="With --collect: re-download already-collected results")
    ap.add_argument("--config",   default=CONFIG,   help="Path to config.json")
    ap.add_argument("--sdg-db",   default=SDG_DB,   help="Path to sdg_hits.duckdb")
    ap.add_argument("--tech-db",  default=TECH_DB,  help="Path to tech_hits.duckdb")
    ap.add_argument("--out-base", default=OUT_BASE, help="Base output dir for JSONL files")
    ap.add_argument("--sdg-table",  default="sdg_hits_classified")
    ap.add_argument("--tech-table", default="tech_hits_classified")
    args = ap.parse_args()

    if not any([args.build, args.push, args.list, args.check, args.collect, args.cancel]):
        ap.error("Specify at least one action: --build, --push, --list, --check, --collect, --cancel")

    if args.file and not args.push:
        ap.error("--file can only be used with --push")
    if args.model and not args.push:
        ap.error("--model can only be used with --push")
    if args.batch and not args.model:
        ap.error("--batch requires --model")
    if args.model and not args.batch:
        ap.error("--model requires --batch")

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
        if args.model and args.batch:
            resolved = f"{args.out_base}/{args.model}/{args.batch}"
            push_file(
                path_str    = resolved,
                config_path = args.config,
                out_base    = args.out_base,
            )
        elif args.file:
            push_file(
                path_str    = args.file,
                config_path = args.config,
                out_base    = args.out_base,
            )
        else:
            push_all(
                config_path   = args.config,
                out_base      = args.out_base,
                filter_entry  = args.entry,
                filter_domain = args.domain,
            )

    if args.list:
        list_batches(
            config_path = args.config,
            out_base    = args.out_base,
        )

    if args.check:
        poll_all(
            config_path = args.config,
            watch       = args.watch,
        )

    if args.collect:
        collect_all(
            config_path  = args.config,
            results_base = "data/classifications/results",
            force        = args.force,
        )

    if args.cancel:
        cancel_all(config_path = args.config)


if __name__ == "__main__":
    main()
