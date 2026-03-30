#!/usr/bin/env python3
"""
run.py — Build, push, check, and collect OpenAI batch classifications.

Usage:
    python3 src/classifications/run.py --build
    python3 src/classifications/run.py --build  --model gpt-4o__tot
    python3 src/classifications/run.py --push   --model gpt-4o__tot
    python3 src/classifications/run.py --check
    python3 src/classifications/run.py --check  --batch batch_abc123
    python3 src/classifications/run.py --collect
    python3 src/classifications/run.py --collect --batch batch_abc123
    python3 src/classifications/run.py --collect --model gpt-4o__tot --part sdg_a
"""

import argparse
from openai import OpenAI
from src.classifications.batch_builder import build_all, load_config
from src.classifications.push_batches import run_queue_loop
from src.classifications.poll_batches import poll_all
from src.classifications.collect_results import collect_all

# ── Paths ──────────────────────────────────────────────────────────────────────
CONFIG       = "src/classifications/config.json"
SDG_DB       = "data/dbs/sdg_hits.duckdb"
TECH_DB      = "data/dbs/tech_hits.duckdb"
OUT_BASE     = "data/classifications/batches"
RESULTS_BASE = "data/classifications/results_v2"

# ── Batch size limits (update when tier changes) ───────────────────────────────
TOKEN_LIMIT = 4_500_000   # input tokens per part file
REQ_LIMIT   = 50_000      # max requests per part file


def main():
    ap = argparse.ArgumentParser(
        description="Build, push, check, and collect OpenAI batch classifications.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--build",   action="store_true", help="Build JSONL part files from DuckDB")
    ap.add_argument("--push",    action="store_true", help="Submit batches (autonomous loop)")
    ap.add_argument("--check",   action="store_true", help="Check status of submitted batches")
    ap.add_argument("--collect", action="store_true", help="Download completed batch results")

    ap.add_argument("--model",  default=None, metavar="ENTRY_ID",
                                help="Filter to one config entry, e.g. gpt-4o__tot")
    ap.add_argument("--part",   default=None, metavar="DOMAIN",
                                help="Filter to one domain: sdg_a, sdg_b, sdg_c, tech")
    ap.add_argument("--batch",  default=None, metavar="BATCH_ID",
                                help="Filter --check / --collect to a specific batch ID")

    args = ap.parse_args()

    if not any([args.build, args.push, args.check, args.collect]):
        ap.error("Specify at least one action: --build, --push, --check, --collect")

    if args.build:
        build_all(
            config_path   = CONFIG,
            sdg_db        = SDG_DB,
            tech_db       = TECH_DB,
            out_base      = OUT_BASE,
            filter_entry  = args.model,
            filter_domain = args.part,
            token_limit   = TOKEN_LIMIT,
            req_limit     = REQ_LIMIT,
        )

    if args.push:
        run_queue_loop(
            config_path   = CONFIG,
            out_base      = OUT_BASE,
            results_base  = RESULTS_BASE,
            filter_entry  = args.model,
            filter_domain = args.part,
        )

    if args.check:
        poll_all(
            config_path = CONFIG,
            batch_id    = args.batch,
        )

    if args.collect:
        collect_all(
            config_path  = CONFIG,
            results_base = RESULTS_BASE,
            batch_id     = args.batch,
        )


if __name__ == "__main__":
    main()
