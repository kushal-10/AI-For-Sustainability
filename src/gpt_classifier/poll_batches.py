#!/usr/bin/env python3
"""
Check active OpenAI batch jobs and their progress.

Usage:
  export OPENAI_API_KEY=your_key
  python3 src/batching/check_batches.py            # show only active batches
  python3 src/batching/check_batches.py --all      # show all batches
  python3 src/batching/check_batches.py --watch 10 # refresh every 10s
  python3 src/batching/check_batches.py --ids BATCH_ID_1 BATCH_ID_2

Notes:
- "Active" = validating, in_progress, finalizing, cancelling
- Progress uses request_counts.{total, completed, failed, expired} if available.
"""

import os
import time
import argparse
from typing import Dict, Any, List
from datetime import datetime, timezone

try:
    from openai import OpenAI
except Exception as e:
    raise SystemExit("openai package not installed. `pip install openai`") from e

ACTIVE_STATES = {
                # "validating",
                #  "in_progress",
                 # "finalizing",
                 # "cancelling",
                 "completed"
                }

def human_ts(ts: Any) -> str:
    try:
        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc).astimezone()
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return str(ts)

def pct(n: int, d: int) -> str:
    return f"{(100.0 * n / d):5.1f}%" if d > 0 else "  n/a "

def summarize_counts(rc: Dict[str, Any]) -> Dict[str, int]:
    rc = rc or {}
    total = int(rc.get("total", 0) or 0)
    completed = int(rc.get("completed", 0) or 0)
    failed = int(rc.get("failed", 0) or 0)
    expired = int(rc.get("expired", 0) or 0)
    processed = completed + failed + expired
    pending = max(0, total - processed)
    return {
        "total": total,
        "completed": completed,
        "failed": failed,
        "expired": expired,
        "processed": processed,
        "pending": pending,
    }

def fetch_batches(client: OpenAI, show_all: bool, ids: List[str] = None) -> List[Dict[str, Any]]:
    out = []
    if ids:
        for bid in ids:
            try:
                out.append(client.batches.retrieve(bid).model_dump())
            except Exception as e:
                print(f"[WARN] Could not retrieve {bid}: {e}")
        return out

    # Simple one-page fetch (limit=100). Most users won't exceed this concurrently.
    resp = client.batches.list(limit=100)
    data = [b.model_dump() for b in resp.data]
    if show_all:
        return data
    return [b for b in data if (b.get("status") or "").lower() in ACTIVE_STATES]

def print_table(rows: List[Dict[str, Any]]):
    if not rows:
        print("No batches found.")
        return
    # Header
    print("\nID                               STATUS        PROGRESS     (proc/total)    COMPLETED | FAILED | EXPIRED   WINDOW  CREATED_AT")
    print("-" * 120)
    for b in rows:
        bid = b.get("id", "")
        status = (b.get("status") or "").lower()
        rc = summarize_counts(b.get("request_counts") or {})
        prog = pct(rc["processed"], rc["total"])
        window = b.get("completion_window", "")
        created = human_ts(b.get("created_at"))
        print(f"{bid:<32}  {status:<12}  {prog:>7}   ({rc['processed']}/{rc['total']:<5})   "
              f"{rc['completed']:<9}| {rc['failed']:<6} | {rc['expired']:<7}  {window:<7} {created}")
        # Show files line if useful
        in_f = b.get("input_file_id") or "-"
        out_f = b.get("output_file_id") or b.get("response_file_id") or "-"
        err_f = b.get("error_file_id") or "-"
        print(f"    files: input={in_f}  output={out_f}  error={err_f}")
    print("-" * 120)

def main():
    ap = argparse.ArgumentParser(description="Check OpenAI batch jobs and progress.")
    ap.add_argument("--all", action="store_true", help="Show all batches (default: only active).")
    ap.add_argument("--watch", type=int, default=0, help="Refresh every N seconds (0 = single snapshot).")
    ap.add_argument("--ids", nargs="*", help="Specific batch IDs to inspect.")
    args = ap.parse_args()

    client = OpenAI()  # uses OPENAI_API_KEY

    if args.watch > 0:
        try:
            while True:
                rows = fetch_batches(client, show_all=args.all, ids=args.ids)
                os.system("clear" if os.name != "nt" else "cls")
                print_table(rows)
                print(f"(watching… refresh {args.watch}s; Ctrl+C to exit)")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            return
    else:
        rows = fetch_batches(client, show_all=args.all, ids=args.ids)
        print_table(rows)

if __name__ == "__main__":
    main()
