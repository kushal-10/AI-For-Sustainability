#!/usr/bin/env python3
import re
from pathlib import Path
from openai import OpenAI

LOG_PATH = Path("src/gpt_filter/submit_requests.log")

def extract_batch_ids(log_path: Path):
    pat = re.compile(r"batch_job\.id:\s*([a-zA-Z0-9_\-]+)")
    ids = []
    if not log_path.exists():
        print(f"No log file at {log_path}")
        return ids
    with open(log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                bid = m.group(1).strip()
                if bid not in ids:
                    ids.append(bid)
    return ids

def as_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def get_counts(rc) -> tuple[int, int, int]:
    """
    Works whether rc is a dict-like or a Pydantic model with attributes.
    Returns (total, completed, failed).
    """
    if rc is None:
        return 0, 0, 0
    # try attribute access first (Pydantic model)
    total = getattr(rc, "total", None)
    completed = getattr(rc, "completed", None)
    failed = getattr(rc, "failed", None)
    if total is not None or completed is not None or failed is not None:
        return as_int(total), as_int(completed), as_int(failed)
    # fallback: dict-like
    try:
        return as_int(rc.get("total")), as_int(rc.get("completed")), as_int(rc.get("failed"))
    except Exception:
        return 0, 0, 0

def main():
    client = OpenAI()
    batch_ids = extract_batch_ids(LOG_PATH)
    if not batch_ids:
        print("No batch IDs found.")
        return

    print(f"{'BATCH_ID':<38} {'STATUS':<12} {'TOTAL':>7} {'DONE':>7} {'FAILED':>7} {'REMAIN':>7}")
    print("-" * 80)

    grand_total = grand_done = grand_failed = 0

    for bid in batch_ids:
        try:
            b = client.batches.retrieve(bid)
        except Exception as e:
            print(f"{bid:<38} {'ERROR':<12} {'-':>7} {'-':>7} {'-':>7} {'-':>7}  # {e}")
            continue

        total, done, failed = get_counts(getattr(b, "request_counts", None))
        remain = max(total - done - failed, 0)
        status = getattr(b, "status", "-")

        print(f"{bid:<38} {status:<12} {total:>7} {done:>7} {failed:>7} {remain:>7}")

        grand_total += total
        grand_done += done
        grand_failed += failed

    grand_remain = max(grand_total - grand_done - grand_failed, 0)
    print("-" * 80)
    print(f"{'OVERALL':<38} {'—':<12} {grand_total:>7} {grand_done:>7} {grand_failed:>7} {grand_remain:>7}")

if __name__ == "__main__":
    main()
