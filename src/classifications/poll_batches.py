"""
poll_batches.py — Check the status of all OpenAI batch jobs listed in config.json.

Reads batch_ids_sdg_a/b/c and batch_ids_tech from each config entry.
Each key is a dict {file_path: batch_id}. Only non-empty batch IDs are polled.

Usage:
    from src.classifications.poll_batches import poll_all
    poll_all("src/classifications/config.json")
    poll_all("src/classifications/config.json", watch=30)
"""

import os
import time
from datetime import datetime, timezone
from typing import Any

from openai import OpenAI

from src.classifications.batch_builder import load_config

# ── Formatting helpers ─────────────────────────────────────────────────────────

def _human_ts(ts: Any) -> str:
    try:
        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc).astimezone()
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)


def _pct(done: int, total: int) -> str:
    return f"{100.0 * done / total:5.1f}%" if total > 0 else "  n/a "


def _counts(rc: dict) -> dict:
    rc       = rc or {}
    total    = int(rc.get("total",     0) or 0)
    complete = int(rc.get("completed", 0) or 0)
    failed   = int(rc.get("failed",    0) or 0)
    expired  = int(rc.get("expired",   0) or 0)
    return {
        "total":    total,
        "complete": complete,
        "failed":   failed,
        "expired":  expired,
        "done":     complete + failed + expired,
    }


# ── Table printer ──────────────────────────────────────────────────────────────

_HDR = (
    f"{'CONFIG ID':<35} {'DOM':<5} {'PART':<25} {'BATCH ID':<34} "
    f"{'STATUS':<12} {'PROGRESS':>8}  {'DONE/TOTAL':<12} "
    f"{'OK':>6} {'FAIL':>5} {'EXP':>5}  {'CREATED'}"
)
_SEP = "-" * 155


def _print_table(rows: list[dict]) -> None:
    print(_HDR)
    print(_SEP)
    for r in rows:
        c = _counts(r.get("request_counts") or {})
        print(
            f"{r['config_id']:<35} {r['domain']:<5} {r['part']:<25} {r['batch_id']:<34} "
            f"{r['status']:<12} {_pct(c['done'], c['total']):>8}  "
            f"{c['done']}/{c['total']:<7} "
            f"{c['complete']:>6} {c['failed']:>5} {c['expired']:>5}  "
            f"{_human_ts(r.get('created_at'))}"
        )
    print(_SEP)
    print(f"({len(rows)} rows)\n")


# ── Core fetch ─────────────────────────────────────────────────────────────────

def _fetch_rows(client: OpenAI, config: list[dict]) -> list[dict]:
    """Retrieve batch status for every submitted batch_id in config."""
    rows = []
    for entry in config:
        for domain, id_key in (
            ("sdg_a", "batch_ids_sdg_a"),
            ("sdg_b", "batch_ids_sdg_b"),
            ("sdg_c", "batch_ids_sdg_c"),
            ("tech",  "batch_ids_tech"),
        ):
            batch_map = entry.get(id_key) or {}
            for file_path, bid in batch_map.items():
                if not bid:
                    continue
                part_name = os.path.basename(file_path)
                try:
                    b = client.batches.retrieve(bid).model_dump()
                except Exception as e:
                    b = {"id": bid, "status": f"ERROR: {e}", "request_counts": {}, "created_at": ""}
                rows.append({
                    "config_id":      entry["id"],
                    "domain":         domain,
                    "part":           part_name,
                    "batch_id":       bid,
                    "status":         (b.get("status") or "").lower(),
                    "request_counts": b.get("request_counts") or {},
                    "created_at":     b.get("created_at"),
                })
    return rows


# ── Public API ─────────────────────────────────────────────────────────────────

def poll_all(
    config_path: str    = "src/classifications/config.json",
    watch: int          = 0,
    batch_id: str | None = None,
) -> None:
    """
    Print a status table for all submitted batches in config.
    If watch > 0, auto-refresh every `watch` seconds (Ctrl+C to stop).
    If batch_id is given, only show that one batch.
    """
    config = load_config(config_path)
    client = OpenAI()

    has_batches = any(
        any(v for v in (entry.get(k) or {}).values())
        for entry in config
        for k in ("batch_ids_sdg_a", "batch_ids_sdg_b", "batch_ids_sdg_c", "batch_ids_tech")
    )
    if not has_batches:
        print("No batch IDs found in config. Run --push first.")
        return

    def _filtered_rows():
        rows = _fetch_rows(client, config)
        if batch_id:
            rows = [r for r in rows if r["batch_id"] == batch_id]
            if not rows:
                print(f"[WARN] Batch ID {batch_id!r} not found in config.")
        return rows

    if watch > 0:
        try:
            while True:
                os.system("clear" if os.name != "nt" else "cls")
                _print_table(_filtered_rows())
                print(f"Watching — refresh every {watch}s. Ctrl+C to exit.")
                time.sleep(watch)
        except KeyboardInterrupt:
            return
    else:
        _print_table(_filtered_rows())
