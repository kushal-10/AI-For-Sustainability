#!/usr/bin/env python3
"""
test.py — Build, push, and check small test batches (10 SDG + 10 Tech samples).

Samples 10 rows with SDG hits and 10 rows with Tech hits from the databases,
then runs the same build → push pipeline as run.py for all config entries.
Batch state is persisted in a separate test_config.json so it does not pollute
the production config.

Usage:
    python3 src/classifications/test.py --build
    python3 src/classifications/test.py --push
    python3 src/classifications/test.py --build --push
    python3 src/classifications/test.py --check
    python3 src/classifications/test.py --check --watch 30
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
from openai import OpenAI

from src.classifications.batch_builder import (
    SDG_COLS_A,
    SDG_COLS_B,
    SDG_COLS_C,
    TECH_HIT_COLS,
    SDG_PROMPTS,
    TECH_PROMPTS,
    _hits_dict,
    build_jsonl,
    load_config,
    save_config,
    jsonl_path,
)

# ── Defaults ───────────────────────────────────────────────────────────────────

CONFIG        = "src/classifications/config.json"
SDG_DB        = "data/dbs/sdg_hits.duckdb"
TECH_DB       = "data/dbs/tech_hits.duckdb"
SDG_TABLE     = "sdg_hits_classified"
TECH_TABLE    = "tech_hits_classified"
OUT_BASE      = "data/classifications/test_batches"
TEST_CONFIG   = "data/classifications/test_batches/test_config.json"
N_SAMPLES     = 10

COMPLETION_WINDOW = "24h"
ENDPOINT          = "/v1/chat/completions"


# ── Sampling ───────────────────────────────────────────────────────────────────

def _load_samples(db_path: str, table: str, hit_cols: list[str], n: int) -> pd.DataFrame:
    """Load up to n rows that have at least one non-empty hit in hit_cols."""
    con = duckdb.connect(db_path, read_only=True)
    df  = con.execute(f"SELECT * FROM {table}").fetchdf()
    con.close()

    def _has_hit(row: pd.Series) -> bool:
        return bool(_hits_dict(row.to_dict(), hit_cols))

    mask = df.apply(_has_hit, axis=1)
    return df[mask].head(n).reset_index(drop=True)


# ── Test config helpers ────────────────────────────────────────────────────────

def _init_test_config(config_path: str, test_config_path: str) -> list[dict]:
    """
    Load or create test_config.json from the production config.
    Each entry gets the same model/prompt/reasoning_effort but empty test batch IDs.
    """
    if os.path.exists(test_config_path):
        with open(test_config_path, encoding="utf-8") as f:
            return json.load(f)

    prod = load_config(config_path)
    test_cfg = [
        {
            "id":                entry["id"],
            "model":             entry["model"],
            "reasoning_effort":  entry.get("reasoning_effort"),
            "prompt_type":       entry["prompt_type"],
            "batch_id_sdg_a":    "",
            "batch_id_sdg_b":    "",
            "batch_id_sdg_c":    "",
            "batch_id_tech":     "",
        }
        for entry in prod
    ]
    Path(test_config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(test_config_path, "w", encoding="utf-8") as f:
        json.dump(test_cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return test_cfg


def _save_test_config(test_config_path: str, config: list[dict]) -> None:
    with open(test_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")


# ── Build ──────────────────────────────────────────────────────────────────────

def build_test(
    config_path: str  = CONFIG,
    sdg_db: str       = SDG_DB,
    tech_db: str      = TECH_DB,
    sdg_table: str    = SDG_TABLE,
    tech_table: str   = TECH_TABLE,
    out_base: str     = OUT_BASE,
    test_config_path: str = TEST_CONFIG,
    n_samples: int    = N_SAMPLES,
) -> None:
    """Sample rows from each DB and build test JSONL for every config entry."""
    test_cfg = _init_test_config(config_path, test_config_path)

    # ── Sample once, reuse for all entries ─────────────────────────────────────
    all_sdg_cols  = [c for c in SDG_COLS_A + SDG_COLS_B + SDG_COLS_C]
    print(f"Sampling {n_samples} SDG rows from {sdg_db} ...")
    df_sdg  = _load_samples(sdg_db,  sdg_table,  all_sdg_cols,               n_samples)
    print(f"  → {len(df_sdg)} rows sampled")

    print(f"Sampling {n_samples} Tech rows from {tech_db} ...")
    df_tech = _load_samples(tech_db, tech_table, list(TECH_HIT_COLS),         n_samples)
    print(f"  → {len(df_tech)} rows sampled\n")

    sdg_id_col      = next(c for c in ("global_id", "id", "uid") if c in df_sdg.columns)
    sdg_passage_col = next(c for c in ("passage", "sentence", "text", "content") if c in df_sdg.columns)
    tech_id_col      = next(c for c in ("global_id", "id", "uid") if c in df_tech.columns)
    tech_passage_col = next(c for c in ("passage", "sentence", "text", "content") if c in df_tech.columns)

    sdg_avail  = set(df_sdg.columns)
    tech_avail = set(df_tech.columns)

    for entry in test_cfg:
        entry_id         = entry["id"]
        model            = entry["model"]
        reasoning_effort = entry.get("reasoning_effort")
        prompt_type      = entry["prompt_type"]
        sdg_prompt       = SDG_PROMPTS[prompt_type]
        tech_prompt      = TECH_PROMPTS[prompt_type]

        for label, cols, span in (
            ("sdg_a", SDG_COLS_A, "sdg1–sdg9"),
            ("sdg_b", SDG_COLS_B, "sdg10–sdg13"),
            ("sdg_c", SDG_COLS_C, "sdg14–sdg17"),
        ):
            p = jsonl_path(out_base, entry_id, label)
            if p.exists():
                print(f"[SKIP] {entry_id}/{label}.jsonl already exists")
            else:
                filtered = [c for c in cols if c in sdg_avail]
                n = build_jsonl(df_sdg, filtered, sdg_id_col, sdg_passage_col,
                                label, sdg_prompt, model, reasoning_effort, p)
                print(f"[OK]   {entry_id}/{label}.jsonl — {n} requests  ({span})")

        # tech
        path_t = jsonl_path(out_base, entry_id, "tech")
        if path_t.exists():
            print(f"[SKIP] {entry_id}/tech.jsonl already exists")
        else:
            t_cols = [c for c in df_tech.columns if c in TECH_HIT_COLS]
            n = build_jsonl(df_tech, t_cols, tech_id_col, tech_passage_col,
                            "tech", tech_prompt, model, reasoning_effort, path_t)
            print(f"[OK]   {entry_id}/tech.jsonl  — {n} requests")

    print("\nDone building test batches.")


# ── Push ───────────────────────────────────────────────────────────────────────

def _submit_file(client: OpenAI, path: Path) -> tuple[str, str]:
    with open(path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id     = file_obj.id,
        endpoint          = ENDPOINT,
        completion_window = COMPLETION_WINDOW,
    )
    return file_obj.id, batch.id


def push_test(
    config_path: str      = CONFIG,
    out_base: str         = OUT_BASE,
    test_config_path: str = TEST_CONFIG,
) -> None:
    """Submit all pending test JSONL files and update test_config.json with batch IDs."""
    test_cfg = _init_test_config(config_path, test_config_path)
    client   = OpenAI()

    _domains = (
        ("sdg_a", "batch_id_sdg_a"),
        ("sdg_b", "batch_id_sdg_b"),
        ("sdg_c", "batch_id_sdg_c"),
        ("tech",  "batch_id_tech"),
    )

    def _is_pending(e: dict) -> bool:
        return any(not e.get(k) for _, k in _domains)

    pending = [e for e in test_cfg if _is_pending(e)]
    print(f"Entries pending submission: {len(pending)} / {len(test_cfg)}")

    for i, entry in enumerate(test_cfg):
        if not _is_pending(entry):
            continue

        entry_id = entry["id"]
        for domain, id_key in _domains:
            if entry.get(id_key):
                print(f"[SKIP] {entry_id}/{domain} — already submitted: {entry[id_key]}")
                continue
            path = jsonl_path(out_base, entry_id, domain)
            if not path.exists():
                print(f"[WARN] {entry_id}/{domain} — JSONL not found at {path}. Run --build first.")
                continue
            file_id, batch_id = _submit_file(client, path)
            entry[id_key] = batch_id
            print(f"[OK]   {entry_id}/{domain} → file_id={file_id}  batch_id={batch_id}")

        test_cfg[i] = entry
        _save_test_config(test_config_path, test_cfg)

    print("\nDone. Run --check to monitor status.")


# ── Check ──────────────────────────────────────────────────────────────────────

def _human_ts(ts) -> str:
    try:
        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc).astimezone()
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)


def _pct(done: int, total: int) -> str:
    return f"{100.0 * done / total:5.1f}%" if total > 0 else "  n/a "


_HDR = (
    f"{'CONFIG ID':<35} {'DOM':<5} {'BATCH ID':<34} "
    f"{'STATUS':<12} {'PROGRESS':>8}  {'DONE/TOTAL':<12} "
    f"{'OK':>6} {'FAIL':>5} {'EXP':>5}  {'CREATED'}"
)
_SEP = "-" * 130


def _print_table(rows: list[dict]) -> None:
    print(_HDR)
    print(_SEP)
    for r in rows:
        rc       = r.get("request_counts") or {}
        total    = int(rc.get("total",     0) or 0)
        complete = int(rc.get("completed", 0) or 0)
        failed   = int(rc.get("failed",    0) or 0)
        expired  = int(rc.get("expired",   0) or 0)
        done     = complete + failed + expired
        print(
            f"{r['config_id']:<35} {r['domain']:<5} {r['batch_id']:<34} "
            f"{r['status']:<12} {_pct(done, total):>8}  "
            f"{done}/{total:<7} "
            f"{complete:>6} {failed:>5} {expired:>5}  "
            f"{_human_ts(r.get('created_at'))}"
        )
    print(_SEP)

    # ── Summary ────────────────────────────────────────────────────────────────
    statuses = [r["status"] for r in rows]
    counts   = {}
    for s in statuses:
        counts[s] = counts.get(s, 0) + 1
    summary = "  ".join(f"{s}: {n}" for s, n in sorted(counts.items()))
    print(f"({len(rows)} batches)  {summary}\n")


def _fetch_rows(client: OpenAI, config: list[dict]) -> list[dict]:
    rows = []
    for entry in config:
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
                b = client.batches.retrieve(bid).model_dump()
            except Exception as e:
                b = {"id": bid, "status": f"error: {e}", "request_counts": {}, "created_at": ""}
            rows.append({
                "config_id":      entry["id"],
                "domain":         domain,
                "batch_id":       bid,
                "status":         (b.get("status") or "").lower(),
                "request_counts": b.get("request_counts") or {},
                "created_at":     b.get("created_at"),
            })
    return rows


CANCELLABLE = {"validating", "in_progress"}


def cancel_test(
    test_config_path: str = TEST_CONFIG,
) -> None:
    """Cancel all cancellable test batches and clear their IDs from test_config.json."""
    if not os.path.exists(test_config_path):
        print(f"No test config found at {test_config_path}.")
        return

    with open(test_config_path, encoding="utf-8") as f:
        test_cfg = json.load(f)

    client  = OpenAI()
    cleared = 0

    for i, entry in enumerate(test_cfg):
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

        test_cfg[i] = entry

    _save_test_config(test_config_path, test_cfg)
    print(f"\nCancelled {cleared} batch(es). Cleared IDs written to {test_config_path}.")


def check_test(
    test_config_path: str = TEST_CONFIG,
    watch: int            = 0,
) -> None:
    """Print a status table for all test batches. Pass watch>0 for auto-refresh."""
    if not os.path.exists(test_config_path):
        print(f"No test config found at {test_config_path}. Run --push first.")
        return

    with open(test_config_path, encoding="utf-8") as f:
        test_cfg = json.load(f)

    submitted = [e for e in test_cfg if e.get("batch_id_sdg_a") or e.get("batch_id_sdg_b") or e.get("batch_id_sdg_c") or e.get("batch_id_tech")]
    if not submitted:
        print("No batch IDs found in test config. Run --push first.")
        return

    client = OpenAI()

    if watch > 0:
        try:
            while True:
                os.system("clear" if os.name != "nt" else "cls")
                _print_table(_fetch_rows(client, submitted))
                print(f"Watching — refresh every {watch}s. Ctrl+C to exit.")
                time.sleep(watch)
        except KeyboardInterrupt:
            return
    else:
        _print_table(_fetch_rows(client, submitted))


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Build, push, and check test batches (10 SDG + 10 Tech samples).")
    ap.add_argument("--build",       action="store_true", help="Sample rows and build test JSONL files")
    ap.add_argument("--push",        action="store_true", help="Submit test JSONL files to OpenAI Batch API")
    ap.add_argument("--check",       action="store_true", help="Check status of submitted test batches")
    ap.add_argument("--cancel",      action="store_true", help="Cancel all in-progress/validating test batches and clear their IDs")
    ap.add_argument("--watch",       type=int, default=0, metavar="SEC",
                                     help="Auto-refresh --check every SEC seconds (0 = once)")
    ap.add_argument("--config",      default=CONFIG,      help="Path to production config.json")
    ap.add_argument("--sdg-db",      default=SDG_DB,      help="Path to sdg_hits.duckdb")
    ap.add_argument("--tech-db",     default=TECH_DB,     help="Path to tech_hits.duckdb")
    ap.add_argument("--sdg-table",   default=SDG_TABLE)
    ap.add_argument("--tech-table",  default=TECH_TABLE)
    ap.add_argument("--out-base",    default=OUT_BASE,    help="Base output dir for test JSONL files")
    ap.add_argument("--test-config", default=TEST_CONFIG, help="Path to test_config.json")
    ap.add_argument("--n-samples",   type=int, default=N_SAMPLES, help="Rows to sample per domain")
    args = ap.parse_args()

    if not args.build and not args.push and not args.check and not args.cancel:
        ap.error("Specify at least one action: --build, --push, --check, and/or --cancel")

    if args.build:
        build_test(
            config_path      = args.config,
            sdg_db           = args.sdg_db,
            tech_db          = args.tech_db,
            sdg_table        = args.sdg_table,
            tech_table       = args.tech_table,
            out_base         = args.out_base,
            test_config_path = args.test_config,
            n_samples        = args.n_samples,
        )

    if args.push:
        push_test(
            config_path      = args.config,
            out_base         = args.out_base,
            test_config_path = args.test_config,
        )

    if args.cancel:
        cancel_test(
            test_config_path = args.test_config,
        )

    if args.check:
        check_test(
            test_config_path = args.test_config,
            watch            = args.watch,
        )


if __name__ == "__main__":
    main()
