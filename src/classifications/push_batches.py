"""
push_batches.py — Submit JSONL part files to OpenAI Batch API and update config.json.

Queue-based workflow (recommended):
  --build   builds all part files and writes data/classifications/queue.json
  --push    submits the next pending part; checks all prior batches are completed first
  --push --file <path>   manual override: submit one specific file

Queue item schema:
  {
    "entry_id":      "gpt-4o__tot",
    "domain":        "sdg_a",
    "file":          "data/classifications/batches/gpt-4o__tot/sdg_a_part0001.jsonl",
    "batch_id":      "",          # empty = not yet submitted
    "openai_status": ""           # synced from OpenAI when polled
  }

Config (config.json) is kept in sync:
  batch_ids_sdg_a / _b / _c / _tech  →  {file_path: batch_id}
"""

import json
import glob
import time
from pathlib import Path

from openai import OpenAI

from src.classifications.batch_builder import (
    load_config,
    save_config,
    part_glob,
    _count_input_tokens,
    TokenCounter,
    TOKEN_LIMIT,
)

COMPLETION_WINDOW = "24h"
ENDPOINT          = "/v1/chat/completions"
OUT_BASE          = "data/classifications/batches"
QUEUE_PATH        = "data/classifications/queue.json"

SDG_DOMAINS = (
    ("sdg_a", "batch_ids_sdg_a"),
    ("sdg_b", "batch_ids_sdg_b"),
    ("sdg_c", "batch_ids_sdg_c"),
    ("tech",  "batch_ids_tech"),
)

# Terminal states — a batch in one of these does not block the queue
_DONE_STATUSES = {"completed", "failed", "expired", "cancelled"}


# ── Queue I/O ──────────────────────────────────────────────────────────────────

def load_queue(queue_path: str = QUEUE_PATH) -> list[dict]:
    p = Path(queue_path)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def save_queue(queue: list[dict], queue_path: str = QUEUE_PATH) -> None:
    p = Path(queue_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(queue, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def rebuild_queue(
    config_path: str = "src/classifications/config.json",
    out_base: str    = OUT_BASE,
    queue_path: str  = QUEUE_PATH,
) -> None:
    """
    Scan disk for all part files, rebuild queue preserving existing batch_ids/statuses.
    Called automatically after --build.
    """
    config   = load_config(config_path)
    existing = {item["file"]: item for item in load_queue(queue_path)}

    # Also pull batch_ids already recorded in config (for migration / manual pushes)
    config_batch_map: dict[str, str] = {}
    for entry in config:
        for _, id_key in SDG_DOMAINS:
            for fp, bid in (entry.get(id_key) or {}).items():
                if bid:
                    config_batch_map[fp] = bid

    new_queue: list[dict] = []
    for entry in config:
        for domain, _ in SDG_DOMAINS:
            for path in part_glob(out_base, entry["id"], domain):
                key = str(path)
                if key in existing:
                    item = dict(existing[key])
                    # Sync batch_id from config if queue entry is missing it
                    if not item.get("batch_id") and config_batch_map.get(key):
                        item["batch_id"] = config_batch_map[key]
                else:
                    bid = config_batch_map.get(key, "")
                    item = {
                        "entry_id":      entry["id"],
                        "domain":        domain,
                        "file":          key,
                        "batch_id":      bid,
                        "openai_status": "",
                    }
                new_queue.append(item)

    save_queue(new_queue, queue_path)

    n_pending   = sum(1 for i in new_queue if not i["batch_id"])
    n_submitted = sum(1 for i in new_queue if i["batch_id"])
    print(f"[OK] Queue saved → {queue_path}")
    print(f"     {len(new_queue)} part(s) total  |  {n_pending} pending  |  {n_submitted} submitted")


# ── OpenAI helpers ─────────────────────────────────────────────────────────────

def _submit_file(client: OpenAI, path: Path) -> tuple[str, str]:
    with open(path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id     = file_obj.id,
        endpoint          = ENDPOINT,
        completion_window = COMPLETION_WINDOW,
    )
    return file_obj.id, batch.id


def _poll_status(client: OpenAI, batch_id: str) -> str:
    try:
        b = client.batches.retrieve(batch_id).model_dump()
        return (b.get("status") or "").lower()
    except Exception as e:
        print(f"  [WARN] Could not poll {batch_id}: {e}")
        return ""


# ── Sync queue item → config ───────────────────────────────────────────────────

def _sync_to_config(file_path: str, batch_id: str, config: list[dict], config_path: str) -> None:
    """Write batch_id for file_path into config[entry][id_key] and save."""
    for i, entry in enumerate(config):
        for domain, id_key in SDG_DOMAINS:
            parts = part_glob(OUT_BASE, entry["id"], domain)
            if any(str(p) == file_path for p in parts):
                submitted = dict(entry.get(id_key) or {})
                submitted[file_path] = batch_id
                config[i][id_key]    = submitted
                save_config(config_path, config)
                return


# ── Queue-based push (next pending) ───────────────────────────────────────────

def push_next(
    config_path: str = "src/classifications/config.json",
    out_base: str    = OUT_BASE,
    queue_path: str  = QUEUE_PATH,
) -> None:
    """
    Submit the next pending part file in the queue.

    Before submitting:
      1. Polls OpenAI to refresh statuses of any in-progress batches ahead in the queue.
      2. Checks all prior parts are in a terminal state (completed/failed/expired).
         - If any are still in_progress → waits and prints status.
         - If any failed/expired → warns and stops (use --push --file to skip ahead).
      3. Submits the next pending part and saves queue + config.
    """
    queue = load_queue(queue_path)
    if not queue:
        print("[INFO] No queue found. Run --build first.")
        return

    # Index of first pending item
    first_pending = next((i for i, item in enumerate(queue) if not item["batch_id"]), None)

    if first_pending is None:
        n_done = sum(1 for item in queue if item.get("openai_status") == "completed")
        print(f"[INFO] All {len(queue)} part(s) have been submitted ({n_done} completed).")
        print("       Run --check to monitor status, --collect when all complete.")
        return

    prior = queue[:first_pending]

    # ── Refresh in-progress batches ────────────────────────────────────────────
    in_progress = [item for item in prior
                   if item["batch_id"] and item["openai_status"] not in _DONE_STATUSES]

    if in_progress:
        client  = OpenAI()
        changed = False
        print(f"Refreshing {len(in_progress)} in-progress batch(es)...")
        for item in in_progress:
            new_status = _poll_status(client, item["batch_id"])
            if new_status and new_status != item["openai_status"]:
                print(f"  {Path(item['file']).name}: {item['openai_status'] or '?'} → {new_status}")
                item["openai_status"] = new_status
                changed = True
            else:
                print(f"  {Path(item['file']).name}: {item['openai_status'] or new_status or 'unknown'}")
        if changed:
            save_queue(queue, queue_path)

    # ── Check blockers ─────────────────────────────────────────────────────────
    still_running = [item for item in prior
                     if item["batch_id"] and item["openai_status"] not in _DONE_STATUSES]
    if still_running:
        print(f"\n[WAIT] {len(still_running)} batch(es) still in progress:")
        for item in still_running:
            print(f"       {Path(item['file']).name}  batch_id={item['batch_id']}  status={item['openai_status'] or 'in_progress'}")
        print("\nRun --push again once they complete.")
        return

    bad = [item for item in prior
           if item["batch_id"] and item["openai_status"] in ("failed", "expired", "cancelled")]
    if bad:
        print(f"\n[ERR] {len(bad)} batch(es) did not complete successfully:")
        for item in bad:
            print(f"      {Path(item['file']).name}  batch_id={item['batch_id']}  status={item['openai_status']}")
        print("\nResolve these before continuing.")
        print("Use --push --file <path> to manually submit a specific part if needed.")
        return

    # ── Submit next pending ────────────────────────────────────────────────────
    next_item = queue[first_pending]
    path      = Path(next_item["file"])

    if not path.exists():
        print(f"[ERR] Part file not found: {path}")
        print("      Run --build to recreate it.")
        return

    client = OpenAI() if not in_progress else client  # reuse if already created above
    client = OpenAI()
    file_id, batch_id = _submit_file(client, path)

    next_item["batch_id"]      = batch_id
    next_item["openai_status"] = "validating"
    save_queue(queue, queue_path)

    config = load_config(config_path)
    _sync_to_config(str(path), batch_id, config, config_path)

    remaining = sum(1 for item in queue[first_pending + 1:] if not item["batch_id"])
    print(f"\n[OK]  Submitted: {path.name}")
    print(f"      entry={next_item['entry_id']}  domain={next_item['domain']}")
    print(f"      file_id={file_id}  batch_id={batch_id}")
    print(f"\n      {remaining} part(s) still pending in queue.")
    if remaining > 0:
        print("      Run --push again once this batch completes.")


# ── Manual single-file push ────────────────────────────────────────────────────

def push_file(
    path_str: str,
    config_path: str = "src/classifications/config.json",
    out_base: str    = OUT_BASE,
    queue_path: str  = QUEUE_PATH,
) -> None:
    """Submit one specific part file; update both config.json and queue.json."""
    path = Path(path_str)
    if not path.exists():
        print(f"[ERR] File not found: {path}")
        return

    client         = OpenAI()
    file_id, batch_id = _submit_file(client, path)
    print(f"[OK]   {path.name} → file_id={file_id}  batch_id={batch_id}")

    # Update config
    config = load_config(config_path)
    _sync_to_config(str(path), batch_id, config, config_path)
    print(f"[OK]   Config updated: {config_path}")

    # Update queue
    queue   = load_queue(queue_path)
    path_str_norm = str(path)
    for item in queue:
        if item["file"] == path_str_norm or Path(item["file"]).resolve() == path.resolve():
            item["batch_id"]      = batch_id
            item["openai_status"] = "validating"
            save_queue(queue, queue_path)
            print(f"[OK]   Queue updated: {queue_path}")
            break


# ── List (from queue) ──────────────────────────────────────────────────────────

def list_batches(
    config_path: str = "src/classifications/config.json",
    out_base: str    = OUT_BASE,
    queue_path: str  = QUEUE_PATH,
) -> None:
    """Print the queue in order with token estimates and submission status."""
    queue = load_queue(queue_path)
    if not queue:
        print("[INFO] No queue found. Run --build first.")
        return

    tc   = TokenCounter()
    rows = []
    for pos, item in enumerate(queue, 1):
        path = Path(item["file"])
        n_req = n_tok = 0
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        n_req += 1
                        n_tok += _count_input_tokens(obj, tc)
                    except Exception:
                        pass

        bid    = item.get("batch_id", "")
        status = item.get("openai_status", "")
        if not bid:
            display = "PENDING"
        elif status:
            display = status
        else:
            display = f"submitted ({bid[:16]}...)"

        rows.append((pos, item["entry_id"], item["domain"], path.name, n_req, n_tok, display))

    w_entry = max(len(r[1]) for r in rows)
    w_file  = max(len(r[3]) for r in rows)
    w_stat  = max(len(r[6]) for r in rows)
    hdr = (f"{'#':>4}  {'ENTRY':<{w_entry}}  {'DOM':<5}  {'FILE':<{w_file}}"
           f"  {'REQS':>7}  {'~TOKENS':>10}  STATUS")
    sep = "-" * len(hdr)
    print(f"\n{hdr}\n{sep}")
    for pos, entry_id, domain, fname, n_req, n_tok, status in rows:
        print(f"{pos:>4}  {entry_id:<{w_entry}}  {domain:<5}  {fname:<{w_file}}"
              f"  {n_req:>7,}  {n_tok:>10,}  {status}")
    print(sep)

    n_pending   = sum(1 for item in queue if not item["batch_id"])
    n_submitted = sum(1 for item in queue if item["batch_id"] and item["openai_status"] not in _DONE_STATUSES)
    n_done      = sum(1 for item in queue if item.get("openai_status") == "completed")
    print(f"\n{len(queue)} parts total  |  {n_pending} pending  |  {n_submitted} in-progress  |  {n_done} completed")
    print(f"Tier 1 batch queue limit: {TOKEN_LIMIT:,} tokens per batch")


# ── Autonomous queue loop ──────────────────────────────────────────────────────

_RESULTS_BASE = "data/classifications/results"


def run_queue_loop(
    config_path:   str      = "src/classifications/config.json",
    out_base:      str      = OUT_BASE,
    queue_path:    str      = QUEUE_PATH,
    results_base:  str      = _RESULTS_BASE,
    poll_interval: int      = 120,
    filter_entry:  str|None = None,
    filter_domain: str|None = None,
) -> None:
    """
    Orchestrate the full push → poll → collect cycle, one batch part at a time.

    Loop behaviour:
      1. Find the first in-flight batch that matches the active filters.
           completed            → collect results, mark done, immediately continue
           failed/expired/cancelled → log, skip to next
           validating/in_progress/finalizing → sleep poll_interval, re-check
      2. No active batch → submit the next PENDING item from the queue.
      3. No pending + no active → exit.

    Ctrl+C saves queue state and exits cleanly.  Re-run --push to resume.
    """
    from src.classifications.collect_results import collect_part

    client = OpenAI()

    def _matches(item: dict) -> bool:
        if filter_entry  and _norm(filter_entry)  not in _norm(item["entry_id"]):
            return False
        if filter_domain and _norm(filter_domain) != _norm(item["domain"]):
            return False
        return True

    def _find_active(q: list) -> dict | None:
        return next(
            (i for i in q
             if i["batch_id"] and i["openai_status"] not in _DONE_STATUSES and _matches(i)),
            None,
        )

    def _find_pending(q: list) -> dict | None:
        return next((i for i in q if not i["batch_id"] and _matches(i)), None)

    def _n_scope(q):   return sum(1 for i in q if _matches(i))
    def _n_done(q):    return sum(1 for i in q if _matches(i) and i.get("openai_status") == "completed")
    def _n_pending(q): return sum(1 for i in q if _matches(i) and not i["batch_id"])

    print("[push] Queue loop started")
    if filter_entry:  print(f"       entry  filter : {filter_entry}")
    if filter_domain: print(f"       domain filter : {filter_domain}")
    print(f"       poll interval  : {poll_interval}s\n")

    try:
        while True:
            queue  = load_queue(queue_path)
            active = _find_active(queue)

            # ── Active batch: check its status ─────────────────────────────────
            if active:
                part   = Path(active["file"]).name
                status = _poll_status(client, active["batch_id"])
                if status:
                    active["openai_status"] = status
                    save_queue(queue, queue_path)
                else:
                    status = active["openai_status"] or "unknown"

                # Progress counter from OpenAI
                rc_str = ""
                try:
                    b   = client.batches.retrieve(active["batch_id"]).model_dump()
                    rc  = b.get("request_counts") or {}
                    tot = int(rc.get("total",     0) or 0)
                    ok  = int(rc.get("completed", 0) or 0)
                    if tot:
                        rc_str = f"  {ok}/{tot} reqs"
                except Exception:
                    pass

                ts = time.strftime("%H:%M:%S")
                print(
                    f"[{ts}] {part}  status={status}{rc_str}  "
                    f"({_n_done(queue)}/{_n_scope(queue)} done, {_n_pending(queue)} pending)"
                )

                if status == "completed":
                    collect_part(client, active, results_base)
                    continue   # check for next active / pending immediately

                if status in _DONE_STATUSES:   # failed / expired / cancelled
                    print(f"  ↳ {active['batch_id']} {status} — skipping to next item")
                    continue

                # Still running — wait
                print(f"  ↳ next check in {poll_interval}s  (Ctrl+C to pause)")
                time.sleep(poll_interval)
                continue

            # ── No active batch: submit next pending ───────────────────────────
            pending = _find_pending(queue)

            if pending is None:
                n_total = _n_scope(queue)
                n_ok    = _n_done(queue)
                n_bad   = sum(
                    1 for i in queue
                    if _matches(i) and i.get("openai_status") in ("failed", "expired", "cancelled")
                )
                print(f"\n[push] All done — {n_ok}/{n_total} completed, {n_bad} failed/expired.")
                break

            path = Path(pending["file"])
            if not path.exists():
                print(f"[ERR]  Missing file: {path} — marking and skipping")
                pending["openai_status"] = "missing"
                save_queue(queue, queue_path)
                continue

            n_left = _n_pending(queue) - 1
            print(f"\n[submit] {path.name}")
            print(f"         entry={pending['entry_id']}  domain={pending['domain']}  "
                  f"{n_left} more pending after this")

            file_id, batch_id = _submit_file(client, path)
            pending["batch_id"]      = batch_id
            pending["openai_status"] = "validating"
            save_queue(queue, queue_path)

            config = load_config(config_path)
            _sync_to_config(str(path), batch_id, config, config_path)

            print(f"         file_id={file_id}")
            print(f"         batch_id={batch_id}")
            print(f"         First status check in {poll_interval}s ...")
            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print(f"\n[push] Paused — queue state saved → {queue_path}")
        print("       Re-run --push to resume.")


# ── Legacy: push all at once (kept for --entry / --domain filtering) ───────────

def _norm(s: str) -> str:
    return s.lower().replace("-", "_")

def _entry_matches(entry_id: str, f: str | None) -> bool:
    return not f or _norm(f) in _norm(entry_id)

def _domain_matches(domain: str, f: str | None) -> bool:
    return not f or _norm(f) == _norm(domain)

def push_all(
    config_path: str          = "src/classifications/config.json",
    out_base: str             = OUT_BASE,
    queue_path: str           = QUEUE_PATH,
    filter_entry: str | None  = None,
    filter_domain: str | None = None,
) -> None:
    """Submit all pending parts at once (filtered). Bypasses queue ordering."""
    config = load_config(config_path)
    client = OpenAI()
    queue  = load_queue(queue_path)
    q_map  = {item["file"]: item for item in queue}

    for i, entry in enumerate(config):
        if not _entry_matches(entry["id"], filter_entry):
            continue
        for domain, id_key in SDG_DOMAINS:
            if not _domain_matches(domain, filter_domain):
                continue
            submitted = dict(entry.get(id_key) or {})
            for path in part_glob(out_base, entry["id"], domain):
                key = str(path)
                if submitted.get(key):
                    print(f"[SKIP] {path.name} — already submitted")
                    continue
                file_id, batch_id = _submit_file(client, path)
                submitted[key] = batch_id
                print(f"[OK]   {path.name} → batch_id={batch_id}")
                if key in q_map:
                    q_map[key]["batch_id"]      = batch_id
                    q_map[key]["openai_status"] = "validating"
            config[i][id_key] = submitted
        save_config(config_path, config)

    save_queue(list(q_map.values()), queue_path)
    print("\nDone. Run --check to monitor status.")
