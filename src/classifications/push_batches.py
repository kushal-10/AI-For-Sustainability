"""
push_batches.py — Submit JSONL batch files to OpenAI Batch API and update config.json
with the returned batch IDs.

Only entries with empty batch_id_sdg / batch_id_tech are submitted.
Config is updated in-place after each successful submission so progress is not lost
if the script is interrupted.

Usage (called from run_push.py or imported):
    from src.classifications.push_batches import push_all
    push_all("src/classifications/config.json")
"""

import json
from pathlib import Path

from openai import OpenAI

from src.classifications.batch_builder import load_config, save_config, jsonl_path

COMPLETION_WINDOW = "24h"
ENDPOINT          = "/v1/chat/completions"
OUT_BASE          = "data/classifications/batches"


def _submit_file(client: OpenAI, path: Path) -> tuple[str, str]:
    """Upload a JSONL file and create a batch. Returns (file_id, batch_id)."""
    with open(path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id     = file_obj.id,
        endpoint          = ENDPOINT,
        completion_window = COMPLETION_WINDOW,
    )
    return file_obj.id, batch.id


def push_entry(
    client: OpenAI,
    entry: dict,
    out_base: str = OUT_BASE,
) -> dict:
    """
    Submit sdg_1.jsonl, sdg_2.jsonl, and tech.jsonl for one config entry.
    Returns the updated entry dict (with batch_ids filled in).
    """
    entry_id = entry["id"]

    for domain, id_key in (
        ("sdg_1", "batch_id_sdg_1"),
        ("sdg_2", "batch_id_sdg_2"),
        ("tech",  "batch_id_tech"),
    ):
        if entry.get(id_key):
            print(f"[SKIP] {entry_id}/{domain} — batch_id already set: {entry[id_key]}")
            continue

        path = jsonl_path(out_base, entry_id, domain)
        if not path.exists():
            print(f"[WARN] {entry_id}/{domain} — JSONL not found at {path}. Run run_build.py first.")
            continue

        file_id, batch_id = _submit_file(client, path)
        entry[id_key] = batch_id
        print(f"[OK]   {entry_id}/{domain} → file_id={file_id}  batch_id={batch_id}")

    return entry


def push_all(
    config_path: str = "src/classifications/config.json",
    out_base: str    = OUT_BASE,
) -> None:
    """Submit all pending entries and update config.json with batch IDs."""
    config = load_config(config_path)
    client = OpenAI()  # reads OPENAI_API_KEY

    pending = [
        e for e in config
        if not e.get("batch_id_sdg_1") or not e.get("batch_id_sdg_2") or not e.get("batch_id_tech")
    ]
    print(f"Entries pending submission: {len(pending)} / {len(config)}")

    for i, entry in enumerate(config):
        if not entry.get("batch_id_sdg_1") or not entry.get("batch_id_sdg_2") or not entry.get("batch_id_tech"):
            config[i] = push_entry(client, entry, out_base)
            # Save after every entry so partial progress is preserved
            save_config(config_path, config)

    print("\nDone. Run poll_batches.py to monitor status.")
