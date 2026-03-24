"""
push_batches.py — Submit JSONL batch files to OpenAI Batch API and update config.json
with the returned batch IDs.

Only entries with empty batch_id_sdg / batch_id_tech are submitted.
Config is updated in-place after each successful submission so progress is not lost
if the script is interrupted.

Usage (called from run.py or imported):
    from src.classifications.push_batches import push_all
    push_all("src/classifications/config.json")
    push_all(..., filter_entry="4o-tot", filter_domain="sdg-a")
"""

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


SDG_DOMAINS = (
    ("sdg_a", "batch_id_sdg_a"),
    ("sdg_b", "batch_id_sdg_b"),
    ("sdg_c", "batch_id_sdg_c"),
    ("tech",  "batch_id_tech"),
)


def _norm(s: str) -> str:
    """Normalise a filter token: lowercase, replace hyphens with underscores."""
    return s.lower().replace("-", "_")


def _entry_matches(entry_id: str, filter_entry: str | None) -> bool:
    """True if filter_entry is a substring of the normalised entry id, or no filter set."""
    if not filter_entry:
        return True
    return _norm(filter_entry) in _norm(entry_id)


def _domain_matches(domain: str, filter_domain: str | None) -> bool:
    """True if filter_domain matches the domain exactly (after normalisation), or no filter set."""
    if not filter_domain:
        return True
    return _norm(filter_domain) == _norm(domain)


def push_entry(
    client: OpenAI,
    entry: dict,
    out_base: str         = OUT_BASE,
    filter_domain: str | None = None,
) -> dict:
    """
    Submit sdg_a.jsonl, sdg_b.jsonl, sdg_c.jsonl, and tech.jsonl for one config entry.
    Pass filter_domain to submit only that domain.
    Returns the updated entry dict (with batch_ids filled in).
    """
    entry_id = entry["id"]

    for domain, id_key in SDG_DOMAINS:
        if not _domain_matches(domain, filter_domain):
            continue

        if entry.get(id_key):
            print(f"[SKIP] {entry_id}/{domain} — batch_id already set: {entry[id_key]}")
            continue

        path = jsonl_path(out_base, entry_id, domain)
        if not path.exists():
            print(f"[WARN] {entry_id}/{domain} — JSONL not found at {path}. Run --build first.")
            continue

        file_id, batch_id = _submit_file(client, path)
        entry[id_key] = batch_id
        print(f"[OK]   {entry_id}/{domain} → file_id={file_id}  batch_id={batch_id}")

    return entry


def _is_pending(entry: dict, filter_domain: str | None = None) -> bool:
    return any(
        not entry.get(id_key)
        for domain, id_key in SDG_DOMAINS
        if _domain_matches(domain, filter_domain)
    )


def push_all(
    config_path: str      = "src/classifications/config.json",
    out_base: str         = OUT_BASE,
    filter_entry: str | None  = None,
    filter_domain: str | None = None,
) -> None:
    """
    Submit pending entries and update config.json with batch IDs.

    filter_entry  — substring match against config entry id (e.g. "4o-tot", "high-cot").
    filter_domain — exact domain to submit (e.g. "sdg-a", "tech"). Omit to submit all.
    """
    config = load_config(config_path)
    client = OpenAI()

    scoped = [e for e in config if _entry_matches(e["id"], filter_entry)]
    if not scoped:
        print(f"No config entries matched filter '{filter_entry}'. Available: {[e['id'] for e in config]}")
        return

    pending = [e for e in scoped if _is_pending(e, filter_domain)]
    label   = f"entry='{filter_entry or 'all'}'  domain='{filter_domain or 'all'}'"
    print(f"Filter: {label}")
    print(f"Entries pending submission: {len(pending)} / {len(scoped)}")

    for i, entry in enumerate(config):
        if not _entry_matches(entry["id"], filter_entry):
            continue
        if _is_pending(entry, filter_domain):
            config[i] = push_entry(client, entry, out_base, filter_domain)
            save_config(config_path, config)

    print("\nDone. Run --check to monitor status.")
