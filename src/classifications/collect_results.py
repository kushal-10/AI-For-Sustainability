"""
collect_results.py — Download completed OpenAI batch results and save per config entry.

For each entry/domain, downloads all submitted part batches and merges them into one
results file:
  data/classifications/results/{entry_id}/sdg_a_results.json
  data/classifications/results/{entry_id}/sdg_b_results.json
  data/classifications/results/{entry_id}/sdg_c_results.json
  data/classifications/results/{entry_id}/tech_results.json

Each results file: { custom_id: {pattern: "symbolic"|"substantive"} }

Skips a domain if its results file already exists (unless --force).
Skips individual parts that are not yet completed.

Usage:
    from src.classifications.collect_results import collect_all
    collect_all("src/classifications/config.json")
"""

import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.classifications.batch_builder import load_config

RESULTS_BASE = "data/classifications/results"

# ── JSON parsing helpers ───────────────────────────────────────────────────────

_CODE_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.DOTALL)


def _strip_fences(s: str) -> str:
    return _CODE_FENCE.sub("", s)


def _parse_content(content: str | None) -> dict | None:
    if not content:
        return None
    s = content.strip()
    for candidate in (s, _strip_fences(s).strip()):
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except Exception:
            pass
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        try:
            result = json.loads(m.group(0))
            if isinstance(result, dict):
                return result
        except Exception:
            pass
    return None


def _extract(rec: dict) -> tuple[str, dict | None, dict | None]:
    custom_id = rec.get("custom_id", "")
    if rec.get("error"):
        return custom_id, None, {"level": "record", "error": rec["error"]}
    resp   = rec.get("response") or {}
    status = int(resp.get("status_code") or 0)
    if status != 200:
        return custom_id, None, {"level": "http", "status_code": status}
    body    = (resp.get("body") or {})
    choices = body.get("choices") or []
    if not choices:
        return custom_id, None, {"level": "body", "reason": "no choices"}
    content = (choices[0].get("message") or {}).get("content", "")
    parsed  = _parse_content(content)
    if parsed is None:
        return custom_id, None, {"level": "parse", "raw_content": content}
    return custom_id, parsed, None


# ── Download helpers ───────────────────────────────────────────────────────────

def _download_text(client: OpenAI, file_id: str) -> str:
    resp = client.files.content(file_id)
    data = resp.read() if hasattr(resp, "read") else getattr(resp, "content", resp)
    return data.decode("utf-8", errors="replace") if isinstance(data, (bytes, bytearray)) else str(data)


def _iter_jsonl(text: str):
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception:
            continue


# ── Per-entry collector ────────────────────────────────────────────────────────

def collect_entry(
    client: OpenAI,
    entry: dict,
    results_base: str = RESULTS_BASE,
    force: bool       = False,
) -> None:
    """
    Download and merge results for all part batches in one config entry.
    Each domain's parts are merged into one results file.
    """
    entry_id = entry["id"]
    out_dir  = Path(results_base) / entry_id
    out_dir.mkdir(parents=True, exist_ok=True)

    for domain, id_key in (
        ("sdg_a", "batch_ids_sdg_a"),
        ("sdg_b", "batch_ids_sdg_b"),
        ("sdg_c", "batch_ids_sdg_c"),
        ("tech",  "batch_ids_tech"),
    ):
        batch_map = entry.get(id_key) or {}
        submitted = {fp: bid for fp, bid in batch_map.items() if bid}

        if not submitted:
            print(f"[SKIP] {entry_id}/{domain} — no submitted batch IDs")
            continue

        out_path = out_dir / f"{domain}_results.json"
        if out_path.exists() and not force:
            print(f"[SKIP] {entry_id}/{domain} — results already collected ({out_path})")
            continue

        # Collect and merge results from all parts
        merged_results: dict[str, Any] = {}
        merged_errors:  dict[str, Any] = {}
        all_complete = True

        for file_path, bid in submitted.items():
            part_name = Path(file_path).name
            b         = client.batches.retrieve(bid).model_dump()
            status    = (b.get("status") or "").lower()

            if status != "completed":
                print(f"[WAIT] {entry_id}/{domain} {part_name} — status: {status}")
                all_complete = False
                continue

            out_file_id = b.get("output_file_id") or b.get("response_file_id") or ""
            if not out_file_id:
                print(f"[WARN] {entry_id}/{domain} {part_name} — completed but no output_file_id")
                continue

            raw = _download_text(client, out_file_id)
            part_ok  = 0
            part_err = 0
            for rec in _iter_jsonl(raw):
                cid, parsed, err = _extract(rec)
                if not cid:
                    continue
                if err is not None:
                    merged_errors[cid] = err
                    part_err += 1
                else:
                    merged_results[cid] = parsed
                    part_ok += 1
            print(f"       {part_name}: {part_ok:,} ok, {part_err:,} errors")

        if not all_complete:
            print(f"[WARN] {entry_id}/{domain} — some parts not yet complete; skipping save")
            continue

        if not merged_results and not merged_errors:
            print(f"[WARN] {entry_id}/{domain} — no results downloaded")
            continue

        out_path.write_text(
            json.dumps(merged_results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        err_path = out_dir / f"{domain}_errors.json"
        err_path.write_text(
            json.dumps(merged_errors, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"[OK]   {entry_id}/{domain} — {len(merged_results):,} ok, "
            f"{len(merged_errors):,} errors → {out_path}"
        )


# ── Collect all ────────────────────────────────────────────────────────────────

def collect_all(
    config_path: str  = "src/classifications/config.json",
    results_base: str = RESULTS_BASE,
    force: bool       = False,
) -> None:
    config = load_config(config_path)
    client = OpenAI()

    submitted = [
        e for e in config
        if any(
            any(v for v in (e.get(k) or {}).values())
            for k in ("batch_ids_sdg_a", "batch_ids_sdg_b", "batch_ids_sdg_c", "batch_ids_tech")
        )
    ]
    print(f"Entries with batch IDs: {len(submitted)} / {len(config)}\n")

    for entry in submitted:
        collect_entry(client, entry, results_base, force)

    print("\nDone. Run analyze_results.py to compute stats.")
