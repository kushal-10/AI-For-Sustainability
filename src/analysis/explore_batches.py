"""
explore_batches.py — Explore built batch JSONL files and print a random object
for human verification.

For each config entry, lists available JSONL files with row counts, then picks
one random request and renders it in a readable format:
  - Custom ID, model, reasoning effort
  - Matched SDG/Tech hit patterns (extracted from user message)
  - Passage text (word-wrapped)
  - System prompt (collapsed to first line + total length)

Usage:
    python3 src/analysis/explore_batches.py
    python3 src/analysis/explore_batches.py --entry gpt-4o__tot --domain sdg
    python3 src/analysis/explore_batches.py --entry gpt-5.2__high__cot --domain tech --seed 42
"""

import argparse
import json
import random
import textwrap
from pathlib import Path

CONFIG_PATH  = "src/classifications/config.json"
BATCHES_BASE = "data/classifications/batches"
WIDTH        = 88   # terminal wrap width

# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_config(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _count_lines(path: Path) -> int:
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _random_line(path: Path, seed: int | None) -> dict:
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng   = random.Random(seed)
    return json.loads(rng.choice(lines))


def _extract_user_content(obj: dict) -> str:
    messages = obj.get("body", {}).get("messages", [])
    for m in messages:
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def _extract_sys_prompt(obj: dict) -> str:
    messages = obj.get("body", {}).get("messages", [])
    for m in messages:
        if m.get("role") == "system":
            return m.get("content", "")
    return ""


def _parse_user_content(content: str) -> tuple[str, dict]:
    """Return (passage_text, hits_dict) parsed from user message."""
    passage = ""
    hits    = {}

    passage_marker = "Passage:\n"
    hits_markers   = ["SDG_HITS (regex patterns already matched):\n",
                      "TECH_HITS (regex patterns already matched):\n"]

    hits_start = -1
    for marker in hits_markers:
        idx = content.find(marker)
        if idx != -1:
            hits_start = idx + len(marker)
            break

    if hits_start != -1:
        p_start = len(passage_marker) if content.startswith(passage_marker) else 0
        passage = content[p_start:hits_start - len(marker) - 1].strip()  # noqa
        # re-parse cleanly
        p_start = content.find(passage_marker)
        hits_raw = content[hits_start:].strip()
        if p_start != -1:
            p_end   = content.find("\n\n", p_start)
            passage = content[p_start + len(passage_marker): p_end if p_end != -1 else hits_start].strip()
        # hits_raw may use single quotes (Python repr) — convert to valid JSON
        try:
            hits = json.loads(hits_raw)
        except Exception:
            try:
                hits = eval(hits_raw, {"__builtins__": {}})  # safe: no builtins
            except Exception:
                hits = {"_raw": hits_raw}
    else:
        passage = content.strip()

    return passage, hits


# ── Single object printer ──────────────────────────────────────────────────────

SEP  = "─" * WIDTH
SEP2 = "═" * WIDTH

def _print_object(obj: dict, entry: dict) -> None:
    custom_id = obj.get("custom_id", "")
    domain    = custom_id.split("||")[0] if "||" in custom_id else "?"
    global_id = custom_id.split("||")[1] if "||" in custom_id else custom_id
    body      = obj.get("body", {})
    model     = body.get("model", "?")
    reasoning = body.get("reasoning_effort") or "—"

    sys_prompt  = _extract_sys_prompt(obj)
    user_content = _extract_user_content(obj)
    passage, hits = _parse_user_content(user_content)

    print()
    print(SEP2)
    print(f"  BATCH OBJECT — {entry['id']}")
    print(SEP2)

    # ── Metadata ───────────────────────────────────────────────────────────────
    print(f"  Custom ID        : {custom_id}")
    print(f"  Global ID        : {global_id}")
    print(f"  Domain           : {domain.upper()}")
    print(f"  Model            : {model}")
    print(f"  Reasoning effort : {reasoning}")
    print(f"  Prompt type      : {entry['prompt_type']}")
    print(SEP)

    # ── System prompt (collapsed) ──────────────────────────────────────────────
    first_line = next((l for l in sys_prompt.splitlines() if l.strip()), "")
    print(f"  SYSTEM PROMPT    : {first_line[:WIDTH - 20]}…")
    print(f"  (total length    : {len(sys_prompt):,} chars)")
    print(SEP)

    # ── Matched hit patterns ───────────────────────────────────────────────────
    print("  HIT PATTERNS:")
    if hits and "_raw" not in hits:
        for col, patterns in hits.items():
            print(f"    {col}:")
            for p in (patterns if isinstance(patterns, list) else [patterns]):
                print(f"      • {p}")
    else:
        print(f"    {hits}")
    print(SEP)

    # ── Passage ────────────────────────────────────────────────────────────────
    print("  PASSAGE:")
    wrapped = textwrap.fill(passage, width=WIDTH - 4, initial_indent="    ",
                            subsequent_indent="    ")
    print(wrapped)
    print(SEP2)
    print()


# ── Overview printer ───────────────────────────────────────────────────────────

def print_overview(config: list[dict], batches_base: str) -> None:
    print()
    print(SEP2)
    print("  BATCH FILES OVERVIEW")
    print(SEP2)
    fmt = f"  {'ENTRY':<35} {'DOMAIN':<6} {'ROWS':>8}  PATH"
    print(fmt)
    print(SEP)
    for entry in config:
        for domain in ("sdg", "tech"):
            path = Path(batches_base) / entry["id"] / f"{domain}.jsonl"
            if path.exists():
                n = _count_lines(path)
                print(f"  {entry['id']:<35} {domain.upper():<6} {n:>8,}  {path}")
            else:
                print(f"  {entry['id']:<35} {domain.upper():<6} {'—':>8}  (not built)")
    print(SEP2)
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def explore(
    config_path:  str        = CONFIG_PATH,
    batches_base: str        = BATCHES_BASE,
    entry_filter: str | None = None,
    domain_filter: str | None = None,
    seed: int | None         = None,
) -> None:
    config = _load_config(config_path)

    # Overview table
    print_overview(config, batches_base)

    # Filter entries
    targets = [
        (entry, domain)
        for entry in config
        for domain in ("sdg", "tech")
        if (entry_filter is None or entry["id"] == entry_filter)
        and (domain_filter is None or domain == domain_filter)
        and (Path(batches_base) / entry["id"] / f"{domain}.jsonl").exists()
    ]

    if not targets:
        print("No matching batch files found.")
        return

    for entry, domain in targets:
        path = Path(batches_base) / entry["id"] / f"{domain}.jsonl"
        obj  = _random_line(path, seed)
        _print_object(obj, entry)


def main():
    ap = argparse.ArgumentParser(description="Explore batch JSONL files and print a random object.")
    ap.add_argument("--config",       default=CONFIG_PATH)
    ap.add_argument("--batches-base", default=BATCHES_BASE)
    ap.add_argument("--entry",        default=None, help="Filter to one config entry ID")
    ap.add_argument("--domain",       default=None, choices=["sdg", "tech"],
                    help="Filter to one domain")
    ap.add_argument("--seed",         type=int, default=None,
                    help="Random seed for reproducible sampling")
    args = ap.parse_args()

    explore(
        config_path   = args.config,
        batches_base  = args.batches_base,
        entry_filter  = args.entry,
        domain_filter = args.domain,
        seed          = args.seed,
    )


if __name__ == "__main__":
    main()
