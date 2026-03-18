#!/usr/bin/env python3
"""
run_batches.py — OpenAI Batch API runner for SDG prompt evaluation.

Usage:
  python run_batches.py --create   # Build batch JSONL files
  python run_batches.py --push     # Upload + submit all batches to OpenAI
  python run_batches.py --check    # Print status of all submitted batches
  python run_batches.py --poll     # Fetch results → 4 CSVs + costs JSON
  python run_batches.py --analyse  # Scan result CSVs → summary result.csv
"""

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE        = Path(__file__).parent
PROMPTS_PY  = HERE / "prompts.py"
DATA_CSV    = HERE / "passage_keyword_truth.csv"
BATCH_DIR   = HERE / "batches"
IDS_FILE    = HERE / "batch_ids.json"
RESULTS_DIR = HERE / "results"
COSTS_FILE  = HERE / "results" / "costs.json"
SUMMARY_CSV = HERE / "results" / "result.csv"

BATCH_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

MODEL = "gpt-4o"

# ---------------------------------------------------------------------------
# Pricing (USD per token) — Batch API is 50% off standard pricing.
# gpt-4o standard: $2.50 input / $10.00 output per 1M tokens
# Batch discount:  $1.25 input /  $5.00 output per 1M tokens
# Source: openai.com/api/pricing (verified March 2026)
# ---------------------------------------------------------------------------
PRICE_INPUT_PER_TOKEN  = 1.25 / 1_000_000
PRICE_OUTPUT_PER_TOKEN = 5.00 / 1_000_000


def token_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens * PRICE_INPUT_PER_TOKEN) + (completion_tokens * PRICE_OUTPUT_PER_TOKEN)


# ---------------------------------------------------------------------------
# Load prompts
# ---------------------------------------------------------------------------
def load_prompts() -> dict:
    spec = importlib.util.spec_from_file_location("prompts", PROMPTS_PY)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {
        "zero_shot": mod.SYS_PROMPT_SDG_ZERO_SHOT,
        "few_shot":  mod.SYS_PROMPT_SDG_FEW_SHOT,
        "cot":       mod.SYS_PROMPT_SDG_COT,
        "tot":       mod.SYS_PROMPT_SDG_TOT,
    }


# ---------------------------------------------------------------------------
# Load source CSV rows
# ---------------------------------------------------------------------------
def load_rows() -> list:
    with open(DATA_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Build user message
# ---------------------------------------------------------------------------
def build_user_message(row: dict) -> str:
    sdg_hits = json.dumps({"hits_sdg": [row["keyword"]]}, ensure_ascii=False)
    return (
        f"Passage:\n{row['passage'].strip()}\n\n"
        f"SDG_HITS:\n{sdg_hits}"
    )


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------
def extract_label(content: str) -> str:
    """
    Returns 'sym', 'sub', or 'invalid'.
    Tries strict JSON parse first, then keyword scan.
    Returns 'invalid' if both words are present (ambiguous), neither is
    present, or content is empty / non-string.
    """
    if not content or not content.strip():
        return "invalid"

    # Attempt JSON parse
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            for v in data.values():
                v = str(v).strip().lower()
                if v == "symbolic":
                    return "sym"
                if v == "substantive":
                    return "sub"
    except (json.JSONDecodeError, TypeError):
        pass

    # Keyword fallback — only unambiguous matches
    low     = content.lower()
    has_sub = "substantive" in low
    has_sym = "symbolic"    in low

    if has_sub and not has_sym:
        return "sub"
    if has_sym and not has_sub:
        return "sym"

    return "invalid"   # both or neither


# ---------------------------------------------------------------------------
# --create
# ---------------------------------------------------------------------------
def cmd_create():
    prompts = load_prompts()
    rows    = load_rows()

    for prompt_name, sys_prompt in prompts.items():
        out_path = BATCH_DIR / f"batch_{prompt_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for idx, row in enumerate(rows):
                request = {
                    "custom_id": f"{prompt_name}_{idx}",
                    "method":    "POST",
                    "url":       "/v1/chat/completions",
                    "body": {
                        "model":       MODEL,
                        "messages": [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user",   "content": build_user_message(row)},
                        ],
                        "temperature": 0,
                        "max_tokens":  256,
                    },
                }
                f.write(json.dumps(request, ensure_ascii=False) + "\n")

        print(f"[create] {prompt_name}: wrote {len(rows)} requests -> {out_path}")

    print("\nDone. Run --push to upload and submit.")


# ---------------------------------------------------------------------------
# --push
# ---------------------------------------------------------------------------
def cmd_push():
    client  = OpenAI()
    prompts = load_prompts()
    ids     = {}

    if IDS_FILE.exists():
        ids = json.loads(IDS_FILE.read_text())
        print(f"[push] Loaded existing batch_ids.json ({len(ids)} entries).")

    for prompt_name in prompts:
        if prompt_name in ids:
            print(f"[push] {prompt_name}: already submitted (id={ids[prompt_name]}), skipping.")
            continue

        jsonl_path = BATCH_DIR / f"batch_{prompt_name}.jsonl"
        if not jsonl_path.exists():
            print(f"[push] {prompt_name}: JSONL not found -- run --create first.", file=sys.stderr)
            continue

        print(f"[push] {prompt_name}: uploading {jsonl_path} ...", end=" ", flush=True)
        with open(jsonl_path, "rb") as f:
            upload = client.files.create(file=f, purpose="batch")
        print(f"file_id={upload.id}")

        batch = client.batches.create(
            input_file_id     = upload.id,
            endpoint          = "/v1/chat/completions",
            completion_window = "24h",
            metadata          = {"prompt": prompt_name},
        )
        ids[prompt_name] = batch.id
        print(f"[push] {prompt_name}: batch created -> id={batch.id}  status={batch.status}")

    IDS_FILE.write_text(json.dumps(ids, indent=2))
    print(f"\n[push] Saved batch_ids.json. Run --check to monitor progress.")


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------
def cmd_check():
    if not IDS_FILE.exists():
        print("No batch_ids.json found -- run --push first.")
        return

    client = OpenAI()
    ids    = json.loads(IDS_FILE.read_text())

    print(f"{'Prompt':<15} {'Batch ID':<32} {'Status':<20} {'Completed':>10} {'Failed':>8}")
    print("-" * 90)
    for prompt_name, batch_id in ids.items():
        b         = client.batches.retrieve(batch_id)
        completed = b.request_counts.completed if b.request_counts else "?"
        failed    = b.request_counts.failed    if b.request_counts else "?"
        print(f"{prompt_name:<15} {batch_id:<32} {b.status:<20} {str(completed):>10} {str(failed):>8}")


# ---------------------------------------------------------------------------
# --poll
# ---------------------------------------------------------------------------
def cmd_poll():
    """
    For each completed batch:
      1. Download JSONL output.
      2. Write results/results_{prompt}.csv with THREE columns:
             passage | keyword | predicted
         where predicted is 'sym', 'sub', or 'invalid'.
      3. Accumulate token counts, compute cost per prompt.
      4. Save results/costs.json.
    """
    if not IDS_FILE.exists():
        print("No batch_ids.json found -- run --push first.")
        return

    client    = OpenAI()
    ids       = json.loads(IDS_FILE.read_text())
    rows      = load_rows()
    row_index = {i: row for i, row in enumerate(rows)}

    # Load or init costs ledger
    costs = {}
    if COSTS_FILE.exists():
        costs = json.loads(COSTS_FILE.read_text())

    for prompt_name, batch_id in ids.items():
        b = client.batches.retrieve(batch_id)

        if b.status != "completed":
            print(f"[poll] {prompt_name}: status={b.status} -- not ready yet, skipping.")
            continue

        if not b.output_file_id:
            print(f"[poll] {prompt_name}: completed but no output_file_id.", file=sys.stderr)
            continue

        print(f"[poll] {prompt_name}: downloading ...", end=" ", flush=True)
        raw = client.files.content(b.output_file_id).text
        print("done.")

        # Parse JSONL output -- collect per-request labels and token counts
        parsed = {}
        for line in raw.strip().splitlines():
            obj       = json.loads(line)
            custom_id = obj["custom_id"]
            try:
                content = obj["response"]["body"]["choices"][0]["message"]["content"] or ""
                usage   = obj["response"]["body"].get("usage", {})
                pt      = usage.get("prompt_tokens",     0)
                ct      = usage.get("completion_tokens", 0)
            except Exception:
                content = ""
                pt, ct  = 0, 0

            parsed[custom_id] = {
                "label": extract_label(content),
                "pt":    pt,
                "ct":    ct,
            }

        # Token counts: prefer batch-level aggregate (available for batches
        # created after Sept 7 2025); fall back to summing per-request JSONL usage.
        if b.usage is not None:
            total_prompt_tokens     = b.usage.input_tokens
            total_completion_tokens = b.usage.output_tokens
            token_source = "batch object"
        else:
            total_prompt_tokens     = sum(r["pt"] for r in parsed.values())
            total_completion_tokens = sum(r["ct"] for r in parsed.values())
            token_source = "JSONL sum"

        total_cost = token_cost(total_prompt_tokens, total_completion_tokens)
        print(f"  tokens [{token_source}]: in={total_prompt_tokens} out={total_completion_tokens} cost=${total_cost:.6f}")

        costs[prompt_name] = {
            "prompt_tokens":     total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens":      total_prompt_tokens + total_completion_tokens,
            "cost_usd":          round(total_cost, 6),
        }

        # Write per-prompt CSV: passage | keyword | predicted
        out_path = RESULTS_DIR / f"results_{prompt_name}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["passage", "keyword", "predicted"])

            for idx, row in row_index.items():
                cid       = f"{prompt_name}_{idx}"
                res       = parsed.get(cid, {"label": "invalid"})
                predicted = res["label"]   # 'sym', 'sub', or 'invalid'
                passage   = row["passage"].replace("\n", " ").strip()
                writer.writerow([passage, row["keyword"], predicted])

        n_valid   = sum(1 for r in parsed.values() if r["label"] != "invalid")
        n_invalid = len(parsed) - n_valid
        print(
            f"[poll] {prompt_name}: saved {out_path.name}  |  "
            f"valid={n_valid}  invalid={n_invalid}  "
            f"tokens={total_prompt_tokens}+{total_completion_tokens}  "
            f"cost=${total_cost:.6f}"
        )

    COSTS_FILE.write_text(json.dumps(costs, indent=2))
    print(f"\n[poll] Costs saved -> {COSTS_FILE}")
    print("Run --analyse to generate summary result.csv.")


# ---------------------------------------------------------------------------
# --analyse
# ---------------------------------------------------------------------------
def cmd_analyse():
    """
    Reads results/results_{prompt}.csv + costs.json.
    Joins on row position against passage_keyword_truth.csv for ground truth.
    Writes results/result.csv:
        prompt_type | prompt_tokens | completion_tokens | total_tokens |
        cost_usd    | total_rows    | valid_rows        | invalid_rows |
        correct     | accuracy_pct
    Also prints a formatted summary table to stdout.
    """
    rows      = load_rows()
    row_index = {i: row for i, row in enumerate(rows)}

    costs = {}
    if COSTS_FILE.exists():
        costs = json.loads(COSTS_FILE.read_text())
    else:
        print("[analyse] Warning: costs.json not found -- cost columns will be empty.")

    prompt_names = ["zero_shot", "few_shot", "cot", "tot"]
    summary_rows = []

    print(f"\n{'Prompt':<12} {'Cost ($)':>12} {'Total':>7} {'Valid':>7} {'Invalid':>9} {'Correct':>9} {'Accuracy':>10}")
    print("-" * 72)

    for prompt_name in prompt_names:
        result_csv = RESULTS_DIR / f"results_{prompt_name}.csv"
        if not result_csv.exists():
            print(f"[analyse] {prompt_name}: results CSV not found, skipping.")
            continue

        with open(result_csv, newline="", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        total_rows   = len(reader)
        valid_rows   = 0
        invalid_rows = 0
        correct      = 0

        for idx, result_row in enumerate(reader):
            predicted = result_row.get("predicted", "invalid").strip().lower()
            gt        = row_index[idx]["ground_truth"].strip().lower()

            if predicted == "invalid":
                invalid_rows += 1
            else:
                valid_rows += 1
                if predicted == gt:
                    correct += 1

        accuracy = (correct / valid_rows * 100) if valid_rows > 0 else 0.0

        c = costs.get(prompt_name, {})
        summary_rows.append({
            "prompt_type":       prompt_name,
            "prompt_tokens":     c.get("prompt_tokens",     ""),
            "completion_tokens": c.get("completion_tokens", ""),
            "total_tokens":      c.get("total_tokens",      ""),
            "cost_usd":          c.get("cost_usd",          ""),
            "total_rows":        total_rows,
            "valid_rows":        valid_rows,
            "invalid_rows":      invalid_rows,
            "correct":           correct,
            "accuracy_pct":      round(accuracy, 2),
        })

        cost_str = f"${c['cost_usd']:.6f}" if "cost_usd" in c else "N/A"
        print(
            f"{prompt_name:<12} {cost_str:>12} {total_rows:>7} {valid_rows:>7} "
            f"{invalid_rows:>9} {correct:>9} {accuracy:>9.1f}%"
        )

    fieldnames = [
        "prompt_type", "prompt_tokens", "completion_tokens", "total_tokens",
        "cost_usd", "total_rows", "valid_rows", "invalid_rows",
        "correct", "accuracy_pct",
    ]
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n[analyse] Summary saved -> {SUMMARY_CSV}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="OpenAI Batch API runner for SDG prompts.")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--create",  action="store_true", help="Build batch JSONL files.")
    group.add_argument("--push",    action="store_true", help="Upload and submit batches to OpenAI.")
    group.add_argument("--check",   action="store_true", help="Check status of submitted batches.")
    group.add_argument("--poll",    action="store_true", help="Fetch results, save CSVs + costs.")
    group.add_argument("--analyse", action="store_true", help="Scan result CSVs -> result.csv.")
    args = parser.parse_args()

    if args.create:
        cmd_create()
    elif args.push:
        cmd_push()
    elif args.check:
        cmd_check()
    elif args.poll:
        cmd_poll()
    elif args.analyse:
        cmd_analyse()


if __name__ == "__main__":
    main()