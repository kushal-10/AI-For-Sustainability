#!/usr/bin/env python3
"""
run_batches.py — OpenAI Batch API runner for SDG prompt evaluation.

Models / reasoning modes (8 batches total):
  gpt-5.2  ×  reasoning_effort ∈ {none, low}
  4 prompt types × 2 reasoning modes = 8 batches

--poll and --analyse only operate on batch keys present in batch_ids.json,
so partial runs (e.g. only some prompts pushed) work cleanly.

Usage:
  python run_batches.py --create            # Build batch JSONL files
  python run_batches.py --push              # Upload + submit all batches to OpenAI
  python run_batches.py --push zero_shot    # Submit only one prompt (all reasoning modes)
  python run_batches.py --check             # Print status of all submitted batches
  python run_batches.py --poll              # Fetch results -> CSVs + costs JSON
  python run_batches.py --analyse           # Scan result CSVs -> summary result.csv
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

# ---------------------------------------------------------------------------
# Model / reasoning-mode matrix
# ---------------------------------------------------------------------------
MODEL = "gpt-5.2"

REASONING_MODES = ["low", "medium", "high", "xhigh"]   # extend with "medium" / "high" / "xhigh" as needed
PROMPT_NAMES    = ["zero_shot", "few_shot", "cot", "tot"]

# Cartesian product: (batch_key, prompt_name, reasoning_effort)
CONFIGS = [
    (f"{p}__{r}", p, r)
    for p in PROMPT_NAMES
    for r in REASONING_MODES
]

# Lookup: batch_key -> (prompt_name, reasoning_effort)
CONFIGS_BY_KEY = {key: (p, r) for key, p, r in CONFIGS}


# ---------------------------------------------------------------------------
PRICE_INPUT_PER_TOKEN  = 0.875 / 1_000_000
PRICE_OUTPUT_PER_TOKEN = 7.00 / 1_000_000


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

    Parsing order:
      1. Strip markdown code fences (``` or ```json).
      2. Attempt strict JSON parse and read the first value.
      3. Keyword fallback: only unambiguous single-word matches.
      4. 'invalid' if both words present, neither present, or empty.
    """
    if not content or not content.strip():
        return "invalid"

    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        inner_lines = lines[1:]
        if inner_lines and inner_lines[-1].strip() == "```":
            inner_lines = inner_lines[:-1]
        stripped = "\n".join(inner_lines).strip()

    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            for v in data.values():
                v = str(v).strip().lower()
                if v == "symbolic":
                    return "sym"
                if v == "substantive":
                    return "sub"
    except (json.JSONDecodeError, TypeError):
        pass

    low     = content.lower()
    has_sub = "substantive" in low
    has_sym = "symbolic"    in low

    if has_sub and not has_sym:
        return "sub"
    if has_sym and not has_sub:
        return "sym"

    return "invalid"


# ---------------------------------------------------------------------------
# Helper: load batch_ids.json, abort if missing/empty
# ---------------------------------------------------------------------------
def load_ids(require: bool = True) -> dict:
    if not IDS_FILE.exists():
        if require:
            print("No batch_ids.json found -- run --push first.")
            sys.exit(1)
        return {}
    ids = json.loads(IDS_FILE.read_text())
    if require and not ids:
        print("[error] batch_ids.json is empty -- run --push first.")
        sys.exit(1)
    return ids


# ---------------------------------------------------------------------------
# Helper: resolve configs to submit based on --push argument
# ---------------------------------------------------------------------------
def resolve_configs(only: str | None) -> list:
    if only is None:
        return CONFIGS

    exact     = [c for c in CONFIGS if c[0] == only]
    by_prompt = [c for c in CONFIGS if c[1] == only]
    by_mode   = [c for c in CONFIGS if c[2] == only]

    if exact:     return exact
    if by_prompt: return by_prompt
    if by_mode:   return by_mode

    print(
        f"[push] ERROR: unknown target '{only}'.\n"
        f"  Valid batch keys:      {[c[0] for c in CONFIGS]}\n"
        f"  Valid prompt names:    {PROMPT_NAMES}\n"
        f"  Valid reasoning modes: {REASONING_MODES}",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# --create
# ---------------------------------------------------------------------------
def cmd_create():
    prompts = load_prompts()
    rows    = load_rows()

    for batch_key, prompt_name, reasoning_effort in CONFIGS:
        out_path = BATCH_DIR / f"batch_{batch_key}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for idx, row in enumerate(rows):
                request = {
                    "custom_id": f"{batch_key}_{idx}",
                    "method":    "POST",
                    "url":       "/v1/chat/completions",
                    "body": {
                        "model":                 MODEL,
                        "reasoning_effort":      reasoning_effort,
                        "messages": [
                            {"role": "system", "content": prompts[prompt_name]},
                            {"role": "user",   "content": build_user_message(row)},
                        ],
                        "max_completion_tokens": 30000,
                    },
                }
                f.write(json.dumps(request, ensure_ascii=False) + "\n")

        print(
            f"[create] {batch_key}: wrote {len(rows)} requests "
            f"(model={MODEL}, reasoning_effort={reasoning_effort}) -> {out_path}"
        )

    print(f"\nDone. {len(CONFIGS)} JSONL files created. Run --push to upload and submit.")


# ---------------------------------------------------------------------------
# --push  (sequential: wait for each batch before starting the next)
# ---------------------------------------------------------------------------
def cmd_push(only: str | None = None):
    client  = OpenAI()
    prompts = load_prompts()  # validate prompts.py is loadable early

    ids: dict = load_ids(require=False)
    if ids:
        print(f"[push] Loaded existing batch_ids.json ({len(ids)} entries).")

    target_configs = resolve_configs(only)
    target_keys    = [c[0] for c in target_configs]
    print(f"[push] Targeting {len(target_configs)} batch(es): {target_keys}")

    missing = [
        key for key in target_keys
        if key not in ids and not (BATCH_DIR / f"batch_{key}.jsonl").exists()
    ]
    if missing:
        print(
            f"[push] ERROR: JSONL files missing for: {missing}\n"
            f"       Run --create first to generate them.",
            file=sys.stderr,
        )
        sys.exit(1)

    submitted = 0
    for batch_key, prompt_name, reasoning_effort in target_configs:
        if batch_key in ids:
            old_id = ids[batch_key]
            try:
                existing_status = client.batches.retrieve(old_id).status
            except Exception as e:
                print(
                    f"[push] {batch_key}: batch {old_id} no longer found "
                    f"({e.__class__.__name__}) — re-submitting."
                )
                del ids[batch_key]
                IDS_FILE.write_text(json.dumps(ids, indent=2))
                existing_status = None

            if existing_status in ("validating", "in_progress", "finalizing"):
                print(f"[push] {batch_key}: already running (id={old_id}, status={existing_status}), skipping.")
                continue
            elif existing_status == "completed":
                print(f"[push] {batch_key}: already completed (id={old_id}), skipping.")
                continue
            elif existing_status in ("failed", "expired", "cancelled"):
                print(f"[push] {batch_key}: previous batch {existing_status} — re-submitting.")
                del ids[batch_key]
            # existing_status == None falls through to submit

        jsonl_path = BATCH_DIR / f"batch_{batch_key}.jsonl"
        print(
            f"[push] {batch_key}: uploading {jsonl_path.name} "
            f"(model={MODEL}, reasoning_effort={reasoning_effort}) ...",
            end=" ", flush=True,
        )
        with open(jsonl_path, "rb") as f:
            upload = client.files.create(file=f, purpose="batch")
        print(f"file_id={upload.id}")

        batch = client.batches.create(
            input_file_id     = upload.id,
            endpoint          = "/v1/chat/completions",
            completion_window = "24h",
            metadata          = {
                "prompt":           prompt_name,
                "reasoning_effort": reasoning_effort,
                "model":            MODEL,
            },
        )
        ids[batch_key] = batch.id
        IDS_FILE.write_text(json.dumps(ids, indent=2))
        print(f"[push] {batch_key}: submitted -> id={batch.id}  status={batch.status}")
        submitted += 1

    print(f"\n[push] Done. {submitted} new batch(es) submitted. Run --check to monitor, --poll when complete.")


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------
def cmd_check():
    ids    = load_ids()
    client = OpenAI()

    print(
        f"\n{'Batch Key':<25} {'Batch ID':<32} {'Status':<20} "
        f"{'Completed':>10} {'Failed':>8} {'Total':>7}"
    )
    print("-" * 107)
    for batch_key, batch_id in ids.items():
        b         = client.batches.retrieve(batch_id)
        rc        = b.request_counts
        completed = rc.completed if rc else "?"
        failed    = rc.failed    if rc else "?"
        total     = rc.total     if rc else "?"
        print(
            f"{batch_key:<25} {batch_id:<32} {b.status:<20} "
            f"{str(completed):>10} {str(failed):>8} {str(total):>7}"
        )


# ---------------------------------------------------------------------------
# --poll   (only processes keys present in batch_ids.json)
# ---------------------------------------------------------------------------
def cmd_poll():
    """
    For each completed batch in batch_ids.json:
      1. Download JSONL output.
      2. Write results/results_{batch_key}.csv  (passage | keyword | predicted).
      3. Accumulate token counts and save results/costs.json.

    Only batch keys present in batch_ids.json are processed — partial runs
    (e.g. only some prompts pushed) are handled cleanly.
    """
    ids    = load_ids()
    client = OpenAI()
    rows   = load_rows()
    row_index = {i: row for i, row in enumerate(rows)}

    costs: dict = {}
    if COSTS_FILE.exists():
        costs = json.loads(COSTS_FILE.read_text())

    print(f"[poll] Checking {len(ids)} batch(es) from batch_ids.json ...")
    completed_batches = {}
    for batch_key, batch_id in ids.items():
        try:
            b = client.batches.retrieve(batch_id)
        except Exception:
            print(f"  {batch_key:<25} EXPIRED/NOT FOUND — re-run --push {batch_key}")
            continue
        status_line = f"  {batch_key:<25} {b.status}"
        if b.request_counts:
            status_line += f"  ({b.request_counts.completed}/{b.request_counts.total} requests)"
        print(status_line)
        if b.status == "completed" and b.output_file_id:
            completed_batches[batch_key] = b

    n = len(completed_batches)
    if n == 0:
        print("\n[poll] No completed batches to fetch yet.")
        return
    print(f"\n[poll] Fetching results for {n}/{len(ids)} completed batch(es) ...\n")

    for batch_key, b in completed_batches.items():
        print(f"[poll] {batch_key}: downloading ...", end=" ", flush=True)
        raw = client.files.content(b.output_file_id).text
        print("done.")

        parsed: dict = {}
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

            parsed[custom_id] = {"label": extract_label(content), "pt": pt, "ct": ct}

        def _get_usage(usage, key):
            if usage is None:           return None
            if isinstance(usage, dict): return usage.get(key)
            return getattr(usage, key, None)

        b_input  = _get_usage(b.usage, "input_tokens")
        b_output = _get_usage(b.usage, "output_tokens")

        if b_input is not None and b_output is not None and (b_input + b_output) > 0:
            total_pt, total_ct, token_source = b_input, b_output, "batch object"
        else:
            total_pt     = sum(r["pt"] for r in parsed.values())
            total_ct     = sum(r["ct"] for r in parsed.values())
            token_source = "JSONL sum"

        total_cost = token_cost(total_pt, total_ct)
        print(
            f"  tokens [{token_source}]: "
            f"in={total_pt}  out={total_ct}  cost=${total_cost:.6f}"
        )

        reasoning_effort = batch_key.split("__")[1] if "__" in batch_key else "unknown"

        costs[batch_key] = {
            "model":             MODEL,
            "reasoning_effort":  reasoning_effort,
            "prompt_tokens":     total_pt,
            "completion_tokens": total_ct,
            "total_tokens":      total_pt + total_ct,
            "cost_usd":          round(total_cost, 6),
        }

        out_path = RESULTS_DIR / f"results_{batch_key}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["passage", "keyword", "predicted"])
            for idx, row in row_index.items():
                cid       = f"{batch_key}_{idx}"
                predicted = parsed.get(cid, {"label": "invalid"})["label"]
                passage   = row["passage"].replace("\n", " ").strip()
                writer.writerow([passage, row["keyword"], predicted])

        n_valid   = sum(1 for r in parsed.values() if r["label"] != "invalid")
        n_invalid = len(parsed) - n_valid
        print(
            f"[poll] {batch_key}: saved {out_path.name}  |  "
            f"valid={n_valid}  invalid={n_invalid}"
        )

    COSTS_FILE.write_text(json.dumps(costs, indent=2))
    print(f"\n[poll] Costs saved -> {COSTS_FILE}")
    print("Run --analyse to generate summary result.csv.")


# ---------------------------------------------------------------------------
# --analyse  (only processes keys present in batch_ids.json)
# ---------------------------------------------------------------------------
def cmd_analyse():
    """
    Reads results/results_{batch_key}.csv + costs.json for every key in
    batch_ids.json. Joins on row position against passage_keyword_truth.csv
    for ground truth. Writes results/result.csv.

    Only batch keys present in batch_ids.json are included — partial runs
    produce a valid (partial) summary without crashing.
    """
    ids       = load_ids()
    rows      = load_rows()
    row_index = {i: row for i, row in enumerate(rows)}

    costs: dict = {}
    if COSTS_FILE.exists():
        costs = json.loads(COSTS_FILE.read_text())
    else:
        print("[analyse] Warning: costs.json not found -- cost columns will be empty.")

    summary_rows = []

    print(
        f"\n[analyse] Processing {len(ids)} batch key(s) from batch_ids.json\n"
        f"\n{'Batch Key':<25} {'Reasoning':>10} {'Cost ($)':>12} "
        f"{'Total':>7} {'Valid':>7} {'Invalid':>9} {'Correct':>9} {'Accuracy':>10}"
    )
    print("-" * 93)

    for batch_key in ids:                          # ← driven by batch_ids.json, not CONFIGS
        result_csv = RESULTS_DIR / f"results_{batch_key}.csv"
        if not result_csv.exists():
            print(f"[analyse] {batch_key}: results CSV not found — run --poll first, skipping.")
            continue

        # Derive prompt_name and reasoning_effort from the batch_key
        if batch_key in CONFIGS_BY_KEY:
            prompt_name, reasoning_effort = CONFIGS_BY_KEY[batch_key]
        else:
            parts            = batch_key.split("__")
            prompt_name      = parts[0] if len(parts) >= 1 else batch_key
            reasoning_effort = parts[1] if len(parts) >= 2 else "unknown"

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
        c        = costs.get(batch_key, {})

        summary_rows.append({
            "batch_key":         batch_key,
            "prompt_type":       prompt_name,
            "reasoning_effort":  reasoning_effort,
            "model":             c.get("model", MODEL),
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
            f"{batch_key:<25} {reasoning_effort:>10} {cost_str:>12} {total_rows:>7} "
            f"{valid_rows:>7} {invalid_rows:>9} {correct:>9} {accuracy:>9.1f}%"
        )

    if not summary_rows:
        print("\n[analyse] Nothing to summarise — run --poll first.")
        return

    fieldnames = [
        "batch_key", "prompt_type", "reasoning_effort", "model",
        "prompt_tokens", "completion_tokens", "total_tokens", "cost_usd",
        "total_rows", "valid_rows", "invalid_rows", "correct", "accuracy_pct",
    ]
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n[analyse] Summary saved -> {SUMMARY_CSV}  ({len(summary_rows)} row(s))")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "OpenAI Batch API runner for SDG prompts.\n"
            f"Model: {MODEL}  |  Reasoning modes: {REASONING_MODES}  |  "
            f"Max batches: {len(CONFIGS)}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--create",  action="store_true", help="Build batch JSONL files.")
    group.add_argument(
        "--push",
        metavar="TARGET",
        nargs="?",
        const="__all__",
        default=None,
        help=(
            "Submit batches sequentially. Optionally narrow the target:\n"
            "  --push                    → all 8 batches\n"
            "  --push zero_shot          → both reasoning modes for zero_shot\n"
            "  --push none               → all prompts at reasoning_effort=none\n"
            "  --push zero_shot__none    → exact single batch"
        ),
    )
    group.add_argument("--check",   action="store_true", help="Check status of submitted batches.")
    group.add_argument("--poll",    action="store_true", help="Fetch results, save CSVs + costs.")
    group.add_argument("--analyse", action="store_true", help="Scan result CSVs -> result.csv.")
    args = parser.parse_args()

    if args.create:
        cmd_create()
    elif args.push is not None:
        only = None if args.push == "__all__" else args.push
        cmd_push(only=only)
    elif args.check:
        cmd_check()
    elif args.poll:
        cmd_poll()
    elif args.analyse:
        cmd_analyse()


if __name__ == "__main__":
    main()