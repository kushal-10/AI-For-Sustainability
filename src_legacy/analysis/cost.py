#!/usr/bin/env python3
"""
Estimate token counts and max costs for gpt-4.1-mini over batch JSONL files.

- Defaults:
  - Looks in data/batches/sdgs and data/batches/tech for *.jsonl
  - Pricing (USD per 1M tokens):
      BATCH:    input = 0.20, output = 0.80
      STANDARD: input = 0.40, output = 1.60
- Tokenization:
  - Tries tiktoken (o200k_base). If unavailable, falls back to ~4 chars/token heuristic.
- Output:
  - Prints per-directory and TOTAL summary.
  - Optionally writes JSON summary (default: data/batches/cost_estimate.json).
"""

import os
import json
import glob
import argparse
from typing import Dict, Any, List, Tuple

# ------- Defaults -------
DEFAULT_DIRS = ["data/batches/sdgs", "data/batches/tech"]
DEFAULT_GLOB = "*.jsonl"
OUT_JSON = "data/batches/cost_estimate.json"

# Pricing: USD per 1M tokens
BATCH_PRICING = {"input": 0.20, "output": 0.80}
STANDARD_PRICING = {"input": 0.40, "output": 1.60}

# ------- Tokenizer (tiktoken if available) -------
class TokenCounter:
    def __init__(self):
        self._tok = None
        try:
            import tiktoken  # type: ignore
            # Prefer a modern encoder; o200k_base matches 4.x/omni families
            try:
                self._tok = tiktoken.get_encoding("o200k_base")
            except Exception:
                # Fallback to a widely available encoding
                self._tok = tiktoken.get_encoding("cl100k_base")
            self._use_tiktoken = True
        except Exception:
            self._use_tiktoken = False

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._use_tiktoken and self._tok:
            try:
                return len(self._tok.encode(text))
            except Exception:
                pass
        # Heuristic fallback: ~4 chars per token
        # Add a small overhead for newlines/spaces in typical prompts
        return max(1, int(len(text) / 4.0))

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def count_request_input_tokens(obj: Dict[str, Any], tc: TokenCounter) -> int:
    """
    Sum tokens across all message contents in a Chat Completions-style request:
    {
      "method": "POST",
      "url": "/v1/chat/completions",
      "body": {
        "model": "...",
        "messages": [{"role": "...", "content": "..."}],
        "max_tokens": 120,
        ...
      }
    }
    """
    body = obj.get("body", {})
    messages = body.get("messages", [])
    total = 0
    for m in messages:
        content = m.get("content", "")
        total += tc.count(str(content))
    return total

def collect_from_dir(dir_path: str, pattern: str, tc: TokenCounter) -> Tuple[int, int, int]:
    """
    Returns: (num_requests, total_input_tokens, total_max_output_tokens)
    """
    num_requests = 0
    input_tokens = 0
    max_output_tokens = 0

    files = sorted(glob.glob(os.path.join(dir_path, pattern)))
    for fp in files:
        for obj in read_jsonl(fp):
            # only consider chat completion requests
            if obj.get("url") != "/v1/chat/completions":
                continue
            num_requests += 1
            input_tokens += count_request_input_tokens(obj, tc)
            max_output_tokens += int(obj.get("body", {}).get("max_tokens", 0) or 0)

    return num_requests, input_tokens, max_output_tokens

def cost_for(pricing: Dict[str, float], input_tokens: int, output_tokens: int) -> Dict[str, float]:
    minv = 1_000_000.0
    input_cost = (input_tokens / minv) * pricing["input"]
    output_cost = (output_tokens / minv) * pricing["output"]
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }

def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", "_")

def fmt_usd(x: float) -> str:
    return f"${x:,.2f}"

def main():
    ap = argparse.ArgumentParser(description="Estimate tokens and costs for gpt-4.1-mini over batch JSONL files.")
    ap.add_argument("--dirs", nargs="+", default=DEFAULT_DIRS, help="Directories to scan (default: data/batches/sdgs data/batches/tech)")
    ap.add_argument("--glob", default=DEFAULT_GLOB, help="Glob pattern (default: *.jsonl)")
    ap.add_argument("--out-json", default=OUT_JSON, help="Write JSON summary here (default: data/batches/cost_estimate.json)")
    args = ap.parse_args()

    tc = TokenCounter()

    results = {}
    total_requests = 0
    total_input = 0
    total_max_output = 0

    for d in args.dirs:
        n, tin, toutmax = collect_from_dir(d, args.glob, tc)
        total_requests += n
        total_input += tin
        total_max_output += toutmax

        dir_batch = cost_for(BATCH_PRICING, tin, toutmax)
        dir_std   = cost_for(STANDARD_PRICING, tin, toutmax)

        results[d] = {
            "requests": n,
            "input_tokens": tin,
            "max_output_tokens": toutmax,
            "batch_pricing_per_1M": BATCH_PRICING,
            "standard_pricing_per_1M": STANDARD_PRICING,
            "cost_batch": dir_batch,
            "cost_standard": dir_std,
        }

    # Totals
    totals_batch = cost_for(BATCH_PRICING, total_input, total_max_output)
    totals_std   = cost_for(STANDARD_PRICING, total_input, total_max_output)

    results["TOTAL"] = {
        "requests": total_requests,
        "input_tokens": total_input,
        "max_output_tokens": total_max_output,
        "batch_pricing_per_1M": BATCH_PRICING,
        "standard_pricing_per_1M": STANDARD_PRICING,
        "cost_batch": totals_batch,
        "cost_standard": totals_std,
    }

    # ---- Print summary ----
    print("\n=== gpt-4.1-mini Token & Cost Estimate ===")
    for d in args.dirs + ["TOTAL"]:
        r = results.get(d, {})
        if not r:
            continue
        print(f"\n[{d}]")
        print(f"Requests            : {fmt_int(r['requests'])}")
        print(f"Input tokens        : {fmt_int(r['input_tokens'])}")
        print(f"Max output tokens   : {fmt_int(r['max_output_tokens'])}")
        cb = r["cost_batch"]
        cs = r["cost_standard"]
        print(f"Batch mode cost     : {fmt_usd(cb['total_cost'])}  (input {fmt_usd(cb['input_cost'])} + output {fmt_usd(cb['output_cost'])})")
        print(f"Standard mode cost  : {fmt_usd(cs['total_cost'])}  (input {fmt_usd(cs['input_cost'])} + output {fmt_usd(cs['output_cost'])})")

    # ---- Write JSON ----
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Wrote JSON summary -> {args.out_json}")

if __name__ == "__main__":
    main()
