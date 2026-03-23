#!/usr/bin/env python3
"""
analyse_accuracy.py — Accuracy + cost summary across all model variants.

Reads:
  passage_keyword_truth.csv           — ground truth (ground_truth col: sym / sub)
  results/results_{key}.csv           — gpt-5.2 predictions (16 files)
  results/costs.json                  — gpt-5.2 costs
  results_4o/results_{prompt}.csv     — gpt-4o predictions (4 files)
  results_4o/costs.json               — gpt-4o costs (if present)

Writes:
  results/accuracy_summary.csv        — one row per model variant

Columns:
  model | prompt | reasoning_effort | total | valid | invalid |
  correct | accuracy_pct | cost_usd | cost_per_correct

Usage:
  python3 analyse_accuracy.py
  python3 analyse_accuracy.py --base path/to/project
"""

import argparse
import csv
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Model matrix
# ---------------------------------------------------------------------------
GPT52_PROMPTS   = ["zero_shot", "few_shot", "cot", "tot"]
GPT52_MODES     = ["low", "medium", "high", "xhigh"]

GPT4O_PROMPTS   = ["zero_shot", "few_shot", "cot", "tot"]


def load_ground_truth(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8") as f:
        return [r["ground_truth"].strip().lower() for r in csv.DictReader(f)]


def load_predictions(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8") as f:
        return [r["predicted"].strip().lower() for r in csv.DictReader(f)]


def accuracy_stats(predictions: list[str], ground_truth: list[str]) -> dict:
    total        = len(predictions)
    valid        = sum(1 for p in predictions if p in ("sym", "sub"))
    invalid      = total - valid
    correct      = sum(
        1 for p, gt in zip(predictions, ground_truth)
        if p in ("sym", "sub") and p == gt
    )
    accuracy_pct = round(correct / valid * 100, 2) if valid > 0 else 0.0
    return {
        "total":        total,
        "valid":        valid,
        "invalid":      invalid,
        "correct":      correct,
        "accuracy_pct": accuracy_pct,
    }


def main(base: Path):
    truth_csv      = base / "passage_keyword_truth.csv"
    results_dir    = base / "results"
    results_4o_dir = base / "results_4o"
    out_csv        = results_dir / "accuracy_summary.csv"

    ground_truth = load_ground_truth(truth_csv)
    print(f"Ground truth rows: {len(ground_truth)}\n")

    # Load costs
    gpt52_costs: dict = {}
    costs_path = results_dir / "costs.json"
    if costs_path.exists():
        gpt52_costs = json.loads(costs_path.read_text())

    gpt4o_costs: dict = {}
    costs_4o_path = results_4o_dir / "costs.json"
    if costs_4o_path.exists():
        gpt4o_costs = json.loads(costs_4o_path.read_text())

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    col_w = [20, 12, 10, 7, 7, 9, 9, 10, 12, 16]
    headers = ["Model", "Prompt", "Reasoning", "Total", "Valid", "Invalid",
               "Correct", "Acc (%)", "Cost ($)", "$/Correct"]
    header_line = "  ".join(h.ljust(col_w[i]) for i, h in enumerate(headers))
    print(header_line)
    print("-" * len(header_line))

    summary_rows = []

    def process(model_name, prompt, reasoning_effort, csv_path, cost_entry):
        if not csv_path.exists():
            print(f"  [MISSING] {csv_path.name}")
            return

        preds  = load_predictions(csv_path)
        stats  = accuracy_stats(preds, ground_truth)
        cost   = cost_entry.get("cost_usd", "") if cost_entry else ""
        cost_per_correct = (
            round(cost / stats["correct"], 6) if cost and stats["correct"] > 0 else ""
        )

        row = {
            "model":            model_name,
            "prompt":           prompt,
            "reasoning_effort": reasoning_effort,
            **stats,
            "cost_usd":         cost,
            "cost_per_correct": cost_per_correct,
        }
        summary_rows.append(row)

        cost_str = f"${cost:.6f}"      if isinstance(cost, float)            else "N/A"
        cpc_str  = f"${cost_per_correct:.6f}" if isinstance(cost_per_correct, float) else "N/A"

        vals = [
            model_name, prompt, reasoning_effort,
            str(stats["total"]), str(stats["valid"]), str(stats["invalid"]),
            str(stats["correct"]), f"{stats['accuracy_pct']:.1f}%",
            cost_str, cpc_str,
        ]
        print("  ".join(v.ljust(col_w[i]) for i, v in enumerate(vals)))

    # ------------------------------------------------------------------
    # gpt-5.2
    # ------------------------------------------------------------------
    print("\n── gpt-5.2 ──────────────────────────────────────────────────────────")
    for prompt in GPT52_PROMPTS:
        for mode in GPT52_MODES:
            key      = f"{prompt}__{mode}"
            csv_path = results_dir / f"results_{key}.csv"
            cost_entry = gpt52_costs.get(key, {})
            process("gpt-5.2", prompt, mode, csv_path, cost_entry)

    # ------------------------------------------------------------------
    # gpt-4o
    # ------------------------------------------------------------------
    print("\n── gpt-4o ───────────────────────────────────────────────────────────")
    for prompt in GPT4O_PROMPTS:
        csv_path   = results_4o_dir / f"results_{prompt}.csv"
        cost_entry = gpt4o_costs.get(prompt, {})
        process("gpt-4o", prompt, "—", csv_path, cost_entry)

    # ------------------------------------------------------------------
    # Totals / best
    # ------------------------------------------------------------------
    if summary_rows:
        print("\n── Summary ──────────────────────────────────────────────────────────")
        valid_rows = [r for r in summary_rows if isinstance(r["accuracy_pct"], float)]

        best       = max(valid_rows, key=lambda r: r["accuracy_pct"])
        cheapest   = min(
            [r for r in valid_rows if isinstance(r["cost_usd"], float)],
            key=lambda r: r["cost_usd"],
        )
        best_value = min(
            [r for r in valid_rows if isinstance(r["cost_per_correct"], float)],
            key=lambda r: r["cost_per_correct"],
        )

        print(f"  Highest accuracy:  {best['model']} / {best['prompt']} / {best['reasoning_effort']}  →  {best['accuracy_pct']}%")
        print(f"  Lowest cost:       {cheapest['model']} / {cheapest['prompt']} / {cheapest['reasoning_effort']}  →  ${cheapest['cost_usd']:.6f}")
        print(f"  Best $/correct:    {best_value['model']} / {best_value['prompt']} / {best_value['reasoning_effort']}  →  ${best_value['cost_per_correct']:.6f}")

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    fieldnames = [
        "model", "prompt", "reasoning_effort",
        "total", "valid", "invalid", "correct", "accuracy_pct",
        "cost_usd", "cost_per_correct",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved -> {out_csv}  ({len(summary_rows)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Accuracy + cost summary for all model variants.")
    parser.add_argument(
        "--base",
        type=Path,
        default=Path(__file__).parent,
        help="Base project directory (default: script directory).",
    )
    args = parser.parse_args()
    main(args.base)