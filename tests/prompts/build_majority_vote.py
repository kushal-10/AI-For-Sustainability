 #!/usr/bin/env python3
"""
build_majority_vote.py — Aggregate classifications from all model variants.

Reads:
  results/results_{batch_key}.csv      — gpt-5.2 variants (16 files)
  results_4o/results_{prompt}.csv      — gpt-4o variants  (4 files)
  passage_keyword_truth.csv            — ground truth

Both CSV formats must have columns: passage | keyword | predicted
Ground truth CSV must have column:  ground_truth  (values: sym / sub)

Writes:
  results/majority_vote.csv

Columns:
  row_id | passage | keyword | ground_truth
  | <model_1> | <model_2> ... <model_20>          (one col per model)
  | models_agreeing_with_gt                        (count of models matching GT)
  | majority_vote                                  (sym / sub / tie)
  | majority_correct                               (1 if majority == GT, 0 otherwise)

Usage:
  python3 build_majority_vote.py
  python3 build_majority_vote.py --base path/to/project   # if run from elsewhere
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# gpt-5.2 batch keys (16 total: 4 prompts × 4 reasoning modes)
# ---------------------------------------------------------------------------
GPT52_PROMPT_NAMES    = ["zero_shot", "few_shot", "cot", "tot"]
GPT52_REASONING_MODES = ["low", "medium", "high", "xhigh"]

GPT52_MODELS = [
    (f"{p}__{r}", f"gpt52_{p}__{r}")
    for p in GPT52_PROMPT_NAMES
    for r in GPT52_REASONING_MODES
]

# ---------------------------------------------------------------------------
# gpt-4o prompt names (4 total)
# ---------------------------------------------------------------------------
GPT4O_PROMPT_NAMES = ["zero_shot", "few_shot", "cot", "tot"]

GPT4O_MODELS = [
    (name, f"gpt4o_{name}")
    for name in GPT4O_PROMPT_NAMES
]


def load_predictions(csv_path: Path) -> list[str]:
    """Return list of predicted labels in row order."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [row["predicted"].strip().lower() for row in csv.DictReader(f)]


def majority(labels: list[str]) -> str:
    """
    Return 'sym', 'sub', or 'tie'.
    Only counts valid labels (sym / sub); ignores 'invalid'.
    """
    valid = [l for l in labels if l in ("sym", "sub")]
    if not valid:
        return "tie"
    counts = Counter(valid)
    top    = counts.most_common(2)
    if len(top) == 1 or top[0][1] > top[1][1]:
        return top[0][0]
    return "tie"


def main(base: Path):
    results_dir    = base / "results"
    results_4o_dir = base / "results_4o"
    truth_csv      = base / "passage_keyword_truth.csv"
    out_csv        = results_dir / "majority_vote.csv"

    # ------------------------------------------------------------------
    # Load ground truth
    # ------------------------------------------------------------------
    with open(truth_csv, newline="", encoding="utf-8") as f:
        truth_rows = list(csv.DictReader(f))

    n = len(truth_rows)
    print(f"Ground truth rows: {n}")

    # ------------------------------------------------------------------
    # Load all model predictions
    # ------------------------------------------------------------------
    all_models   = []   # list of (col_name, predictions_list)
    missing_files = []

    # gpt-5.2
    for batch_key, col_name in GPT52_MODELS:
        path = results_dir / f"results_{batch_key}.csv"
        if not path.exists():
            print(f"  [MISSING] {path.name}")
            missing_files.append(col_name)
            all_models.append((col_name, ["invalid"] * n))
        else:
            preds = load_predictions(path)
            print(f"  [OK] {path.name}  ({len(preds)} rows)")
            all_models.append((col_name, preds))

    # gpt-4o
    for prompt_name, col_name in GPT4O_MODELS:
        path = results_4o_dir / f"results_{prompt_name}.csv"
        if not path.exists():
            print(f"  [MISSING] {path.name}")
            missing_files.append(col_name)
            all_models.append((col_name, ["invalid"] * n))
        else:
            preds = load_predictions(path)
            print(f"  [OK] {path.name}  ({len(preds)} rows)")
            all_models.append((col_name, preds))

    total_models = len(all_models)
    print(f"\nTotal models loaded: {total_models}  ({len(missing_files)} missing)")
    if missing_files:
        print(f"  Missing: {missing_files}")

    # ------------------------------------------------------------------
    # Build output rows
    # ------------------------------------------------------------------
    output_rows = []

    for idx, truth_row in enumerate(truth_rows):
        gt      = truth_row["ground_truth"].strip().lower()
        passage = truth_row["passage"].replace("\n", " ").strip()
        keyword = truth_row.get("keyword", "").strip()

        # Collect per-model predictions for this row
        row_preds = {col: preds[idx] if idx < len(preds) else "invalid"
                     for col, preds in all_models}

        # How many models agree with ground truth
        agree_count = sum(
            1 for pred in row_preds.values()
            if pred == gt
        )

        # Majority vote across all models
        maj = majority(list(row_preds.values()))

        # Is majority correct?
        maj_correct = 1 if maj == gt else (0 if maj != "tie" else "tie")

        out_row = {
            "row_id":       idx,
            "passage":      passage,
            "keyword":      keyword,
            "ground_truth": gt,
        }
        out_row.update(row_preds)
        out_row["models_agreeing_with_gt"] = agree_count
        out_row["majority_vote"]           = maj
        out_row["majority_correct"]        = maj_correct

        output_rows.append(out_row)

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    model_col_names = [col for col, _ in all_models]
    fieldnames = (
        ["row_id", "passage", "keyword", "ground_truth"]
        + model_col_names
        + ["models_agreeing_with_gt", "majority_vote", "majority_correct"]
    )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------
    valid_rows       = [r for r in output_rows if r["majority_vote"] != "tie"]
    tie_rows         = [r for r in output_rows if r["majority_vote"] == "tie"]
    correct_rows     = [r for r in valid_rows   if r["majority_correct"] == 1]
    majority_acc     = len(correct_rows) / len(valid_rows) * 100 if valid_rows else 0

    avg_agree        = sum(r["models_agreeing_with_gt"] for r in output_rows) / n
    full_agree_rows  = [r for r in output_rows if r["models_agreeing_with_gt"] == total_models]
    full_disagree    = [r for r in output_rows if r["models_agreeing_with_gt"] == 0]

    print(f"\n{'='*55}")
    print(f"  Rows:                     {n}")
    print(f"  Total models:             {total_models}")
    print(f"  Majority accuracy:        {majority_acc:.1f}%  ({len(correct_rows)}/{len(valid_rows)} non-tie rows)")
    print(f"  Tie rows:                 {len(tie_rows)}")
    print(f"  Avg models agreeing GT:   {avg_agree:.1f} / {total_models}")
    print(f"  All {total_models} models agree:      {len(full_agree_rows)} rows")
    print(f"  No models agree with GT:  {len(full_disagree)} rows")
    if full_disagree:
        ids = [str(r["row_id"]) for r in full_disagree]
        print("    Row IDs: " + ", ".join(ids))
    print(f"{'='*55}")
    print(f"\nSaved -> {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build majority vote CSV across all model variants.")
    parser.add_argument(
        "--base",
        type=Path,
        default=Path(__file__).parent,
        help="Base project directory (default: script's directory). "
             "Should contain results/, results_4o/, passage_keyword_truth.csv",
    )
    args = parser.parse_args()
    main(args.base)