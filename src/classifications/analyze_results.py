"""
analyze_results.py — Compute distribution stats and (optionally) accuracy across
all classification runs defined in config.json.

For each config entry with collected results, reports:
  • Symbolic / Substantive counts and share per domain
  • Overall totals across all models

If --ground-truth is provided (tests/prompts/passage_keyword_truth.csv), also
computes accuracy (correct / valid) for each entry against the ground truth labels.

Ground truth CSV format:  passage, keyword, ground_truth
  ground_truth values:  "sym" or "sub"

Results CSV format (per entry):  custom_id, pattern, predicted
  predicted values:  "symbolic" or "substantive"

Usage:
    python3 src/classifications/analyze_results.py
    python3 src/classifications/analyze_results.py --ground-truth tests/prompts/passage_keyword_truth.csv
    python3 src/classifications/analyze_results.py --results-base data/classifications/results
"""

import argparse
import csv
import json
from pathlib import Path

from src.classifications.batch_builder import load_config

RESULTS_BASE = "data/classifications/results"
GT_PATH      = "tests/prompts/passage_keyword_truth.csv"


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_results(results_base: str, entry_id: str, domain: str) -> dict | None:
    """Load {custom_id: {pattern: label}} for one entry/domain. Returns None if missing."""
    path = Path(results_base) / entry_id / f"{domain}_results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_ground_truth(gt_path: str) -> list[dict]:
    """
    Load ground truth CSV.
    Returns list of {passage, keyword, ground_truth} dicts.
    ground_truth is normalised to "symbolic" / "substantive".
    """
    mapping = {"sym": "symbolic", "sub": "substantive",
               "symbolic": "symbolic", "substantive": "substantive"}
    rows = []
    with open(gt_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gt = r.get("ground_truth", "").strip().lower()
            rows.append({
                "passage":      r.get("passage", ""),
                "keyword":      r.get("keyword", ""),
                "ground_truth": mapping.get(gt, gt),
            })
    return rows


# ── Distribution stats ─────────────────────────────────────────────────────────

def distribution_stats(results: dict) -> dict:
    """
    Compute symbolic/substantive counts across all patterns in results.
    results: {custom_id: {pattern: label}}
    """
    symbolic = substantive = invalid = 0
    for label_map in results.values():
        if not isinstance(label_map, dict):
            invalid += 1
            continue
        for label in label_map.values():
            if label == "symbolic":
                symbolic += 1
            elif label == "substantive":
                substantive += 1
            else:
                invalid += 1
    total = symbolic + substantive
    return {
        "symbolic":    symbolic,
        "substantive": substantive,
        "invalid":     invalid,
        "total":       total,
        "sub_share":   round(substantive / total * 100, 1) if total > 0 else 0.0,
    }


# ── Accuracy ───────────────────────────────────────────────────────────────────

def accuracy_stats(results: dict, ground_truth: list[dict]) -> dict:
    """
    Match results to ground truth by keyword (pattern) position and compute accuracy.

    The ground truth CSV and results are aligned by keyword order: the Nth entry in
    ground truth corresponds to the Nth matched pattern label in results (flattened
    in insertion order). This mirrors the approach in tests/prompts/analysis.py.
    """
    # Flatten results to an ordered list of labels
    predicted: list[str] = []
    for label_map in results.values():
        if isinstance(label_map, dict):
            predicted.extend(label_map.values())

    gt_labels = [r["ground_truth"] for r in ground_truth]

    n     = min(len(predicted), len(gt_labels))
    valid = sum(1 for p in predicted[:n] if p in ("symbolic", "substantive"))
    correct = sum(
        1 for p, gt in zip(predicted[:n], gt_labels[:n])
        if p in ("symbolic", "substantive") and p == gt
    )
    return {
        "n_compared": n,
        "valid":      valid,
        "correct":    correct,
        "accuracy":   round(correct / valid * 100, 2) if valid > 0 else 0.0,
    }


# ── Report ─────────────────────────────────────────────────────────────────────

_SEP = "-" * 110

_HDR_DIST = (
    f"{'CONFIG ID':<35} {'DOM':<5} {'SYMBOLIC':>9} {'SUBST.':>9} "
    f"{'TOTAL':>8} {'SUB%':>7} {'INVALID':>8}"
)

_HDR_ACC = (
    f"{'CONFIG ID':<35} {'DOM':<5} {'COMPARED':>9} {'VALID':>7} "
    f"{'CORRECT':>8} {'ACC%':>7}"
)


def analyze_all(
    config_path:  str        = "src/classifications/config.json",
    results_base: str        = RESULTS_BASE,
    gt_path:      str | None = None,
    save_summary: bool       = True,
) -> None:
    config     = load_config(config_path)
    ground_truth = load_ground_truth(gt_path) if gt_path else None

    dist_rows: list[dict] = []
    acc_rows:  list[dict] = []

    for entry in config:
        entry_id = entry["id"]
        for domain in ("sdg", "tech"):
            results = load_results(results_base, entry_id, domain)
            if results is None:
                continue

            d = distribution_stats(results)
            dist_rows.append({
                "config_id":   entry_id,
                "model":       entry["model"],
                "reasoning":   entry.get("reasoning_effort") or "—",
                "prompt_type": entry["prompt_type"],
                "domain":      domain,
                **d,
            })

            if ground_truth:
                a = accuracy_stats(results, ground_truth)
                acc_rows.append({
                    "config_id":   entry_id,
                    "model":       entry["model"],
                    "reasoning":   entry.get("reasoning_effort") or "—",
                    "prompt_type": entry["prompt_type"],
                    "domain":      domain,
                    **a,
                })

    # ── Print distribution table ───────────────────────────────────────────────
    print("\n=== Distribution (symbolic vs substantive) ===\n")
    if not dist_rows:
        print("No results found. Run collect_results.py first.")
        return

    print(_HDR_DIST)
    print(_SEP)
    for r in dist_rows:
        print(
            f"{r['config_id']:<35} {r['domain']:<5} {r['symbolic']:>9,} "
            f"{r['substantive']:>9,} {r['total']:>8,} {r['sub_share']:>6.1f}% {r['invalid']:>8,}"
        )
    print(_SEP)

    # Totals per domain
    for domain in ("sdg", "tech"):
        rows = [r for r in dist_rows if r["domain"] == domain]
        if rows:
            print(
                f"  {domain.upper()} total across {len(rows)} entries — "
                f"symbolic: {sum(r['symbolic'] for r in rows):,}  "
                f"substantive: {sum(r['substantive'] for r in rows):,}"
            )

    # ── Print accuracy table ───────────────────────────────────────────────────
    if acc_rows:
        print("\n=== Accuracy vs Ground Truth ===\n")
        print(_HDR_ACC)
        print(_SEP)
        for r in acc_rows:
            print(
                f"{r['config_id']:<35} {r['domain']:<5} {r['n_compared']:>9,} "
                f"{r['valid']:>7,} {r['correct']:>8,} {r['accuracy']:>6.2f}%"
            )
        print(_SEP)

    # ── Save CSV summary ───────────────────────────────────────────────────────
    if save_summary and dist_rows:
        out_dir = Path(results_base)
        out_dir.mkdir(parents=True, exist_ok=True)

        dist_csv = out_dir / "distribution_summary.csv"
        _write_csv(dist_csv, dist_rows)
        print(f"\nSaved distribution summary → {dist_csv}")

        if acc_rows:
            acc_csv = out_dir / "accuracy_summary.csv"
            _write_csv(acc_csv, acc_rows)
            print(f"Saved accuracy summary    → {acc_csv}")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Analyze classification results.")
    ap.add_argument("--config",       default="src/classifications/config.json")
    ap.add_argument("--results-base", default=RESULTS_BASE)
    ap.add_argument("--ground-truth", default=None,
                    help="Path to passage_keyword_truth.csv for accuracy scoring")
    ap.add_argument("--no-save",      action="store_true", help="Skip saving CSV summaries")
    args = ap.parse_args()

    analyze_all(
        config_path  = args.config,
        results_base = args.results_base,
        gt_path      = args.ground_truth,
        save_summary = not args.no_save,
    )


if __name__ == "__main__":
    main()
