#!/usr/bin/env python3
"""
Analyze and (optionally) auto-fix parse-level errors from Batch results.

- Reads:
    data/batches/results/results_errors.json   (custom_id -> {"level":"parse","raw_content": ...})
    data/batches/results/results_map.json      (custom_id -> parsed JSON)  [optional, for merging]
- Detects common causes:
    • code fences ```...```
    • single quotes (Python dict style)
    • invalid JSON escapes like \s, \w, \d, \S, \W, \+ (anything not in ["\/bfnrtu"])
    • trailing commas before } or ]
    • extra text around the JSON (e.g., prose, “Expected output:”)
    • multiple JSON blocks / none found
- Tries to salvage by:
    1) stripping fences & extracting first {...}
    2) escaping invalid backslash sequences: \X  -> \\X  (when X not a valid JSON escape)
    3) normalizing single quotes to double quotes for keys/values
    4) removing trailing commas
    5) json.loads; if still failing, regex-pick key/value pairs of the form
       "some regex": "symbolic|substantive"
- Writes:
    data/batches/results/results_parse_analysis.json  (summary + per-item diagnosis)
    data/batches/results/results_fixed_map.json       (custom_id -> salvaged dict)
    data/batches/results/results_all_map.json         (merged original + fixed)
- CLI:
    --dry-run    : analyze only; do not write fixed/merged maps
    --errors     : path to errors json (default as above)
    --good       : path to existing good results (optional)
    --outdir     : directory for outputs (default: data/batches/results)

Usage:
  python3 src/batching/analyze_parse_errors.py
"""
import os
import re
import json
import argparse
from typing import Dict, Any, Tuple

DEFAULT_ERRORS = "data/batches/results/results_errors.json"
DEFAULT_GOOD   = "data/batches/results/results_map.json"
DEFAULT_OUTDIR = "data/batches/results"

# --------- Helpers ---------

CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.DOTALL)

def strip_code_fences(s: str) -> str:
    return CODE_FENCE_RE.sub("", s)

def has_code_fence(s: str) -> bool:
    return "```" in s

def find_first_json_object(s: str) -> Tuple[str, str]:
    """
    Return (json_block, context) where json_block is the first {...} found
    and context is the full original string (for diagnostics).
    If none, return ("","").
    """
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return "", ""
    return s[m.start():m.end()], s

def has_trailing_commas(s: str) -> bool:
    return re.search(r",\s*[}\]]", s) is not None

def fix_trailing_commas(s: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", s)

def count_invalid_json_escapes(s: str) -> int:
    # Backslash not followed by a valid JSON escape char: " \/ b f n r t u
    # This flags things like \s, \w, \d, \+, etc.
    return len(re.findall(r'\\(?!["\\/bfnrtu])', s))

def fix_invalid_json_escapes(s: str) -> str:
    # Turn \X into \\X for any X not in the valid JSON escape set
    return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)

def looks_python_dict_style(s: str) -> bool:
    return bool(re.search(r"'[^']*'\s*:", s) or re.search(r":\s*'[^']*'", s))

def normalize_single_quotes_to_double(s: str) -> str:
    # Keys: 'key':  -> "key":
    s = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'(?=\s*:)", r'"\1"', s)
    # Values: : 'value'
    s = re.sub(r":\s*'([^'\\]*(?:\\.[^'\\]*)*)'", r': "\1"', s)
    return s

def summarize_context_noise(original: str, json_block: str) -> bool:
    # If there is significant non-JSON text around, mark as noisy context
    return len(original.strip()) > len(json_block.strip())

def try_json_loads(s: str):
    try:
        return json.loads(s), None
    except Exception as e:
        return None, str(e)

def salvage_by_regex_pairs(s: str):
    """
    Last-resort: extract pairs like "REGEX": "symbolic|substantive".
    Returns dict or None.
    """
    pairs = re.findall(r'"([^"]+)"\s*:\s*"(symbolic|substantive)"', s, flags=re.IGNORECASE)
    if not pairs:
        return None
    out = {}
    for k, v in pairs:
        v_norm = "substantive" if v.lower().startswith("sub") else "symbolic"
        out[k] = v_norm
    return out

# --------- Diagnosis & Salvage ---------

def diagnose(raw: str) -> Dict[str, Any]:
    flags = {
        "code_fence": has_code_fence(raw),
        "single_quotes": looks_python_dict_style(raw),
        "invalid_escapes_count": count_invalid_json_escapes(raw),
        "trailing_commas": has_trailing_commas(raw),
        "no_json_found": False,
        "context_noise": False,
        "extra_text_likely": ("Expected output" in raw) or ("Passage:" in raw) or ("SDG_HITS" in raw) or ("TECH_HITS" in raw),
    }
    stripped = strip_code_fences(raw)
    block, ctx = find_first_json_object(stripped)
    if not block:
        flags["no_json_found"] = True
    else:
        flags["context_noise"] = summarize_context_noise(stripped, block)
    return flags

def salvage(raw: str) -> Tuple[Dict[str, Any], str]:
    """
    Attempt to repair and parse; return (obj, reason_if_failed).
    """
    s = strip_code_fences(raw)
    block, _ = find_first_json_object(s)
    if not block:
        # Nothing that looks like JSON; try last-resort extraction
        obj = salvage_by_regex_pairs(s)
        return (obj, None) if obj is not None else (None, "no_json_found")

    candidate = block

    # Fix invalid escapes first (\s, \w, \d, \+ ...)
    if count_invalid_json_escapes(candidate) > 0:
        candidate = fix_invalid_json_escapes(candidate)

    # Normalize Pythonic single quotes in keys/values
    if looks_python_dict_style(candidate):
        candidate = normalize_single_quotes_to_double(candidate)

    # Remove trailing commas
    if has_trailing_commas(candidate):
        candidate = fix_trailing_commas(candidate)

    # Try direct loads
    obj, err = try_json_loads(candidate)
    if obj is not None:
        return obj, None

    # Last resort: scrape pairs
    fallback = salvage_by_regex_pairs(candidate)
    if fallback is not None:
        return fallback, None

    return None, f"json_loads_failed: {err}"

# --------- Main ---------

def main():
    ap = argparse.ArgumentParser(description="Analyze parse-level errors and attempt to auto-fix them.")
    ap.add_argument("--errors", default=DEFAULT_ERRORS, help="Path to results_errors.json")
    ap.add_argument("--good", default=DEFAULT_GOOD, help="Path to existing results_map.json (optional)")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Directory to write analysis and fixes")
    ap.add_argument("--dry-run", action="store_true", help="Analyze only; do not write fixed/merged maps")
    args = ap.parse_args()

    if not os.path.exists(args.errors):
        raise SystemExit(f"[ERR] errors file not found: {args.errors}")

    with open(args.errors, "r", encoding="utf-8") as f:
        errors_map = json.load(f)

    good_map = {}
    if os.path.exists(args.good):
        try:
            with open(args.good, "r", encoding="utf-8") as f:
                good_map = json.load(f)
        except Exception:
            good_map = {}

    analysis = {
        "tot_errors": len(errors_map),
        "by_flag": {
            "code_fence": 0,
            "single_quotes": 0,
            "invalid_escapes_gt0": 0,
            "trailing_commas": 0,
            "no_json_found": 0,
            "context_noise": 0,
            "extra_text_likely": 0,
        },
        "items": {}
    }

    fixed_map = {}
    failed = {}

    for cid, info in errors_map.items():
        raw = info.get("raw_content", "") or ""
        diag = diagnose(raw)

        # tally
        for k in analysis["by_flag"].keys():
            if k == "invalid_escapes_gt0":
                if diag["invalid_escapes_count"] > 0:
                    analysis["by_flag"][k] += 1
            else:
                if isinstance(diag.get(k), bool) and diag[k]:
                    analysis["by_flag"][k] += 1

        obj, fail = salvage(raw)
        item = {
            "diagnosis": diag,
            "salvaged": obj is not None,
            "reason": None if obj is not None else fail
        }
        analysis["items"][cid] = item

        if obj is not None:
            fixed_map[cid] = obj
        else:
            failed[cid] = {"raw_preview": raw[:400], "reason": fail}

    # Summaries
    analysis["fixed_count"] = len(fixed_map)
    analysis["failed_count"] = len(failed)

    # Output
    os.makedirs(args.outdir, exist_ok=True)
    analysis_path = os.path.join(args.outdir, "results_parse_analysis.json")
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote analysis -> {analysis_path}")
    print(f"     total errors: {analysis['tot_errors']}, fixed: {analysis['fixed_count']}, failed: {analysis['failed_count']}")
    print("     flags:", {k: v for k, v in analysis["by_flag"].items()})

    if not args.dry_run:
        fixed_path = os.path.join(args.outdir, "results_fixed_map.json")
        with open(fixed_path, "w", encoding="utf-8") as f:
            json.dump(fixed_map, f, ensure_ascii=False, indent=2)
        print(f"[OK] Wrote fixed map -> {fixed_path}  (items: {len(fixed_map)})")

        # Merge with good_map (do not overwrite existing good entries)
        merged = dict(good_map)
        merged.update({k: v for k, v in fixed_map.items() if k not in merged})
        merged_path = os.path.join(args.outdir, "results_all_map.json")
        with open(merged_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"[OK] Wrote merged map -> {merged_path}  (items: {len(merged)})")

        # Persist remaining failures for manual inspection
        if failed:
            remain_path = os.path.join(args.outdir, "results_parse_unfixed.json")
            with open(remain_path, "w", encoding="utf-8") as f:
                json.dump(failed, f, ensure_ascii=False, indent=2)
            print(f"[OK] Wrote unfixed details -> {remain_path}  (items: {len(failed)})")

if __name__ == "__main__":
    main()
