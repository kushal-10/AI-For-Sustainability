#!/usr/bin/env python3
"""
Analyze sentence token lengths across JSON files (multiprocessing + tqdm).

- Loads all .json files from data/jsons/
- Tokenizes sentences with tiktoken (cl100k_base) where available
- Uses ProcessPoolExecutor for parallelism (good on Apple M3 8-core)
- Prints overall stats: total sentences, mean, median, min, max
- Saves histogram (10–512 token range) to src/analysis/plots/token_length_hist.png
"""

import os
import json
from typing import Any, Dict, List, Tuple, Union, Iterable
from statistics import mean, median
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import multiprocessing as mp


# ----------------- JSON loading -----------------
def _fallback_load_json(path: str) -> Union[Dict[str, Any], List[Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_json_any(path: str) -> Union[Dict[str, Any], List[Any]]:
    # Try project loader if available, else fallback
    try:
        from src.utils.file_utils import load_json as project_load_json  # type: ignore
        return project_load_json(path)
    except Exception:
        return _fallback_load_json(path)


# ----------------- Token counting -----------------
def _get_token_counter():
    """
    Create and return a function count_tokens(text) bound to either tiktoken
    (cl100k_base) or a whitespace fallback. Done inside worker to avoid
    pickling state across processes.
    """
    try:
        import tiktoken  # local import per process
        enc = tiktoken.get_encoding("cl100k_base")
        def count_tokens(text: str) -> int:
            return len(enc.encode(text or ""))
        tokenizer_name = "tiktoken cl100k_base"
    except Exception:
        def count_tokens(text: str) -> int:
            return len(text.split()) if text else 0
        tokenizer_name = "fallback (whitespace)"
    return count_tokens, tokenizer_name


def iter_sentences(obj: Union[Dict[str, Any], List[Any]]) -> Iterable[str]:
    """Extract sentence strings from a dict or list JSON structure."""
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, str):
                yield v
            elif isinstance(v, list):
                for vv in v:
                    if isinstance(vv, str):
                        yield vv
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, str):
                yield v


# ----------------- Worker -----------------
def process_file(path: str) -> Tuple[int, List[int], str, str]:
    """
    Process a single JSON file.

    Returns:
      (sent_count, token_lengths, tokenizer_name, error_msg)
    """
    try:
        content = load_json_any(path)
    except Exception as e:
        return 0, [], "", f"read_error:{e}"

    count_tokens, tok_name = _get_token_counter()

    lengths: List[int] = []
    sent_count = 0
    try:
        for s in iter_sentences(content):
            sent_count += 1
            lengths.append(count_tokens(s))
        return sent_count, lengths, tok_name, ""
    except Exception as e:
        return 0, [], tok_name, f"proc_error:{e}"


# ----------------- Main -----------------
def main():
    parser = argparse.ArgumentParser(description="Token stats over JSON sentences (MP + tqdm)")
    parser.add_argument("--base_dir", default=os.path.join("data", "jsons"),
                        help="Directory to scan for .json files")
    parser.add_argument("--out_dir", default=os.path.join("src", "analysis", "plots"),
                        help="Directory to save plots")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of worker processes (M3 8-core: use 8)")
    parser.add_argument("--max_files", type=int, default=0,
                        help="Optionally limit number of files for a quick run (0 = all)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Gather all JSON files
    json_files: List[str] = []
    for dirname, _, filenames in os.walk(args.base_dir):
        for filename in filenames:
            if filename.endswith(".json"):
                json_files.append(os.path.join(dirname, filename))
    if args.max_files > 0:
        json_files = json_files[:args.max_files]

    if not json_files:
        print(f"No JSON files found under {args.base_dir}")
        return

    # Use spawn on macOS to be safe
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        # already set
        pass

    token_lengths: List[int] = []
    total_sentences = 0
    tokenizer_names = set()
    read_errors = 0
    proc_errors = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_file, p): p for p in json_files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing JSON files"):
            path = futures[fut]
            try:
                sent_count, lengths, tok_name, err = fut.result()
            except Exception as e:
                proc_errors += 1
                continue

            if err:
                if err.startswith("read_error"):
                    read_errors += 1
                else:
                    proc_errors += 1

            if tok_name:
                tokenizer_names.add(tok_name)
            total_sentences += sent_count
            if lengths:
                token_lengths.extend(lengths)

    if not token_lengths:
        print("No sentences found or tokenized. "
              f"Read errors: {read_errors}, Proc errors: {proc_errors}")
        return

    # Summary stats
    stats = {
        "files_scanned": len(json_files),
        "total_sentences": total_sentences,
        "mean_tokens": float(mean(token_lengths)),
        "median_tokens": float(median(token_lengths)),
        "min_tokens": int(min(token_lengths)),
        "max_tokens": int(max(token_lengths)),
        "tokenizer_used": ", ".join(sorted(tokenizer_names)) or "unknown",
        "read_errors": read_errors,
        "proc_errors": proc_errors,
        "workers": args.workers,
    }

    print("=== Summary ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    # Histogram (10–512)
    bins = list(range(10, 513))  # 10..512 inclusive
    plt.figure(figsize=(9, 5))
    plt.hist(token_lengths, bins=bins)
    plt.xlim(10, 512)
    plt.ylim(0, 10000)  # <-- cap y-axis at 10k
    plt.xlabel("Token length (per sentence)")
    plt.ylabel("Count")
    plt.title("Histogram of sentence token lengths (10–512)")

    out_path = os.path.join(args.out_dir, "token_length_hist.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Histogram saved to: {out_path}")


if __name__ == "__main__":
    main()


"""
files_scanned: 1420
total_sentences: 620177
mean_tokens: 350.4602911749388
median_tokens: 431.0
min_tokens: 1
max_tokens: 512
tokenizer_used: tiktoken cl100k_base
read_errors: 0
proc_errors: 0
workers: 8
"""