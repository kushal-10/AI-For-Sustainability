#!/usr/bin/env python3
import argparse
import random
import shutil
from pathlib import Path

def copy_if_exists(src_path: Path, dst_path: Path, overwrite: bool) -> bool:
    if not src_path.exists():
        return False
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists() and not overwrite:
        return True  # treat as success (already there)
    shutil.copy2(src_path, dst_path)
    return True

def main(n: int = 10, seed: int = 42, overwrite: bool = False):
    src_root = Path("data/texts")
    dst_root = Path("data/sample_texts")

    # Expect structure: data/texts/COMPANY/YEAR/{results.txt,splits.json}
    candidates = [p for p in src_root.glob("*/*") if p.is_dir()]

    if not candidates:
        raise SystemExit("No folders found at data/texts/*/*")

    random.seed(seed)
    picks = candidates if len(candidates) <= n else random.sample(candidates, n)

    copied_rows = []
    for src_dir in picks:
        company = src_dir.parent.name
        year = src_dir.name

        src_results = src_dir / "results.txt"
        src_splits  = src_dir / "splits.json"

        dst_dir = dst_root / company / year
        dst_results = dst_dir / "results.txt"
        dst_splits  = dst_dir / "splits.json"

        ok_results = copy_if_exists(src_results, dst_results, overwrite)
        ok_splits  = copy_if_exists(src_splits,  dst_splits,  overwrite)

        copied_rows.append((str(src_dir), str(dst_results) if ok_results else "",
                            str(dst_splits) if ok_splits else ""))

    # Manifest
    manifest = dst_root / "sample_manifest.tsv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as f:
        f.write("source_dir\tdest_results\tdest_splits\n")
        for row in copied_rows:
            f.write("\t".join(row) + "\n")

    print(f"Sampled {len(picks)} report folders -> {dst_root}")
    print(f"Manifest written to: {manifest}")
    missing = sum(1 for _, r, s in copied_rows if not r or not s)
    if missing:
        print(f"Note: {missing} sampled folders were missing one or both files.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Copy a sample of reports (results.txt & splits.json).")
    ap.add_argument("--n", type=int, default=10, help="number of report folders to sample")
    ap.add_argument("--seed", type=int, default=42, help="random seed for sampling")
    ap.add_argument("--overwrite", action="store_true", help="overwrite files in sample_texts if present")
    args = ap.parse_args()
    main(n=args.n, seed=args.seed, overwrite=args.overwrite)
