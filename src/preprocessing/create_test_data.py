# Create a sample subset of data for testing filtering/data processing

# save as scripts/sample_reports.py (or run directly)
import random, shutil
from pathlib import Path
import argparse

def main(n=10, seed=42):
    src_root = Path("data/texts")
    dst_root = Path("data/sample_texts")
    files = list(src_root.glob("*/*/results.txt"))  # COMPANY/YEAR/results.txt

    if not files:
        raise SystemExit("No files found at data/texts/*/*/results.txt")

    random.seed(seed)
    pick = files if len(files) <= n else random.sample(files, n)

    copied = []
    for src in pick:
        company = src.parents[1].name  # COMPANY
        year = src.parents[0].name     # YEAR
        dst_dir = dst_root / company / year
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "results.txt"
        shutil.copy2(src, dst)
        copied.append((str(src), str(dst)))

    # Optional: write a manifest for traceability
    manifest = dst_root / "sample_manifest.tsv"
    with manifest.open("w", encoding="utf-8") as f:
        f.write("source_path\tdest_path\n")
        for s, d in copied:
            f.write(f"{s}\t{d}\n")

    print(f"Copied {len(copied)} files to {dst_root}")
    print(f"Manifest: {manifest}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="number of reports to sample")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    args = ap.parse_args()
    main(n=args.n, seed=args.seed)
