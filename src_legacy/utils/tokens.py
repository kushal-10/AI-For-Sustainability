import sys
from pathlib import Path
import csv
import pandas as pd

ROOT = Path("data/texts")
OUT_CSV = Path("src/utils/results/token_counts.csv")

def get_encoding(name: str):
    import tiktoken
    try:
        # Try as encoding name first (e.g., "o200k_base", "cl100k_base")
        return tiktoken.get_encoding(name)
    except Exception:
        # Try as a model name (e.g., "gpt-4.1")
        try:
            return tiktoken.encoding_for_model(name)
        except Exception:
            # Fallback for older tiktoken installs
            print(f"[WARN] Unknown encoding/model '{name}', falling back to 'cl100k_base'.", file=sys.stderr)
            return tiktoken.get_encoding("cl100k_base")

def main():
    # Single optional arg: encoding/model (default geared for GPT-4.1)
    enc_name = sys.argv[1] if len(sys.argv) > 1 else "o200k_base"
    try:
        enc = get_encoding(enc_name)
    except ImportError:
        print("[ERR] tiktoken is not installed. Run: pip install tiktoken", file=sys.stderr)
        sys.exit(1)

    files = sorted(ROOT.glob("*/*/results.txt"))
    if not files:
        print(f"[ERR] No files found under {ROOT}/<COMPANY>/<YEAR>/results.txt", file=sys.stderr)
        sys.exit(1)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    total_tokens = 0
    for fp in files:
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[WARN] Failed to read {fp}: {e}", file=sys.stderr)
            continue

        # Expect path: data/texts/<COMPANY>/<YEAR>/results.txt
        year = fp.parent.name
        company = fp.parent.parent.name

        try:
            tokens = len(enc.encode(text))
        except Exception as e:
            print(f"[WARN] Failed to tokenize {fp}: {e}", file=sys.stderr)
            continue

        rows.append({"company": company, "year": year, "tokens": tokens})
        total_tokens += tokens

    # Write CSV
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["company", "year", "tokens"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Wrote {len(rows)} rows to {OUT_CSV}")
    print(f"[OK] Total tokens: {total_tokens:,}")

if __name__ == "__main__":
    main() # 211M approx tokens
