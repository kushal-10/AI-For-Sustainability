import json
import os
import statistics

# Try to load tiktoken, else fallback to word-split
try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(enc.encode(text))

except ImportError:
    print("⚠️ tiktoken not installed. Falling back to word-based token counts.")

    def count_tokens(text: str) -> int:
        return len(text.split())

paths = []
base_dir = os.path.join("data", "sample_texts")
for dirname, _, filenames in os.walk(base_dir):
    for filename in filenames:
        if filename.endswith(".json"):
            paths.append(os.path.join(dirname, filename))

# Compute stats per file
for path in paths:
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        continue

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    token_counts = [count_tokens(s) for s in data.values() if s.strip()]

    if not token_counts:
        print(f"⚠️ No sentences found in {path}")
        continue

    stats = {
        "file": path,
        "total_sentences": len(token_counts),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "mean_tokens": statistics.mean(token_counts),
        "median_tokens": statistics.median(token_counts),
    }

    print(stats)

"""
python3 src/preprocessing/analyze_splitter.py
"""
