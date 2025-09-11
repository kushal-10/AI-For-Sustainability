import json
import statistics
from pathlib import Path

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

# Files to analyze
paths = [
    Path("results/sample_splits/basf_semantic_splits.json"),
    Path("results/sample_splits/basf_semantic_splits_v2.json"),
    Path("results/sample_splits/basf_naive_splits.json")
]

# Compute stats per file
for path in paths:
    if not path.exists():
        print(f"❌ File not found: {path}")
        continue

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    token_counts = [count_tokens(s) for s in data.values() if s.strip()]

    if not token_counts:
        print(f"⚠️ No sentences found in {path}")
        continue

    stats = {
        "file": path.name,
        "total_sentences": len(token_counts),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "mean_tokens": statistics.mean(token_counts),
        "median_tokens": statistics.median(token_counts),
    }

    print(stats)

"""
{'file': 'basf_semantic_splits.json', 'total_sentences': 313, 'min_tokens': 8, 'max_tokens': 4274, 'mean_tokens': 850.0383386581469, 'median_tokens': 524}
{'file': 'basf_semantic_splits_v2.json', 'total_sentences': 309, 'min_tokens': 8, 'max_tokens': 4274, 'mean_tokens': 861.042071197411, 'median_tokens': 531}
{'file': 'basf_naive_splits.json', 'total_sentences': 6025, 'min_tokens': 1, 'max_tokens': 2084, 'mean_tokens': 41.35551867219917, 'median_tokens': 27}
"""
