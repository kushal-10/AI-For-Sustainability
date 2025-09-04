#!/usr/bin/env python3
import os

import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

# ---------------- Config ----------------
CSV_DIR = Path("data/scores_csv")        # directory with all csv files
AI_JSON = Path("kw_data/ai.json")  # filtered AI keywords
THRESHOLDS = [0.5, 0.3]

# ---------------- Load AI keywords ----------------
with open(AI_JSON, "r", encoding="utf-8") as f:
    ai_keywords = list(json.load(f).keys())

# ---------------- SDG columns (from your schema) ----------------
sdg_columns = [
    "sdg_1","sdg_2","sdg_3","sdg_4","sdg_5","sdg_6","sdg_7","sdg_8","sdg_9","sdg_10",
    "sdg_11","sdg_12","sdg_13","sdg_14","sdg_15","sdg_16","sdg_17"
]

# ---------------- Collect results ----------------
total_rows = 0
results = {thr: 0 for thr in THRESHOLDS}

csv_files = []
for dirpath, _, filenames in os.walk(CSV_DIR):
    for filename in filenames:
        if filename.endswith(".csv"):
            csv_files.append(os.path.join(dirpath, filename))

for csv_file in tqdm(csv_files):
    df = pd.read_csv(csv_file)

    # Relevant columns = SDGs + AI keywords
    cols = [c for c in df.columns if c in sdg_columns or c in ai_keywords]
    data = df[cols]

    total_rows += len(data)

    for thr in THRESHOLDS:
        mask = (data >= thr).any(axis=1)
        results[thr] += mask.sum()

# ---------------- Print summary ----------------
print(f"Total sentences across CSVs: {total_rows}")
for thr in THRESHOLDS:
    print(f"Sentences with ≥ {thr} in SDG+AI columns: {results[thr]}")
