#!/usr/bin/env python3
import json
import random
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# ---------------- Config ----------------
SCORES_ROOT = Path("data/scores_csv")   # CSVs live here (recursive)
AI_JSON = Path("kw_data/ai.json")
NUM_FILES = 20
SEED = 42

# SDG columns (sdg_1..sdg_17)
SDG_COLS = [f"sdg_{i}" for i in range(1, 18)]

# ---------------- Load AI keyword columns ----------------
with open(AI_JSON, "r", encoding="utf-8") as f:
    ai_cols = list(json.load(f).keys())

# ---------------- Utilities ----------------
def splits_path_for(csv_path: Path) -> Path:
    """
    data/scores_csv/COMPANY/YEAR/xxx.csv -> data/texts/COMPANY/YEAR/splits.json
    """
    parts = list(csv_path.parts)
    try:
        i = parts.index("scores_csv")
        parts[i] = "texts"
    except ValueError:
        # fallback: just swap the first occurrence in the string path
        return Path(str(csv_path).replace("scores_csv", "texts")).with_name("splits.json")
    return Path(*parts).with_name("splits.json")

def load_splits(splits_path: Path) -> dict:
    with open(splits_path, "r", encoding="utf-8") as f:
        return json.load(f)  # keys are sentence_id as strings

def firm_label(csv_path: Path) -> str:
    """
    Uses folder structure scores_csv/COMPANY/YEAR/file.csv
    """
    parts = csv_path.parts
    try:
        i = parts.index("scores_csv")
        company = parts[i+1] if len(parts) > i+1 else "UNKNOWN_COMPANY"
        year = parts[i+2] if len(parts) > i+2 else "UNKNOWN_YEAR"
        return f"{company}/{year}"
    except ValueError:
        return csv_path.stem

# ---------------- Drive ----------------
all_csvs = list(SCORES_ROOT.rglob("*.csv"))
if not all_csvs:
    raise SystemExit("No CSV files found under data/scores_csv")

random.seed(SEED)
sample_csvs = all_csvs if len(all_csvs) <= NUM_FILES else random.sample(all_csvs, NUM_FILES)

for csv_path in tqdm(sample_csvs, desc="Scanning random CSVs"):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Skip unreadable: {csv_path} ({e})")
        continue

    # Check columns
    if "sentence_id" not in df.columns:
        print(f"Skip (no sentence_id): {csv_path}")
        continue

    relevant_cols = [c for c in df.columns if c in SDG_COLS or c in ai_cols]
    if not relevant_cols:
        print(f"Skip (no SDG/AI cols): {csv_path}")
        continue

    # numeric subframe
    num_df = df[relevant_cols].apply(pd.to_numeric, errors="coerce")

    # masks
    mask_low  = num_df.gt(0.3).any(axis=1) & num_df.lt(0.5).any(axis=1)  # any (0.3, 0.5)
    mask_high = num_df.ge(0.5).any(axis=1)                               # any >= 0.5

    if not (mask_low.any() or mask_high.any()):
        # nothing to show in this file
        continue

    splits_path = splits_path_for(csv_path)
    if not splits_path.exists():
        print(f"Skip (missing splits.json): {splits_path}")
        continue

    try:
        splits = load_splits(splits_path)
    except Exception as e:
        print(f"Skip (bad splits.json): {splits_path} ({e})")
        continue

    # helper to extract first sentence from a mask (returns (sid, text, top_col, top_score))
    def first_example(mask):
        idxs = df.index[mask].tolist()
        for idx in idxs:
            sid = df.loc[idx, "sentence_id"]
            text = splits.get(str(int(sid)))
            if not text:
                continue
            # top contributing column + score
            row_vals = num_df.loc[idx]
            top_col = row_vals.idxmax()
            top_score = float(row_vals[top_col]) if pd.notna(row_vals[top_col]) else float("nan")
            return (int(sid), text, top_col, top_score)
        return None

    ex_low = first_example(mask_low)
    ex_high = first_example(mask_high)

    # ensure we print up to 2 per file, preferring one low + one high
    picked = []
    if ex_low:  picked.append(("0.3<score<0.5",) + ex_low)
    if ex_high and (not ex_low or ex_high[0] != ex_low[0]):
        picked.append(("score≥0.5",) + ex_high)

    # if we still have <2, fill from whichever bucket has extras
    if len(picked) < 2:
        # union mask
        mask_any = mask_low | mask_high
        idxs = df.index[mask_any].tolist()
        used_sids = {p[1] for p in picked}  # p[1] is sid
        for idx in idxs:
            sid = int(df.loc[idx, "sentence_id"])
            if sid in used_sids:
                continue
            text = splits.get(str(sid))
            if not text:
                continue
            row_vals = num_df.loc[idx]
            top_col = row_vals.idxmax()
            top_score = float(row_vals[top_col]) if pd.notna(row_vals[top_col]) else float("nan")
            bucket = "score≥0.5" if (row_vals >= 0.5).any() else "0.3<score<0.5"
            picked.append((bucket, sid, text, top_col, top_score))
            if len(picked) == 2:
                break

    if not picked:
        continue

    # ---- Print nicely ----
    print("\n" + "="*88)
    print(f"Firm/Year: {firm_label(csv_path)}")
    print(f"File: {csv_path.relative_to(SCORES_ROOT)}")
    for tag, sid, text, top_col, top_score in picked[:2]:
        print("-"*88)
        print(f"{tag} | sentence_id={sid} | top_col={top_col} | top_score={top_score:.3f}")
        print(text)
