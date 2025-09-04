import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Set, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

from src.classification.objects import create_batch_object
from src.utils.file_utils import load_json  # your existing util

# ---------------- Config ----------------
MODEL = "gpt-4.1-mini"
BASE_DIR = Path("data/texts")
SCORES_ROOT = Path("data/scores_csv")
BATCH_ROOT = Path("data/batches_41_mini")
BATCH_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_THRESHOLD = 0.5
DEFAULT_BATCH_LINES = 20_000

# ---------------- Logging ----------------
logging.basicConfig(
    filename=str(Path("src/classification/submit_requests.log")),
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------- Helpers ----------------
def _paired_csv_path(splits_path: Path) -> Path:
    """Map data/texts/.../splits.json -> data/scores_csv/.../similarity_scores.csv"""
    rel = splits_path.relative_to(Path("data") / "texts")
    return SCORES_ROOT / rel.parent / "similarity_scores.csv"

def _jsonl_rotating_writer(batch_dir: Path, prefix: str = "batch", batch_lines: int = DEFAULT_BATCH_LINES):
    """Yield write_line(obj) that rotates files every `batch_lines` lines."""
    batch_idx = 0
    line_count = 0
    f = None

    def _open_new():
        nonlocal f, batch_idx, line_count
        if f:
            f.close()
        path = batch_dir / f"{prefix}_{batch_idx}.jsonl"
        f = open(path, "w", encoding="utf-8")
        line_count = 0

    _open_new()

    def write_line(obj):
        nonlocal batch_idx, line_count, f
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        line_count += 1
        if line_count >= batch_lines:
            f.close()
            batch_idx += 1
            _open_new()

    def close_and_report():
        nonlocal f, batch_idx, line_count
        if f:
            f.close()
        # number of files created (batch_idx is 0-based)
        return batch_idx + (1 if line_count > 0 else 0)

    return write_line, close_and_report

def _sdg_columns() -> Set[str]:
    return {f"sdg_{i}" for i in range(1, 18)}

def _collect_ai_names_from_dict(ai_obj: Dict) -> Set[str]:
    """
    ai.json is a dict like { "english term": "deutscher Begriff", ... }.
    Return a set containing ALL keys and ALL string values.
    """
    names: Set[str] = set()
    for k, v in ai_obj.items():
        if isinstance(k, str) and k.strip():
            names.add(k.strip())
        if isinstance(v, str) and v.strip():
            names.add(v.strip())
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, str) and item.strip():
                    names.add(item.strip())
    return names

def _resolve_existing_columns(df: pd.DataFrame, allowed: Set[str]) -> List[str]:
    """
    Match allowed names against df columns (excluding the first id column).
    Case-insensitive exact match. Returns actual df column names present.
    """
    if df.shape[1] <= 1:
        return []
    df_cols = list(df.columns[1:])
    # build case-insensitive map from normalized -> actual
    norm_map = {c.lower(): c for c in df_cols}
    resolved = []
    for name in allowed:
        key = name.lower()
        if key in norm_map:
            resolved.append(norm_map[key])
    return resolved

# ---------------- Core ----------------
def create_batches(
    ai_json_path: Path,
    threshold: float = DEFAULT_THRESHOLD,
    batch_lines: int = DEFAULT_BATCH_LINES,
    batch_dir: Path = BATCH_ROOT,
    model: str = MODEL,
) -> None:
    """
    For every splits.json, load paired scores CSV. Select rows where ANY of the
    columns in (AI names from ai.json keys+values ∪ sdg_1..sdg_17) is ≥ threshold.
    No fuzzy matching.
    """
    # Load allowed AI names (English + German)
    ai_obj = load_json(str(ai_json_path))
    if not isinstance(ai_obj, dict):
        raise ValueError("ai.json must be a dict mapping English→German (or similar).")
    ai_names = _collect_ai_names_from_dict(ai_obj)

    allowed_names = ai_names | _sdg_columns()

    split_paths: List[Path] = sorted(BASE_DIR.rglob("splits.json"))
    write_line, close_and_report = _jsonl_rotating_writer(batch_dir, batch_lines=batch_lines)

    total_selected = 0
    total_files = 0

    for sp in tqdm(split_paths, desc="Scanning files", unit="file"):
        csv_path = _paired_csv_path(sp)
        if not csv_path.exists():
            logger.warning(f"Missing scores CSV for {sp}: {csv_path}")
            continue

        sentences = load_json(str(sp))  # keys are str(int sentence_id)
        df = pd.read_csv(csv_path)

        if df.shape[1] < 2:
            logger.warning(f"No score columns in {csv_path}")
            continue

        keep_cols = _resolve_existing_columns(df, allowed_names)
        if not keep_cols:
            logger.info(f"No allowed AI/SDG columns present in {csv_path}. Skipping.")
            continue

        # Vectorized threshold check on the restricted columns
        scores = df[keep_cols].to_numpy(dtype=np.float32, copy=False)
        mask = (scores >= threshold).any(axis=1)
        if not mask.any():
            total_files += 1
            continue

        # Produce batch objects for the matched rows
        id_col = df.columns[0]
        sel_ids = df.loc[mask, id_col].to_numpy()
        for sid in sel_ids:
            sid_str = str(int(sid))
            sentence = sentences.get(sid_str)
            if not sentence:
                continue
            obj = create_batch_object(sentence, sid_str, str(csv_path), model=model)
            write_line(obj)
            total_selected += 1

        total_files += 1

    num_batches = close_and_report()
    print(f"Created {num_batches} batch file(s) in {batch_dir}")
    print(f"Selected rows (≥ {threshold}): {total_selected}")
    print(f"Processed split files: {total_files}")

def submit_requests(batch_dir: Path = BATCH_ROOT, completion_window: str = "24h"):
    """
    Upload every .jsonl file in batch_dir and create a batch job per file.
    """
    client = OpenAI()
    jsonl_files = sorted(p for p in batch_dir.glob("*.jsonl") if p.is_file())

    for path in tqdm(jsonl_files, desc="Submitting batches", unit="file"):
        with open(path, "rb") as fh:
            batch_file = client.files.create(file=fh, purpose="batch")

        logger.info("/" * 50)
        logger.info(f"BATCH file: {batch_file.id} for path: {path}")
        logger.info("/" * 50)

        job = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window
        )
        logger.info(f"batch_job.id: {job.id}")
        logger.info(f"batch_job.output_file_id: {job.output_file_id}")

def main():
    parser = argparse.ArgumentParser(description="Create and/or submit batch jobs from SDG/AI scores.")
    parser.add_argument("--ai_json", type=str, required=True, help="Path to ai.json (dict English->German).")
    parser.add_argument("--make_batches", action="store_true", help="Create JSONL batches.")
    parser.add_argument("--submit", action="store_true", help="Submit existing JSONL batches.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Score threshold (default: 0.5).")
    parser.add_argument("--batch-lines", type=int, default=DEFAULT_BATCH_LINES, help="Max lines per JSONL file.")
    parser.add_argument("--model", type=str, default=MODEL, help="OpenAI model for create_batch_object.")

    args = parser.parse_args()

    if args.make_batches or args.submit:
        pass  # just to keep lints happy

    if args.make_batches:
        create_batches(
            ai_json_path=Path(args.ai_json),
            threshold=args.threshold,
            batch_lines=args.batch_lines,
            model=args.model,
        )

    if args.submit:
        submit_requests()

if __name__ == "__main__":
    main()

# python3 src/classification/batches.py --make-batches --ai-json kw_data/ai.json
# python3 src/classification/batches.py --submit  --ai-json kw_data/ai.json
