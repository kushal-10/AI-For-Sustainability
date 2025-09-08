#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List

import duckdb
from openai import OpenAI

# Reuse your object builder
from src.gpt_classifier.objects import create_batch_object

TEXTS_ROOT = Path("data/texts")
SCORES_ROOT = Path("data/scores_csv")  # only to synthesize csv_path for create_batch_object
OUT_ROOT = Path("data/classification_batches")


def parse_custom_id(custom_id: str) -> Tuple[str, str, str]:
    """
    custom_id format: task||{sentence_id}||{COMPANY}||{YEAR}
    Returns (sentence_id, company, year)
    """
    parts = (custom_id or "").split("||")
    if len(parts) < 4:
        raise ValueError(f"Bad custom_id: {custom_id!r}")
    _, sid, company, year = parts[:4]
    return sid, company, year


def _jsonl_rotating_writer(batch_dir: Path, prefix: str = "batch", batch_lines: int = 20000):
    """Yield write_line(obj) that rotates files every `batch_lines` lines."""
    batch_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    count = 0
    f = None

    def _open_new():
        nonlocal f, idx, count
        if f:
            f.close()
        path = batch_dir / f"{prefix}_{idx}.jsonl"
        f = open(path, "w", encoding="utf-8")
        count = 0

    _open_new()

    def write_line(obj: dict):
        nonlocal f, idx, count
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        count += 1
        if count >= batch_lines:
            f.close()
            idx += 1
            _open_new()

    def close_and_report():
        nonlocal f, idx, count
        if f:
            f.close()
        return idx + (1 if count > 0 else 0)

    return write_line, close_and_report


# cache for splits.json per (company, year)
_SPLITS_CACHE: Dict[Tuple[str, str], Dict[str, str]] = {}


def load_sentence(company: str, year: str, sentence_id: str) -> str | None:
    """
    Load sentence text from data/texts/{company}/{year}/splits.json by string(sentence_id).
    """
    key = (company, year)
    if key not in _SPLITS_CACHE:
        p = TEXTS_ROOT / company / year / "splits.json"
        if not p.exists():
            return None
        try:
            with open(p, "r", encoding="utf-8") as fh:
                _SPLITS_CACHE[key] = json.load(fh)
        except Exception:
            return None
    try:
        return _SPLITS_CACHE[key].get(str(int(sentence_id)))
    except Exception:
        return None


def synth_csv_path(company: str, year: str) -> str:
    """
    `create_batch_object` expects a csv_path whose last segments resolve to .../{company}/{year}/...
    We synthesize: data/scores_csv/{company}/{year}/similarity_scores.csv
    """
    return str(SCORES_ROOT / company / year / "similarity_scores.csv")


def create_batches(
    db_path: Path,
    out_dir: Path,
    batch_lines: int = 20000,
    model: str = "gpt-4.1-mini",
) -> None:
    """
    Create JSONL batch files for all results where raw != '[0, False]'.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB not found: {db_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(db_path.as_posix())
    try:
        rows: List[Tuple[str]] = con.execute(
            "SELECT custom_id FROM results WHERE raw != '[0, False]'"
        ).fetchall()
    finally:
        con.close()

    if not rows:
        print("No rows to enqueue (all are [0, False]).")
        return

    write_line, close_and_report = _jsonl_rotating_writer(out_dir, prefix="batch", batch_lines=batch_lines)

    total = 0
    written = 0
    missing_sentence = 0

    for (custom_id,) in rows:
        total += 1
        try:
            sid, company, year = parse_custom_id(custom_id)
        except Exception:
            continue

        sent = load_sentence(company, year, sid)
        if not sent or not isinstance(sent, str) or not sent.strip():
            missing_sentence += 1
            continue

        csv_path = synth_csv_path(company, year)
        obj = create_batch_object(sent.strip(), sid, csv_path, model=model)
        # Force the custom_id to match exactly
        obj["custom_id"] = custom_id

        write_line(obj)
        written += 1

    files = close_and_report()
    print(f"Created {files} batch file(s) in {out_dir}")
    print(f"Candidates: {total} | Written: {written} | Missing sentences: {missing_sentence}")


def submit_batches(
    batch_dir: Path,
    completion_window: str = "24h",
) -> None:
    """
    Upload each .jsonl in `batch_dir` and start a Batch job.
    Prints: file path -> batch_file_id -> batch_job.id
    """
    client = OpenAI()
    batch_files = sorted(p for p in batch_dir.glob("*.jsonl") if p.is_file())
    if not batch_files:
        print(f"No .jsonl files found in {batch_dir}")
        return

    for path in batch_files:
        try:
            with open(path, "rb") as fh:
                up = client.files.create(file=fh, purpose="batch")
            job = client.batches.create(
                input_file_id=up.id,
                endpoint="/v1/chat/completions",
                completion_window=completion_window,
            )
            print("=" * 80)
            print(f"Local file: {path}")
            print(f"Uploaded file_id: {up.id}")
            print(f"Batch job id: {job.id}")
            print(f"Status: {getattr(job, 'status', '-')}")
        except Exception as e:
            print("=" * 80)
            print(f"ERROR submitting {path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Create and/or submit classifier batches for non-[0, False] rows.")
    parser.add_argument("--db", default="data/outputs_merged/classifications.duckdb", help="DuckDB path (with table `results`).")
    parser.add_argument("--outdir", default=str(OUT_ROOT), help="Output dir for JSONL batches.")
    parser.add_argument("--batch-lines", type=int, default=20000, help="Max lines per JSONL file.")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model for chat completions.")
    parser.add_argument("--make-batches", action="store_true", help="Create JSONL batches.")
    parser.add_argument("--submit", action="store_true", help="Submit existing JSONL batches from --outdir.")
    parser.add_argument("--completion-window", default="24h", help="Batch completion window (e.g., 24h, 48h).")

    args = parser.parse_args()
    out_dir = Path(args.outdir)

    if args.make-batches:
        create_batches(
            db_path=Path(args.db),
            out_dir=out_dir,
            batch_lines=args.batch_lines,
            model=args.model,
        )

    if args.submit:
        submit_batches(
            batch_dir=out_dir,
            completion_window=args.completion_window,
        )

    if not args.make-batches and not args.submit:
        parser.print_help()


if __name__ == "__main__":
    main()

"""
python3 src/gpt_classifier/batches.py \
  --make-batches \
  --db data/outputs_merged/classifications.duckdb \
  --outdir data/classification_batches \
  --batch-lines 20000 \
  --model gpt-4.1-mini

python3 src/gpt_classifier/batches.py \
  --submit \
  --outdir data/classification_batches \
  --completion-window 24h
"""