#!/usr/bin/env python3
"""
Collect OpenAI Batch results and build a map: custom_id -> parsed output (JSON).

Usage:
  export OPENAI_API_KEY=your_key
  python3 src/batching/collect_results.py --ids BATCH_ID_1 BATCH_ID_2 ...
  # optional flags:
  #   --outdir data/batches/results
  #   --save-raw   (store raw output jsonl files to disk)
"""

import os
import re
import json
import argparse
from typing import Dict, Any, List, Tuple

try:
    from openai import OpenAI
except Exception as e:
    raise SystemExit("openai package not installed. `pip install openai`") from e


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def fetch_batch(client: OpenAI, batch_id: str) -> Dict[str, Any]:
    return client.batches.retrieve(batch_id).model_dump()


def get_output_file_id(batch: Dict[str, Any]) -> str:
    # Support different field names used historically
    return batch.get("output_file_id") or batch.get("response_file_id") or ""


def download_file_content(client: OpenAI, file_id: str) -> str:
    """Return the raw text of a file (JSONL expected)."""
    if not file_id:
        return ""
    resp = client.files.content(file_id)
    # Support both streamlike and direct return types
    data = None
    if hasattr(resp, "read"):
        data = resp.read()
    elif hasattr(resp, "content"):
        data = resp.content
    else:
        data = resp
    if isinstance(data, (bytes, bytearray)):
        return data.decode("utf-8", errors="replace")
    return str(data)


def iter_jsonl(text: str):
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception:
            # Skip non-JSON line safely
            continue


_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.DOTALL)


def strip_code_fences(s: str) -> str:
    # Remove triple-backtick fences if present
    return _CODE_FENCE_RE.sub("", s)


def try_parse_json_content(content: str):
    """
    Try to parse the assistant content into JSON.
    1) direct json.loads
    2) strip code fences and try again
    3) extract the first {...} block and json.loads it
    Return parsed object or None.
    """
    if content is None:
        return None
    s = content.strip()
    # 1) direct
    try:
        return json.loads(s)
    except Exception:
        pass
    # 2) strip code fences
    s2 = strip_code_fences(s).strip()
    if s2 != s:
        try:
            return json.loads(s2)
        except Exception:
            pass
    # 3) find first JSON object
    m = re.search(r"\{.*\}", s2, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def extract_choice_content(rec: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """
    Returns (custom_id, parsed_json_output or None, error_info or None)
    """
    custom_id = rec.get("custom_id", "")

    # If record has an error at the batch layer
    if rec.get("error"):
        return custom_id, None, {"level": "record", "error": rec.get("error")}

    resp = rec.get("response") or {}
    status = int(resp.get("status_code") or 0)
    if status != 200:
        return custom_id, None, {"level": "http", "status_code": status, "response": resp}

    body = (resp.get("body") or {})
    choices = (body.get("choices") or [])
    if not choices:
        return custom_id, None, {"level": "body", "reason": "no choices", "body": body}

    msg = choices[0].get("message") or {}
    content = msg.get("content", "")

    parsed = try_parse_json_content(content)
    if parsed is None:
        # Return raw content for inspection if parsing failed
        return custom_id, None, {"level": "parse", "raw_content": content}

    return custom_id, parsed, None


def collect_from_batches(batch_ids: List[str], outdir: str, save_raw: bool = False) -> Dict[str, Any]:
    ensure_dir(outdir)
    client = OpenAI()  # reads OPENAI_API_KEY

    results_map: Dict[str, Any] = {}
    errors_map: Dict[str, Any] = {}

    for bid in batch_ids:
        print(f"[INFO] Processing batch {bid} ...")
        b = fetch_batch(client, bid)
        out_file_id = get_output_file_id(b)
        if not out_file_id:
            print(f"  [WARN] No output file for {bid}. Status={b.get('status')}")
            continue

        raw = download_file_content(client, out_file_id)
        if not raw:
            print(f"  [WARN] Empty output for {bid}.")
            continue

        if save_raw:
            ensure_dir(os.path.join(outdir, "raw"))
            raw_path = os.path.join(outdir, "raw", f"{bid}.jsonl")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(raw)
            print(f"  [OK] Saved raw -> {raw_path}")

        # Parse each line
        ok_count = 0
        err_count = 0
        for rec in iter_jsonl(raw):
            custom_id, parsed, err = extract_choice_content(rec)
            if not custom_id:
                err_count += 1
                continue
            if err is not None:
                errors_map[custom_id] = err
                err_count += 1
            else:
                results_map[custom_id] = parsed
                ok_count += 1

        print(f"  [OK] {ok_count} parsed, {err_count} errors for {bid}")

    # Save consolidated maps
    results_path = os.path.join(outdir, "results_map.json")
    errors_path = os.path.join(outdir, "results_errors.json")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_map, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote results map -> {results_path}  (items: {len(results_map)})")

    with open(errors_path, "w", encoding="utf-8") as f:
        json.dump(errors_map, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote errors map  -> {errors_path}  (items: {len(errors_map)})")

    return {"results": results_map, "errors": errors_map}


def main():
    ap = argparse.ArgumentParser(description="Collect OpenAI Batch results into a custom_id -> output map.")
    ap.add_argument("--ids", nargs="+", required=True, help="Batch IDs to collect.")
    ap.add_argument("--outdir", default="data/batches/results", help="Directory to save results.")
    ap.add_argument("--save-raw", action="store_true", help="Also save raw output JSONL per batch.")
    args = ap.parse_args()

    collect_from_batches(args.ids, args.outdir, args.save_raw)


if __name__ == "__main__":
    main()

"""
python3 src/gpt_classifier/collect_results.py \
  --ids batch_68cfe37201788190a684f8e3e3d45a96 \
       batch_68cfe37d0e3c81909c077bd2cb9e35df \
       batch_68cfe386e8308190a0d39757614ee0cb \
       batch_68cfe38b0ce08190b04def0d90fdde6f \
       batch_68cfe393e43c81908efe18f43c1dbbb1 \
       batch_68cfe39917fc8190a739b4680edc2cd0 \
  --save-raw
"""