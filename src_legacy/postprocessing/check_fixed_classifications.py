#!/usr/bin/env python3
# src/batching/collect_and_verify_fix.py
"""
Collect FIX batch results into data/batches/fix/results/, then verify them.

Usage:
  export OPENAI_API_KEY=your_key

  # 1) Collect results for the fix batches (saves into data/batches/fix/results/)
  python3 src/batching/collect_and_verify_fix.py --collect \
    --ids batch_... batch_...

  # 2) Verify the collected results (ignore extraneous; require all OG hits labeled)
  python3 src/batching/collect_and_verify_fix.py --verify
  # optional flags:
  #   --normalize         (collapse '\\\\' -> '\\' in keys before comparing)
  #   --export-csv PATH   (write issues as CSV)
"""

import os
import re
import json
import argparse
from typing import Any, Dict, List, Tuple, Set

import duckdb
import pandas as pd
from openai import OpenAI

# ---------- Defaults ----------
OUTDIR = "data/batches/fix/results"

# Fix targets come from original DBs
SDG_DB  = "data/dbs/sdg_hits.duckdb"
SDG_TBL = "sdg_hits"
TECH_DB = "data/dbs/tech_hits.duckdb"
TECH_TBL= "tech_hits"

VALID_LABELS = {"symbolic", "substantive"}

# ---------- IO helpers ----------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def read_jsonl_text(text: str):
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception:
            continue

def save_json(path: str, obj: Any):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------- Parsing assistant content ----------
_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.DOTALL)

def strip_code_fences(s: str) -> str:
    return _CODE_FENCE_RE.sub("", s)

def try_parse_result_content(content: str):
    """
    Return parsed dict or None. Tolerates:
      - code fences
      - extra prose around JSON
      - invalid escapes by doubling unknown \X
    """
    if content is None:
        return None
    s = content.strip()
    # 1) direct
    try:
        return json.loads(s)
    except Exception:
        pass
    # 2) strip fences
    s2 = strip_code_fences(s).strip()
    try:
        return json.loads(s2)
    except Exception:
        pass
    # 3) extract first {...}
    m = re.search(r"\{.*\}", s2, flags=re.DOTALL)
    if m:
        candidate = m.group(0)
        # fix invalid escapes like \s, \w
        candidate2 = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', candidate)
        try:
            return json.loads(candidate2)
        except Exception:
            return None
    return None

# ---------- Collect fix batch results ----------
def collect_fix_results(batch_ids: List[str], outdir: str = OUTDIR, save_raw: bool = True):
    ensure_dir(outdir)
    client = OpenAI()  # needs OPENAI_API_KEY

    results_map: Dict[str, Dict[str, str]] = {}
    errors_map: Dict[str, Any] = {}

    for bid in batch_ids:
        print(f"[INFO] Retrieving batch {bid} ...")
        b = client.batches.retrieve(bid)
        out_file_id = getattr(b, "output_file_id", None) or getattr(b, "response_file_id", None)
        if not out_file_id:
            print(f"  [WARN] No output for {bid} yet (status={b.status}). Skipping.")
            continue

        # download file content
        content = client.files.content(out_file_id)
        data = getattr(content, "read", None)
        if callable(data):
            raw = content.read()
        else:
            raw = getattr(content, "content", content)
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="replace")
        if not isinstance(raw, str):
            raw = str(raw)

        if save_raw:
            raw_dir = os.path.join(outdir, "raw")
            ensure_dir(raw_dir)
            with open(os.path.join(raw_dir, f"{bid}.jsonl"), "w", encoding="utf-8") as f:
                f.write(raw)
            print(f"  [OK] Saved raw -> {os.path.join(raw_dir, f'{bid}.jsonl')}")

        ok, err = 0, 0
        for rec in read_jsonl_text(raw):
            custom_id = rec.get("custom_id", "")
            if rec.get("error"):
                errors_map[custom_id] = {"level": "record", "error": rec.get("error")}
                err += 1
                continue
            resp = rec.get("response") or {}
            if int(resp.get("status_code", 0)) != 200:
                errors_map[custom_id] = {"level": "http", "status": resp.get("status_code"), "response": resp}
                err += 1
                continue
            body = resp.get("body") or {}
            choices = body.get("choices") or []
            if not choices:
                errors_map[custom_id] = {"level": "body", "reason": "no choices", "body": body}
                err += 1
                continue
            content = (choices[0].get("message") or {}).get("content", "")
            parsed = try_parse_result_content(content)
            if parsed is None or not isinstance(parsed, dict):
                errors_map[custom_id] = {"level": "parse", "raw_content": content}
                err += 1
                continue
            results_map[custom_id] = parsed
            ok += 1

        print(f"  [OK] Parsed {ok}, errors {err} for {bid}")

    # write consolidated
    save_json(os.path.join(outdir, "results_map.json"), results_map)
    save_json(os.path.join(outdir, "results_errors.json"), errors_map)
    print(f"[OK] Wrote results -> {os.path.join(outdir, 'results_map.json')} (items: {len(results_map)})")
    print(f"[OK] Wrote errors  -> {os.path.join(outdir, 'results_errors.json')} (items: {len(errors_map)})")

# ---------- Verify fix results (ignore extraneous; require all OG hits labeled) ----------
def get_df(db_path: str, table: str) -> pd.DataFrame:
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute(f"SELECT * FROM {table}").fetchdf()
    con.close()
    return df

def guess_hit_cols_sdg(cols: List[str]) -> List[str]:
    lower = {c: c.lower() for c in cols if isinstance(c, str)}
    hits = [c for c in cols if isinstance(c, str) and lower[c].startswith("hits_sdg")]
    if hits: return hits
    hits = [c for c in cols if isinstance(c, str) and lower[c].startswith("hits") and "sdg" in lower[c]]
    if hits: return hits
    return [c for c in cols if isinstance(c, str) and lower[c].startswith("hits")]

def guess_hit_cols_tech(cols: List[str]) -> List[str]:
    lower = {c: c.lower() for c in cols if isinstance(c, str)}
    prefixes = ("hits_ai_ml", "hits_cloud_computing", "hits_big_data_blockchain", "hits_applications_practice")
    hits = [c for c in cols if isinstance(c, str) and any(lower[c].startswith(p) for p in prefixes)]
    if hits: return hits
    return [c for c in cols if isinstance(c, str) and lower[c].startswith("hits") and "sdg" not in lower[c]]

def parse_hits_cell_to_list(cell: Any) -> List[str]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    if isinstance(cell, list):
        return [s for s in cell if isinstance(s, str) and s.strip()]
    if isinstance(cell, dict):
        return [k for k in cell.keys() if isinstance(k, str) and k.strip()]
    if isinstance(cell, str):
        s = cell.strip()
        if not s:
            return []
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                obj = json.loads(s)
            except Exception:
                try:
                    s2 = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)
                    obj = json.loads(s2)
                except Exception:
                    return []
            if isinstance(obj, list):
                return [x for x in obj if isinstance(x, str) and x.strip()]
            if isinstance(obj, dict):
                return [k for k in obj.keys() if isinstance(k, str) and k.strip()]
        return []
    return []

def normalize_key(s: str) -> str:
    if s is None: return ""
    out = s
    while "\\\\" in out:
        out = out.replace("\\\\", "\\")
    return out.strip()

def verify_fix_results(outdir: str = OUTDIR, normalize: bool = False, export_csv: str = None):
    # Load collected fix results
    results_map = load_json(os.path.join(outdir, "results_map.json"))

    # Separate by mode and strip the trailing "||fix"
    by_row: Dict[Tuple[str, str], Dict[str, str]] = {}  # (mode, global_id) -> {pattern: label}
    for cid, mapping in results_map.items():
        # expected form: "sdg||<gid>||fix" or "tech||<gid>||fix"
        parts = str(cid).split("||")
        if len(parts) < 2:
            # fallback: try "sdg||gid" style
            continue
        mode = parts[0]
        gid = parts[1]
        if mode not in ("sdg", "tech"):
            continue
        # normalize label values
        row_map = {}
        for k, v in (mapping or {}).items():
            key = normalize_key(k) if normalize else k
            lbl = str(v).strip().lower()
            row_map[key] = lbl
        by_row[(mode, gid)] = row_map

    # Load original DBs
    df_sdg = get_df(SDG_DB, SDG_TBL)
    df_tech = get_df(TECH_DB, TECH_TBL)

    hit_cols_sdg = guess_hit_cols_sdg(df_sdg.columns.tolist())
    hit_cols_tech = guess_hit_cols_tech(df_tech.columns.tolist())

    issues: List[Dict[str, Any]] = []
    summary = {
        "rows_checked": 0,
        "patterns_checked": 0,
        "invalid_labels": 0,
        "missing_classifications": 0
    }

    def verify_mode_df(mode: str, df: pd.DataFrame, hit_cols: List[str]):
        if "global_id" not in df.columns:
            return
        nonlocal issues, summary
        # focus only on rows present in results_map (fix rows)
        df_sub = df[df["global_id"].astype(str).isin([gid for (m, gid) in by_row.keys() if m == mode])]
        for _, row in df_sub.iterrows():
            gid = str(row["global_id"])
            res = by_row.get((mode, gid), {})
            summary["rows_checked"] += 1

            # build original patterns for row
            row_patterns: Set[str] = set()
            for col in hit_cols:
                for p in parse_hits_cell_to_list(row.get(col)):
                    row_patterns.add(normalize_key(p) if normalize else p)

            for p in sorted(row_patterns):
                summary["patterns_checked"] += 1
                lbl = res.get(p)
                if lbl is None:
                    summary["missing_classifications"] += 1
                    issues.append({
                        "type": "missing_classification",
                        "mode": mode,
                        "global_id": gid,
                        "pattern": p,
                        "reason": "present in OG hits but absent in fix result"
                    })
                elif lbl not in VALID_LABELS:
                    summary["invalid_labels"] += 1
                    issues.append({
                        "type": "invalid_label",
                        "mode": mode,
                        "global_id": gid,
                        "pattern": p,
                        "label": lbl,
                        "reason": "label must be symbolic|substantive"
                    })
            # NOTE: extraneous classifications in res are allowed/ignored by design.

    verify_mode_df("sdg", df_sdg, hit_cols_sdg)
    verify_mode_df("tech", df_tech, hit_cols_tech)

    # Write report
    report = {"summary": summary, "issues": issues}
    save_json(os.path.join(outdir, "fix_verification_report.json"), report)

    print("\n=== FIX VERIFICATION (ignore extraneous; require all OG hits labeled) ===")
    print(f"Rows checked            : {summary['rows_checked']}")
    print(f"Patterns checked        : {summary['patterns_checked']}")
    print(f"Missing classifications : {summary['missing_classifications']}")
    print(f"Invalid labels          : {summary['invalid_labels']}")

    if issues:
        print("\nSample issues (up to 20):")
        print(pd.DataFrame(issues[:20]).to_string(index=False, max_rows=20, max_cols=0))

    if export_csv:
        ensure_dir(os.path.dirname(export_csv))
        pd.DataFrame(issues).to_csv(export_csv, index=False)
        print(f"[OK] Wrote CSV issues -> {export_csv} (rows: {len(issues)})")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Collect fix batch results to data/batches/fix/results/ and verify them.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--collect", action="store_true", help="Collect results for provided fix batch IDs.")
    g.add_argument("--verify", action="store_true", help="Verify previously collected fix results.")
    ap.add_argument("--ids", nargs="+", help="Batch IDs for --collect.")
    ap.add_argument("--outdir", default=OUTDIR, help="Results output dir (default: data/batches/fix/results)")
    ap.add_argument("--normalize", action="store_true", help="Collapse '\\\\' -> '\\' in keys for tolerant matching (verification only).")
    ap.add_argument("--export-csv", default=None, help="Optional CSV export of verification issues.")
    args = ap.parse_args()

    if args.collect:
        if not args.ids:
            raise SystemExit("Please provide --ids for --collect.")
        collect_fix_results(args.ids, outdir=args.outdir, save_raw=True)

    elif args.verify:
        verify_fix_results(outdir=args.outdir, normalize=args.normalize, export_csv=args.export_csv)

if __name__ == "__main__":
    main()

"""
python3 src/postprocessing/check_fixed_classifications.py --collect \
  --ids batch_68d1083103288190a5e8fceaeaf66093 batch_68d1082f8e80819096bd0402df036093

# 2) Verify (allow extra patterns; require all OG hits labeled)
python3 src/postprocessing/check_fixed_classifications.py --verify
"""