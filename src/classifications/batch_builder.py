"""
batch_builder.py — Build JSONL batch files from SDG/Tech DuckDB hits.

Each domain is split into numbered part files to stay under the Tier 1
OpenAI batch queue limit (1,500,000 input tokens per batch):

  {out_base}/{entry_id}/sdg_a_part0001.jsonl
  {out_base}/{entry_id}/sdg_a_part0002.jsonl
  ...
  {out_base}/{entry_id}/sdg_b_part0001.jsonl
  {out_base}/{entry_id}/sdg_c_part0001.jsonl
  {out_base}/{entry_id}/tech_part0001.jsonl

Splitting rules (whichever is hit first):
  - TOKEN_LIMIT = 900,000 input tokens  (Tier 1 batch queue limit)
  - REQ_LIMIT   = 500 requests         (OpenAI batch file limit)

Config tracking:
  batch_ids_sdg_a / _b / _c / _tech  are dicts:  {file_path: batch_id_or_""}
  Empty string = built but not yet submitted.
  Filled string = submitted; value is the OpenAI batch ID.
"""

import json
import glob
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from src.classifications.prompts import (
    SYS_PROMPT_SDG_ZERO_SHOT,
    SYS_PROMPT_SDG_FEW_SHOT,
    SYS_PROMPT_SDG_COT,
    SYS_PROMPT_SDG_TOT,
    SYS_PROMPT_TECH_ZERO_SHOT,
    SYS_PROMPT_TECH_FEW_SHOT,
    SYS_PROMPT_TECH_COT,
    SYS_PROMPT_TECH_TOT,
)

# ── Tier 1 limits ──────────────────────────────────────────────────────────────

TOKEN_LIMIT = 900_000   # input tokens per batch (Tier 1 batch queue limit)
REQ_LIMIT   = 500      # max requests per batch file

# ── Prompt lookup tables ───────────────────────────────────────────────────────

SDG_PROMPTS: dict[str, str] = {
    "zero_shot": SYS_PROMPT_SDG_ZERO_SHOT,
    "few_shot":  SYS_PROMPT_SDG_FEW_SHOT,
    "cot":       SYS_PROMPT_SDG_COT,
    "tot":       SYS_PROMPT_SDG_TOT,
}

TECH_PROMPTS: dict[str, str] = {
    "zero_shot": SYS_PROMPT_TECH_ZERO_SHOT,
    "few_shot":  SYS_PROMPT_TECH_FEW_SHOT,
    "cot":       SYS_PROMPT_TECH_COT,
    "tot":       SYS_PROMPT_TECH_TOT,
}

# ── SDG column splits ──────────────────────────────────────────────────────────

SDG_COLS_A = [f"hits_sdg{i}" for i in range(1, 10)]    # sdg1–sdg9
SDG_COLS_B = [f"hits_sdg{i}" for i in range(10, 14)]   # sdg10–sdg13
SDG_COLS_C = [f"hits_sdg{i}" for i in range(14, 18)]   # sdg14–sdg17
TECH_HIT_COLS = {
    "hits_ai_ml",
    "hits_cloud_computing",
    "hits_big_data_blockchain",
    "hits_applications_practice",
}

# ── Token counter ──────────────────────────────────────────────────────────────

class TokenCounter:
    def __init__(self):
        self._tok = None
        try:
            import tiktoken
            try:
                self._tok = tiktoken.get_encoding("o200k_base")
            except Exception:
                self._tok = tiktoken.get_encoding("cl100k_base")
            self._use_tiktoken = True
        except Exception:
            self._use_tiktoken = False

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._use_tiktoken and self._tok:
            try:
                return len(self._tok.encode(text))
            except Exception:
                pass
        return max(1, int(len(text) / 4.0))


def _count_input_tokens(obj: dict, tc: TokenCounter) -> int:
    """Estimate input tokens for one batch request (sum of all message contents)."""
    body     = obj.get("body", {})
    messages = body.get("messages", [])
    return sum(tc.count(str(m.get("content", ""))) for m in messages)

# ── Config helpers ─────────────────────────────────────────────────────────────

def load_config(config_path: str) -> list[dict]:
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def save_config(config_path: str, config: list[dict]) -> None:
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")


# ── Path helpers ───────────────────────────────────────────────────────────────

def entry_dir(out_base: str, entry_id: str) -> Path:
    return Path(out_base) / entry_id


def part_glob(out_base: str, entry_id: str, domain: str) -> list[Path]:
    """Return sorted list of existing part files for an entry/domain."""
    pattern = str(entry_dir(out_base, entry_id) / f"{domain}_part*.jsonl")
    return sorted(Path(p) for p in glob.glob(pattern))


# ── Hit column helpers ─────────────────────────────────────────────────────────

def _maybe_parse_json(v: Any) -> list | dict | None:
    if v is None:
        return None
    if isinstance(v, (list, dict)):
        return v
    if isinstance(v, str):
        s = v.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                return json.loads(s)
            except Exception:
                pass
    return None


def _hits_dict(row: dict, hit_cols: list[str]) -> dict[str, list[str]]:
    out = {}
    for col in hit_cols:
        parsed = _maybe_parse_json(row.get(col))
        if isinstance(parsed, list):
            patterns = [p for p in parsed if isinstance(p, str) and p.strip()]
        elif isinstance(parsed, dict):
            patterns = [k for k in parsed if isinstance(k, str) and k.strip()]
        else:
            continue
        if patterns:
            out[col] = patterns
    return out


# ── Batch object builder ───────────────────────────────────────────────────────

def make_batch_object(
    passage: str,
    global_id: str,
    hits_dict: dict,
    domain: str,
    sys_prompt: str,
    model: str,
    reasoning_effort: str | None,
) -> dict:
    hits_label   = "SDG_HITS" if domain.startswith("sdg") else "TECH_HITS"
    user_content = f"Passage:\n{passage}\n\n{hits_label}:\n{hits_dict}"

    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_content},
        ],
    }

    if reasoning_effort:
        body["reasoning_effort"]      = reasoning_effort
        body["max_completion_tokens"] = 30000
    else:
        body["temperature"]           = 0
        body["max_completion_tokens"] = 4096

    return {
        "custom_id": f"{domain}||{global_id}",
        "method":    "POST",
        "url":       "/v1/chat/completions",
        "body":      body,
    }


# ── DuckDB → JSONL (with token-based splitting) ────────────────────────────────

def build_jsonl(
    df: pd.DataFrame,
    hit_cols: list[str],
    id_col: str,
    passage_col: str,
    domain: str,
    sys_prompt: str,
    model: str,
    reasoning_effort: str | None,
    out_dir: Path,
    token_limit: int = TOKEN_LIMIT,
    req_limit: int   = REQ_LIMIT,
) -> list[Path]:
    """
    Build batch objects from df, splitting into numbered part files whenever
    TOKEN_LIMIT input tokens OR REQ_LIMIT requests would be exceeded.

    Returns list of written part file paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tc = TokenCounter()

    parts: list[Path] = []
    part_num   = 1
    buf: list[dict] = []
    buf_tokens = 0

    def _flush():
        nonlocal part_num
        path = out_dir / f"{domain}_part{part_num:04d}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for obj in buf:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        parts.append(path)
        part_num += 1

    for _, row in df.iterrows():
        row_d = row.to_dict()
        hits  = _hits_dict(row_d, hit_cols)
        if not hits:
            continue
        obj = make_batch_object(
            passage          = str(row_d[passage_col]),
            global_id        = str(row_d[id_col]),
            hits_dict        = hits,
            domain           = domain,
            sys_prompt       = sys_prompt,
            model            = model,
            reasoning_effort = reasoning_effort,
        )
        obj_tokens = _count_input_tokens(obj, tc)
        if buf and (len(buf) >= req_limit or buf_tokens + obj_tokens > token_limit):
            _flush()
            buf        = []
            buf_tokens = 0
        buf.append(obj)
        buf_tokens += obj_tokens

    if buf:
        _flush()

    return parts


def _load_db(db_path: str, table: str) -> tuple[pd.DataFrame, str, str]:
    con = duckdb.connect(db_path, read_only=True)
    df  = con.execute(f"SELECT * FROM {table}").fetchdf()
    con.close()
    cols = list(df.columns)
    id_col = next((c for c in ("global_id", "id", "uid") if c in cols), None)
    if id_col is None:
        raise ValueError(f"No id column found in {db_path} ({table})")
    passage_col = next((c for c in ("passage", "sentence", "text", "content") if c in cols), None)
    if passage_col is None:
        raise ValueError(f"No passage column found in {db_path} ({table})")
    return df, id_col, passage_col


# ── Per-entry builder ──────────────────────────────────────────────────────────

def build_for_entry(
    entry: dict,
    sdg_db: str,
    tech_db: str,
    out_base: str,
    sdg_table: str            = "sdg_hits_classified",
    tech_table: str           = "tech_hits_classified",
    filter_domain: str | None = None,
) -> None:
    """
    Build sdg_a, sdg_b, sdg_c, and tech part files for one config entry.
    Skips a domain if part0001 already exists on disk.
    """
    entry_id         = entry["id"]
    model            = entry["model"]
    reasoning_effort = entry.get("reasoning_effort")
    prompt_type      = entry["prompt_type"]
    out_dir          = entry_dir(out_base, entry_id)

    def _want(domain: str) -> bool:
        return not filter_domain or _norm(filter_domain) == _norm(domain)

    sdg_domains = [
        (SDG_COLS_A, "sdg_a", "sdg1–sdg9"),
        (SDG_COLS_B, "sdg_b", "sdg10–sdg13"),
        (SDG_COLS_C, "sdg_c", "sdg14–sdg17"),
    ]

    sdg_sys_prompt  = SDG_PROMPTS[prompt_type]
    tech_sys_prompt = TECH_PROMPTS[prompt_type]

    # ── SDG ────────────────────────────────────────────────────────────────────
    wanted_sdg = [(c, lbl, span) for c, lbl, span in sdg_domains if _want(lbl)]
    if wanted_sdg:
        df_sdg, id_col, passage_col = _load_db(sdg_db, sdg_table)
        available_cols = set(df_sdg.columns)
        for cols, lbl, span in wanted_sdg:
            first_part = out_dir / f"{lbl}_part0001.jsonl"
            if first_part.exists():
                existing = part_glob(out_base, entry_id, lbl)
                print(f"[SKIP] {entry_id}/{lbl} — {len(existing)} part(s) already on disk")
                continue
            filtered = [c for c in cols if c in available_cols]
            parts = build_jsonl(df_sdg, filtered, id_col, passage_col,
                                lbl, sdg_sys_prompt, model, reasoning_effort, out_dir)
            total = sum(1 for _ in open(parts[0]) if _.strip()) if parts else 0
            print(f"[OK]   {entry_id}/{lbl} ({span}) — {len(parts)} part(s)")
            for p in parts:
                n = sum(1 for _ in open(p) if _.strip())
                print(f"         {p.name}  ({n:,} requests)")

    # ── Tech ───────────────────────────────────────────────────────────────────
    if _want("tech"):
        first_part = out_dir / "tech_part0001.jsonl"
        if first_part.exists():
            existing = part_glob(out_base, entry_id, "tech")
            print(f"[SKIP] {entry_id}/tech — {len(existing)} part(s) already on disk")
        else:
            df_tech, id_col_t, passage_col_t = _load_db(tech_db, tech_table)
            tech_cols = [c for c in df_tech.columns if c in TECH_HIT_COLS]
            parts = build_jsonl(df_tech, tech_cols, id_col_t, passage_col_t,
                                "tech", tech_sys_prompt, model, reasoning_effort, out_dir)
            print(f"[OK]   {entry_id}/tech — {len(parts)} part(s)")
            for p in parts:
                n = sum(1 for _ in open(p) if _.strip())
                print(f"         {p.name}  ({n:,} requests)")


# ── Build all ──────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return s.lower().replace("-", "_")


def build_all(
    config_path: str,
    sdg_db: str,
    tech_db: str,
    out_base: str,
    sdg_table: str            = "sdg_hits_classified",
    tech_table: str           = "tech_hits_classified",
    filter_entry: str | None  = None,
    filter_domain: str | None = None,
) -> None:
    config = load_config(config_path)
    scoped = [e for e in config if not filter_entry or _norm(filter_entry) in _norm(e["id"])]
    if not scoped:
        print(f"No config entries matched filter '{filter_entry}'. Available: {[e['id'] for e in config]}")
        return

    label = f"entry='{filter_entry or 'all'}'  domain='{filter_domain or 'all'}'"
    print(f"Config: {len(scoped)} / {len(config)} entries  (filter: {label})")
    print(f"Limits: {TOKEN_LIMIT:,} tokens / {REQ_LIMIT:,} requests per part file\n")
    for entry in scoped:
        build_for_entry(entry, sdg_db, tech_db, out_base, sdg_table, tech_table,
                        filter_domain=filter_domain)
    print("\nDone. Run --push to submit to OpenAI (one part file at a time recommended).")
