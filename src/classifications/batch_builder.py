"""
batch_builder.py — Methods for building JSONL batch files from SDG/Tech DuckDB hits.

SDG is split into three files per entry to stay under the OpenAI 100 MB batch limit:
  sdg_a.jsonl  — SDG categories 1–9
  sdg_b.jsonl  — SDG categories 10–13
  sdg_c.jsonl  — SDG categories 14–17
  tech.jsonl   — all 4 Tech categories

Output paths:
  data/classifications/batches/{entry_id}/sdg_a.jsonl
  data/classifications/batches/{entry_id}/sdg_b.jsonl
  data/classifications/batches/{entry_id}/sdg_c.jsonl
  data/classifications/batches/{entry_id}/tech.jsonl

Prompt selection:
  SDG  — variant from src/classifications/prompts.py (zero_shot, few_shot, cot, tot)
  Tech — variant from src/classifications/prompts.py (zero_shot, few_shot, cot, tot)

Only entries with all batch_ids empty are processed (skips already-submitted).
"""

import json
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

# ── SDG split ──────────────────────────────────────────────────────────────────
# Three-way split keeps each JSONL file under OpenAI's 100 MB batch file limit.

SDG_COLS_A = [f"hits_sdg{i}" for i in range(1, 10)]    # sdg1–sdg9
SDG_COLS_B = [f"hits_sdg{i}" for i in range(10, 14)]   # sdg10–sdg13
SDG_COLS_C = [f"hits_sdg{i}" for i in range(14, 18)]   # sdg14–sdg17
TECH_HIT_COLS = {
    "hits_ai_ml",
    "hits_cloud_computing",
    "hits_big_data_blockchain",
    "hits_applications_practice",
}

# ── Config helpers ─────────────────────────────────────────────────────────────

def load_config(config_path: str) -> list[dict]:
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def save_config(config_path: str, config: list[dict]) -> None:
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")


# ── Path helpers ───────────────────────────────────────────────────────────────

def jsonl_path(out_base: str, entry_id: str, domain: str) -> Path:
    """
    domain: "sdg_a" | "sdg_b" | "sdg_c" | "tech"
    """
    return Path(out_base) / entry_id / f"{domain}.jsonl"


# ── Hit column helpers ─────────────────────────────────────────────────────────

def _maybe_parse_json(v: Any) -> list | dict | None:
    """Parse a hit cell. Handles both list (unclassified) and dict (classified) formats."""
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
    """
    Build {col: [matched_patterns]} for non-empty hit columns.
    Handles both formats:
      - Unclassified DB: JSON list  → ["pattern1", ...]
      - Classified DB:   JSON object → {"pattern1": "symbolic", ...}  (keys are patterns)
    """
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
    """Return one OpenAI Batch API request object."""
    # domain is "sdg_a", "sdg_b", "sdg_c", or "tech"
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
        # Reasoning models (gpt-5.2 / o-series) — large budget for chain-of-thought
        body["reasoning_effort"]      = reasoning_effort
        body["max_completion_tokens"] = 30000
    else:
        # Standard models (gpt-4o etc.) — max output is 4096 tokens
        body["temperature"]           = 0
        body["max_completion_tokens"] = 4096

    return {
        "custom_id": f"{domain}||{global_id}",
        "method":    "POST",
        "url":       "/v1/chat/completions",
        "body":      body,
    }


# ── DuckDB → JSONL ─────────────────────────────────────────────────────────────

def build_jsonl(
    df: pd.DataFrame,
    hit_cols: list[str],
    id_col: str,
    passage_col: str,
    domain: str,
    sys_prompt: str,
    model: str,
    reasoning_effort: str | None,
    out_path: Path,
) -> int:
    """
    Build batch objects from a pre-loaded DataFrame using only the given hit_cols.
    Writes to out_path as JSONL. Returns number of objects written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
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
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
    return count


def _load_db(db_path: str, table: str) -> tuple[pd.DataFrame, str, str]:
    """Load table from DuckDB, return (df, id_col, passage_col)."""
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
    sdg_table: str        = "sdg_hits_classified",
    tech_table: str       = "tech_hits_classified",
    filter_domain: str | None = None,
) -> None:
    """
    Build sdg_a.jsonl, sdg_b.jsonl, sdg_c.jsonl, and tech.jsonl for one config entry.
    Pass filter_domain to build only that domain (e.g. "sdg-a", "tech").
    Skips if all targeted batch_ids are already set or files already exist on disk.
    """
    entry_id         = entry["id"]
    model            = entry["model"]
    reasoning_effort = entry.get("reasoning_effort")
    prompt_type      = entry["prompt_type"]

    def _want(domain: str) -> bool:
        return not filter_domain or _norm(filter_domain) == _norm(domain)

    sdg_domains = [
        (jsonl_path(out_base, entry_id, "sdg_a"), SDG_COLS_A, "sdg_a", "sdg1–sdg9"),
        (jsonl_path(out_base, entry_id, "sdg_b"), SDG_COLS_B, "sdg_b", "sdg10–sdg13"),
        (jsonl_path(out_base, entry_id, "sdg_c"), SDG_COLS_C, "sdg_c", "sdg14–sdg17"),
    ]

    all_submitted = (
        entry.get("batch_id_sdg_a")
        and entry.get("batch_id_sdg_b")
        and entry.get("batch_id_sdg_c")
        and entry.get("batch_id_tech")
    )
    if all_submitted and not filter_domain:
        print(f"[SKIP] {entry_id} — already submitted")
        return

    sdg_sys_prompt  = SDG_PROMPTS[prompt_type]
    tech_sys_prompt = TECH_PROMPTS[prompt_type]

    # ── SDG ────────────────────────────────────────────────────────────────────
    wanted_sdg = [(p, c, lbl, span) for p, c, lbl, span in sdg_domains if _want(lbl)]
    if wanted_sdg:
        if all(p.exists() for p, _, _, _ in wanted_sdg):
            for _, _, lbl, _ in wanted_sdg:
                print(f"[SKIP] {entry_id}/{lbl}.jsonl already exists")
        else:
            df_sdg, id_col, passage_col = _load_db(sdg_db, sdg_table)
            available_cols = set(df_sdg.columns)
            for out_path, cols, lbl, span in wanted_sdg:
                if out_path.exists():
                    print(f"[SKIP] {entry_id}/{lbl}.jsonl already exists")
                else:
                    filtered = [c for c in cols if c in available_cols]
                    n = build_jsonl(df_sdg, filtered, id_col, passage_col,
                                    lbl, sdg_sys_prompt, model, reasoning_effort, out_path)
                    print(f"[OK]   {entry_id}/{lbl}.jsonl — {n:,} requests  ({span})")

    # ── Tech ───────────────────────────────────────────────────────────────────
    if _want("tech"):
        tech_out = jsonl_path(out_base, entry_id, "tech")
        if tech_out.exists():
            print(f"[SKIP] {entry_id}/tech.jsonl already exists")
        else:
            df_tech, id_col_t, passage_col_t = _load_db(tech_db, tech_table)
            tech_cols = [c for c in df_tech.columns if c in TECH_HIT_COLS]
            n_tech = build_jsonl(df_tech, tech_cols, id_col_t, passage_col_t,
                                 "tech", tech_sys_prompt, model, reasoning_effort, tech_out)
            print(f"[OK]   {entry_id}/tech.jsonl  — {n_tech:,} requests")


# ── Build all ──────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return s.lower().replace("-", "_")


def build_all(
    config_path: str,
    sdg_db: str,
    tech_db: str,
    out_base: str,
    sdg_table: str        = "sdg_hits_classified",
    tech_table: str       = "tech_hits_classified",
    filter_entry: str | None  = None,
    filter_domain: str | None = None,
) -> None:
    """
    Build JSONL files for every config entry that hasn't been submitted yet.

    filter_entry  — substring match on entry id (e.g. "4o-tot", "high-cot").
    filter_domain — restrict to one domain ("sdg-a", "sdg-b", "sdg-c", "tech").
    """
    config = load_config(config_path)
    scoped = [
        e for e in config
        if not filter_entry or _norm(filter_entry) in _norm(e["id"])
    ]
    if not scoped:
        print(f"No config entries matched filter '{filter_entry}'. Available: {[e['id'] for e in config]}")
        return

    label = f"entry='{filter_entry or 'all'}'  domain='{filter_domain or 'all'}'"
    print(f"Config: {len(scoped)} / {len(config)} entries  (filter: {label})\n")
    for entry in scoped:
        build_for_entry(entry, sdg_db, tech_db, out_base, sdg_table, tech_table,
                        filter_domain=filter_domain)
    print("\nDone. Run --push to submit to OpenAI.")
