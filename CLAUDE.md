# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end LLM pipeline to classify corporate sustainability disclosures (passages from German firm reports) as **Symbolic** vs **Substantive** mentions of SDGs and AI/Tech topics, grounded in legitimacy theory (Ashforth & Gibbs 1990, Suchman 1995).

**v2.0.0** (current `main` branch): Starts from pre-computed SDG/Tech keyword hits in DuckDB. Focuses on classification, post-processing, evaluation, and analysis.
**v1.0.0** (`v1-full-pipeline` branch): Full pipeline from raw PDFs through OCR/splitting to DuckDB. Legacy source code lives in `src_legacy/`.

**Dataset:** 153 German companies, 1,420 sustainability reports (multi-year). Passages split at semantic boundaries. Languages: German (`de`), English (`en`).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
source prepare_path.sh   # sets PYTHONPATH to include repo root
export OPENAI_API_KEY=your_key
```

## Repository Structure

```
├── kw_data/              # Keyword JSON files for SDG and Tech categories
├── data/
│   ├── dbs/              # DuckDB files (hits + classified)
│   ├── batches/          # JSONL batch files and OpenAI results
│   └── exports/          # Final CSV outputs
├── src/                  # v2.0.0 new code (analysis utilities)
│   └── analysis/
│       └── explore_dbs.py
├── src_legacy/           # Full pipeline code (used in both v1 and v2)
│   ├── preprocessing/    # PDF → text → passages (v1.0 only)
│   ├── filtering/        # Keyword matching → DuckDB (v1.0 only)
│   ├── gpt_classifier/   # OpenAI Batch API classification
│   ├── postprocessing/   # Result merging and aggregation
│   ├── analysis/         # Cost estimation, filtering stats
│   └── utils/            # File I/O, firm metadata, token utilities
└── tests/
    └── prompts/          # Prompt strategy evaluation
```

## Pipeline Commands

### 1. Classification (Batch Submission)
```bash
python3 src_legacy/gpt_classifier/batches.py -- build    # DuckDB → JSONL batch files
python3 src_legacy/gpt_classifier/batches.py -- submit   # submit to OpenAI Batch API (24h window)
python3 src_legacy/gpt_classifier/poll_batches.py        # monitor batch status
python3 src_legacy/gpt_classifier/collect_results.py     # collect completed results → results_all_map.json
python3 src_legacy/gpt_classifier/fix_results.py         # fix JSON parse errors in results (if needed)
```

### 2. Post-Processing
```bash
python3 src_legacy/postprocessing/check_classifications.py          # verify all passages classified
python3 src_legacy/postprocessing/fix_err_classifications.py        # resubmit unclassified passages (if needed)
python3 src_legacy/postprocessing/apply_classifications.py          # merge results → classified DuckDBs
python3 src_legacy/postprocessing/apply_fixed_classifications.py    # apply fixed results (only if fix_err was run)
python3 src_legacy/postprocessing/build_company_year_summary.py     # aggregate to firm-year level CSV
python3 src_legacy/postprocessing/merge_kw_filter.py                # merge keyword + classification data
python3 src_legacy/postprocessing/add_lang.py                       # add language detection metadata
python3 src_legacy/postprocessing/add_token.py                      # add token count estimates
```

### 3. Analysis
```bash
python3 src/analysis/explore_dbs.py           # explore classified DBs, print column names + stats
python3 src_legacy/analysis/cost.py           # estimate token costs for batch files
```

### 4. Prompt Evaluation (tests/prompts/)
```bash
python3 tests/prompts/run_batches.py --create            # create batch files for all prompt variants
python3 tests/prompts/run_batches.py --push [prompt_name] # submit batches
python3 tests/prompts/run_batches.py --poll              # check status
python3 tests/prompts/analysis.py                        # compute accuracy vs ground truth
```

### Legacy: Full Pipeline from PDFs (v1.0.0 — `v1-full-pipeline` branch)
```bash
python3 src_legacy/preprocessing/pdf2text.py      # PDF → text
python3 src_legacy/preprocessing/splitter.py      # text → passages
python3 src_legacy/filtering/sdg_filter.py        # passages → sdg_hits.duckdb
python3 src_legacy/filtering/tech_filter.py       # passages → tech_hits.duckdb
# then follow classification + post-processing steps above
```

## Architecture

### Data Flow
```
data/dbs/sdg_hits.duckdb
data/dbs/tech_hits.duckdb
  → batches.py: DuckDB → JSONL (10k items/file, data/batches/{sdgs,tech}/)
  → OpenAI Batch API (gpt-4.1-mini, 24h window)
  → collect_results.py: → data/batches/results/results_all_map.json
  → fix_results.py: (if needed) → results_fixed_map.json
  → apply_classifications.py: → data/dbs/{sdg,tech}_hits_classified.duckdb
  → build_company_year_summary.py + merge_kw_filter.py: → data/exports/*.csv
```

### Database Schema

**`sdg_hits` / `tech_hits` (pre-classification):**
- `global_id` (VARCHAR) — unique ID: `YYYY<company_normalized><sentence_id>`
- `passage` (VARCHAR) — text passage
- `company` (VARCHAR) — firm name
- `year` (INTEGER) — report year
- `language` (VARCHAR) — `en`, `de`, or `unknown`
- Hit columns store a JSON list of matched regex patterns:
  - SDG: `hits_sdg1` … `hits_sdg17`
  - Tech: `hits_ai_ml`, `hits_cloud_computing`, `hits_big_data_blockchain`, `hits_applications_practice`

**`sdg_hits_classified` / `tech_hits_classified` (post-classification):**
- Same schema, but hit columns store a JSON object mapping each pattern to its label:
  `{"<pattern>": "symbolic" | "substantive", ...}` — empty `{}` means no match for that category.

### Key Modules

**`src_legacy/filtering/`** — Keyword filtering pipeline (v1.0 only)
- `base_filter.py`: Abstract base — DuckDB Appender pattern, wildcard regex expansion (`*` → `\w*` suffix, `.*` span), language detection via pycld3/langdetect
- `sdg_filter.py` / `tech_filter.py`: Concrete filters for SDG (17 categories) and Tech (4 categories)
- Keywords: `kw_data/keywords_{sdg,tech}_{en,de}.json`

**`src_legacy/gpt_classifier/`** — OpenAI batch classification
- `objects.py`: `SYS_PROMPT_SDG`, `SYS_PROMPT_TECH`, `create_batch_object_sdg()`, `create_batch_object_tech()`
- `batches.py`: Orchestrates DuckDB → JSONL → OpenAI Batch API (model: `gpt-4.1-mini`, temperature=0, max_tokens=150)
- `fix_results.py`: Detects and salvages JSON parse errors in batch responses
- Custom IDs: `sdg||<global_id>` or `tech||<global_id>`

**`src_legacy/postprocessing/`** — Result aggregation and cleanup
- Produces `{sdg,tech}_hits_classified.duckdb` and firm-year CSV summaries (`data/exports/`)

**`src_legacy/analysis/`** — Metrics and cost estimation
- `cost.py`: Token cost estimation using tiktoken `o200k_base` encoding

**`src_legacy/utils/`** — Shared utilities
- `firms.py`: Metadata for 153 companies, 1,420 report files
- `tokens.py`: Token counting with tiktoken (`o200k_base` → `cl100k_base` → heuristic fallback)

**`src/analysis/explore_dbs.py`** — Explore classified DBs
- Prints column names, row counts, company/year/language breakdown, classification counts per category, sample passages

**`tests/prompts/`** — Prompt strategy evaluation
- `prompts.py`: Four strategies — `zero_shot`, `few_shot`, `CoT` (Chain of Thought), `ToT` (Tree of Thought)
- `run_batches.py`: Tests gpt-5.2 (4 reasoning modes × 4 prompts) and gpt-4o (4 prompts)
- Ground truth: `tests/prompts/passage_keyword_truth.csv`
- Results: `tests/prompts/results/` (gpt-5.2) and `tests/prompts/results_4o/` (gpt-4o)

### Classification Task

Each matched keyword pattern is classified as either:
- **Symbolic** — vague/aspirational, policy/compliance-only, legal citations, no concrete actions, resources, timelines, or KPIs
- **Substantive** — concrete implementation/results, projects/pilots, budgets/teams, timelines, KPIs/metrics, quantified impact, verified reporting

Prompts are in `src_legacy/gpt_classifier/objects.py` (production) and `tests/prompts/prompts.py` (evaluation variants).
