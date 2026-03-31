# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end LLM pipeline to classify corporate sustainability disclosures (passages from German firm reports) as **Symbolic** vs **Substantive** mentions of SDGs and AI/Tech topics, grounded in legitimacy theory (Ashforth & Gibbs 1990, Suchman 1995).

**v2.3.0** (current `main` branch): New `src/postprocessing/` module — fix results, generate classified DuckDBs, aggregate to CSV.
**v2.2.0** (`v2.2.0` tag): New `src/classifications/` module with autonomous batch loop. `src_legacy/` fully removed.
**v2.0.0** (`v2.0.0` tag): Starts from pre-computed SDG/Tech keyword hits in DuckDB. Focuses on classification, post-processing, evaluation, and analysis.
**v1.0.0** (`v1-full-pipeline` branch): Full pipeline from raw PDFs through OCR/splitting to DuckDB.

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
├── data/
│   ├── dbs/                    # DuckDB files (hits + classified)
│   ├── classifications/
│   │   ├── batches/            # JSONL batch part files per model/domain
│   │   └── results_v2/         # Collected classification results
│   └── exports/                # Final CSV outputs
├── src/
│   ├── classifications/        # v2.2.0 — batch classification pipeline
│   │   ├── run.py              # Unified CLI (--build/--push/--check/--collect)
│   │   ├── batch_builder.py    # DuckDB → JSONL part files
│   │   ├── push_batches.py     # Autonomous submit loop
│   │   ├── poll_batches.py     # Batch status polling
│   │   ├── collect_results.py  # Result collection
│   │   ├── prompts.py          # Prompt definitions
│   │   └── analyze_results.py  # Result analysis utilities
│   └── analysis/
│       └── explore_dbs.py      # Explore classified DuckDBs
└── tests/
    └── prompts/                # Prompt strategy evaluation
```

## Pipeline Commands

### 1. Classification (Batch Submission)
```bash
python3 src/classifications/run.py --build                          # DuckDB → JSONL batch part files
python3 src/classifications/run.py --push [--model <id>]            # autonomous submit → poll → collect loop
python3 src/classifications/run.py --check [--batch <id>]           # check batch status
python3 src/classifications/run.py --collect [--model <id> --part <domain>]  # collect results
```

Config: `src/classifications/config.json` (gitignored)
Queue state: `data/classifications/queue.json` (gitignored)

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
Checkout the `v1-full-pipeline` branch for the full PDF → DuckDB preprocessing pipeline.
The `src_legacy/` directory has been removed from `main` as of v2.2.0.

## Architecture

### Data Flow
```
data/dbs/sdg_hits.duckdb
data/dbs/tech_hits.duckdb
  → batch_builder.py: DuckDB → JSONL part files (data/classifications/batches/<model>/<domain>_partNNNN.jsonl)
  → push_batches.py: autonomous loop → OpenAI Batch API (configured model, 24h window)
  → collect_results.py: → data/classifications/results_v2/<model>/<domain>/
  → (post-processing TBD) → data/dbs/{sdg,tech}_hits_classified.duckdb
  → data/exports/*.csv
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

**`src/classifications/`** — OpenAI batch classification pipeline (v2.2.0)
- `run.py`: Unified CLI entry point (`--build`, `--push`, `--check`, `--collect`); optional `--model` / `--part` / `--batch` filters
- `batch_builder.py`: DuckDB → JSONL part files, split by Tier 1 limits (50M tokens, 10k requests per part)
- `push_batches.py`: Autonomous submit loop — submits, polls, collects, and repeats per batch part
- `poll_batches.py`: Polls OpenAI Batch API status for pending batches
- `collect_results.py`: Downloads completed batch results to `data/classifications/results_v2/`
- `prompts.py`: Prompt definitions (system prompts for SDG and Tech classification)
- `config.json` / `queue.json`: Runtime state (gitignored — local only)

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
