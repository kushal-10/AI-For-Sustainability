# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end LLM pipeline to classify corporate sustainability disclosures (passages from German firm reports) as **Symbolic** vs **Substantive** mentions of SDGs and AI/Tech topics, grounded in legitimacy theory (Ashforth & Gibbs 1990, Suchman 1995).

**v2.0.0** (current `main` branch): Starts from pre-computed SDG/Tech keyword hits in DuckDB. Focuses on prompting strategies, evaluation, and analysis.
**v1.0.0** (`v1-full-pipeline` branch): Full pipeline from raw PDFs through OCR/splitting to DuckDB. Legacy source code lives in `src_legacy/`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
source prepare_path.sh   # sets PYTHONPATH to include repo root
export OPENAI_API_KEY=your_key
```

## Pipeline Commands

### 1. Classification (Batch Submission)
```bash
python3 src_legacy/gpt_classifier/batches.py -- build    # DuckDB → JSONL batch files
python3 src_legacy/gpt_classifier/batches.py -- submit   # submit to OpenAI Batch API (24h window)
python3 src_legacy/gpt_classifier/poll_batches.py        # monitor batch status
python3 src_legacy/gpt_classifier/collect_results.py     # collect completed results
```

### 2. Post-Processing
```bash
python3 src_legacy/postprocessing/check_classifications.py          # verify all passages classified
python3 src_legacy/postprocessing/fix_err_classifications.py        # resubmit unclassified passages
python3 src_legacy/postprocessing/apply_classifications.py          # merge results into classified DuckDBs
python3 src_legacy/postprocessing/apply_fixed_classifications.py    # run if fix_err was needed
python3 src_legacy/postprocessing/build_company_year_summary.py     # aggregate to firm-year level
python3 src_legacy/postprocessing/merge_kw_filter.py                # merge keyword + classification data
python3 src_legacy/postprocessing/add_lang.py                       # add language detection metadata
python3 src_legacy/postprocessing/add_token.py                      # add token count estimates
```

### 3. Prompt Evaluation (tests/prompts/)
```bash
python3 tests/prompts/run_batches.py --create     # create batch files for all prompt variants
python3 tests/prompts/run_batches.py --push [prompt_name]  # submit batches
python3 tests/prompts/run_batches.py --poll       # check status
python3 tests/prompts/analysis.py                 # compute accuracy vs ground truth
```

### 4. Analysis
```bash
python3 src_legacy/analysis/cost.py                      # estimate token costs for batch files
```

## Architecture

### Data Flow
```
DuckDB (sdg_hits / tech_hits)
  → batches.py: exports to JSONL (10k items/file, data/batches/{sdgs,tech}/)
  → OpenAI Batch API (gpt-4.1-mini, 24h window)
  → collect_results.py: aggregates to results_map.json
  → apply_classifications.py: merges back into {sdg,tech}_hits_classified.duckdb
  → build_company_year_summary.py + merge_kw_filter.py: final CSVs
```

### Key Modules

**`src_legacy/filtering/`** — Keyword filtering pipeline
- `base_filter.py`: Abstract base with DuckDB Appender pattern, wildcard regex expansion, language detection
- `sdg_filter.py` / `tech_filter.py`: SDG (17 categories sdg1–sdg17) and Tech (4 categories: ai_ml, cloud_computing, big_data_blockchain, applications_practice) filters
- Keywords stored in `kw_data/keywords_{sdg,tech}_{en,de}.json`

**`src_legacy/gpt_classifier/`** — OpenAI batch classification
- `objects.py`: System prompts and batch object builders for SDG/Tech (`create_batch_object_sdg`, `create_batch_object_tech`)
- `batches.py`: Orchestrates DuckDB → JSONL → OpenAI Batch API submission
- Model: `gpt-4.1-mini`

**`src_legacy/postprocessing/`** — Result aggregation and cleanup
- Produces `{sdg,tech}_hits_classified.duckdb` and final firm-year CSV summaries

**`src_legacy/analysis/`** — Metrics and cost estimation
- `cost.py`: Uses tiktoken `o200k_base` encoding for accurate token counting

**`tests/prompts/`** — Prompt strategy evaluation
- `prompts.py`: Four strategies — zero_shot, few_shot, CoT (Chain of Thought), ToT (Tree of Thought)
- `run_batches.py`: Tests gpt-5.2 (4 reasoning modes × 4 prompts) and gpt-4o (4 prompts)
- Ground truth: `tests/prompts/passage_keyword_truth.csv`
- Results in `tests/prompts/results/` and `tests/prompts/results_4o/`

### Database Schema
All intermediate data stored as DuckDB files in `data/dbs/`. Hit columns follow naming patterns: `hits_sdg*` for SDG categories, `hits_ai_ml` / `hits_cloud_computing` / etc. for Tech categories.

### Classification Task
Each passage is classified as either **Symbolic** (ceremonial/legitimacy-seeking, no concrete action) or **Substantive** (concrete actions, measurable commitments) per legitimacy theory. Prompts are in `src/gpt_classifier/objects.py` (production) and `tests/prompts/prompts.py` (evaluation variants).
