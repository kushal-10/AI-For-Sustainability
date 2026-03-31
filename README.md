# AI-For-Sustainability

End-to-end LLM pipeline to classify corporate sustainability disclosures (passages from German firm reports) as **Symbolic** vs **Substantive** mentions of SDGs and AI/Tech topics, grounded in legitimacy theory (Ashforth & Gibbs 1990, Suchman 1995).

**Dataset:** 153 German companies, 1,420 sustainability reports (multi-year). Passages split at semantic boundaries. Languages: German (`de`) and English (`en`).


## Setup

```bash
git clone https://github.com/kushal-10/AI-For-Sustainability.git
cd AI-For-Sustainability
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
source prepare_path.sh        # adds repo root to PYTHONPATH
export OPENAI_API_KEY=your_key
```

`prepare_path.sh` must be sourced in every new shell before running any `src/` script. Alternatively, add your key to `.env`.


## Versions

| Version | Branch/Tag | Description |
|---------|------------|-------------|
| v1.0.0 | `v1-full-pipeline` | Full pipeline from raw PDFs → OCR → splitting → DuckDB → classification |
| v2.0.0 | `v2.0.0` | Starts from pre-computed SDG/Tech keyword hits; new prompts and evaluation |
| v2.1.0 | `v2.1.0` | gpt-5.2 classification results and prompt evaluation |
| v2.2.0 | `main` | New `src/classifications/` module with autonomous batch loop; `src_legacy/` removed |

To reproduce the full PDF → DuckDB preprocessing, checkout the `v1-full-pipeline` branch.


## Classification Task

Each keyword-matched passage is classified as:

- **Symbolic** — vague or aspirational language; policy/compliance-only; legal citations; no concrete actions, budgets, timelines, or KPIs
- **Substantive** — concrete implementation or results; specific projects/pilots; allocated budgets or teams; timelines; KPIs/metrics; quantified impact; verified reporting


---


## End-to-End Workflow

Starting point: two source DuckDB files under `data/dbs/` containing all passages with keyword hits. The full pipeline runs in four stages.

```
data/dbs/sdg_hits.duckdb
data/dbs/tech_hits.duckdb
       │
       │  Stage 1 — src/classifications/
       │             Build JSONL batches + submit to OpenAI Batch API
       ▼
data/classifications/results/<model_config>/
       │
       │  Stage 2 — src/postprocessing/fix_results.py
       │             Detect + fix malformed result rows
       ▼
data/classifications/results/<model_config>/   (fixed in-place, originals backed up)
       │
       │  Stage 3 — src/postprocessing/generate_results_db.py
       │             Merge results JSON → classified DuckDBs
       ▼
data/dbs/<model_config>/sdg_hits_classified.duckdb
data/dbs/<model_config>/tech_hits_classified.duckdb
       │
       │  Stage 4 — src/postprocessing/generate_sym_sub_results.py
       │             Aggregate to company×year symbolic/substantive counts
       ▼
data/exports/<model_config>_sym_sub.csv
```


---


## Stage 1 — Classification (`src/classifications/`)

Submits passages to the OpenAI Batch API and collects symbolic/substantive labels per regex pattern.

### Scripts

| Script | Role |
|--------|------|
| `run.py` | Unified CLI — entry point for all classification operations |
| `batch_builder.py` | Reads source DuckDBs → writes JSONL part files under `data/classifications/batches/` |
| `push_batches.py` | Autonomous loop: submits one part file, polls until complete, collects results, repeats |
| `poll_batches.py` | Polls OpenAI Batch API status for all pending batches |
| `collect_results.py` | Downloads completed batch output and merges all parts into per-domain result JSONs |
| `prompts.py` | System prompt definitions for SDG and Tech classification (zero_shot / few_shot / cot / tot) |

### Config (`src/classifications/config.json`)

Gitignored — create locally before running. Each entry defines one model config:

```json
[
  {
    "id": "gpt-5.2__high__cot",
    "model": "gpt-5.2",
    "reasoning_effort": "high",
    "prompt_type": "cot",
    "batch_ids_sdg_a": {},
    "batch_ids_sdg_b": {},
    "batch_ids_sdg_c": {},
    "batch_ids_tech":  {}
  }
]
```

`batch_ids_*` maps are filled in automatically as part files are submitted (`{ "path/to/part.jsonl": "batch_abc123" }`). State persists between runs so the push loop can resume after interruptions.

### Domain Splits

SDG passages are split into three domains to stay within OpenAI Tier 1 batch limits (50M tokens / 10k requests per part file):

| Domain | Hit columns | SDGs covered |
|--------|-------------|--------------|
| `sdg_a` | `hits_sdg1` – `hits_sdg9` | SDG 1–9 |
| `sdg_b` | `hits_sdg10` – `hits_sdg13` | SDG 10–13 |
| `sdg_c` | `hits_sdg14` – `hits_sdg17` | SDG 14–17 |
| `tech`  | `hits_ai_ml`, `hits_cloud_computing`, `hits_big_data_blockchain`, `hits_applications_practice` | All tech |

### Commands

```bash
# Build JSONL part files from source DuckDBs
python3 src/classifications/run.py --build

# Submit → poll → collect (autonomous loop, resumes from last saved state)
python3 src/classifications/run.py --push

# Check status of all submitted batches
python3 src/classifications/run.py --check

# Manually collect completed results
python3 src/classifications/run.py --collect

# Scope any command to a specific model config or domain
python3 src/classifications/run.py --push    --model gpt-5.2__high__cot
python3 src/classifications/run.py --collect --model gpt-5.2__high__cot --part sdg_a
python3 src/classifications/run.py --check   --batch batch_abc123
```

### Output

```
data/classifications/
├── batches/
│   └── <model_config>/
│       ├── sdg_a_part0001.jsonl
│       ├── sdg_a_part0002.jsonl   # split when > 50M tokens or 10k requests
│       ├── sdg_b_part0001.jsonl
│       ├── sdg_c_part0001.jsonl
│       └── tech_part0001.jsonl
└── results/
    └── <model_config>/
        ├── sdg_a_results.json     # { "<domain>||<global_id>": {"<pattern>": "symbolic"|"substantive"} }
        ├── sdg_a_errors.json      # rows that failed to parse or returned HTTP errors
        ├── sdg_b_results.json
        ├── sdg_c_results.json
        └── tech_results.json
```


---


## Stage 2 — Fix Malformed Results (`src/postprocessing/fix_results.py`)

Detects and fixes rows where the model returned column-level labels instead of pattern-level labels. This occurs occasionally when the model aggregates all patterns in a column into a single key (e.g. `{"hits_big_data_blockchain": "substantive"}` instead of `{"\\bblockchain\\b": "substantive", "big\\ data": "substantive"}`).

### Anomaly types

| Type | Description | Fixable? |
|------|-------------|----------|
| Column-keyed rows | Result key is a DB column name instead of a regex pattern | Yes — expands the label to all patterns in that column for that row |
| Missing rows | A passage with hits has no result entry at all | No — must be resubmitted |

### Commands

```bash
# Report all anomalies across all model configs (no writes)
python3 src/postprocessing/fix_results.py --analyze

# Preview fixes without writing
python3 src/postprocessing/fix_results.py --fix --dry-run

# Apply fixes in-place (backs up originals as *_backup_<timestamp>.json before overwriting)
python3 src/postprocessing/fix_results.py --fix

# Scope to one model config
python3 src/postprocessing/fix_results.py --fix --model gpt-5.2__low__zero_shot
```


---


## Stage 3 — Build Classified DuckDBs (`src/postprocessing/generate_results_db.py`)

Merges result JSONs back into DuckDB files. Uses the source DuckDBs to determine which regex patterns belong to which column for each row, then writes new databases with updated labels.

### Commands

```bash
# Build classified DBs for all model configs found under data/classifications/results/
python3 src/postprocessing/generate_results_db.py

# One model config only
python3 src/postprocessing/generate_results_db.py --model gpt-5.2__high__cot

# Dry run — print coverage stats without writing
python3 src/postprocessing/generate_results_db.py --dry-run
```

### Output

```
data/dbs/
└── <model_config>/
    ├── sdg_hits_classified.duckdb
    └── tech_hits_classified.duckdb
```

Hit columns store classified JSON dicts:
```
hits_sdg6  →  {"\\bwater\\s+short\\w*\\b": "symbolic", "\\bsewage\\s+treat\\w*\\b": "substantive"}
hits_ai_ml →  {"\\bmachine\\s+learn\\w*\\b": "substantive"}
```

An empty `{}` means no keyword from that category matched for that passage.

### Hit Column Schema

**`sdg_hits_classified`**

| Column | Type | Description |
|--------|------|-------------|
| `global_id` | VARCHAR | Unique passage ID: `YYYY<company_normalized><sentence_id>` |
| `passage` | VARCHAR | Text of the passage |
| `company` | VARCHAR | Company name |
| `year` | VARCHAR | Report year |
| `language` | VARCHAR | `en`, `de`, or `unknown` |
| `hits_sdg1` – `hits_sdg17` | VARCHAR | JSON: `{"<pattern>": "symbolic"\|"substantive"}` per SDG |

**`tech_hits_classified`:** same base columns plus:

| Column | Type | Description |
|--------|------|-------------|
| `hits_ai_ml` | VARCHAR | AI & machine learning patterns |
| `hits_cloud_computing` | VARCHAR | Cloud computing patterns |
| `hits_big_data_blockchain` | VARCHAR | Big data & blockchain patterns |
| `hits_applications_practice` | VARCHAR | Practical application patterns |


---


## Stage 4 — Generate Summary CSV (`src/postprocessing/generate_sym_sub_results.py`)

Aggregates classified DuckDBs into one CSV per model config. Each row is one company×year combination; columns are passage-level symbolic/substantive counts for every SDG and tech category.

**Counting rule:** for each (column × label) pair, counts the number of passages where that column has at least one pattern with that label. A passage with both symbolic and substantive patterns in the same column increments both counters independently.

### Commands

```bash
# Generate CSVs for all model configs under data/dbs/
python3 src/postprocessing/generate_sym_sub_results.py

# One model config only
python3 src/postprocessing/generate_sym_sub_results.py --model gpt-5.2__high__cot

# Point directly at specific DB files
python3 src/postprocessing/generate_sym_sub_results.py \
  --sdg-db  data/backup_dbs/sdg_hits_classified.duckdb \
  --tech-db data/backup_dbs/tech_hits_classified.duckdb \
  --model   backup
```

### Output

`data/exports/<model_config>_sym_sub.csv` — 44 columns:

```
company, year,
sdg1_symbolic,  sdg1_substantive,
sdg2_symbolic,  sdg2_substantive,
...
sdg17_symbolic, sdg17_substantive,
ai_ml_symbolic,                 ai_ml_substantive,
cloud_computing_symbolic,       cloud_computing_substantive,
big_data_blockchain_symbolic,   big_data_blockchain_substantive,
applications_practice_symbolic, applications_practice_substantive
```

Each value is a non-negative integer (passage count for that company×year).


---


## Analysis (`src/analysis/`)

Exploratory scripts for inspecting DuckDB contents. Do not modify any data.

| Script | Description |
|--------|-------------|
| `explore_dbs.py` | Prints column names, row counts, company/year/language breakdown, classification counts per category, and sample passages from `sdg_hits_classified` and `tech_hits_classified` |
| `explore_batches.py` | Inspects raw JSONL batch files — useful for debugging batch content before submission |

```bash
python3 src/analysis/explore_dbs.py
```

By default reads from `data/dbs/`. Edit `DB_DIR` at the top of the script to point at a model config subfolder (e.g. `data/dbs/gpt-5.2__high__cot`).


---


## Prompt Evaluation (`tests/prompts/`)

Evaluates all prompt strategies against a hand-labeled ground-truth set **before** committing to a full classification run. Results inform which model config and prompt type to use in production.

### Ground truth

`tests/prompts/passage_keyword_truth.csv` — manually labeled sample with `symbolic` / `substantive` ground truth labels.

`tests/prompts/sample_1000.csv` — larger unlabeled sample used for qualitative inspection.

### Evaluated combinations

| Axis | Values |
|------|--------|
| Models | `gpt-5.2`, `gpt-4o` |
| Reasoning effort (gpt-5.2 only) | `low`, `medium`, `high`, `xhigh` |
| Prompt types | `zero_shot`, `few_shot`, `cot`, `tot` |

**gpt-5.2:** 4 prompt types × 4 reasoning modes = 16 batches
**gpt-4o:** 4 prompt types = 4 batches

### Prompt types

| Type | Description |
|------|-------------|
| `zero_shot` | Definitions only — no examples |
| `few_shot` | Labeled examples of symbolic and substantive passages included in the prompt |
| `cot` | Chain-of-Thought — model reasons step-by-step before classifying |
| `tot` | Tree-of-Thought — model explores multiple reasoning paths before deciding |

### Scripts

| Script | Description |
|--------|-------------|
| `prompts.py` | Prompt definitions for all four strategies |
| `run_batches.py` | Build JSONL files, submit to OpenAI Batch API, poll status, collect results |
| `analysis.py` | Compute accuracy vs ground truth → `results/accuracy_summary.csv` |
| `build_majority_vote.py` | Aggregate predictions across multiple runs via majority vote |

### Commands

```bash
cd tests/prompts

# Build JSONL batch files for all prompt × reasoning mode combos
python3 run_batches.py --create

# Submit all batches
python3 run_batches.py --push

# Submit only one prompt type (all reasoning modes)
python3 run_batches.py --push zero_shot

# Submit a single exact combination
python3 run_batches.py --push cot__high

# Check status of submitted batches
python3 run_batches.py --check

# Download results → per-batch CSVs + costs.json
python3 run_batches.py --poll

# Compute accuracy → results/accuracy_summary.csv
python3 run_batches.py --analyse
```

### Output

```
tests/prompts/
├── batches/                             # JSONL files for gpt-5.2
├── batches_4o/                          # JSONL files for gpt-4o
├── results/
│   ├── results_<prompt>__<mode>.csv     # Per-batch predictions
│   ├── costs.json                       # Token usage and cost per batch
│   └── accuracy_summary.csv            # Accuracy across all batches
└── results_4o/                          # Same structure for gpt-4o
```


---


## Data Directory

```
data/
├── dbs/
│   ├── sdg_hits.duckdb                     # SOURCE — SDG passages + keyword hits (required)
│   ├── tech_hits.duckdb                    # SOURCE — Tech passages + keyword hits (required)
│   └── <model_config>/                     # Generated by Stage 3, one folder per model config
│       ├── sdg_hits_classified.duckdb
│       └── tech_hits_classified.duckdb
│
├── classifications/
│   ├── batches/
│   │   └── <model_config>/                 # JSONL part files sent to OpenAI (Stage 1)
│   │       ├── sdg_a_part0001.jsonl
│   │       ├── sdg_b_part0001.jsonl
│   │       ├── sdg_c_part0001.jsonl
│   │       └── tech_part0001.jsonl
│   └── results/
│       └── <model_config>/                 # Collected result JSONs (Stage 1 output)
│           ├── sdg_a_results.json
│           ├── sdg_a_errors.json
│           ├── sdg_b_results.json
│           ├── sdg_c_results.json
│           └── tech_results.json
│
├── exports/
│   └── <model_config>_sym_sub.csv          # Final aggregated CSV (Stage 4 output)
│
└── backup_dbs/
    ├── sdg_hits_classified.duckdb          # Reference classified DB from a prior run
    └── tech_hits_classified.duckdb
```

**Required before running Stage 1:**
- `data/dbs/sdg_hits.duckdb`
- `data/dbs/tech_hits.duckdb`
- `src/classifications/config.json` (gitignored — create manually)

Everything else is generated by the pipeline.

**Gitignored runtime files (not committed):**
- `src/classifications/config.json` — model configs and accumulated batch ID state
- `data/classifications/queue.json` — push loop queue state


---


## Full Pipeline — Quick Reference

```bash
# 0. One-time setup
source .venv/bin/activate && source prepare_path.sh

# 1. (Optional) Evaluate prompts on ground truth before full run
cd tests/prompts && python3 run_batches.py --create && python3 run_batches.py --push
python3 run_batches.py --poll && python3 run_batches.py --analyse
cd ../..

# 2. Build JSONL batches from source DuckDBs
python3 src/classifications/run.py --build

# 3. Submit to OpenAI and collect results autonomously
python3 src/classifications/run.py --push

# 4. Detect and fix malformed result rows
python3 src/postprocessing/fix_results.py --analyze
python3 src/postprocessing/fix_results.py --fix

# 5. Build classified DuckDBs
python3 src/postprocessing/generate_results_db.py

# 6. Generate final company×year CSVs
python3 src/postprocessing/generate_sym_sub_results.py
```

Final output: `data/exports/<model_config>_sym_sub.csv` — one file per model config, ready for analysis.
