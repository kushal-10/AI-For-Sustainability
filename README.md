# AI-For-Sustainability
End-to-end LLM pipeline to classify corporate sustainability disclosures (passages from German firm reports) as **Symbolic** vs **Substantive** mentions of SDGs and AI/Tech topics, grounded in legitimacy theory (Ashforth & Gibbs 1990, Suchman 1995).

**Dataset:** 153 German companies, 1,420 sustainability reports (multi-year). Languages: German (`de`) and English (`en`).


## Setup

Clone this repo and install requirements:

```bash
git clone https://github.com/kushal-10/AI-For-Sustainability.git
cd AI-For-Sustainability
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
source prepare_path.sh   # sets PYTHONPATH to include repo root
export OPENAI_API_KEY=your_key # Or alternatively, set up in .env
```


## Versions

| Version | Branch/Tag | Description |
|---------|-----------|-------------|
| v1.0.0 | `v1-full-pipeline` | Full pipeline from raw PDFs → Filtering → DuckDB → SDG/Tech classification |
| v2.0.0 | `v2.0.0` | Starts from SDG/Tech hits — new prompts, evaluation, analysis |
| v2.1.0 | `v2.1.0` | gpt-5.2 classification results and prompt evaluation |
| v2.2.0 | `main` | New `src/classifications/` module, autonomous batch loop, legacy code removed |

> To reproduce v1.0.0 preprocessing from scratch, checkout the `v1-full-pipeline` branch.


## Classification into Symbolic/Substantive

Each matched keyword passage is classified as either:
- **Symbolic** — vague/aspirational, policy/compliance-only, no concrete actions or KPIs
- **Substantive** — concrete implementation, projects, budgets, timelines, KPIs, quantified impact


### Running Classification

```bash
# Build JSONL batch files from DuckDB
python3 src/classifications/run.py --build

# Submit batches (autonomous loop: submit → poll → collect → repeat)
python3 src/classifications/run.py --push

# Check batch status
python3 src/classifications/run.py --check

# Collect completed results
python3 src/classifications/run.py --collect
```

Filter by model or domain:
```bash
# Push only batches for a specific model config
python3 src/classifications/run.py --push --model gpt-5.2__tot

# Collect results for a specific model + domain part
python3 src/classifications/run.py --collect --model gpt-5.2__tot --part sdg_a

# Check status of a specific batch ID
python3 src/classifications/run.py --check --batch batch_abc123
```

Configuration is managed in `src/classifications/config.json` (gitignored — local only).

### Batch Architecture

- Batches are split into part files based on Tier 1 limits: **50M tokens** and **10k requests** per part
- Submitted to OpenAI Batch API with a **24h completion window**
- Results collected to `data/classifications/results_v2/<model>/<domain>/`


## Prompt Evaluation (`tests/prompts/`)

Evaluates prompt strategies against a labeled ground-truth set (`tests/prompts/passage_keyword_truth.csv`).

### Model

| Model | Notes |
|-------|-------|
| `gpt-5.2` | Used for all prompt evaluation runs |

### Reasoning Modes

| Mode | Description |
|------|-------------|
| `low` | Minimal reasoning budget |
| `medium` | Moderate reasoning |
| `high` | Extended reasoning |
| `xhigh` | Maximum reasoning budget |

### Prompt Types

| Type | Description |
|------|-------------|
| `zero_shot` | Direct classification with definitions only — no examples |
| `few_shot` | Includes labeled examples of symbolic and substantive passages |
| `cot` | Chain-of-Thought — model reasons step-by-step before classifying |
| `tot` | Tree-of-Thought — model explores multiple reasoning paths before deciding |

All combinations are evaluated: **4 prompt types × 4 reasoning modes = 16 batches total**.

### Running Evaluation

```bash
# Build JSONL batch files for all 16 prompt × reasoning mode combos
python3 tests/prompts/run_batches.py --create

# Submit all 16 batches to OpenAI Batch API
python3 tests/prompts/run_batches.py --push

# Submit only one prompt type (all reasoning modes)
python3 tests/prompts/run_batches.py --push zero_shot

# Submit a single exact combo
python3 tests/prompts/run_batches.py --push cot__high

# Check status of all submitted batches
python3 tests/prompts/run_batches.py --check

# Download results and save per-batch CSVs + costs.json
python3 tests/prompts/run_batches.py --poll

# Compute accuracy vs ground truth → results/result.csv
python3 tests/prompts/run_batches.py --analyse
```

Results are saved to `tests/prompts/results/`:
- `results_{prompt}__{reasoning}.csv` — per-batch predictions
- `costs.json` — token usage and cost per batch
- `result.csv` — accuracy summary across all batches
