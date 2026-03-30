# AI-For-Sustainability
End-to-end LLM pipeline to quantify the impact of AI towards SDGs in German Firms.


## Setup

Clone this repo and install requirements

```bash
git clone https://github.com/kushal-10/AI-For-Sustainability.git
cd AI-For-Sustainability
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
source prepare_path.sh   # sets PYTHONPATH to include repo root
export OPENAI_API_KEY=your_key
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
python3 src/classifications/run.py --push    --model gpt-4o__tot
python3 src/classifications/run.py --collect --model gpt-4o__tot --part sdg_a
python3 src/classifications/run.py --check   --batch batch_abc123
```

Configuration is managed in `src/classifications/config.json` (gitignored — local only).

### Batch Architecture

- Batches are split into mini-parts based on Tier 1 limits: **50M tokens** and **10k requests** per part file
- Submitted to OpenAI Batch API (`gpt-4.1-mini` or configured model, 24h window)
- Results collected to `data/classifications/results_v2/`

## Prompt Evaluation (tests/prompts/)

Four prompt strategies are evaluated: `zero_shot`, `few_shot`, `CoT`, `ToT`.

```bash
python3 tests/prompts/run_batches.py --create
python3 tests/prompts/run_batches.py --push [prompt_name]
python3 tests/prompts/run_batches.py --poll
python3 tests/prompts/analysis.py
```

Ground truth: `tests/prompts/passage_keyword_truth.csv`
