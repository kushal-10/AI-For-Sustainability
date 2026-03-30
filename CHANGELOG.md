# Changelog

## [v2.2.0] - 2026-03-30
### Changed
- New `src/classifications/` module replaces `src_legacy/gpt_classifier/` with autonomous push loop (submit → poll → collect → repeat per batch part)
- Batch files split into mini-parts based on Tier 1 token/request limits (50M tokens, 10k requests per part)
- `run.py` unified CLI for `--build`, `--push`, `--check`, `--collect` with optional `--model` and `--part` filters
- Duplicate batch object handling added

### Removed
- `src_legacy/` directory fully removed (preprocessing, filtering, gpt_classifier, postprocessing, analysis, utils)
- Legacy keyword JSON files (`kw_data/keywords_*.json`) removed
- `config.json` and `queue.json` removed from git tracking (batch runtime state, gitignored)

## [v2.1.0] - 2026-03-28
### Added
- gpt-5.2 classification results across low/medium/high/xhigh reasoning modes
- Prompt evaluation results in `tests/prompts/results/`

## [v2.0.0] - 2026-03-23
### Changed
- Pipeline now starts from pre-computed SDG and Tech hits, skipping PDF preprocessing
- New prompt variants: zero_shot, few_shot, CoT, ToT
- Evaluation against gpt-5.2 (low/medium/high/xhigh) and gpt-4o

### Removed
- PDF ingestion and DuckDB preprocessing scripts (see v1.0.0 / legacy branch)

## [v1.0.0] - legacy branch
### Features
- Raw PDF → DuckDB preprocessing pipeline
- SDG and Tech keyword extraction
- Initial classification scripts