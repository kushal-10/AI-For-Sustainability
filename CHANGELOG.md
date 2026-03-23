# Changelog

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