
from typing import Any, Dict, List, Tuple

# Flexible import whether run as module or script
try:
    from base_filter import Filter, build_cli, _json_str  # type: ignore
except Exception:
    from .base_filter import Filter, build_cli, _json_str  # type: ignore

class TechFilter(Filter):
    _CATS = [
        "ai_ml",
        "cloud_computing",
        "big_data_blockchain",
        "applications_practice",
    ]

    def category_names(self) -> List[str]:
        return list(self._CATS)

    def table_name(self) -> str:
        return "tech_hits"

    def extra_columns(self) -> List[Tuple[str, str]]:
        # No extra columns beyond BASE + hits_* for this use case.
        return []

    def make_row(
        self,
        *,
        global_id: str,
        passage: str,
        company: str,
        year: str,
        language: str,
        hits_by_cat: Dict[str, List[str]],
    ) -> Tuple[Any, ...]:
        base = (global_id, passage, company, year, language)
        hit_cols = tuple(_json_str(hits_by_cat.get(cat, [])) for cat in self.category_names())
        return base + hit_cols  # no extras

def main():
    ap = build_cli("Filter passages by tech keywords (EN/DE) and write to DuckDB (progress bar, errors only).")
    args = ap.parse_args()

    TechFilter(
        root_path=args.root,
        kw_en_path=args.kw_en,
        kw_de_path=args.kw_de,
        out_db=args.out_db,
        table=args.table,
    ).run()

if __name__ == "__main__":
    main()

"""
python3 src/filtering/tech_filter.py \
  --root data/jsons \
  --kw_en kw_data/keywords_tech.json \
  --kw_de kw_data/keywords_tech_de.json \
  --out_db data/dbs/tech_hits.duckdb \
  --table tech_hits
"""