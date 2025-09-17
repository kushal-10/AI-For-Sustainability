#!/usr/bin/env python3
"""
python3 src/filtering/sdg_filter.py \
  --root data/jsons \
  --kw_en kw_data/keywords_sdg.json \
  --kw_de kw_data/keywords_sdg_de.json \
  --out_db data/dbs/sdg_hits.duckdb \
  --table sdg_hits \
  --wildcard
"""

from typing import Any, Dict, List, Tuple

try:
    from base_filter import Filter, build_cli, _json_str  # type: ignore
except Exception:
    from .base_filter import Filter, build_cli, _json_str  # type: ignore


class SDGFilter(Filter):
    _CATS = [f"sdg{i}" for i in range(1, 18)]  # sdg1..sdg17

    def category_names(self) -> List[str]:
        return list(self._CATS)

    def table_name(self) -> str:
        return "sdg_hits"

    def extra_columns(self) -> List[Tuple[str, str]]:
        return []  # BASE + hits_sdg* only

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
        return base + hit_cols


def main():
    ap = build_cli("Filter passages by SDG keywords (17 categories) and write to DuckDB.")
    args = ap.parse_args()

    SDGFilter(
        root_path=args.root,
        kw_en_path=args.kw_en,
        kw_de_path=args.kw_de,
        out_db=args.out_db,
        table=args.table,
        star_is_wildcard=args.wildcard,  # enable wildcard behavior if requested
    ).run()


if __name__ == "__main__":
    main()
