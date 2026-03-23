import duckdb, json
import pandas as pd

def get_df(db_path="data/dbs/sdg_hits.duckdb", table="sdg_hits") -> pd.DataFrame:
    con = duckdb.connect(db_path)
    df = con.execute(f"SELECT * FROM {table}").fetchdf()
    con.close()
    return df

sdg_df = get_df()

# Auto-detect per-SDG hit columns (works for hits_sdg1 ... hits_sdg17)
hit_cols = [c for c in sdg_df.columns if c.lower().startswith("hits_sdg")]
if not hit_cols:
    raise SystemExit("No per-SDG hit columns found (hits_sdg1..hits_sdg17).")

def parse_json_list(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    try:
        return json.loads(x)
    except Exception:
        return []

# Parse all hit columns into lists and compute counts
for c in hit_cols:
    sdg_df[c + "__list"] = sdg_df[c].apply(parse_json_list)
    sdg_df[c + "__count"] = sdg_df[c + "__list"].apply(lambda lst: len(lst))

# Total # of matched terms (across all SDGs) and # of SDGs that got at least one hit
count_cols = [c + "__count" for c in hit_cols]
sdg_df["total_hit_terms"] = sdg_df[count_cols].sum(axis=1)
sdg_df["num_sdgs_hit"] = (sdg_df[count_cols] > 0).sum(axis=1)

# Keep rows with multiple hits: either ≥2 total terms or ≥2 SDGs
multi_hits = sdg_df[(sdg_df["total_hit_terms"] >= 2) | (sdg_df["num_sdgs_hit"] >= 2)]

# Shuffle for variety, then take 10
examples = multi_hits.sample(n=min(10, len(multi_hits)), random_state=42)

# Pretty print
for i, row in examples.iterrows():
    # Build compact dict of non-empty hits per SDG
    hits_dict = {}
    for c in hit_cols:
        lst = row[c + "__list"]
        if lst:
            hits_dict[c] = lst
    print("=" * 80)
    print(f"global_id: {row.get('global_id')}")
    print(f"company  : {row.get('company')} | year: {row.get('year')} | lang: {row.get('language')}")
    print(f"SDGs hit : {row['num_sdgs_hit']} | total terms: {row['total_hit_terms']}")
    print("- passage --------------------------------------------------------------")
    print((row.get('passage') or "").strip())
    print("- hits --------------------------------------------------------------")
    # Show a compact JSON-style view
    # If it's too long, trim each list to first 5 items for readability
    trimmed = {k: (v if len(v) <= 5 else v[:5] + ["…+%d more" % (len(v)-5)]) for k, v in hits_dict.items()}
    print(json.dumps(trimmed, ensure_ascii=False, indent=2))

print("\nShown:", len(examples), "rows. Total candidates with multiple hits:", len(multi_hits))
