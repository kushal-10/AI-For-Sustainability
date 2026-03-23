import duckdb, json
import pandas as pd

def get_df(db_path="data/dbs/tech_hits.duckdb", table="tech_hits") -> pd.DataFrame:
    con = duckdb.connect(db_path)
    df = con.execute(f"SELECT * FROM {table}").fetchdf()
    con.close()
    return df

tech_df = get_df()

# Column sanity
col = None
for c in tech_df.columns:
    if c.lower() == "hits_ai_ml":
        col = c
        break
if col is None:
    raise SystemExit("Could not find 'hits_ai_ml' column in tech_hits table.")

# Parse JSON list
def parse_json_list(x):
    if pd.isna(x): return []
    if isinstance(x, list): return x
    try: return json.loads(x)
    except Exception: return []

tech_df["ai_ml_hits_list"] = tech_df[col].apply(parse_json_list)
tech_df["ai_ml_hits_count"] = tech_df["ai_ml_hits_list"].apply(len)

# Prefer rows with multiple AI/ML terms; if fewer than 10, pad with singles
multi = tech_df[tech_df["ai_ml_hits_count"] >= 2]
single = tech_df[(tech_df["ai_ml_hits_count"] == 1)]

n_multi = min(10, len(multi))
examples = multi.sample(n=n_multi, random_state=42) if n_multi > 0 else pd.DataFrame(columns=tech_df.columns)

if len(examples) < 10 and len(single) > 0:
    need = 10 - len(examples)
    examples = pd.concat([examples, single.sample(n=min(need, len(single)), random_state=42)], ignore_index=True)

# Print
for _, row in examples.iterrows():
    hits = row["ai_ml_hits_list"]
    # trim long lists for readability
    show_hits = hits if len(hits) <= 8 else hits[:8] + [f"…+{len(hits)-8} more"]
    print("=" * 80)
    print(f"global_id: {row.get('global_id')}")
    print(f"company  : {row.get('company')} | year: {row.get('year')} | lang: {row.get('language')}")
    print(f"AI/ML terms matched: {len(hits)}")
    print("- passage --------------------------------------------------------------")
    print((row.get('passage') or "").strip())
    print("- ai_ml hits -----------------------------------------------------------")
    print(json.dumps(show_hits, ensure_ascii=False))

print("\nShown:", len(examples), "rows (", len(multi), "with ≥2 AI/ML terms available).")

"""
Siemens Healthineers 2023

.Further investments into efficiency measures, and the use of new technologies such as machine learning, digital twins 
and artificial intelligence, could potentially drive additional improvements in our processes and cost structures.Increased 
harmonization, collaboration and transparency throughout the entire organization could create synergies, lead to faster decision-
making processes and reduce redundant efforts.

Mentions it in a symbolic way
"""