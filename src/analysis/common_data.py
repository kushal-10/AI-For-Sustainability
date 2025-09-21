import duckdb

def get_df(db_path="data/dbs/sdg_hits.duckdb", db_name="sdg_hits"):
    con = duckdb.connect(db_path)
    df = con.execute(f"SELECT * FROM {db_name}").fetchdf()
    con.close()
    return df

# Load both tables
sdg_df = get_df(db_path="data/dbs/sdg_hits.duckdb", db_name="sdg_hits")
tech_df = get_df(db_path="data/dbs/tech_hits.duckdb", db_name="tech_hits")

# Counts
print("SDG rows :", len(sdg_df))
print("Tech rows:", len(tech_df))

# Common global_ids
sdg_ids = set(sdg_df["global_id"])
tech_ids = set(tech_df["global_id"])
common_ids = sdg_ids & tech_ids

print("Common global_ids:", len(common_ids))

"""
SDG rows : 31480
Tech rows: 13670
Common global_ids: 1165
"""