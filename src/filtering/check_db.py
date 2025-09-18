import duckdb

con = duckdb.connect("data/dbs/sdg_hits.duckdb")
df = con.execute("SELECT * FROM sdg_hits LIMIT 20").fetchdf()

print(df)   # prints in a table-like format