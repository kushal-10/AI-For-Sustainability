import duckdb

con = duckdb.connect("data/dbs/sdg_hits.duckdb")
df = con.execute("SELECT * FROM sdg_hits").fetchdf()

print(len(df))
print(df.tail())   # prints in a table-like format
print(df.head())