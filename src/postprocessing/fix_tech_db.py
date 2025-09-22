import duckdb

con_sdg = duckdb.connect("data/dbs/sdg_hits.duckdb")
df = con_sdg.execute("SELECT * FROM sdg_hits").fetchdf()

print(len(df))
print(df.tail())   # prints in a table-like format
print(df.head())

con_tech = duckdb.connect("data/dbs/tech_hits.duckdb")
df = con_tech.execute("SELECT * FROM tech_hits").fetchdf()
print(df.columns)
print(len(df))
print(df.tail())   # prints in a table-like format
print(df.head())