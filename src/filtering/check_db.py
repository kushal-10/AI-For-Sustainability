import duckdb

con = duckdb.connect("data/dbs/tech_hits.duckdb")
df = con.execute("SELECT * FROM tech_hits LIMIT 20").fetchdf()

print(df)   # prints in a table-like format