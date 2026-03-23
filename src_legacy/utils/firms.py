import pandas as pd

source_csv = "src/utils/results/token_counts.csv"

df = pd.read_csv(source_csv)

print(len(df)) # 1420 text files
print(len(df["company"].unique())) # 153 firms