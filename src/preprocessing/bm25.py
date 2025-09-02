import pandas as pd

# Pick one report
base = "data/sample_texts/sgl carbon/2023"


sentences = pd.read_parquet(f"{base}/sentences.parquet")
matches   = pd.read_parquet(f"{base}/matches.parquet")

total = len(sentences)
matched = matches['sent_id'].nunique()   # how many unique sentences had ≥1 hit
print(f"Sentences total: {total}, matched: {matched}, ratio: {matched/total:.1%}")
