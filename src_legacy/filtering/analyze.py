import duckdb
import pandas as pd
import numpy as np
import json

con_tech = duckdb.connect("data/dbs/tech_hits.duckdb")
con_sdgs = duckdb.connect("data/dbs/sdg_hits.duckdb")

df_sdgs = con_sdgs.execute("SELECT * FROM sdg_hits").fetchdf()
df_tech = con_tech.execute("SELECT * FROM tech_hits").fetchdf()


cols_sdgs = ['hits_sdg1',
       'hits_sdg2', 'hits_sdg3', 'hits_sdg4', 'hits_sdg5', 'hits_sdg6',
       'hits_sdg7', 'hits_sdg8', 'hits_sdg9', 'hits_sdg10', 'hits_sdg11',
       'hits_sdg12', 'hits_sdg13', 'hits_sdg14', 'hits_sdg15', 'hits_sdg16',
       'hits_sdg17']

cols_tech = ['hits_ai_ml',
       'hits_cloud_computing', 'hits_big_data_blockchain',
       'hits_applications_practice']

def passage_stats(df, cols):
    stats = []
    for col in cols:
        lengths = []
        for _, row in df.iterrows():
            hits = row[col]
            if hits != "[]" and isinstance(row["passage"], str):
                token_count = len(row["passage"].split())
                lengths.append(token_count)
        if lengths:  # only compute if we have hits
            stats.append({
                "column": col,
                "count": len(lengths),
                "avg_tokens": np.mean(lengths),
                "min_tokens": np.min(lengths),
                "max_tokens": np.max(lengths),
                "median_tokens": np.median(lengths),
            })
    return pd.DataFrame(stats)

# compute for SDGs and Tech
sdg_stats = passage_stats(df_sdgs, cols_sdgs)
tech_stats = passage_stats(df_tech, cols_tech)

print("=== SDG Passage Stats ===")
print(sdg_stats)

print("\n=== Tech Passage Stats ===")
print(tech_stats)

"""
31476 - SDGs
13670 - Tech
1165 common between Tech and SDGs

{'hits_ai_ml': 4230, 
'hits_cloud_computing': 2435, 
'hits_big_data_blockchain': 2167, 
'hits_applications_practice': 7084}

{'hits_sdg1': 30, 
'hits_sdg2': 59, 
'hits_sdg3': 100, 
'hits_sdg4': 149, 
'hits_sdg5': 190, 
'hits_sdg6': 807, 
'hits_sdg7': 1634, 
'hits_sdg8': 4328, 
'hits_sdg9': 1674, 
'hits_sdg10': 316, 
'hits_sdg11': 1239,
'hits_sdg12': 6780,
'hits_sdg13': 12850,
'hits_sdg14': 127, 
'hits_sdg15': 35, 
'hits_sdg16': 6647, 
'hits_sdg17': 1156}

Passage Analysis

=== SDG Passage Stats ===
        column  count  avg_tokens  min_tokens  max_tokens  median_tokens
0    hits_sdg1     30  288.033333          56         374          315.5
1    hits_sdg2     59  303.423729          56         388          338.0
2    hits_sdg3    100  298.860000          83         385          330.0
3    hits_sdg4    149  287.812081          42         395          321.0
4    hits_sdg5    190  292.442105          13         398          334.0
5    hits_sdg6    807  283.724907          20         402          312.0
6    hits_sdg7   1634  295.241126          17         411          321.0
7    hits_sdg8   4328  286.529113           9         405          315.0
8    hits_sdg9   1674  284.166667          17         405          314.0
9   hits_sdg10    316  288.876582          16         407          317.0
10  hits_sdg11   1239  299.920904          20         403          330.0
11  hits_sdg12   6780  290.834661           8         410          321.0
12  hits_sdg13  12850  281.321089           4         418          313.0
13  hits_sdg14    127  292.070866          12         376          323.0
14  hits_sdg15     35  316.971429         122         379          331.0
15  hits_sdg16   6647  290.906274           5         420          325.0
16  hits_sdg17   1156  291.598616          18         423          324.0

=== Tech Passage Stats ===
                       column  count  avg_tokens  min_tokens  max_tokens  median_tokens
0                  hits_ai_ml   4230  291.069031          10         405          325.0
1        hits_cloud_computing   2435  275.602875           9         404          306.0
2    hits_big_data_blockchain   2167  284.105215           2         412          320.0
3  hits_applications_practice   7084  272.127894           7         411          305.0

TODO: Remove passages < 10 tokens
"""