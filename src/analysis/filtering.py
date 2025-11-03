import duckdb
import ast
import pandas as pd

# Load tech hits
tech_con = duckdb.connect('data/dbs/tech_hits.duckdb')
tech_df = tech_con.execute("SELECT * FROM tech_hits").fetchdf()
tech_con.close()

# Load SDG hits
sdg_con = duckdb.connect('data/dbs/sdg_hits.duckdb')
sdg_df = sdg_con.execute("SELECT * FROM sdg_hits").fetchdf()
sdg_con.close()

passage_metadata = {}

tech_hits_cols = ["hits_ai_ml",
                  "hits_cloud_computing",
                  "hits_big_data_blockchain",
                  "hits_applications_practice"
                  ]

base_counters = [0,0,0,0]
total_passages = set()
for i in range(len(tech_df)):
    row_data = tech_df.iloc[i]
    passage_id = row_data['global_id']
    for j in range(len(tech_hits_cols)):
        kw_col = tech_hits_cols[j]
        row_data_value = ast.literal_eval(row_data[kw_col])
        if row_data_value != []:
            base_counters[j] += 1
            if row_data['global_id'] not in total_passages:
                total_passages.add(row_data['global_id'])

print(f"\nCounts for each set of tech hits: {tech_hits_cols} are - {base_counters}")
print(f"\nTotal passages that mention Tech related keywords: {len(total_passages)}")

sdg_hits_cols = [f"hits_sdg{i+1}" for i in range(17)]
base_sdg_counters = [0]*17
sdg_passages = set()
for i in range(len(sdg_df)):
    row_data = sdg_df.iloc[i]
    for j in range(len(sdg_hits_cols)):
        kw_col = sdg_hits_cols[j]
        row_data_value = ast.literal_eval(row_data[kw_col])
        if row_data_value != []:
            base_sdg_counters[j] += 1
            if row_data['global_id'] not in sdg_passages:
                sdg_passages.add(row_data['global_id'])

print(f"\nCounts for each SDG related keywords in order: {base_sdg_counters}")

print(f"\nTotal passages that mention SDG related keywords: {len(sdg_passages)}")

commons_passages = total_passages.intersection(sdg_passages)

print(f"\nTotal common passages between the two sets: {len(commons_passages)}")

"""

Counts for each set of tech hits: ['hits_ai_ml', 'hits_cloud_computing', 'hits_big_data_blockchain', 'hits_applications_practice'] are - [4230, 2435, 2167, 7084]

Total passages that mention Tech related keywords: 13670

Counts for each SDG related keywords in order: [30, 59, 100, 149, 190, 808, 1634, 4329, 1674, 317, 1239, 6781, 12850, 127, 35, 6648, 1156]

Total passages that mention SDG related keywords: 31476

Total common passages between the two sets: 1165

"""