import os

import pandas as pd
from tqdm import tqdm


base_dir = os.path.join("data", "scores_csv")

csvs = []
for dirname, _, filenames in os.walk(base_dir):
    for filename in filenames:
        if filename.endswith(".csv"):
            csvs.append(os.path.join(dirname, filename))

T = 0.3

# total_sdgs = 0
# total_ais = 0
filtered_sents = 0
total_sents = 0

for csv in tqdm(csvs):
    df = pd.read_csv(csv)
    for i in range(len(df)):
        row_data = df.iloc[i]
        l = len(row_data)
        total_sents += l
        for j in range(1, l):
            curr_val = row_data.iloc[j]
            if curr_val >= T:
                filtered_sents += 1
                break
                # if j <= 17:
                #     total_sdgs += 1
                # else:
                #     total_ais += 1

# print(total_sdgs, total_ais) # 3935049 319298
print(total_sents, filtered_sents) # 110,707,703 || 1,747,704










